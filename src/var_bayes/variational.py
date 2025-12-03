import numpy as np
from scipy.interpolate import CubicSpline
from src.numerics.utilities import chol_inv


class VarGP(object):
    """
    Variational Gaussian Process Approximation class.
    """

    __slots__ = ("model", "fwd_ode", "bwd_ode", "kl0", "likelihood",
                 "obs_y", "obs_t", "dt", "dim_n", "dim_d", "dim_tot",
                 "output")

    def __init__(self, model, m0, s0, fwd_ode, bwd_ode, likelihood,
                 kl0, obs_y, obs_t):
        """
        Default constructor of VGPA object.

        :param model: objects represents the dynamical system.

        :param m0: initial marginal mean m(t=0). For the moment
        this is kept fixed, but it can also be optimized.

        :param s0: initial marginal co-variance s(t=0). For the
        moment this is kept fixed, but it can also be optimized.

        :param fwd_ode: forward ode solver.

        :param bwd_ode: backward ode solver.

        :param likelihood: likelihood object.

        :param kl0: Kullback-Liebler at initial moment KL(t=0).

        :param obs_y: observation values (including noise).

        :param obs_t: observation times (discrete).
        """

        # Stochastic model.
        self.model = model

        # Forward / backward integrators.
        self.fwd_ode = fwd_ode
        self.bwd_ode = bwd_ode

        # Prior / likelihood functions.
        self.kl0 = kl0
        self.likelihood = likelihood

        # Observations / times.
        self.obs_y = obs_y
        self.obs_t = obs_t

        # Extract auxiliary variables.
        self.dt = self.model.time_step

        # Get the dimensions.
        if self.model.single_dim:
            self.dim_n, self.dim_d = self.model.sample_path.size, 1
        else:
            self.dim_n, self.dim_d = self.model.sample_path.shape
        # _end_if_

        # Total number of linear variables: a(t).
        self.dim_tot = self.dim_n * self.dim_d * self.dim_d

        # Output variables.
        self.output = {"m0": m0, "s0": s0}
    # _end_def_

    def initialization(self):
        """
        This function initializes  the variational parameters A(t) and b(t).
        This is  done with a simple interpolation technique (Cubic Splines).
        In the case where the dimensions are more than one the interpolation
        happens on each dimension separately.

        :return: a single array containing all the variational parameters.
        """

        # Time window of inference.
        time_window = self.model.time_window

        # Replicate the first and last time points.
        time_x = [time_window[0], *time_window[self.obs_t], time_window[-1]]

        # Switch according to the dimensionality.
        if self.model.single_dim:
            # Replicate the first and last observations.
            obs_z = np.hstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Linear variational parameters.
            a0 = 0.5 * (self.model.sigma / 0.25) * np.ones(self.dim_n)

            # Build a uni-variate extrapolation
            # (with cubic splines).
            fb0 = CubicSpline(time_x, obs_z)

            # Generate the offset parameters on the whole time window.
            b0 = fb0(time_window)
        else:
            # Replicate the first and last observations.
            obs_z = np.vstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Cubic spline extrapolation for
            # each dimension separately.
            fb0 = CubicSpline(time_x, obs_z)
            mt0 = fb0(time_window)

            # Preallocate variational parameters.
            a0 = np.zeros((self.dim_n, self.dim_d, self.dim_d))
            b0 = np.zeros((self.dim_n, self.dim_d))

            # Initial covariance matrix S(t=0)
            s0 = 0.25 * np.eye(self.dim_d)

            # Compute the discrete differences
            # (approximation of Dm(t)/Dt).
            dmt0 = np.diff(mt0, axis=0) / self.dt

            # System Noise / S(t=0).
            diag_k = np.diag(self.model.sigma.diagonal() / s0.diagonal())

            # Construct a0(t) and b0(t) assuming a0(t) and s(t) are diagonal.
            for k in range(self.dim_n - 1):
                a0[k] = 0.5 * diag_k
                b0[k] = dmt0[k] + a0[k].diagonal() * mt0[k]
            # _end_for_

            # At the last point (t=tf) we assume the gradient Dmt0 is zero.
            a0[-1] = 0.5 * diag_k
            b0[-1] = a0[-1].diagonal() * mt0[-1]
        # _end_if_

        # Concatenate the results into one (big) array.
        return np.concatenate((a0.ravel(), b0.ravel()))
    # _end_def_

    def free_energy(self, x):
        """
        Computes the variational free energy, along with parameters
        related to the variational posterior process defined by the
        linear / offset parameters a(t) and b(t).

        :param x: initial variational linear and offset parameters.

        :return: E0 + Esde + Eobs (scalar).
        """

        # Switch according to dim_d.
        if self.dim_d == 1:
            # Extract a(t) and b(t).
            linear_a = x[:self.dim_tot]
            offset_b = x[self.dim_tot:]
        else:
            # Extract a(t) and b(t).
            linear_a = x[:self.dim_tot].reshape(self.dim_n,
                                                self.dim_d, self.dim_d)
            offset_b = x[self.dim_tot:].reshape(self.dim_n, self.dim_d)
        # _end_if_

        # Initial posterior moments.
        m0 = self.output["m0"]
        s0 = self.output["s0"]

        # Forward sweep to get consistent 'm(t)' and 's(t)'.
        mt, st = self.fwd_ode(linear_a, offset_b, m0, s0, self.model.sigma)

        # Energy from the observations (Likelihood).
        Eobs = self.likelihood(mt, st)

        # Energy from the SDE, along with expectations and gradients.
        Esde, (Efx, Edf), (dEsde_dm, dEsde_ds, *_) = self.model.energy(linear_a, offset_b,
                                                                       mt, st, self.obs_t)
        # Compute the required gradients from the Eobs.
        dEobs_dm, dEobs_ds, *_ = self.likelihood.gradients(mt, st)

        # Backward sweep to ensure constraints are satisfied.
        lamt, psit = self.bwd_ode(linear_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds)

        # Energy from the initial moment (t=0). If "m0" and "s0" are
        # not optimized, this value is going to be constant in time.
        E0 = self.kl0(m0, s0)

        # Store the parameters that will be
        # used later in the gradient method.
        self.output["mt"] = mt
        self.output["st"] = st

        self.output["Efx"] = Efx
        self.output["Edf"] = Edf

        self.output["lamt"] = lamt
        self.output["psit"] = psit

        # Total free energy value.
        return np.array(E0 + Esde + Eobs).item()
    # _end_def_

    def gradient(self, x, eval_fun=False):
        """
        Returns the gradient of the Lagrangian w.r.t.
        the variational parameters a(t) (linear) and
        b(t) (offset).

        :param x: variational linear + offset parameters
        (dim_n * dim_d * (dim_d + 1)).

        :param eval_fun: it determines whether we have to
        evaluate first the variational free energy to update
        the parameters before the gradients.

        :return: grouped gradient of the Lagrangian w.r.t.
        the variational linear parameters 'a(t)' (dim_n x dim_d x dim_d)
        and the variational offset parameters 'b(t)' (dim_n x dim_d).
        """

        # Occasionally we have to  evaluate the gradient
        # at different input parameters. In this case we
        # need to  make sure that  all the  marginal and
        # Lagrangian parameters are consistent.
        if eval_fun:
            _ = self.free_energy(x)
        # _end_if_

        # Switch to single dimension.
        if self.model.single_dim:
            # Unpack data.
            at = x[:self.dim_tot]
            bt = x[self.dim_tot:]

            # Preallocate the return arrays.
            gLa = np.zeros(self.dim_n)
            gLb = np.zeros(self.dim_n)
        else:
            # Unpack data.
            at = x[:self.dim_tot].reshape(self.dim_n, self.dim_d, self.dim_d)
            bt = x[self.dim_tot:].reshape(self.dim_n, self.dim_d)

            # Preallocate the return arrays.
            gLa = np.zeros((self.dim_n, self.dim_d, self.dim_d))
            gLb = np.zeros((self.dim_n, self.dim_d))
        # _end_if_

        # Posterior moments: m(t), S(t).
        mt = self.output["mt"]
        st = self.output["st"]

        # Lagrange multipliers: lam(t), psi(t).
        lamt = self.output["lamt"]
        psit = self.output["psit"]

        # Expectation values.
        Efx = self.output["Efx"]
        Edf = self.output["Edf"]

        # Inverse of Sigma noise.
        inv_sigma = self.model.inverse_sigma

        # Main loop.
        for k in range(self.dim_n):
            # Get the values at time 'tk'.
            ak = at[k]
            sk = st[k]
            mk = mt[k]
            lamk = lamt[k]

            # Gradient of Esde w.r.t. 'b' -Eq(29)-
            dEsde_dbt = self._dEsde_db(inv_sigma, Efx[k], mk, ak, bt[k])

            # Gradient of Esde w.r.t. 'A' -Eq(28)-
            dEsde_dat = self._dEsde_da(inv_sigma, ak, mk, sk, Edf[k], dEsde_dbt)

            # Gradient of Lagrangian w.r.t. 'a(t)' -Eq(12)-
            gLa[k] = self._grad_at(dEsde_dat, lamk, mk, psit[k], sk)

            # Gradient of Lagrangian w.r.t. 'b(t)' -Eq(13)-
            gLb[k] = dEsde_dbt + lamk
        # _end_for_

        # Scale the results with the time increment.
        gLa = self.dt * gLa
        gLb = self.dt * gLb

        # Group the gradients together and exit.
        return np.concatenate((gLa.flatten(), gLb.flatten()))
    # _end_def_

    @property
    def arg_out(self):
        """
        Accessor for the output dictionary.
        :return: the output dictionary.
        """
        return self.output
    # _end_def_

    def _grad_at(self, dEsde_dak, lamk, mk, psik, sk):
        """
        Auxiliary function. Return automatically
        the 1D or nD version of the calculation.
        """
        if self.model.single_dim:
            return dEsde_dak - (lamk * mk) - (2.0 * psik * sk)
        else:
            return dEsde_dak - np.outer(lamk, mk) - 2.0 * psik.dot(sk)
        # _end_if_
    # _end_def_

    def _dEsde_da(self, inv_sigma, at, mt, st, Edf, dEsde_dbt):
        """
        Auxiliary function. Return automatically
        the 1D or nD version of the calculation.
        """
        if self.model.single_dim:
            return inv_sigma * (Edf + at) * st - (dEsde_dbt * mt)
        else:
            return inv_sigma.dot(Edf + at).dot(st) - np.outer(dEsde_dbt, mt)
        # _end_if_
    # _end_def_

    def _dEsde_db(self, inv_sigma, Efx, mt, at, bt):
        """
        Auxiliary function. Return automatically
        the 1D or nD version of the calculation.
        """
        if self.model.single_dim:
            return inv_sigma * (-Efx - (at * mt) + bt)
        else:
            return inv_sigma.dot(-Efx - at.dot(mt) + bt)
        # _end_if_
    # _end_def_

# _end_class_

# Auxiliary function.
def grad_Esde_dm_ds(x, fun, mt, st, at, bt, diag_inv_sigma):
    """
    Returns the gradient of the -SDE- energy function with respect
    to the marginal means and variances.  This method is used when
    the analytic expressions for the gradients are difficult to be
    computed, hence we use approximations such as the unscented
    transformation.

    :param x: input state samples (dim_n x dim_d).

    :param fun: drift function.

    :param mt: marginal mean at time 't' (dim_d).

    :param st: marginal covar. at time 't' (dim_d x dim_d).

    :param at: linear parameter (dim_d x dim_d).

    :param bt: offset parameter (dim_d).

    :param diag_inv_sigma: diagonal elements of inverse system
    noise (dim_d)

    :return: gradients w.r.t. to 'mt' and 'st' with dimensions:
    [dim_n x dim_d * (dim_d + 1)].
    """

    # Get the dimensions of the input array.
    dim_n, dim_d = x.shape

    # Preallocate array: [dim_n x dim_d^2].
    dst = np.zeros((dim_n, dim_d * dim_d))

    # Compute auxiliary quantity:
    x_mat = (fun(x) + x.dot(at.T) - np.tile(bt, (dim_n, 1))) ** 2
    var = diag_inv_sigma.dot(x_mat.T)

    # Gradient w.r.t. 'mt': [dim_n x dim_d]
    dmt = np.linalg.solve(st, (np.tile(var, (dim_d, 1)) * x.T)).T

    # Inverse of marginal covariance.
    inv_st, _ = chol_inv(st)

    # Calculate the gradients w.r.t. 'st'.
    for k in range(dim_n):
        # Take the values at sample 'k'.
        zt = x[k] - mt

        # Square matrix.
        zk = np.outer(zt, zt)

        # Gradient w.r.t. 'st'.
        dst[k] = var[k] * np.linalg.solve(st, zk).dot(inv_st).ravel()
    # _end_for_

    # Scale the results.
    dmt = 0.5 * dmt
    dst = 0.5 * dst

    # Group the gradients together and exit.
    return np.concatenate((dmt, dst), axis=1)
# _end_def_
