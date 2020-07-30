import numpy as np
from scipy import interpolate


class VarGP(object):
    """
    TBD
    """

    __slots__ = ("model", "fwd_ode", "bwd_ode", "kl0", "likelihood",
                 "obs_y", "obs_t", "dt", "dim_n", "dim_d", "dim_tot",
                 "output")

    def __init__(self, model, m0, s0, fwd_ode, bwd_ode, likelihood,
                 kl0, obs_y, obs_t):
        """
        Default constructor of VGPA object.

        :param model:

        :param m0:

        :param s0:

        :param fwd_ode:

        :param bwd_ode:

        :param likelihood:

        :param kl0:

        :param obs_y:

        :param obs_t:
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
            self.dim_n, self.dim_d = self.model.sample_path[0].size, 1
        else:
            self.dim_n, self.dim_d = self.model.sample_path[0].shape
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
        time_x = [time_window[0]]
        time_x.extend(time_window[self.obs_t])
        time_x.extend(time_window[-1])

        # Switch according to the dimensionality.
        if self.model.single_dim:
            # Replicate the first and last observations.
            obs_z = np.hstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Linear variational parameters.
            a0 = 0.5 * (self.model.sigma / 0.25) * np.ones(self.dim_n)

            # Build a uni-variate extrapolation (with cubic splines).
            fb0 = interpolate.CubicSpline(time_x, obs_z)

            # Generate the offset parameters on the whole time window.
            b0 = fb0(time_window)
        else:
            # Replicate the first and last observations.
            obs_z = np.vstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Cubic spline extrapolation for each dimension separately.
            fb0 = interpolate.CubicSpline(time_x, obs_z)
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

        # Concatenate the results into one (large) array.
        return np.concatenate((a0.ravel()[:, np.newaxis],
                               b0.ravel()[:, np.newaxis]))
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
        # _end_if

        # Initial posterior moments.
        # For the moment are not optimized.
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

        # Energy from the initial moment (t=0).
        # If m0 and s0 are not optimized, this
        # value is going to be constant in time.
        E0 = self.kl0(m0, s0)

        # Store the parameters that will be
        # used later from the gradient method.
        self.output["mt"] = mt
        self.output["st"] = st

        self.output["Efx"] = Efx
        self.output["Edf"] = Edf

        self.output["lamt"] = lamt
        self.output["psit"] = psit

        # Total free energy value.
        return np.asscalar(E0 + Esde + Eobs)
    # _end_def_

    def gradient(self, x):
        # return grad
        pass
    # _end_def_

# _end_def_
