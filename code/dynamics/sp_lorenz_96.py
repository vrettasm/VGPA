import numpy as np
from numba import njit
from ..src.variational import grad_Esde_dm_ds
from scipy.linalg import cholesky, LinAlgError
from .stochastic_process import StochasticProcess
from ..numerics.utilities import my_trapz, ut_approx, chol_inv


@njit
def fwd_1(x):
    # Shift forward by one.
    return np.roll(x, -1)
# _end_def_


@njit
def bwd_1(x):
    # Shift backward by one.
    return np.roll(x, +1)
# _end_def_


@njit
def bwd_2(x):
    # Shift backward by two.
    return np.roll(x, +2)
# _end_def


@njit
def shift_vectors(x):
    # Return ALL the shifted
    # vectors: (-1, +1, +2).
    return np.roll(x, -1),\
           np.roll(x, +1),\
           np.roll(x, +2)
# _end_def_


@njit
def E96_drift_dx(x):
    """
    Returns the mean value of the gradient of the drift
    function with respect to the state vector: <df(x)/dx>.

    :param x: input state samples (dim_d x 1).

    :return: mean gradient w.r.t. 'x' (dim_d x dim_d).
    """

    # Size of the state vector.
    dim_d = x.size

    # Preallocate return matrix.
    Ex = np.zeros((dim_d, dim_d))

    # Local index array: [0, 1, 2, ... , dim_d-1]
    idx = np.arange(0, dim_d)

    # Get the shifted indices.
    fwd_1i, bwd_1i, bwd_2i = shift_vectors(idx)

    # Get the shifted vector.
    fwd_1x, bwd_1x, bwd_2x = shift_vectors(x)

    # Compute the k-th gradient of the
    # state vector at each dimension.
    for k in range(dim_d):
        # Generate zeros.
        Gx = np.zeros(dim_d)

        # Diff: x_{i}
        Gx[k] = -1

        # Diff: x_{i+1}
        Gx[fwd_1i[k]] = bwd_1x[k]

        # Diff: x_{i-2}
        Gx[bwd_2i[k]] = -bwd_1x[k]

        # Diff: x_{i-1}
        Gx[bwd_1i[k]] = fwd_1x[k] - bwd_2x[k]

        # Store i-th gradient.
        Ex[k] = Gx
    # _end_for_

    # Return: <df(x)/dx>
    return Ex
# _end_def_

@njit
def l96(x, u):
    """
    The Lorenz 96 model function.

    :param x: state vector (dim_d x 1).

    :param u: additional parameters (theta).

    :return: One step ahead in the equation.
    """

    # Get the shifted values.
    fwd_1x, bwd_1x, bwd_2x = shift_vectors(x)

    # Return one step ahead Diff.Eq.
    return (fwd_1x - bwd_2x) * bwd_1x - x + u
# _end_def_


class Lorenz96(StochasticProcess):
    """
    Class that model the Lorenz 40D (1996) dynamical system.

    https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    __slots__ = ("sigma_", "theta_", "sig_inv", "dim_d")

    def __init__(self, sigma, theta, dim_d=40, r_seed=None):
        """
        Default constructor of the L96 object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model vector.

        :param dim_d: dimensionality of the model (default = 40).

        :param r_seed: random seed (default = None).
        """

        # Display class info.
        print(" Creating Lorenz-96 (D={0)) process.".format(dim_d))

        # Call the constructor of the parent class.
        super().__init__(r_seed, single_dim=False)

        # Make sure the inputs are arrays.
        sigma = np.asarray(sigma)
        theta = np.asarray(theta)

        # Store the model dimensions.
        self.dim_d = dim_d

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Create a diagonal matrix.
            self.sigma_ = sigma * np.eye(dim_d)

        elif sigma.ndim == 1:
            # Create a diagonal matrix.
            self.sigma_ = np.diag(sigma)

        elif sigma.ndim == 2:
            # Store the array.
            self.sigma_ = sigma

        else:
            raise ValueError(" {0}: Wrong input dimensions:"
                             " {1}".format(self.__class__.__name__,
                                           sigma.ndim))
        # _end_if_

        # Check the dimensionality.
        if self.sigma_.shape != (dim_d, dim_d):
            raise ValueError(" {0}: Wrong matrix dimensions:"
                             " {1}".format(self.__class__.__name__,
                                           self.sigma_.shape))
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(self.sigma_) > 0.0):

            # This is not the best way to invert.
            self.sig_inv, _ = chol_inv(self.sigma_)
        else:
            raise RuntimeError(" {0}: Noise matrix is not"
                               " positive definite.".format(self.__class__.__name__,
                                                            self.sigma_))
        # _end_if_

        # Store the drift vector.
        self.theta_ = theta
    # _end_def_

    @property
    def theta(self):
        """
        Accessor method.

        :return: the drift parameter.
        """
        return self.theta_
    # _end_def_

    @theta.setter
    def theta(self, new_value):
        """
        Accessor method.

        :param new_value: for the drift parameter.

        :return: None.
        """
        self.theta_ = new_value
    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method.

        :return: the system noise parameter.
        """
        return self.sigma_
    # _end_def_

    @sigma.setter
    def sigma(self, new_value):
        """
        Accessor method.

        :param new_value: for the sigma diffusion.

        :return: None.
        """

        # Check the dimensionality.
        if new_value.shape != (self.dim_d, self.dim_d):
            raise ValueError(" {0}: Wrong matrix dimensions:"
                             " {1}".format(self.__class__.__name__,
                                           new_value.shape))
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(new_value) > 0.0):
            # Make the change.
            self.sigma_ = new_value

            # Update the inverse value.
            self.sig_inv = chol_inv(self.sigma_)
        else:
            raise RuntimeError(" {0}: Noise matrix is not"
                               " positive definite.".format(self.__class__.__name__,
                                                            new_value))
        # _end_if_
    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method.

        :return: the inverse of diffusion noise parameter.
        """
        return self.sig_inv
    # _end_def_

    def make_trajectory(self, t0, tf, dt=0.01):
        """
        Generates a realizations of the Lorenz96 (40D)
        dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Default starting point.
        x0 = self.theta_ * np.ones(self.dim_d)

        # Initial conditions time step.
        delta_t = 1.0e-3

        # Perturb the middle of the vector by "dt".
        x0[int(self.dim_d / 2.0)] += delta_t

        # BURN IN:
        for t in range(5000):
            x0 = x0 + l96(x0, self.theta_) * delta_t
        # _end_for_

        # Preallocate array.
        x = np.zeros((dim_t, self.dim_d))

        # Start with the new point.
        x[0] = x0

        # Compute the Cholesky decomposition of input matrix.
        try:
            ek = cholesky(self.sigma_ * dt)
        except LinAlgError:
            # Show a warning message.
            print(" Warning : The input matrix was not positive definite."
                  " The diagonal elements will be used instead.")

            # If it fails use the diagonal only.
            ek = np.sqrt(np.eye(self.dim_d) * self.sigma_ * dt)
        # _end_try_

        # Random variables.
        ek = ek.dot(self.rng.standard_normal((self.dim_d, dim_t))).T

        # Create the path by solving the "stochastic" Diff.Eq. iteratively.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + l96(x[t - 1], self.theta_) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def energy(self, linear_a, offset_b, m, s, obs_t):
        """
        Energy for the stochastic Lorenz 96 DE (dim_d = 40)
        and related quantities (including gradients).

        :param linear_a: variational linear parameters (dim_t x dim_d x dim_d).

        :param offset_b: variational offset parameters (dim_t x dim_d).

        :param m: marginal means (dim_t x dim_d).

        :param s: marginal variances (dim_t x dim_d x dim_d).

        :param obs_t: observation times.

        :return: Esde       : total energy of the sde.

                 Ef         : average drift (dim_t x dim_d).
                 Edf        : average differentiated drift (dim_t x dim_d).

                 dEsde_dm   : gradient of Esde w.r.t. the means (dim_t x dim_d).
                 dEsde_dS   : gradient of Esde w.r.t. the covariance (dim_t x dim_d x dim_d).
                 dEsde_dtheta : gradient of Esde w.r.t. the parameter theta.
                 dEsde_dsigma : gradient of Esde w.r.t. the parameter Sigma.
        """

        # System dimensions.
        dim_d = self.dim_d

        # Number of discrete time points.
        dim_t = self.time_window.size

        # Get the time step.
        dt = self.time_step

        # Inverse System Noise.
        inv_sigma = self.sig_inv

        # Drift (forcing) parameter.
        theta = self.theta

        # Diagonal elements of inverse Sigma.
        diag_inv_sig = np.diag(inv_sigma)

        # Energy from the sDyn.
        Esde = np.zeros(dim_t)

        # Average drift.
        Ef = np.zeros((dim_t, dim_d))

        # Average gradient of drift.
        Edf = np.zeros((dim_t, dim_d, dim_d))

        # Gradients of Esde w.r.t. 'm' and 'S'.
        dEsde_dm = np.zeros((dim_t, dim_d))
        dEsde_ds = np.zeros((dim_t, dim_d, dim_d))

        # Gradients of Esde w.r.t. 'Theta'.
        dEsde_dth = np.zeros((dim_t, dim_d))

        # Gradients of Esde w.r.t. 'Sigma'.
        dEsde_dSig = np.zeros((dim_t, dim_d))

        # Define lambda functions:
        Fx = {'1': lambda x, at, bt: (l96(x, theta) + x.dot(at.T) - np.tile(bt, (x.shape[0], 1))) ** 2,
              '2': lambda x, _: l96(x, theta)}

        # Compute the quantities iteratively.
        for t in range(dim_t):
            # Get the values at time 't'.
            at = linear_a[t]
            bt = offset_b[t]

            # Marginal Moments.
            mt, st = m[t], s[t]

            # Compute: <(f(xt)-g(xt))'*(f(xt)-g(xt))>.
            m_bar, _ = ut_approx(Fx['1'], mt, st, at, bt)

            # Esde energy: Esde(t) = 0.5*<(f(xt)-g(xt))'*SigInv*(f(xt)-g(xt))>.
            Esde[t] = 0.5 * diag_inv_sig.dot(m_bar.T)

            # Average drift: <f(Xt)>
            Ef[t] = self.E96_drift(mt, st)

            # Average gradient of drift: <Df(Xt)>
            Edf[t] = E96_drift_dx(mt)

            # Approximate the expectation of the gradients.
            # x, fun, mt, st, at, bt, diag_inv_sigma
            dmS, _ = ut_approx(grad_Esde_dm_ds, mt, st,
                               Fx['2'], mt, st, at, bt,
                               diag_inv_sig)

            # Gradient w.r.t. mean mt: dEsde(t)_dmt
            dEsde_dm[t] = dmS[:dim_d] - Esde[t] * np.linalg.solve(st, mt)

            #  Gradient w.r.t. covariance St: dEsde(t)_dSt
            dEsde_ds[t] = 0.5 * (dmS[dim_d:].reshape(dim_d, dim_d) - Esde[t] * np.linalg.inv(st))

            # Gradients of Esde w.r.t. 'Theta': dEsde(t)_dtheta
            dEsde_dth[t] = Ef[t] + mt.dot(at.T) - bt

            # Gradients of Esde w.r.t. 'Sigma': dEsde(t)_dSigma
            dEsde_dSig[t] = m_bar
        # _end_for_

        # Compute energy using numerical integration.
        Esde = my_trapz(Esde, dt, obs_t)

        # Final adjustments for the (hyper)parameters.
        dEsde_dth = diag_inv_sig * my_trapz(dEsde_dth, dt, obs_t)

        # Final adjustments for the System noise.
        dEsde_dSig = -0.5 * inv_sigma.dot(np.diag(my_trapz(dEsde_dSig, dt, obs_t))).dot(inv_sigma)

        # --->
        return Esde, (Ef, Edf), (dEsde_dm, dEsde_ds, dEsde_dth, dEsde_dSig)
    # _end_def_

    def E96_drift(self, mt, st):
        """
        Returns the mean value of the drift function <f(x)>.

        :param mt: mean vector (dim_d x 1).

        :param st: covariance matrix (dim_d x dim_d).

        :return: mean of the drift function (dim_d x 1).
        """

        # Local index array: [0, 1, 2, ... , dim_d-1]
        idx = np.arange(0, self.dim_d)

        # Get the shifted indices.
        fwd_1i, bwd_1i, bwd_2i = shift_vectors(idx)

        # Get access to the covariances at the desired points.
        Cxx = st[fwd_1i, bwd_1i] - st[bwd_2i, bwd_1i]

        # Compute the expected value.
        return Cxx + (fwd_1(mt) - bwd_2(mt)) * bwd_1(mt) - mt + self.theta
    # _end_def_

# _end_class_
