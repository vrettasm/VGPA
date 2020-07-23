import numpy as np
from code.numerics.utilities import my_trapz
from .stochastic_process import StochasticProcess
from code.src.gaussian_moments import GaussianMoments


class OrnsteinUhlenbeck(StochasticProcess):
    """

    Information about the Ornstein - Uhlenbeck process:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    __slots__ = ("sigma_", "theta_", "sig_inv")

    def __init__(self, sigma, theta, r_seed=None):
        """
        Default constructor of the DW object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model parameter.

        :param r_seed: random seed.
        """
        # Call the constructor of the parent class.
        super().__init__(r_seed)

        # Store the diffusion noise.
        if sigma > 0.0:
            self.sigma_ = sigma
        else:
            raise ValueError(" {0}: The diffusion noise value:"
                             " {1}, should be strictly positive.".format(self.__class__.__name__,
                                                                         sigma))
        # _end_if_

        # Store the drift parameter.
        if theta > 0.0:
            self.theta_ = theta
        else:
            raise ValueError(" {0}: The drift parameter:"
                             " {1}, should be strictly positive.".format(self.__class__.__name__,
                                                                         theta))
        # _end_if_

        # Inverse of sigma noise.
        self.sig_inv = 1.0 / sigma
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

        # Accept only positive values.
        if new_value > 0.0:
            # Make the change.
            self.theta_ = new_value
        else:
            # Raise an error with a message.
            raise ValueError(" {0}: The drift value:"
                             " {1}, should be strictly positive.".format(self.__class__.__name__,
                                                                         new_value))
        # _end_if_
    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method.

        :return: the diffusion noise parameter.
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

        # Accept only positive values.
        if new_value > 0.0:
            # Make the change.
            self.sigma_ = new_value

            # Update the inverse value.
            self.sig_inv = 1.0 / self.sigma_
        else:
            # Raise an error with a message.
            raise ValueError(" {0}: The sigma value:"
                             " {1}, should be strictly positive.".format(self.__class__.__name__,
                                                                         new_value))
        # _end_if_
    # _end_def_

    def make_trajectory(self, t0, tf, mu=0.0, dt=0.01):
        """
        Generates a realizations of the Ornstein - Uhlenbeck
        (OU) dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param mu: default mean value is zero.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create a time-window (for inference).
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value X(t=0) = 0 or X(t=0) ~ N(mu,K)
        x[0] = mu

        # Random variables.
        ek = np.sqrt(self.sigma_ * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + self.theta_ * (mu - x[t - 1]) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def energy(self, linear_a, offset_b, m, s, dt, obs_t):
        """
        Energy for the OU SDE and related quantities (including gradients).

        :param linear_a: variational linear parameters (dim_n x 1).

        :param offset_b: variational offset parameters (dim_n x 1).

        :param m: marginal means (dim_n x 1).

        :param s: marginal variances (dim_n x 1).

        :param dt: discrete time step.

        :param obs_t: observation times.

        :return:
            Esde       : total energy of the sde.
            Ef         : average drift (dim_n x 1).
            Edf        : average differentiated drift (dim_n x 1).
            dEsde_dm   : gradient of Esde w.r.t. the means (dim_n x 1).
            dEsde_dS   : gradient of Esde w.r.t. the covariance (dim_n x 1).
            dEsde_dth  : gradient of Esde w.r.t. the parameter theta.
            dEsde_dSig : gradient of Esde w.r.t. the parameter Sigma.
        """

        # Gaussian Moments object.
        gauss_mom = GaussianMoments(m, s)

        # Higher order Gaussian Moments.
        Ex2 = gauss_mom(order=2)

        # Pre-compute these quantities only once.
        Q1 = (self.theta_ - linear_a) ** 2
        Q2 = linear_a * offset_b

        # Energy from the sDyn: Eq(7)
        var_q = Ex2 * Q1 + 2 * m * (self.theta_ - linear_a) * offset_b + offset_b ** 2
        Esde = 0.5 * self.sig_inv * my_trapz(var_q, dt, obs_t)

        # Average drift.
        Ef = -self.theta_ * m

        # Average gradient of drift.
        Edf = -self.theta_ * np.ones(m.shape)

        # Gradients of Esde w.r.t. 'means'.
        dEsde_dm = self.sig_inv * (m * (self.theta_ - linear_a) ** 2 + self.theta_ * offset_b - Q2)

        # Gradients of Esde w.r.t. 'variances'.
        dEsde_dS = 0.5 * self.sig_inv * Q1

        # Gradients of Esde w.r.t. 'theta'.
        dEsde_dth = self.sig_inv * my_trapz(Ex2 * (self.theta_ - linear_a) + m * offset_b, dt, obs_t)

        # Gradients of Esde w.r.t. 'sigma'.
        dEsde_dSig = -self.sig_inv * Esde

        # --->
        return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig
    # _end_def_

# _end_class_
