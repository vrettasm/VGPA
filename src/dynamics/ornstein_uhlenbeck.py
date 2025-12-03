import numpy as np
from src.numerics.utilities import my_trapz
from src.dynamics.stochastic_process import StochasticProcess
from src.var_bayes.gaussian_moments import GaussianMoments


class OrnsteinUhlenbeck(StochasticProcess):
    """

    Information about the Ornstein - Uhlenbeck process:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    __slots__ = ("_sigma", "_theta", "sig_inv")

    def __init__(self, sigma, theta, r_seed=None) -> None:
        """
        Default constructor of the DW object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model parameter.

        :param r_seed: random seed.
        """
        # Call the constructor of the parent class.
        super().__init__(r_seed, single_dim=True)

        # Display class info.
        print(" Creating Ornstein-Uhlenbeck process.")

        # Check noise value.
        if sigma <= 0.0:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" The diffusion noise value: {sigma},"
                             f" should be strictly positive.")
        # Store the noise.
        self._sigma = sigma

        # Check the drift parameter.
        if theta <= 0.0:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" The drift parameter: {theta},"
                             f" should be strictly positive.")
        # Store the drift.
        self._theta = theta

        # Inverse of sigma noise.
        self.sig_inv = 1.0 / sigma
    # _end_def_

    @property
    def theta(self) -> float:
        """
        Accessor method.

        :return: the drift parameter.
        """
        return self._theta
    # _end_def_

    @theta.setter
    def theta(self, new_value: float) -> None:
        """
        Accessor method.

        :param new_value: for the drift parameter.

        :return: None.
        """
        # Accept only positive values.
        if new_value <= 0.0:
            # Raise an error with a message.
            raise ValueError(f" {self.__class__.__name__}: The drift value"
                             f" {new_value}, should be strictly positive.")
        # Make the change.
        self._theta = new_value
    # _end_def_

    @property
    def sigma(self) -> float:
        """
        Accessor method.

        :return: the diffusion noise parameter.
        """
        return self._sigma
    # _end_def_

    @sigma.setter
    def sigma(self, new_value: float) -> None:
        """
        Accessor method.

        :param new_value: for the sigma diffusion.

        :return: None.
        """

        # Accept only positive values.
        if new_value <= 0.0:
            # Raise an error with a message.
            raise ValueError(f"{self.__class__.__name__}: The sigma value "
                             f"{new_value}, should be strictly positive.")
        # Make the change.
        self._sigma = new_value

        # Update the inverse value.
        self.sig_inv = 1.0 / self._sigma
    # _end_def_

    @property
    def inverse_sigma(self) -> float:
        """
        Accessor method.

        :return: the inverse of diffusion noise parameter.
        """
        return self.sig_inv
    # _end_def_

    def make_trajectory(self, t0, tf, dt: float = 0.01, mu: float = 0.0) -> None:
        """
        Generates a realizations of the Ornstein - Uhlenbeck
        (OU) dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :param mu: default mean value is zero.

        :return: None.
        """
        # Create a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value X(t=0) = 0 or X(t=0) ~ N(mu,K)
        x[0] = mu

        # Random variables (notice the scale of noise with the 'dt').
        ek = np.sqrt(self._sigma * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + self._theta * (mu - x[t - 1]) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def energy(self, linear_a, offset_b, m, s, obs_t) -> tuple:
        """
        Energy for the OU SDE and related quantities (including gradients).

        :param linear_a: variational linear parameters (dim_n x 1).

        :param offset_b: variational offset parameters (dim_n x 1).

        :param m: marginal means (dim_n x 1).

        :param s: marginal variances (dim_n x 1).

        :param obs_t: observation times.

        :return:
            Esde       : total energy of the sde.

            Ef         : average drift (dim_n x 1).
            Edf        : average differentiated drift (dim_n x 1).

            dEsde_dm   : gradient of Esde w.r.t. the means (dim_n x 1).
            dEsde_dS   : gradient of Esde w.r.t. the covariance (dim_n x 1).
            dEsde_dtheta : gradient of Esde w.r.t. the parameter theta.
            dEsde_dsigma : gradient of Esde w.r.t. the parameter Sigma.
        """
        # Gaussian Moments object.
        gauss_mom = GaussianMoments(m, s)

        # Get the time step from the parent class.
        dt = self.time_step

        # Higher order Gaussian Moments.
        Ex2 = gauss_mom(order=2)

        # Pre-compute these quantities only once.
        Q1 = (self._theta - linear_a) ** 2
        Q2 = linear_a * offset_b

        # Auxiliary variable.
        var_q = Ex2 * Q1 + 2.0 * m * (self._theta - linear_a) * offset_b + (offset_b ** 2)

        # Energy from the sDyn: Eq(7)
        Esde = 0.5 * my_trapz(var_q, dt, obs_t) / self._sigma

        # Average drift.
        Ef = - self._theta * m

        # Average gradient of drift.
        Edf = - self._theta * np.ones(m.shape)

        # Gradients of Esde w.r.t. 'means'.
        dEsde_dm = (m * (self._theta - linear_a) ** 2 +
                    self._theta * offset_b - Q2) / self._sigma

        # Gradients of Esde w.r.t. 'variances'.
        dEsde_dS = 0.5 * Q1 / self._sigma

        # Gradients of Esde w.r.t. 'theta'.
        dEsde_dth = my_trapz(Ex2 * (self._theta - linear_a) +
                             m * offset_b, dt, obs_t) / self._sigma

        # Gradients of Esde w.r.t. 'sigma'.
        dEsde_dSig = - Esde / self._sigma

        # --->
        return Esde, (Ef, Edf), (dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig)
    # _end_def_

# _end_class_
