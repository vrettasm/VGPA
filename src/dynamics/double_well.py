import numpy as np
from src.numerics.utilities import my_trapz
from src.dynamics.stochastic_process import StochasticProcess
from src.var_bayes.gaussian_moments import GaussianMoments


class DoubleWell(StochasticProcess):
    """
    Information about the double-well potential:

    https://en.wikipedia.org/wiki/Double-well_potential

    NOTE: The equation numbers correspond to the paper:

    Cedric Archambeau, Manfred Opper, Yuan Shen, Dan Cornford, and
    John Shawe-Taylor (2007). "Variational Inference for Diffusion
    Processes". Annual Conference on Neural Information Processing
    Systems (NIPS).
    """

    __slots__ = ("_sigma", "_theta", "sig_inv")

    def __init__(self, sigma: float, theta: float, r_seed=None) -> None:
        """
        Default constructor of the DW object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model parameter.

        :param r_seed: random seed.
        """
        # Call the constructor of the parent class.
        super().__init__(r_seed, single_dim=True)

        # Display class info.
        print(" Creating Double-Well process.")

        # Check value.
        if sigma <= 0.0:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" The diffusion noise value: {sigma},"
                             f" should be strictly positive.")
        # Store the noise.
        self._sigma = sigma

        # Inverse of sigma noise.
        self.sig_inv = 1.0 / sigma

        # Store the drift parameter.
        self._theta = theta
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
            raise ValueError(f" {self.__class__.__name__}: The sigma value:"
                             f" {new_value}, should be strictly positive.")
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

    def make_trajectory(self, t0: float, tf: float, dt: float = 0.01) -> None:
        """
        Generates a realizations of the double well (DW) dynamical system,
        within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create locally a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual time points.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value is chosen from the
        #    "Equilibrium Distribution":
        # x0 = 0.5*N(+mu,K) + 0.5*N(-mu,K)
        if self.rng.random() > 0.5:
            x[0] = +self._theta
        else:
            x[0] = -self._theta
        # _end_if_

        # Add Gaussian noise.
        x[0] += np.sqrt(0.5 * self._sigma * dt) * self.rng.standard_normal()

        # Random variables (notice the scale of noise with the 'dt').
        ek = np.sqrt(self._sigma * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + \
                   4.0 * x[t - 1] * (self._theta - x[t - 1] ** 2) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def energy(self, linear_a: np.ndarray, offset_b: np.ndarray,
               m: np.ndarray, s: np.ndarray, obs_t: np.ndarray):
        """
        Energy for the Double-Well SDE and related quantities.

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
            dEsde_ds   : gradient of Esde w.r.t. the covariance (dim_n x 1).
            dEsde_dtheta : gradient of Esde w.r.t. the parameter theta.
            dEsde_dsigma : gradient of Esde w.r.t. the parameter Sigma.
        """
        # Gaussian Moments object.
        gauss_mom = GaussianMoments(m, s)

        # Constant value.
        c = (4.0 * self._theta) + linear_a

        # Get the time step from the parent class.
        dt = self.time_step

        # Auxiliary constant.
        c2 = c ** 2

        # Higher order Gaussian Moments.
        Ex2 = gauss_mom(order=2)
        Ex3 = gauss_mom(order=3)
        Ex4 = gauss_mom(order=4)
        Ex6 = gauss_mom(order=6)

        # Auxiliary variable.
        var_q = 8.0 * (Ex6 - c * Ex4 + offset_b * Ex3) + (c2 * Ex2) - (2.0 * offset_b * c * m) + (offset_b ** 2)

        # Energy from the sDyn: Eq(7)
        Esde = 0.5 * my_trapz(var_q, dt, obs_t) / self._sigma

        # Average drift: Eq(20) -> f(t,x) = 4*x*(theta -x^2).
        Ef = 4.0 * (self._theta * m - Ex3)

        # Average gradient of drift: -> df(t,x)_dx = 4*theta - 12*x^2.
        Edf = 4.0 * (self._theta - 3.0 * Ex2)

        # Derivatives of higher order Gaussian moments
        # w.r.t. marginal moments 'm(t)' and 's(t)'.
        Dm2 = gauss_mom.dm(order=2)
        Ds2 = gauss_mom.ds(order=2)

        # ---
        Dm3 = gauss_mom.dm(order=3)
        Ds3 = gauss_mom.ds(order=3)

        # ---
        Dm4 = gauss_mom.dm(order=4)
        Ds4 = gauss_mom.ds(order=4)

        # ---
        Dm6 = gauss_mom.dm(order=6)
        Ds6 = gauss_mom.ds(order=6)

        # Gradients of Esde w.r.t. 'means'.
        dEsde_dm = 0.5 * (16.0 * Dm6 - 8.0 * c * Dm4 +
                          8.0 * offset_b * Dm3 + c2 * Dm2 -
                          2.0 * offset_b * c) / self._sigma

        # Gradients of Esde w.r.t. 'variances'.
        dEsde_ds = 0.5 * (16.0 * Ds6 - 8.0 * c * Ds4 +
                          8.0 * offset_b * Ds3 + c2 * Ds2) / self._sigma

        # Gradients of Esde w.r.t. 'theta'.
        dEsde_dtheta = 4.0 * my_trapz(c * Ex2 - 4.0 * Ex4 -
                                      offset_b * m, dt, obs_t) / self._sigma

        # Gradients of Esde w.r.t. 'sigma'.
        dEsde_dsigma = - Esde / self._sigma

        # --->
        return Esde, (Ef, Edf), (dEsde_dm, dEsde_ds, dEsde_dtheta, dEsde_dsigma)
    # _end_def_

# _end_class_
