import numpy as np
from .stochastic_process import StochasticProcess
from code.src.gaussian_moments import GaussianMoments


class DoubleWell(StochasticProcess):
    """
    Information about the double-well potential:

    https://en.wikipedia.org/wiki/Double-well_potential

    NOTE: The equation numbers correspond to the paper:

    Cedric A. Opper M. Shen Y. Cornford D. and Shawe-Taylor J. (2007).
    "Variational Inference for Diffusion Processes". Annual Conference
    on Neural Information Processing Systems (NIPS).

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

        # Inverse of sigma noise.
        self.sig_inv = 1.0 / sigma

        # Store the drift parameter.
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

    def make_trajectory(self, t0, tf, dt=0.01):
        """
        Generates a realizations of the double well (DW)
        dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create a time-window (for inference).
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value X(t=0), is chosen from the
        # "Equilibrium Distribution": x0 = 0.5*N(+mu,K) + 0.5*N(-mu,K)
        if self.rng.random() > 0.5:
            x[0] = +self.theta_ + \
                   np.sqrt(0.5 * self.sigma_ * dt) * self.rng.standard_normal()
        else:
            x[0] = -self.theta_ + \
                   np.sqrt(0.5 * self.sigma_ * dt) * self.rng.standard_normal()
        # _end_if_

        # Random variables.
        ek = np.sqrt(self.sigma_ * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + \
                   4 * x[t - 1] * (self.theta_ - x[t - 1] ** 2) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def energy(self, linear_a, bias_b, m, s, dt, obs_t):
        """
        Energy for the Double-Well SDE and related quantities (including gradients).

        :param linear_a: variational linear parameters (dim_n x 1).

        :param bias_b: variational offset parameters (dim_n x 1).

        :param m: marginal means (dim_n x 1).

        :param s: marginal variances (dim_n x 1).

        :param dt: discrete time step.

        :param obs_t: observation times.

        :return:
            Esde      : total energy of the sde.
            Ef        : average drift (dim_n x 1).
            Edf       : average differentiated drift (dim_n x 1).
            dEsde_dm  : gradient of Esde w.r.t. the means (dim_n x 1).
            dEsde_dS  : gradient of Esde w.r.t. the covariance (dim_n x 1).
            dEsde_dth : gradient of Esde w.r.t. the parameter theta.
            dEsde_dSig: gradient of Esde w.r.t. the parameter Sigma.
        """

        # Gaussian Moments object.
        gauss_mom = GaussianMoments(m, s)

        # Constant value.
        c = 4.0 * self.theta_ + linear_a

        # Auxiliary constant.
        c2 = c ** 2

        # Higher order Gaussian Moments.
        Ex2 = gauss_mom(order=2)
        Ex3 = gauss_mom(order=3)
        Ex4 = gauss_mom(order=4)
        Ex6 = gauss_mom(order=6)

        # Auxiliary variable.
        var_q = 8.0 * (Ex6 - c * Ex4 + bias_b * Ex3) +\
                c2 * Ex2 - 2.0 * bias_b * c * m + bias_b ** 2

        # Energy from the sDyn: Eq(7)
        Esde = 0.5 * self.sig_inv * np.trapz(var_q, dt, obs_t)

        # Average drift: Eq(20) -> f(t,x) = 4*x*(theta -x^2).
        Ef = 4.0 * (self.theta_ * m - Ex3)

        # Average gradient of drift: -> df(t,x)_dx = 4*theta - 12*x^2.
        Edf = 4.0 * (self.theta_ - 3 * Ex2)

        # Derivatives of higher order Gaussian moments w.r.t. 'm' and 's'.
        Dm2 = gauss_mom.dM(order=2)
        DS2 = gauss_mom.dS(order=2)

        # ---
        Dm3 = gauss_mom.dM(order=3)
        DS3 = gauss_mom.dS(order=3)

        # ---
        Dm4 = gauss_mom.dM(order=4)
        DS4 = gauss_mom.dS(order=4)

        # ---
        Dm6 = gauss_mom.dM(order=6)
        DS6 = gauss_mom.dS(order=6)

        # Gradients of Esde w.r.t. 'm' and 's'.
        dEsde_dm = 0.5 * self.sig_inv * (16.0*Dm6 - 8.0*c*Dm4 + 8.0*bias_b*Dm3 + c2*Dm2 - 2.0*bias_b*c)
        dEsde_dS = 0.5 * self.sig_inv * (16.0*DS6 - 8.0*c*DS4 + 8.0*bias_b*DS3 + c2*DS2)

        # Gradients of Esde w.r.t. 'Theta'.
        dEsde_dth = 4.0 * self.sig_inv * np.trapz(c*Ex2 - 4.0*Ex4 - bias_b*m, dt, obs_t)

        # Gradients of Esde w.r.t. 'sigma'.
        dEsde_dSig = -Esde * self.sig_inv

        # --->
        return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig
    # _end_def_

# _end_class_
