import numpy as np
from ..numerics.utilities import my_trapz, ut_approx
from ..src.variational import grad_Esde_dm_ds
from .stochastic_process import StochasticProcess
from scipy.linalg import cholesky, inv, LinAlgError

# Generates the shifted state vectors.
fwd_1 = lambda x: np.roll(x, -1)
bwd_1 = lambda x: np.roll(x, +1)
bwd_2 = lambda x: np.roll(x, +2)

# Helper function.
def l96(x, u):
    """
    The Lorenz 96 model function.

    :param x: state vector (dim_d x 1).

    :param u: additional parameters (theta).

    :return: One step ahead in the equation.
    """

    # Differential equations.
    dx = (fwd_1(x) - bwd_2(x)) * bwd_1(x) - x + u

    # Return dx.
    return dx
# _end_def_


class Lorenz96(StochasticProcess):
    """
    Class that model the Lorenz 40D (1996) dynamical system.

    https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    __slots__ = ("sigma_", "theta_", "sig_inv")

    def __init__(self, sigma, theta, r_seed=None):
        """
        Default constructor of the L96 object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model vector.

        :param r_seed: random seed.
        """

        # Display class info.
        print(" Creating Lorenz-96 process.")

        # Call the constructor of the parent class.
        super().__init__(r_seed, single_dim=False)

        # Make sure the inputs are arrays.
        sigma = np.asarray(sigma)
        theta = np.asarray(theta)

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Create a diagonal matrix.
            self.sigma_ = sigma * np.eye(40)

        elif sigma.ndim == 1:
            # Create a diagonal matrix.
            self.sigma_ = np.diag(sigma)

        elif sigma.ndim == 2:
            # Store the array.
            self.sigma_ = sigma

        else:
            raise ValueError(" {0}: Wrong input dimensions:"
                             " {1}".format(self.__class__.__name__, sigma.ndim))
        # _end_if_

        # Check the dimensionality.
        if self.sigma_.shape != (40, 40):
            raise ValueError(" {0}: Wrong matrix dimensions:"
                             " {1}".format(self.__class__.__name__, self.sigma_.shape))
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(self.sigma_) > 0.0):

            # This is not the best way to invert.
            self.sig_inv = inv(self.sigma_)
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
        if new_value.shape != (40, 40):
            raise ValueError(" {0}: Wrong matrix dimensions:"
                             " {1}".format(self.__class__.__name__,
                                           new_value.shape))
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(new_value) > 0.0):
            # Make the change.
            self.sigma_ = new_value

            # Update the inverse value.
            self.sig_inv = inv(self.sigma_)
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
        x0 = self.theta_ * np.ones(40)

        # Initial conditions time step.
        delta_t = 1.0e-3

        # Perturb the one dimension by dt.
        x0[20] += delta_t

        # BURN IN:
        for t in range(5000):
            x0 = x0 + l96(x0, self.theta_) * delta_t
        # _end_for_

        # Preallocate array.
        x = np.zeros(dim_t, 40)

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
            ek = np.sqrt(np.eye(40) * self.sigma_ * dt)
        # _end_try_

        # Random variables.
        ek *= self.rng.standard_normal(dim_t, 40)

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
        Energy for the stochastic Lorenz 96 DE (40 dimensional)
        and related quantities (including gradients).

        :param linear_a: variational linear parameters (dim_t x 40 x 40).

        :param offset_b: variational offset parameters (dim_t x 40).

        :param m: marginal means (dim_t x 40).

        :param s: marginal variances (dim_t x 40 x 40).

        :param obs_t: observation times.

        :return: Esde       : total energy of the sde.

                 Ef         : average drift (dim_t x 40).
                 Edf        : average differentiated drift (dim_t x 40).

                 dEsde_dm   : gradient of Esde w.r.t. the means (dim_t x 40).
                 dEsde_dS   : gradient of Esde w.r.t. the covariance (dim_t x 40 x 40).
                 dEsde_dtheta : gradient of Esde w.r.t. the parameter theta.
                 dEsde_dsigma : gradient of Esde w.r.t. the parameter Sigma.
        """

        # Number of discrete time points.
        dim_t = self.time_window.size

        # Get the time step from the parent class.
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
        Ef = np.zeros((dim_t, 40))

        # Average gradient of drift.
        Edf = np.zeros((dim_t, 40, 40))

        # Gradients of Esde w.r.t. 'm' and 'S'.
        dEsde_dm = np.zeros((dim_t, 40))
        dEsde_ds = np.zeros((dim_t, 40, 40))

        # Gradients of Esde w.r.t. 'Theta'.
        dEsde_dth = np.zeros((dim_t, 40))

        # Gradients of Esde w.r.t. 'Sigma'.
        dEsde_dSig = np.zeros((dim_t, 40))

        # Define lambda functions:
        Fx = {'1': lambda x, at, bt: (l96(x, theta) + x.dot(at.T) - np.tile(bt, (x.shape[0], 1))) ** 2,
              '2': lambda x, _: l96(x, theta)}

        # Compute the quantities iteratively.
        for t in range(dim_t):
            # Get the values at time 't'.
            at = linear_a[t]
            bt = offset_b[t]
            st = s[t]
            mt = m[t]

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
            dEsde_dm[t] = dmS[:40] - Esde[t] * np.linalg.solve(st, mt)

            #  Gradient w.r.t. covariance St: dEsde(t)_dSt
            dEsde_ds[t] = 0.5 * (dmS[40:].reshape(40, 40) - Esde[t] * np.linalg.inv(st))

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

        :param mt: mean vector (40 x 1).

        :param st: covariance matrix (40 x 40).

        :return: mean of the drift function (40 x 1).
        """

        # Local index array: [0, 1, 2, ... , 39]
        idx = np.arange(0, 40)

        # Get access to the covariances at the desired points.
        Cxx = st[fwd_1(idx), bwd_1(idx)] - st[bwd_2(idx), bwd_1(idx)]

        # Compute the expected value.
        return Cxx + (fwd_1(mt) - bwd_2(mt)) * bwd_1(mt) - mt + self.theta
    # _end_def_

# _end_class_

# Helper function.
def E96_drift_dx(x):
    """
    Returns the mean value of the gradient of the drift
    function with respect to the state vector: <df(x)/dx>.

    :param x: input state samples (40 x 1).

    :return: mean gradient w.r.t. to x (40 x 40).
    """
    
    # Preallocate return matrix.
    Ex = np.zeros((40, 40))
    
    # Local index array: [0, 1, 2, ... , 39]
    idx = np.arange(0, 40)
    
    # Compute the gradient of the state vector
    # at each time point.
    for i in range(40):
        # Generate zeros.
        Gx = np.zeros(40)
        
        # Compute the i-th ODE gradient.
        Gx[i] = -1
        Gx[fwd_1(idx)[i]] = +bwd_1(x)[i]
        Gx[bwd_2(idx)[i]] = -bwd_1(x)[i]
        Gx[bwd_1(idx)[i]] = +fwd_1(x)[i] - bwd_2(x)[i]
        
        # Store i-th gradient.
        Ex[i] = Gx
    # --->
    return Ex
# _end_def_