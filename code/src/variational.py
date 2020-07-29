import numpy as np
from scipy import interpolate


class VarGP(object):
    """
    TBD
    """

    __slots__ = ("model", "fwd_ode", "bwd_ode", "kl0", "likelihood",
                 "obs_y", "obs_t")

    def __init__(self, model, fwd_ode, bwd_ode, likelihood, kl0, obs_y, obs_t):
        self.model = model

        self.fwd_ode = fwd_ode
        self.bwd_ode = bwd_ode

        self.kl0 = kl0
        self.likelihood = likelihood

        self.obs_y = obs_y
        self.obs_t = obs_t
    # _end_def_

    def initialization(self):
        """
        TBD

        :return:
        """

        # Time window of inference.
        time_window = self.model.time_window

        # Replicate the first and last time points.
        time_x = [time_window[0]]
        time_x.extend(time_window[self.obs_t])
        time_x.extend(time_window[-1])

        # Switch according to the dimensionality.
        if self.model.single_dim:
            # Get the dimensions of the dynamical system.
            dim_n = self.model.sample_path[0].size

            # Replicate the first and last observations.
            obs_z = np.hstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Linear variational parameters.
            a0 = 0.5 * (self.model.sigma / 0.25) * np.ones(dim_n)

            # Build a uni-variate extrapolation (with cubic splines).
            fb0 = interpolate.CubicSpline(time_x, obs_z)

            # Generate the offset parameters on the whole time window.
            b0 = fb0(time_window)
        else:
            # Get the dimensions of the dynamical system.
            dim_n, dim_d = self.model.sample_path[0].shape

            # Replicate the first and last observations.
            obs_z = np.vstack((self.obs_y[0], self.obs_y, self.obs_y[-1]))

            # Discrete time step.
            dt = self.model.time_step

            # Cubic spline extrapolation for each dimension separately.
            fb0 = interpolate.CubicSpline(time_x, obs_z)
            mt0 = fb0(time_window)

            # Preallocate variational parameters.
            a0 = np.zeros((dim_n, dim_d, dim_d))
            b0 = np.zeros((dim_n, dim_d))

            # Initial covariance matrix S(t=0)
            s0 = 0.25 * np.eye(dim_d)

            # Compute the discrete differences
            # (approximation of Dm(t)/Dt).
            dmt0 = np.diff(mt0, axis=0) / dt

            # System Noise / S(t=0).
            diag_k = np.diag(self.model.sigma.diagonal() / s0.diagonal())

            # Construct a0(t) and b0(t) assuming a0(t) and s(t) are diagonal.
            for k in range(dim_n - 1):
                a0[k] = 0.5 * diag_k
                b0[k] = dmt0[k] + a0[k].diagonal() * mt0[k]
            # _end_for_

            # At the last point (t=tf) we assume the gradient Dmt0 is zero.
            a0[-1] = 0.5 * diag_k
            b0[-1] = a0[-1].diagonal() * mt0[-1]
        # _end_if_

        # Concatenate the results into one (big) array before exit.
        return np.concatenate((a0.ravel()[:, np.newaxis],
                               b0.ravel()[:, np.newaxis]))
    # _end_def_

    def free_energy(self, x):
        # return KL0 + Eobs + Esde
        pass
    # _end_def_

    def gradient(self, x):
        # return grad
        pass
    # _end_def_

# _end_def_
