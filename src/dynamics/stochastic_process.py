import numpy as np
from numpy.random import default_rng, SeedSequence


class StochasticProcess(object):
    """
    This is a base (parent) class for all the stochastic process models.
    """

    __slots__ = ("xt", "tk", "rand_g", "single_dimension")

    def __init__(self, r_seed=None, single_dim=True):
        """
        Default constructor.

        :param r_seed: random seed.

        :param single_dim: single dimensional flag.
        """

        # Create a random number generator.
        if r_seed is not None:
            self.rand_g = default_rng(SeedSequence(r_seed))
        else:
            self.rand_g = default_rng()
        # _end_if_

        # Single dimensional process.
        self.single_dimension = single_dim

        # Sample-path.
        self.xt = None

        # Time-window.
        self.tk = None
    # _end_def_

    @property
    def single_dim(self):
        """
        Accessor method.

        :return: True if the stochastic process is 1-D.
        """
        return self.single_dimension
    # _end_def_

    @property
    def sample_path(self):
        """
        Accessor method.

        :return: the sample path.
        """
        # Check if the sample path is created.
        if self.xt is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Sample path has not been created.")
        # _end_def_

        return self.xt
    # _end_def_

    @sample_path.setter
    def sample_path(self, new_value):
        """
        Accessor method.

        :param new_value: of the sample path (trajectory).

        :return: None.
        """
        self.xt = new_value
    # _end_def_

    @property
    def time_window(self):
        """
        Accessor method.

        :return: the time window of the path.
        """
        # Check if the sample path is created.
        if self.tk is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Time window has not been created yet.")
        # _end_def_

        return self.tk
    # _end_def_

    @time_window.setter
    def time_window(self, new_value):
        """
        Accessor method.

        :param new_value: of the time window (for inference).

        :return: None.
        """
        self.tk = new_value
    # _end_def_

    @property
    def time_step(self):
        """
        Accessor method.

        :return: the time step of the discrete path.
        """
        # Check if the sample path is created.
        if self.tk is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Time window has not been created yet.")
        # _end_def_

        # Return the 'dt'.
        return np.abs(self.tk[1] - self.tk[0])
    # _end_def_

    @property
    def rng(self):
        """
        Accessor method.

        :return: the random number generator.
        """
        return self.rand_g
    # _end_def_

    def collect_obs(self, n_obs, rn, h_mask=None):
        """
        This function collects a number of noisy observations
        from a sample path (trajectory). The observations are
        collected at equidistant time points.

        :param n_obs: Observations density (# of observations
        per time unit).

        :param rn: Observation noise (co)-variance.

        :param h_mask: boolean that masks only the observed values.

        :return: observation times / noisy values / noise (matrix).
        """

        # Check if the sample path is created.
        if self.tk is None or self.xt is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Sample path (or time window) have not been created.")
        # _end_def_

        # Make sure input is numpy array.
        rn = np.asarray(rn)

        # Get the discrete time step.
        dt = np.diff(self.tk)[0]

        # Check if the required number of observations
        # per time unit exceeds the available capacity
        # of samples.
        if n_obs > int(1.0 / dt):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Observation density exceeds the number of samples.")
        # _end_def_

        # Total number of observations.
        dim_m = int(np.floor(np.abs(self.tk[0] - self.tk[-1]) * n_obs))

        # Number of discrete time points.
        dim_t = self.tk.size

        # Observation indexes.
        idx = np.linspace(0, dim_t, dim_m + 2, dtype=int)

        # Make sure they are unique and sorted.
        obs_t = sorted(np.unique(idx[1:-1]))

        # Extract the complete observations (d = D) at times obs_t.
        obs_y = np.take(self.xt, obs_t, axis=0)

        # Check if a mask has been given.
        if h_mask is not None:
            # Here we have (d < D)
            obs_y = obs_y[:, h_mask]
        # _end_if_

        # Dimensionality of observations.
        dim_d = 1 if obs_y.ndim == 1 else obs_y.shape[-1]

        # Add noise according to the observation dimensions.
        if dim_d == 1:
            # Scalar value.
            obs_noise = rn

            # Sqrt(Rn).
            sq_rn = np.sqrt(obs_noise)

            # Fixed Gaussian noise.
            obs_y += sq_rn * self.rand_g.standard_normal(dim_m)

        else:
            # Multidimensional case.
            if rn.ndim == 0:
                # Diagonal matrix (from scalar).
                obs_noise = rn * np.eye(dim_d)

                # Sqrt(Rn).
                sq_rn = np.sqrt(obs_noise)

            elif rn.ndim == 1:
                # Diagonal matrix (from vector).
                obs_noise = np.diag(rn)

                # Sqrt(Rn).
                sq_rn = np.sqrt(obs_noise)

            else:
                # Diagonal matrix (from matrix).
                obs_noise = rn * np.eye(dim_d)

                # Sqrt(Rn).
                sq_rn = np.sqrt(obs_noise)
            # _end_if_

            # Add fixed Gaussian noise.
            obs_y += sq_rn.dot(self.rand_g.standard_normal((dim_d, dim_m))).T
        # _end_if_

        # Observation (times / values / noise).
        return obs_t, obs_y, obs_noise
    # _end_def_

# _end_class_
