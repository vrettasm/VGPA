import numpy as np

class StochasticProcess(object):
    """
    This is a base (parent) class for all the stochastic process models.
    """

    __slots__ = ("xt", "tk", "rand_g")

    def __init__(self, r_seed=None):
        """
        Default constructor.

        :param r_seed: random seed.
        """

        # Create a random number generator.
        if r_seed is not None:
            self.rand_g = np.random.default_rng(r_seed)
        else:
            self.rand_g = np.random.default_rng()
        # _end_if_

        # Sample-path.
        self.xt = None

        # Time-window.
        self.tk = None
    # _end_def_

    @property
    def sample_path(self):
        """
        Accessor method.

        :return: the sample path.
        """
        # Check if the sample path is created.
        if self.xt is None:
            raise NotImplementedError(" {0}:"
                                      " Sample path is not created.".format(self.__class__.__name__))
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
            raise NotImplementedError(" {0}:"
                                      " Time window is not created.".format(self.__class__.__name__))
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
    def rng(self):
        """
        Accessor method.

        :return: the random number generator.
        """
        return self.rand_g
    # _end_def_

    def collect_obs(self, n_obs, rn, h_mask=None):
        """
        This function collects a number of noisy observations from a sample path
        (trajectory). The observations are collected at equidistant time points.

        :param n_obs: Observations density (# of observations per time unit).

        :param rn: Observation noise (co)-variance.

        :param h_mask: List that masks only the observed values.

        :return: observation times / observation values (with i.i.d. white noise).
        """

        # Check if the sample path is created.
        if self.tk is None or self.xt is None:
            raise NotImplementedError(" {0}: Sample path or time window"
                                      " are not created.".format(self.__class__.__name__))
        # _end_def_

        # Make sure input is numpy array.
        rn = np.asarray(rn)

        # Get the discrete time step.
        dt = np.diff(self.tk)[0]

        # Check if the required number of observations
        # per time unit exceeds the available capacity
        # of samples.
        if n_obs > (1.0 / dt):
            raise ValueError(" {0}: Observation density"
                             " exceeds the number of samples.".format(self.__class__.__name__))
        # _end_def_

        # Total number of observations.
        dim_m = int(np.floor(np.abs(self.tk[0] - self.tk[-1]) * n_obs))

        # Number of discrete time points.
        dim_t = self.tk.shape[0]

        # Observation indexes.
        idx = np.linspace(0, dim_t, dim_m + 2, dtype=np.int)

        # Convert it to list so you can use it as index.
        # Make sure they are unique and sorted.
        obs_t = np.array(sorted(set(idx[1:-1].tolist())))

        # Extract the complete observations (d = D) at times obs_t.
        obs_y = np.take(self.xt, obs_t, axis=0)

        # Check if a mask has been given.
        if h_mask is not None:
            # Here we have (d < D)
            obs_y = obs_y[:, h_mask]
        # _end_if_

        # Check if (co)-variance vector/matrix is given.
        if rn.ndim == 0:
            # Add fixed Gaussian noise.
            obs_y += np.sqrt(rn) * self.rand_g.standard_normal(dim_m)
        else:
            # Dimensionality of observations.
            dim_d = 1 if obs_y.ndim == 1 else obs_y.shape[-1]

            # For the moment consider only diagonal matrices.
            if rn.ndim == 1:
                # Vector.
                rn = np.diag(rn)
            else:
                # Square matrix.
                rn *= np.eye(dim_d)
            # _end_if_

            # Get the square root of the noise matrix.
            sq_rn = np.sqrt(rn)

            # Add fixed Gaussian noise.
            obs_y += sq_rn.dot(self.rand_g.standard_normal(dim_d, dim_m)).T
        # _end_if_

        # Observation (times / values).
        return obs_t, obs_y
    # _end_def_

# _end_class_
