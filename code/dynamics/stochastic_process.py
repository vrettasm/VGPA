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

# _end_class_
