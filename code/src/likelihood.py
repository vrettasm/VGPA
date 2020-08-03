import numpy as np


class Likelihood(object):
    """
    This is a base (parent) class for all the likelihood functions.
    It provides basic (common) functionality, such as storing and
    retrieving the observation values and times.
    """

    __slots__ = ("obs_v", "obs_t", "obs_n", "obs_h")

    def __init__(self, values, times, noise, operator=None):
        """
        Default constructor. No checks are performed for the
        validity of the input data.

        :param values: observation values. This is a numpy
        array that contains the observations.

        :param times: observation times. This is a vector that
        contains the discrete times that the observations occur.

        :param noise: observation noise co-variance.

        :param operator: observation operator. This is an object
        (function or matrix) that maps the observation space. It
        is usually assumed to be the identity matrix (for simplicity).
        """

        # Store observation times/values.
        self.obs_t = np.asarray(times)
        self.obs_v = np.asarray(values)
        self.obs_n = np.asarray(noise)

        # Default operator is identity matrix.
        if operator is None:
            self.obs_h = np.asarray(1)
        else:
            self.obs_h = np.asarray(operator)
        # _end_if_
    # _end_def_

    @property
    def values(self):
        """
        Accessor method.

        :return: the observation values.
        """
        return self.obs_v
    # _end_def_

    @property
    def times(self):
        """
        Accessor method.

        :return: the observation times.
        """
        return self.obs_t
    # _end_def_

    @property
    def noise(self):
        """
        Accessor method.

        :return: the observation noise.
        """
        return self.obs_n
    # _end_def_

    @noise.setter
    def noise(self, new_value):
        """
        Accessor method.

        :param new_value: for the noise parameter.

        :return: None.
        """
        self.obs_n = new_value
    # _end_def_

    @property
    def operator(self):
        """
        Accessor method.

        :return: the observation operator.
        """
        return self.obs_h
    # _end_def_

# _end_class_
