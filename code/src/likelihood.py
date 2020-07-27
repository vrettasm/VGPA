
class Likelihood(object):
    """
    This is a base (parent) class for all the likelihood functions.
    It provides basic (common) functionality, such as storing and
    retrieving the observation values and times.
    """

    __slots__ = ("obs_v", "obs_t", "obs_h")

    def __init__(self, values, times, operator=None):
        """
        Default constructor. No checks are performed
        for the validity of the input data.

        :param values: observation values. This is a
        numpy array that contains the observations.

        :param times: observation times. This is a
        vector that contains the discrete times that
        the observations occur.

        :param operator: observation operator. This is
        an object (function or matrix) that maps the
        observation space. It is usually assumed to be
        the identity matrix (for simplicity).
        """

        # Store observation times/values.
        self.obs_t = times
        self.obs_v = values

        # Default operator is identity matrix.
        if operator is None:
            self.obs_h = 1
        else:
            self.obs_h = operator
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
    def operator(self):
        """
        Accessor method.

        :return: the observation operator.
        """
        return self.obs_h
    # _end_def_

# _end_class_
