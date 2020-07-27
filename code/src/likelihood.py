
class Likelihood(object):
    """
    This is a base (parent) class for all the likelihood functions.
    It provides basic (common) functionality, such as storing and
    retrieving the observation values and times.
    """

    __slots__ = ("obs_v", "obs_t")

    def __init__(self, values, times):
        """
        Default constructor. No checks are performed
        for the validity of the input data.

        :param values: observation values.

        :param times: observation times.
        """
        self.obs_v = values
        self.obs_t = times
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

# _end_class_
