import numpy as np


class GaussianMoments(object):
    """
    This class creates an object that returns the higher order
    moments, of a non-central Gaussian, up to the 8-th order.

        https://en.wikipedia.org/wiki/Normal_distribution

    """

    __slots__ = ("m_arr", "v_arr", "n_size")

    def __init__(self, m_arr, v_arr):
        """
        Constructs an object that holds the marginal means
        and variances at all times (t).

        :param m_arr: marginal means array (N x 1).

        :param v_arr: marginal variances array (N x 1).
        """

        # Make sure the inputs are at least 1-D.
        m_arr, v_arr = np.atleast_1d(m_arr, v_arr)

        # The shapes must match.
        if m_arr.shape != v_arr.shape:
            raise RuntimeError(" {0}: Input arrays shape mismatch."
                               " {1} != {2)".format(self.__class__.__name__,
                                                    m_arr.shape, v_arr.shape))
        # _end_if_

        # Store the marginal means and variances.
        self.m_arr = m_arr
        self.v_arr = v_arr

        # Get the size of the arrays.
        self.n_size = m_arr.shape[0]
    # _end_def_

    def __call__(self, order=0):
        """
        Compute the correct non-central moment up to 8-th order.

        :param order: of the un-centered Gaussian moment.

        :return: the un-centered Gaussian moment.

        :raises ValueError: if the input order is out of bounds.
        """

        if order == 0:
            x_out = np.ones(self.n_size)
        elif order == 1:
            x_out = self.m_arr
        elif order == 2:
            x_out = self.m_arr ** 2 + self.v_arr
        elif order == 3:
            x_out = self.m_arr ** 3 +\
                    3 * self.m_arr * self.v_arr
        elif order == 4:
            x_out = self.m_arr ** 4 +\
                    6 * (self.m_arr ** 2) * self.v_arr +\
                    3 * (self.v_arr ** 2)
        elif order == 5:
            x_out = self.m_arr ** 5 +\
                    10 * (self.m_arr ** 3) * self.v_arr +\
                    15 * self.m_arr * (self.v_arr ** 2)
        elif order == 6:
            x_out = self.m_arr ** 6 +\
                    15 * (self.m_arr ** 4) * self.v_arr +\
                    45 * (self.m_arr ** 2) * (self.v_arr ** 2) +\
                    15 * (self.v_arr ** 3)
        elif order == 7:
            x_out = self.m_arr ** 7 +\
                    21 * (self.m_arr ** 5) * self.v_arr +\
                    105 * (self.m_arr ** 3) * (self.v_arr ** 2) +\
                    105 * self.m_arr * (self.v_arr ** 3)
        elif order == 8:
            x_out = self.m_arr ** 8 +\
                    28 * (self.m_arr ** 6) * self.v_arr +\
                    210 * (self.m_arr ** 4) * (self.v_arr ** 2) +\
                    420 * (self.m_arr ** 2) * (self.v_arr ** 3) +\
                    105 * (self.v_arr ** 4)
        else:
            raise ValueError(" {0}: Wrong order value."
                             " Use values 0-8.".format(self.__class__.__name__))
        # _end_if_

        return x_out
    # _end_def_

    def dM(self, order=1):
        """
        Compute the derivative with respect to the marginal
        means, of the non-central moment, up to 8-th order.

        :param order: of the un-centered Gaussian moment.

        :return: the derivative with respect to the marginal
        means.

        :raises ValueError: if the input order is out of bounds.
        """

        if order == 1:
            x_out = np.ones(self.n_size)
        elif order == 2:
            x_out = 2 * self.m_arr
        elif order == 3:
            x_out = 3 * (self.m_arr ** 2 + self.v_arr)
        elif order == 4:
            x_out = 4 * (self.m_arr ** 3 +
                         3 * self.m_arr * self.v_arr)
        elif order == 5:
            x_out = 5 * (self.m_arr ** 4 +
                         6 * (self.m_arr ** 2) * self.v_arr +
                         3 * (self.v_arr ** 2))
        elif order == 6:
            x_out = 6 * (self.m_arr ** 5 +
                         10 * (self.m_arr ** 3) * self.v_arr +
                         15 * self.m_arr * (self.v_arr ** 2))
        elif order == 7:
            x_out = 7 * (self.m_arr ** 6 +
                         15 * (self.m_arr ** 4) * self.v_arr +
                         45 * (self.m_arr ** 2) * (self.v_arr ** 2) +
                         15 * (self.v_arr ** 3))
        elif order == 8:
            x_out = 8 * (self.m_arr ** 7 +
                         21 * (self.m_arr ** 5) * self.v_arr +
                         105 * (self.m_arr ** 3) * (self.v_arr ** 2) +
                         105 * self.m_arr * (self.v_arr ** 3))
        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong order value. Use values 1-8.")
        # _end_if_

        return x_out
    # _end_def_

    def dS(self, order=1):
        """
        Compute the derivative with respect to the marginal
        variances, of the un-centered moment, up to 8-th order.

        :param order: of the un-centered Gaussian moment.

        :return: the derivative with respect to the marginal
        variances.

        :raises ValueError: if the input order is out of bounds.
        """

        if order == 1:
            x_out = np.zeros(self.n_size)
        elif order == 2:
            x_out = np.ones(self.n_size)
        elif order == 3:
            x_out = 3 * self.m_arr
        elif order == 4:
            x_out = 6 * (self.m_arr ** 2 + self.v_arr)
        elif order == 5:
            x_out = 10 * (self.m_arr ** 3) +\
                    30 * (self.m_arr * self.v_arr)
        elif order == 6:
            x_out = 15 * (self.m_arr ** 4) +\
                    90 * (self.m_arr ** 2) * self.v_arr +\
                    45 * (self.v_arr ** 2)
        elif order == 7:
            x_out = 21 * (self.m_arr ** 5) +\
                    210 * (self.m_arr ** 3) * self.v_arr +\
                    315 * self.m_arr * (self.v_arr ** 2)
        elif order == 8:
            x_out = 28 * (self.m_arr ** 6) +\
                    420 * (self.m_arr ** 4) * self.v_arr +\
                    1260 * (self.m_arr ** 2) * (self.v_arr ** 2) +\
                    420 * (self.v_arr ** 3)
        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong order value. Use values 1-8.")
        # _end_if_

        return x_out
    # _end_def_

# _end_class_
