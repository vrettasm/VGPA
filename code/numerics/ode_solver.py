
class OdeSolver(object):
    """
    Parent class of all ODE solvers.
    """

    __slots__ = ("dt",)

    def __init__(self, dt=0.01):
        """
        Default (parent) class constructor.

        :param dt:  discrete time step.
        """
        # Check if time step is positive.
        if dt > 0.0:
            self.dt = dt
        else:
            raise ValueError(" {0}: Discrete time step should be"
                             " strictly positive -> {1}.".format(self.__class__.__name__, dt))
        # _end_if_
    # _end_def_

    # Define (static) methods for the computation of the marginal
    # means and covariances for the "nD" versions. These are very
    # helpful for the Runge-Kutta (2/4) implementations!

    @staticmethod
    def fun_mt(mt, at, bt):
        return -mt.dot(at.T) + bt
    # _end_def_

    @staticmethod
    def fun_ct(st, at, sn):
        return -st.dot(st) - st.dot(at.T) + sn
    # _end_def_

# _end_class_
