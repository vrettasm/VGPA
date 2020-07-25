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
    def fun_mt_1D(mt, at, bt):
        return -at * mt + bt
    # _end_def_

    @staticmethod
    def fun_mt_nD(mt, at, bt):
        return -mt.dot(at.T) + bt
    # _end_def_

    @staticmethod
    def fun_st_1D(st, at, sn):
        return -2.0 * at * st + sn
    # _end_def_

    @staticmethod
    def fun_st_nD(st, at, sn):
        return -at.dot(st) - st.dot(at.T) + sn
    # _end_def_

    @staticmethod
    def fun_lam_1D(df_dm, at, lamt):
        return -df_dm + lamt * at
    # _end_def_

    @staticmethod
    def fun_lam_nD(df_dm, at, lamt):
        return -df_dm + lamt.dot(at.T)
    # _end_def_

    @staticmethod
    def fun_psi_1D(df_ds, at, psit):
        return -df_ds + 2.0 * psit * at
    # _end_def_

    @staticmethod
    def fun_psi_nD(df_ds, at, psit):
        return -df_ds + psit.dot(at) + at.T.dot(psit)
    # _end_def_

# _end_class_
