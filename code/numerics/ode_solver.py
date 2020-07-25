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
            raise ValueError(" {0}: Discrete time step should be positive -->"
                             " {1}.".format(self.__class__.__name__, dt))
        # _end_if_
    # _end_def_

    # Define (static) methods for the computation of the marginal
    # means and covariances for the "nD" versions. These are very
    # helpful for the Runge-Kutta (2/4) implementations!

    @staticmethod
    def fun_mt(mt, at, bt, single_dim=True):
        # Switch according to single_dim.
        if single_dim:
            return -at * mt + bt
        else:
            return -at.dot(mt.T) + bt
    # _end_def_

    @staticmethod
    def fun_st(st, at, sn, single_dim=True):
        # Switch according to single_dim.
        if single_dim:
            return -2.0 * at * st + sn
        else:
            return -at.dot(st) - st.dot(at.T) + sn
    # _end_def_

    @staticmethod
    def fun_lam(df_dm, at, lamt, single_dim=True):
        # Switch according to single_dim.
        if single_dim:
            return -df_dm + lamt * at
        else:
            return -df_dm + lamt.dot(at.T)
    # _end_def_

    @staticmethod
    def fun_psi(df_ds, at, psit, single_dim=True):
        # Switch according to single_dim.
        if single_dim:
            return -df_ds + 2.0 * psit * at
        else:
            return -df_ds + psit.dot(at) + at.T.dot(psit)
    # _end_def_

# _end_class_
