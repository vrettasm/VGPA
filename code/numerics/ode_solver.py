class OdeSolver(object):
    """
    Parent class of all ODE solvers.
    """

    __slots__ = ("dt", "single_dim")

    def __init__(self, dt=0.01, single_dim=True):
        """
        Default (parent) class constructor.

        :param dt: discrete time step.

        :param single_dim: flags the ode as 1D or nD.
        """
        # Check if time step is positive.
        if dt > 0.0:
            self.dt = dt
        else:
            raise ValueError(" {0}: Discrete time step should be positive -->"
                             " {1}.".format(self.__class__.__name__, dt))
        # _end_if_

        # Store the boolean flag.
        self.single_dim = single_dim
    # _end_def_

    def fun_mt(self, mt, at, bt):
        """
        Auxiliary mean function.

        :param mt: marginal mean m(t).

        :param at: variational parameter a(t).

        :param bt: variational parameter b(t).

        :return: - (A * m.T) + B
        """
        # Switch according to single_dim.
        if self.single_dim:
            return -at * mt + bt
        else:
            return -at.dot(mt.T) + bt
    # _end_def_

    def fun_st(self, st, at, sn):
        """
        Auxiliary co-variance function.

        :param st: marginal co-variance S(t).

        :param at: variational parameter a(t).

        :param sn: system noise.

        :return: - (A * S) - (S * A.T) + Sigma
        """
        # Switch according to single_dim.
        if self.single_dim:
            return -2.0 * at * st + sn
        else:
            return -at.dot(st) - st.dot(at.T) + sn
    # _end_def_

    def fun_lam(self, df_dm, at, lamt):
        """
        Auxiliary Lagrange function.

        :param df_dm: expectation w.r.t. m(t)

        :param at: variational parameter a(t).

        :param lamt: Lagrange multiplier lam(t).

        :return: - dEf_dm + (lam * A.T)
        """
        # Switch according to single_dim.
        if self.single_dim:
            return -df_dm + lamt * at
        else:
            return -df_dm + lamt.dot(at.T)
    # _end_def_

    def fun_psi(self, df_ds, at, psit):
        """
        Auxiliary Lagrange function.

        :param df_ds: expectation w.r.t. s(t)

        :param at: variational parameter a(t).

        :param psit: Lagrange multiplier psi(t).

        :return: - dEf_dS + (psi * A) + (A.T * psi).
        """
        # Switch according to single_dim.
        if self.single_dim:
            return -df_ds + 2.0 * psit * at
        else:
            return -df_ds + psit.dot(at) + at.T.dot(psit)
    # _end_def_

# _end_class_
