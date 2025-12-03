
class OdeSolver(object):
    """
    Parent class of all ODE solvers.
    """

    __slots__ = ("dt", "single_dim")

    def __init__(self, dt: float = 0.01, single_dim: bool = True) -> None:
        """
        Default (parent) class constructor.

        :param dt: discrete time step.

        :param single_dim: flags the ode as 1D or nD.
        """

        # Check if time step is positive.
        if dt <= 0.0:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Discrete time step should be positive --> {dt}.")
        # Store the time discretization step.
        self.dt = dt

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
        return -(at * mt) + bt if self.single_dim else -at.dot(mt) + bt
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
        return -(2.0 * at * st) + sn if self.single_dim else -at.dot(st) - st.dot(at.T) + sn
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
        return -df_dm + (lamt * at) if self.single_dim else -df_dm + lamt.dot(at.T)
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
        return -df_ds + (2.0 * psit * at) if self.single_dim else -df_ds + psit.dot(at) + at.T.dot(psit)
    # _end_def_

# _end_class_
