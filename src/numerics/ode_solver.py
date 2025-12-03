import numpy as np


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

    def fun_mt(self, mt: np.ndarray, at: np.ndarray, bt: np.ndarray) -> np.ndarray:
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

    def fun_st(self, st: np.ndarray, at: np.ndarray, sn: np.ndarray) -> np.ndarray:
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

    def fun_lam(self, df_dm: np.ndarray, at: np.ndarray,
                lam_t: np.ndarray) -> np.ndarray:
        """
        Auxiliary Lagrange function.

        :param df_dm: expectation w.r.t. m(t)

        :param at: variational parameter a(t).

        :param lam_t: Lagrange multiplier lam(t).

        :return: - dEf_dm + (lam * A.T)
        """
        # Switch according to single_dim.
        return -df_dm + (lam_t * at) if self.single_dim else -df_dm + lam_t.dot(at.T)
    # _end_def_

    def fun_psi(self, df_ds: np.ndarray, at: np.ndarray,
                psi_t: np.ndarray) -> np.ndarray:
        """
        Auxiliary Lagrange function.

        :param df_ds: expectation w.r.t. s(t)

        :param at: variational parameter a(t).

        :param psi_t: Lagrange multiplier psi(t).

        :return: - dEf_dS + (psi * A) + (A.T * psi).
        """
        # Switch according to single_dim.
        return -df_ds + (2.0 * psi_t * at) if self.single_dim else -df_ds + psi_t.dot(at) + at.T.dot(psi_t)
    # _end_def_

# _end_class_
