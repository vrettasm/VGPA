import numpy as np
from .ode_solver import OdeSolver


class Euler(OdeSolver):
    """
    Euler's method of integration:

    y_{n+1} = y_{n} + h * f(t_{n}, y_{n})

    https://en.wikipedia.org/wiki/Euler_method

    """

    def __init__(self, dt):
        """
        Default constructor.

        :param dt: discrete time step.
        """
        # Call the constructor of the parent class.
        super().__init__(dt)
    # _end_def_

    def solve_fwd(self, lin_a, off_b, m0, s0, sigma, single_dim=True):
        """
        Euler integration method. This provides the actual solution.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param off_b: Offset variational parameters (dim_n x dim_d).

        :param m0: Initial marginal mean (dim_d x 1).

        :param s0: Initial marginal variance (dim_d x dim_d).

        :param sigma: System noise variance (dim_d x dim_d).

        :param single_dim: Boolean flag. Determines which version of the
        code will be called.

        :return: 1) mt: posterior means values (dim_n x dim_d).
                 2) st: posterior variance values (dim_n x dim_d x dim_d).
        """

        # Pre-allocate memory according to single_dim.
        if single_dim:
            # Number of discrete time points.
            dim_n = off_b.shape[0]

            # Return arrays.
            mt = np.zeros(dim_n)
            st = np.zeros(dim_n)
        else:
            # Get the dimensions.
            dim_n, dim_d = off_b.shape

            # Return arrays.
            mt = np.zeros((dim_n, dim_d))
            st = np.zeros((dim_n, dim_d, dim_d))
        # _end_if_

        # Initialize the first moments.
        mt[0], st[0] = m0, s0

        # Discrete time step.
        dt = self.dt

        # Local copies of auxiliary functions.
        fun_mt = self.fun_mt
        fun_st = self.fun_st

        # Run through all time points.
        for k in range(dim_n - 1):
            # Get the values at time 'tk'.
            ak = lin_a[k]
            bk = off_b[k]

            # Marginal moments.
            sk = st[k]
            mk = mt[k]

            # -Eq(09)- NEW "mean" point.
            mt[k + 1] = mk + fun_mt(mk, ak, bk, single_dim) * dt

            # -Eq(10)- NEW "covariance" point.
            st[k + 1] = sk + fun_st(sk, ak, sigma, single_dim) * dt
        # _end_for_

        # Marginal moments.
        return mt, st
    # _end_def_

    def solve_bwd(self, lin_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds, single_dim=True):
        """
        Euler integration method. Provides the actual solution.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param dEsde_dm: Derivative of Esde w.r.t. m(t), (dim_n x dim_d).

        :param dEsde_ds: Derivative of Esde w.r.t. s(t), (dim_n x dim_d x dim_d).

        :param dEobs_dm: Derivative of Eobs w.r.t. m(t), (dim_n x dim_d).

        :param dEobs_ds: Derivative of Eobs w.r.t. s(t), (dim_n x dim_d x dim_d).

        :param single_dim: Boolean flag. Determines which version of the
        code will be called.

        :return: 1) lam: Lagrange multipliers for the mean  values (dim_n x dim_d),
                 2) psi: Lagrange multipliers for the var values (dim_n x dim_d x dim_d).
        """

        # Pre-allocate memory according to single_dim.
        if single_dim:
            # Number of discrete points.
            dim_n = dEsde_dm.shape[0]

            # Return arrays.
            lam = np.zeros(dim_n)
            psi = np.zeros(dim_n)
        else:
            # Get the dimensions.
            dim_n, dim_d = dEsde_dm.shape

            # Return arrays.
            lam = np.zeros((dim_n, dim_d))
            psi = np.zeros((dim_n, dim_d, dim_d))
        # _end_if_

        # Discrete time step.
        dt = self.dt

        # Local copies of auxiliary functions.
        fun_lam = self.fun_lam
        fun_psi = self.fun_psi

        # Run through all time points.
        for t in range(dim_n - 1, 0, -1):
            # Get the values at time 't'.
            at = lin_a[t]
            lamt = lam[t]
            psit = psi[t]

            # -Eq(14)- NEW "Lambda" point.
            lam[t - 1] = lamt - fun_lam(dEsde_dm[t], at, lamt, single_dim) * dt + dEobs_dm[t - 1]

            # -Eq(15)- NEW "Psi" point.
            psi[t - 1] = psit - fun_psi(dEsde_ds[t], at, psit, single_dim) * dt + dEobs_ds[t - 1]
        # _end_for_

        # Lagrange multipliers.
        return lam, psi
    # _end_def_

# _end_class_
