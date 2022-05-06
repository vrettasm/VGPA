import numpy as np
from src.numerics.ode_solver import OdeSolver


class Heun(OdeSolver):
    """
    Heun's method of integration:

    c_{n+1} = y_{n} + h * f(t_{n}, y_{n})
    y_{n+1} = y_{n} + h/2 * (f(t_{n}, y_{n}) * f(t_{n+1}, c_{n+1}) )

    https://en.wikipedia.org/wiki/Heun%27s_method

    """

    def __init__(self, dt, single_dim):
        """
        Default constructor.

        :param dt: discrete time step.

        :param single_dim: flags the ode as 1D or nD.
        """

        # Call the constructor of the parent class.
        super().__init__(dt, single_dim)
    # _end_def_

    def solve_fwd(self, lin_a, off_b, m0, s0, sigma):
        """
        Heun integration method. This provides the actual solution.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param off_b: Offset variational parameters (dim_n x dim_d).

        :param m0: Initial marginal mean (dim_d x 1).

        :param s0: Initial marginal variance (dim_d x dim_d).

        :param sigma: System noise variance (dim_d x dim_d).

        :return: 1) mt: posterior means values (dim_n x dim_d).
                 2) st: posterior variance values (dim_n x dim_d x dim_d).
        """

        # Pre-allocate memory according to single_dim.
        if self.single_dim:
            # Number of discrete time points.
            dim_n = off_b.size

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

        # Half step-size.
        h = 0.5 * dt

        # Run through all-time points.
        for k in range(dim_n - 1):
            # Get the values at time 'tk'.
            ak = lin_a[k]
            bk = off_b[k]

            # Marginal moments.
            sk = st[k]
            mk = mt[k]

            # Get the value at time 't+1'.
            a_plus = lin_a[k + 1]
            b_plus = off_b[k + 1]

            # -Eq(09)- Prediction step:
            f_predict = fun_mt(mk, ak, bk)

            # Correction step:
            f_correct = fun_mt((mk + f_predict * dt), a_plus, b_plus)

            # NEW "mean" point.
            mt[k + 1] = mk + h * (f_predict + f_correct)

            # -Eq(10)- Prediction step:
            f_predict = fun_st(sk, ak, sigma)

            # Correction step:
            f_correct = fun_st((sk + f_predict * dt), a_plus, sigma)

            # NEW "variance" point.
            st[k + 1] = sk + h * (f_predict + f_correct)
        # _end_for_

        # Marginal moments.
        return mt, st
    # _end_def_

    def solve_bwd(self, lin_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds):
        """
        Heun integration method. Provides the actual solution.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param dEsde_dm: Derivative of Esde w.r.t. m(t), (dim_n x dim_d).

        :param dEsde_ds: Derivative of Esde w.r.t. s(t), (dim_n x dim_d x dim_d).

        :param dEobs_dm: Derivative of Eobs w.r.t. m(t), (dim_n x dim_d).

        :param dEobs_ds: Derivative of Eobs w.r.t. s(t), (dim_n x dim_d x dim_d).

        :return: 1) lam: Lagrange multipliers for the mean  values (dim_n x dim_d),
                 2) psi: Lagrange multipliers for the var values (dim_n x dim_d x dim_d).
        """

        # Pre-allocate memory according to single_dim.
        if self.single_dim:
            # Number of discrete points.
            dim_n = dEsde_dm.size

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

        # Half step-size.
        h = 0.5 * dt

        # Run through all-time points.
        for t in range(dim_n - 1, 0, -1):
            # Get the values at time 't'.
            at = lin_a[t]
            lamt = lam[t]
            psit = psi[t]

            # Get the value at time 't-1'.
            ak = lin_a[t - 1]

            # -Eq(14)- "Lambda" Prediction step.
            f_predict = fun_lam(dEsde_dm[t], at, lamt)

            # "Lambda" Correction step.
            f_correct = fun_lam(dEsde_dm[t - 1], ak, (lamt - f_predict * dt))

            # NEW "Lambda" point.
            lam[t - 1] = lamt - h * (f_predict + f_correct) + dEobs_dm[t - 1]

            # -Eq(15)- "Psi" Prediction step.
            f_predict = fun_psi(dEsde_ds[t], at, psit)

            # "Psi" Correction step.
            f_correct = fun_psi(dEsde_ds[t - 1], ak, (psit - f_predict * dt))

            # NEW "Psi" point:
            psi[t - 1] = psit - h * (f_predict + f_correct) + dEobs_ds[t - 1]
        # _end_for_

        # Lagrange multipliers.
        return lam, psi
    # _end_def_

# _end_class_
