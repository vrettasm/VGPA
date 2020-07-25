import numpy as np
from .ode_solver import OdeSolver


class RungeKutta4(OdeSolver):
    """
    Runge-Kutta (4th order) method of integration:

    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

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
        Runge-Kutta 4 integration method. This provides the actual solution.

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

        # Compute the midpoints at time 't + 0.5*dt'.
        ak_mid = 0.5 * (lin_a[0:-1] + lin_a[1:])
        bk_mid = 0.5 * (off_b[0:-1] + off_b[1:])

        # Run through all time points.
        for k in range(dim_n - 1):
            # Get the values at time 'tk'.
            ak = lin_a[k]
            bk = off_b[k]

            # Marginal moments.
            sk = st[k]
            mk = mt[k]

            # Get the midpoints at time 't + 0.5*dt'.
            a_mid = ak_mid[k]
            b_mid = bk_mid[k]

            # Intermediate steps.
            K1 = fun_mt(mk, ak, bk, single_dim) * dt
            K2 = fun_mt((mk + 0.5 * K1), a_mid, b_mid, single_dim) * dt
            K3 = fun_mt((mk + 0.5 * K2), a_mid, b_mid, single_dim) * dt
            K4 = fun_mt((mk + K3), lin_a[k + 1], off_b[k + 1], single_dim) * dt

            # NEW "mean" point.
            mt[k + 1] = mk + (K1 + 2.0 * (K2 + K3) + K4) / 6.0

            # Intermediate steps.
            L1 = fun_st(sk, ak, sigma, single_dim) * dt
            L2 = fun_st((sk + 0.5 * L1), a_mid, sigma, single_dim) * dt
            L3 = fun_st((sk + 0.5 * L2), a_mid, sigma, single_dim) * dt
            L4 = fun_st((sk + L3), lin_a[k + 1], sigma, single_dim) * dt

            # NEW "variance" point
            st[k + 1] = sk + (L1 + 2.0 * (L2 + L3) + L4) / 6.0
        # _end_for_

        # Marginal moments.
        return mt, st
    # _end_def_

    def solve_bwd(self, lin_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds, single_dim=True):
        """
        RK4 integration method. Provides the actual solution.

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

        # Compute the midpoints at time 't + 0.5*dt'.
        ak_mid = 0.5 * (lin_a[0:-1] + lin_a[1:])
        dEmk_mid = 0.5 * (dEsde_dm[0:-1] + dEsde_dm[1:])
        dEsk_mid = 0.5 * (dEsde_ds[0:-1] + dEsde_ds[1:])

        # Run through all time points.
        for t in range(dim_n - 1, 0, -1):
            # Get the values at time 't'.
            at = lin_a[t]
            lamt = lam[t]
            psit = psi[t]

            # Get the midpoints at time 't - 0.5*dt'.
            ak = ak_mid[t]
            dEmk = dEmk_mid[t]
            dEsk = dEsk_mid[t]

            # Lambda (backward) propagation: Intermediate steps.
            K1 = fun_lam(dEsde_dm[t], at, lamt, single_dim) * dt
            K2 = fun_lam(dEmk, ak, (lamt - 0.5 * K1), single_dim) * dt
            K3 = fun_lam(dEmk, ak, (lamt - 0.5 * K2), single_dim) * dt
            K4 = fun_lam(dEsde_dm[t - 1], lin_a[t - 1], (lamt - K3), single_dim) * dt

            # NEW "Lambda" point.
            lam[t - 1] = lamt - (K1 + 2.0 * (K2 + K3) + K4) / 6.0 + dEobs_dm[t - 1]

            # Psi (backward) propagation: Intermediate steps.
            L1 = fun_psi(dEsde_ds[t], at, psit, single_dim) * dt
            L2 = fun_psi(dEsk, ak, (psit - 0.5 * L1), single_dim) * dt
            L3 = fun_psi(dEsk, ak, (psit - 0.5 * L2), single_dim) * dt
            L4 = fun_psi(dEsde_ds[t - 1], lin_a[t - 1], (psit - L3), single_dim) * dt

            # NEW "Psi" point.
            psi[t - 1] = psit - (L1 + 2.0 * (L2 + L3) + L4) / 6.0 + dEobs_ds[t - 1]
        # _end_for_

        # Lagrange multipliers.
        return lam, psi
    # _end_def_

# _end_class_
