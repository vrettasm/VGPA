import numpy as np
from .ode_solver import OdeSolver


class Euler(OdeSolver):
    """
    Euler's method of integration:

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

    def fwd(self, *args):
        """
        Forward solution of the ode.

        :param args: dictionary with the variational parameters.

        :return: the result of the solver (marginal moments).
        """
        # Get the list of parameters.
        p_list = args[0]

        # Unpack the list.
        at = p_list["at"]
        bt = p_list["bt"]
        m0 = p_list["m0"]
        s0 = p_list["s0"]
        sigma = p_list["sigma"]

        # Dimensionality of the system.
        if at.shape[-1] == 1:
            return self.solve_fwd_1D(at, bt, m0, s0, sigma)
        else:
            return self.solve_fwd_nD(at, bt, m0, s0, sigma)
        # _end_if_

    # _end_def_

    def solve_fwd_1D(self, lin_a, off_b, m0, s0, sigma):
        """
        Euler integration method 1D.

        :param lin_a: Linear variational parameters (dim_n x 1).

        :param off_b: Offset variational parameters (dim_n x 1).

        :param m0: Initial marginal mean (scalar).

        :param s0: Initial marginal variance (scalar).

        :param sigma: System noise variance (scalar).

        :return: 1) mt: posterior means values (dim_n x 1).
                 2) st: posterior variance values (dim_n x 1).
        """

        # Get the number of discrete time points.
        dim_n = off_b.shape[0]

        # Preallocate the return arrays.
        mt = np.zeros(dim_n)
        st = np.zeros(dim_n)

        # Initialize the first moments.
        mt[0], st[0] = m0, s0

        # Run through all time points.
        for k in range(dim_n - 1):
            # Get the values at time 'tk'.
            ak = lin_a[k]
            bk = off_b[k]

            # Marginal moments.
            sk = st[k]
            mk = mt[k]

            # -Eq(09)- NEW "mean" point.
            mt[k + 1] = mk + (-ak * mk + bk) * self.dt

            # -Eq(10)- NEW "covariance" point.
            st[k + 1] = sk + (-2.0 * ak * sk + sigma) * self.dt
        # _end_for_

        # Return the marginal moments.
        return mt, st
    # _end_def_

    def solve_fwd_nD(self, lin_a, off_b, m0, s0, sigma):
        """
        Euler integration method nD.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param off_b: Offset variational parameters (dim_n x dim_d).

        :param m0: Initial marginal mean (dim_d x 1).

        :param s0: Initial marginal variance (dim_d x dim_d).

        :param sigma: System noise variance (dim_d x dim_d).

        :return: 1) mt: posterior means values (dim_n x dim_d).
                 2) st: posterior variance values (dim_n x dim_d x dim_d).
        """
        # Get the dimensions.
        dim_n, dim_d = off_b.shape

        # Preallocate the return arrays.
        mt = np.zeros((dim_n, dim_d))
        st = np.zeros((dim_n, dim_d, dim_d))

        # Initialize the first moments.
        mt[0], st[0] = m0, s0

        # Run through all time points.
        for k in range(dim_n - 1):
            # Get the values at time 'tk'.
            ak = lin_a[k]
            bk = off_b[k]

            # Marginal moments.
            sk = st[k]
            mk = mt[k]

            # -Eq(09)- NEW "mean" point.
            mt[k + 1] = mk + self.fun_mt(mk, ak, bk) * self.dt

            # -Eq(10)- NEW "covariance" point.
            st[k + 1] = sk + self.fun_ct(sk, ak, sigma) * self.dt
        # _end_for_

        # Return the marginal moments.
        return mt, st
    # _end_def_

    def bwd(self, *args):
        """
        Backward solution of the ode.

        :param args: dictionary with the variational parameters.

        :return: the result of the solver (Lagrange multipliers).
        """
        # Get the list of parameters.
        p_list = args[0]

        # Unpack the list.
        at = p_list["at"]
        dEsde_dm = p_list["dEsde_dm"]
        dEsde_ds = p_list["dEsde_ds"]
        dEobs_dm = p_list["dEobs_dm"]
        dEobs_ds = p_list["dEobs_ds"]

        # Dimensionality of the system.
        if at.shape[-1] == 1:
            return self.solve_bwd_1D(at, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds)
        else:
            return self.solve_bwd_nD(at, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds)
        # _end_if_
    # _end_def_

    def solve_bwd_1D(self, lin_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds):
        """
        Euler integration method 1D.

        :param lin_a: Linear variational parameters (dim_n x 1).

        :param dEsde_dm: Derivative of Esde w.r.t. m(t), (dim_n x 1).

        :param dEsde_ds: Derivative of Esde w.r.t. s(t), (dim_n x 1).

        :param dEobs_dm: Derivative of Eobs w.r.t. m(t), (dim_n x 1).

        :param dEobs_ds: Derivative of Eobs w.r.t. s(t), (dim_n x 1).

        :return: 1) lam: Lagrange multipliers for the mean  values (dim_n x 1),
                 2) psi: Lagrange multipliers for the var values (dim_n x 1).
        """

        # Get the dimensions.
        dim_n = dEsde_dm.shape[0]

        # Preallocate the return arrays.
        lam = np.zeros(dim_n)
        psi = np.zeros(dim_n)

        # Run through all time points.
        for t in range(dim_n - 1, 0, -1):
            # Get the values at time 't'.
            at = lin_a[t]
            lamt = lam[t]
            psit = psi[t]

            # -Eq(14)- NEW "Lambda" point.
            lam[t - 1] = lamt - (-dEsde_dm[t] + at * lamt) * self.dt + dEobs_dm[t - 1]

            # -Eq(15)- NEW "Psi" point.
            psi[t - 1] = psit - (-dEsde_ds[t] + 2.0 * at * psit) * self.dt + dEobs_ds[t - 1]
        # _end_for_

        return lam, psi
    # _end_def_

    def solve_bwd_nD(self, lin_a, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds):
        """
        Euler integration method 1D.

        :param lin_a: Linear variational parameters (dim_n x dim_d x dim_d).

        :param dEsde_dm: Derivative of Esde w.r.t. m(t), (dim_n x dim_d).

        :param dEsde_ds: Derivative of Esde w.r.t. s(t), (dim_n x dim_d x dim_d).

        :param dEobs_dm: Derivative of Eobs w.r.t. m(t), (dim_n x dim_d).

        :param dEobs_ds: Derivative of Eobs w.r.t. s(t), (dim_n x dim_d x dim_d).

        :return: 1) lam: Lagrange multipliers for the mean  values (dim_n x 1),
                 2) psi: Lagrange multipliers for the var values (dim_n x 1).
        """
        # Get the dimensions.
        dim_n, dim_d = dEsde_dm.shape

        # Preallocate the return arrays.
        lam = np.zeros((dim_n, dim_d))
        psi = np.zeros((dim_n, dim_d, dim_d))

        # Run through all time points.
        for t in range(dim_n - 1, 0, -1):
            # Get the values at time 't'.
            at = lin_a[t]
            lamt = lam[t]
            psit = psi[t]

            # -Eq(14)- NEW "Lambda" point.
            lam[t - 1] = lamt - self.fun_lam(dEsde_dm[t], at, lamt) * self.dt + dEobs_dm[t - 1]

            # -Eq(15)- NEW "Psi" point.
            psi[t - 1] = psit - self.fun_psi(dEsde_ds[t], at, psit) * self.dt + dEobs_ds[t - 1]
        # _end_for_

        # Lagrange multipliers.
        return lam, psi
    # _end_def_

# _end_class_
