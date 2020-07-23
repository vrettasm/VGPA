import numpy as np
from .ode_solver import OdeSolver


class Heun(OdeSolver):
    """
    Heun's method of integration:

    https://en.wikipedia.org/wiki/Heun%27s_method

    """

    def __init__(self, dt):
        """
        Default constructor.

        :param dt: discrete time step.
        """
        # Call the constructor of the parent class.
        super().__init__(dt)
    # _end_def_

    def __call__(self, *args):
        """
        Solution of the ode.

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
            return self.solve_1D(at, bt, m0, s0, sigma)
        else:
            return self.solve_nD(at, bt, m0, s0, sigma)
        # _end_id_

    # _end_def_

    def solve_1D(self, lin_a, off_b, m0, s0, sigma):
        """
        Heun integration method 1D.

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

        # Half step-size.
        h = 0.5 * self.dt

        # Run through all time points.
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
            ftp = -ak * mk + bk

            # Correction step:
            ftc = -a_plus * (mk + ftp * self.dt) + b_plus

            # NEW "mean" point.
            mt[k + 1] = mk + h * (ftp + ftc)

            # -Eq(10)- Prediction step:
            ftp = -2.0 * ak * sk + sigma

            # Correction step:
            ftc = -2.0 * a_plus * (sk + ftp * self.dt) + sigma

            # NEW "variance" point.
            st[k + 1] = sk + h * (ftp + ftc)
        # _end_for_

        # Return the marginal moments.
        return mt, st
    # _end_def_

    def solve_nD(self, lin_a, off_b, m0, s0, sigma):
        """
        Heun integration method nD.

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

        # Half step-size.
        h = 0.5 * self.dt

        # Run through all time points.
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
            ftp = self.fun_mt(mk, ak, bk)

            # Correction step:
            ftc = self.fun_mt((mk + ftp * self.dt), a_plus, b_plus)

            # NEW "mean" point.
            mt[k + 1] = mk + h * (ftp + ftc)

            # -Eq(10)- Prediction step:
            ftp = self.fun_ct(sk, ak, sigma)

            # Correction step:
            ftc = self.fun_ct((sk + ftp * self.dt), a_plus, sigma)

            # NEW "variance" point.
            st[k + 1] = sk + h * (ftp + ftc)
        # _end_for_

        # Return the marginal moments.
        return mt, st
    # _end_def_

# _end_class_
