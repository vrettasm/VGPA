import numpy as np
from src.numerics.utilities import log_det, chol_inv


class PriorKL0(object):
    """
    Prior (Gaussian) moments at t=0 - KL0.
    """

    __slots__ = ("mu0", "tau0", "single_dim")

    def __init__(self, mu0, tau0, single_dim: bool = True) -> None:
        """
        Default constructor.

        :param mu0: prior moment for the mean (dim_d x 1).

        :param tau0: prior moment for the co-variance (dim_d x dim_d).
        """
        # Prior mean values (t=0).
        self.mu0 = np.asarray(mu0)

        # Prior co-variance (t=0).
        self.tau0 = np.asarray(tau0)

        # Single / multiple dimensional.
        self.single_dim = single_dim
    # _end_def_

    def __call__(self, m0, s0):
        """
        Compute the KL0. This is a convenience method that
        will call automatically the correct version of the
        gaussian function.

        :param m0: marginal mean at t=0, (dim_d x 1).

        :param s0: marginal co-variance at t=0, (dim_d x dim_d).

        :return: energy of the initial state KL0, (scalar).
        """
        # Switch according to the system dimensions.
        return self.gauss_1d(m0, s0) if self.single_dim else self.gauss_nd(m0, s0)
    # _end_def_

    def gauss_1d(self, m0: np.ndarray, s0: np.ndarray) -> np.ndarray:
        """
        1D Gaussian version.

        :param m0: marginal mean at t=0, (scalar).

        :param s0:marginal variance at t=0, (scalar).

        :return: energy of the initial state KL0, (scalar).
        """
        # Pre-compute once.
        z0 = m0 - self.mu0

        # Energy of the initial moment.
        kl0 = - np.log(s0) - 0.5 * (1.0 - np.log(self.tau0)) +\
              0.5 / self.tau0 * (z0 ** 2 + s0)

        # Kullback-Liebler at t=0.
        return kl0
    # _end_def_

    def gauss_nd(self, m0: np.ndarray, s0: np.ndarray) -> np.ndarray:
        """
        nD Gaussian version.

        :param m0: marginal mean at t=0, (dim_d x 1).

        :param s0: marginal variance at t=0, (dim_d x dim_d).

        :return: energy of the initial state KL0, (scalar).
        """
        # Inverse of tau0 matrix.
        inv_tau0, _ = chol_inv(self.tau0)

        # Inverted.
        inv_s0, _ = chol_inv(s0)

        # Energy of the initial moment.
        z0 = m0 - self.mu0

        # Energy of the initial moment.
        kl0 = 0.5 * (log_det(self.tau0.dot(inv_s0)) +
                     np.sum(np.diag(inv_tau0.dot(z0.T.dot(z0) +
                                                 s0 - self.tau0))))
        # Kullback-Liebler at t=0.
        return kl0
    # _end_def_

    def gradients(self, m0, s0, lam0, psi0):
        """
        Gradient of KL0 w.r.t. the initial mean / variance.

        :param m0: marginal mean at t=0, (dim_d x 1).

        :param s0: marginal variance at t=0, (dim_d x dim_d).

        :param lam0: Lagrange multiplier (lambda) at t=0, (dim_d x 1).

        :param psi0: Lagrange multiplier (Psi) at t=0, (dim_d x dim_d).

        :return: Gradient of KL0 w.r.t. the initial mean / variance.
        """
        # Switch according to the system dimensions.
        if self.single_dim:
            return self.gradients_1d(m0, s0, lam0, psi0)
        else:
            return self.gradients_nd(m0, s0, lam0, psi0)
    # _end_def_

    def gradients_1d(self, m0: np.ndarray, s0: np.ndarray,
                     lam0: np.ndarray, psi0: np.ndarray):
        """
        1D Gradient of KL0.

        :param m0: marginal mean at t=0, (scalar).

        :param s0: marginal variance at t=0, (scalar).

        :param lam0: Lagrange multiplier (lambda) at t=0, (scalar).

        :param psi0: Lagrange multiplier (Psi) at t=0, (scalar).

        :return: Gradient of KL0 w.r.t. the initial mean / variance.
        """
        # Auxiliary variable.
        z0 = m0 - self.mu0

        # Gradient w.r.t. 'm(t=0)'.
        dKL0_dm0 = lam0 + z0 / self.tau0

        # Gradient w.r.t. 's(t=0)'.
        dKL0_ds0 = psi0 + 0.5 * (1.0 / self.tau0 - 1.0 / s0)

        # Gradient w.r.t marginal moments.
        return dKL0_dm0, dKL0_ds0
    # _end_def_

    def gradients_nd(self, m0: np.ndarray, s0: np.ndarray,
                     lam0: np.ndarray, psi0: np.ndarray):
        """
        nD Gradient of KL0.

        :param m0: marginal mean at t=0, (dim_d x 1).

        :param s0: marginal variance at t=0, (dim_d x dim_d).

        :param lam0: Lagrange multiplier (lambda) at t=0, (dim_d x 1).

        :param psi0: Lagrange multiplier (Psi) at t=0, (dim_d x dim_d).

        :return: Gradient of KL0 w.r.t. the initial mean / variance.
        """
        # Inverse of tau0 matrix.
        inv_tau0, _ = chol_inv(self.tau0)

        # Inverted marginal variance.
        inv_s0, _ = chol_inv(s0)

        # Auxiliary variable.
        z0 = m0 - self.mu0

        # Gradient w.r.t. 'm(t=0)'.
        dKL0_dm0 = lam0 + np.linalg.solve(self.tau0, z0.T).T

        # Gradient w.r.t. 's(t=0)'.
        dKL0_ds0 = psi0 + 0.5 * (inv_tau0 - inv_s0)

        # Gradient w.r.t marginal moments.
        return dKL0_dm0, dKL0_ds0
    # _end_def_

# _end_class_
