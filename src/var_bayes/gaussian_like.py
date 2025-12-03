import numpy as np
from src.var_bayes.likelihood import Likelihood
from src.numerics.utilities import log_det, chol_inv


class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood function.
    """

    # Store log(2*pi), as class member.
    LOG2PI = np.log(2.0 * np.pi)

    __slots__ = ("single_dim",)

    def __init__(self, values, times, noise, operator,
                 single_dim: bool = True) -> None:
        """
        Default constructor of the Gaussian Likelihood object.

        :param values: observation values.

        :param times: observation times.

        :param noise: observation noise.

        :param operator: observation operator.

        :param single_dim: single dimension flag.
        """
        # Call the constructor of the parent class.
        super().__init__(values, times, noise, operator)

        # Marks whether the likelihood corresponds to
        # a single or multiple dimensions observations.
        self.single_dim = single_dim
    # _end_def_

    def __call__(self, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian likelihood function. This is a convenience
        method that will call automatically the correct version of the
        Gaussian likelihood.

        :param m: marginal means m(t), (dim_n x dim_d).

        :param s: marginal variances s(t), (dim_n x dim_d x dim_d).

        :return: Energy from the observation likelihood, (scalar).
        """
        return self.gauss_1d(m, s) if self.single_dim else self.gauss_nd(m, s)
    # _end_def_

    def gradients(self, m: np.ndarray, s: np.ndarray):
        """
        Compute the gradients of the Gaussian likelihood function.
        This is a convenience method that will call automatically
        the correct version of the gradient.

        :param m: marginal means m(t), (dim_n x dim_d).

        :param s: marginal variances s(t), (dim_n x 1).

        :return: Energy from the observation likelihood, (scalar).
        """
        return self.gradients_1d(m, s) if self.single_dim else self.gradients_nd(m)
    # _end_def_

    def gauss_1d(self, m: np.ndarray, s: np.ndarray):
        """
        Gaussian 1D method.

        :param m: marginal means m(t), (dim_n x 1).

        :param s: marginal variances s(t), (dim_n x 1).

        :return: Energy from the observation likelihood, (scalar).
        """
        # Extract the observation values and times
        # from the parent class (Likelihood).
        obs_y, obs_t = self.values, self.times

        # Total number of observations.
        dim_m = obs_t.size

        # Second order Gaussian moments,
        # at observation times only.
        Ex2 = (m[obs_t] ** 2) + s[obs_t]

        # Energy from the observations.
        Eobs = 0.5 * np.sum((obs_y ** 2) - 2.0 * obs_y * m[obs_t] + Ex2) / self.noise +\
               0.5 * dim_m * (GaussianLikelihood.LOG2PI + np.log(self.noise))

        # Energy term from the observations.
        return Eobs
    # _end_def_

    def gauss_nd(self, m: np.ndarray, s: np.ndarray):
        """
        Gaussian nD method.

        :param m: marginal means m(t), (dim_n x dim_d).

        :param s: marginal variances s(t), (dim_n x dim_d x dim_d).

        :return: Energy from the observation likelihood, (scalar).
        """
        # Extract the observation values and times
        # from the parent class (Likelihood).
        obs_y, obs_t = self.values, self.times

        # Total number of observations.
        dim_m, dim_o = obs_y.shape

        # Observation operator.
        H = self.operator

        # Pre-compute this only once.
        W = (obs_y - m[obs_t]).dot(H)

        # Inverted Cholesky factor.
        inv_rn, inv_chol_factor = chol_inv(self.noise)

        # Extract the diagonal elements.
        diag_inv_rn = inv_rn.diagonal()

        # Pre-compute this only once.
        Z = W.dot(inv_chol_factor.T)

        # Energy term from the observations.
        Eobs = 0.0

        # Get the diagonal elements.
        # NOTE: Array s is 3-dimensional. The following command
        # returns the diagonal elements on the first dimensions.
        # I.e. np.diag(s[0, :, :]), ... , np.diag(s[n-1, :, :])
        sn_diag = np.diagonal(s, axis1=1, axis2=2)

        # Sum the energy iteratively.
        for n, tn in enumerate(obs_t):
            # Inner product: z * z.T
            zn_sq = np.inner(Z[n], Z[n])

            # Energy term.
            Eobs += zn_sq + np.inner(diag_inv_rn, sn_diag[n])
        # _end_for_

        # Compute the final including the constants.
        Eobs = 0.5 * (Eobs + dim_m * (dim_o * GaussianLikelihood.LOG2PI + log_det(self.noise)))

        # Energy from the (noisy) observation set.
        return Eobs
    # _end_def_

    def gradients_1d(self, m: np.ndarray, s: np.ndarray):
        """
        1D gradients of Eobs.

        :param m: marginal means m(t), (dim_n x 1).

        :param s: marginal variances s(t), (dim_n x 1).

        :return: dEobs_dm, dEobs_ds, dEobs_dr.
        """
        # Extract the observation values and times
        # from the parent class (Likelihood).
        obs_y, obs_t = self.values, self.times

        # Observation operator.
        H = self.operator

        # Second order Gaussian moments,
        # at observation times only.
        Ex2 = (m[obs_t] ** 2) + s[obs_t]

        # Get the total number of discrete times.
        dim_n = m.shape[0]

        # Pre-compute this only once.
        W = (obs_y - H * m[obs_t])

        # Gradients of Eobs w.r.t. marginal m(t) and s(t).
        dEobs_dm = np.zeros(dim_n)
        dEobs_ds = np.zeros(dim_n)
        dEobs_dr = np.zeros(dim_n)

        # Jumps -Eq(31)- NOT IN THE PAPER.
        dEobs_dm[obs_t] = - W / self.noise

        # Jumps -Eq(32)- NOT IN THE PAPER.
        dEobs_ds[obs_t] = 0.5 / self.noise

        # Calculate the gradient at 'M' observation times.
        dEobs_dr[obs_t] = -0.5 * ((obs_y ** 2) - 2.0 * obs_y * m[obs_t] + Ex2 + 1.0) / self.noise

        # Gradients of energy Eobs.
        return dEobs_dm, dEobs_ds, dEobs_dr
    # _end_def

    def gradients_nd(self, m: np.ndarray):
        """
        nD gradients of Eobs.

        :param m: marginal means m(t), (dim_n x dim_d).

        :return: Energy from the observation likelihood, (scalar).
        """
        # Extract the observation values and times
        # from the parent class (Likelihood).
        obs_y, obs_t = self.values, self.times

        # Dimensions of observations.
        dim_o, _ = obs_y.shape

        # Dimensions of input array.
        dim_n, dim_d = m.shape

        # Observation operator.
        H = self.operator

        # Pre-compute this only once.
        W = (obs_y - m[obs_t]).dot(H)

        # Inverted Cholesky factor.
        inv_rn, _ = chol_inv(self.noise)

        # Preallocate memory for the gradients.
        dEobs_dm = np.zeros((dim_n, dim_d))
        dEobs_ds = np.zeros((dim_n, dim_d, dim_d))
        dEobs_dr = np.zeros((dim_n, dim_o, dim_o))

        # Sum the energy iteratively.
        for n, tn in enumerate(obs_t):
            # Gradient of E_{obs} w.r.t. m(tn).
            dEobs_dm[tn] = -H.T.dot(inv_rn).dot(W[n])

            # Gradient of E_{obs} w.r.t. S(tn).
            dEobs_ds[tn] = 0.5 * H.T.dot(inv_rn).dot(H)
        # _end_for_

        # Gradients of energy Eobs.
        return dEobs_dm, dEobs_ds, dEobs_dr
    # _end_def_

# _end_class_
