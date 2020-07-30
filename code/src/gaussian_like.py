import numpy as np
from .likelihood import Likelihood
from ..numerics.utilities import log_det, chol_inv

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood function.
    """

    __slots__ = ("single_dim", "log2pi")

    def __init__(self, values, times, noise, operator, single_dim=True):
        """
        Default constructor.

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

        # Store log(2*pi), as class member.
        self.log2pi = np.log(2.0 * np.pi)
    # _end_def_

    def __call__(self, m, s):
        """
        Compute the Gaussian likelihood function. This is a
        convenience method that will call automatically the
        correct version of the gauss-likelihood.

        :param m: marginal means m(t), (dim_n x dim_d).

        :param s: marginal variances s(t), (dim_n x dim_d x dim_d).

        :return: Energy from the observation likelihood, (scalar).
        """
        if self.single_dim:
            return self.gauss_1D(m, s)
        else:
            return self.gauss_nD(m, s)
        # _end_if_
    # _end_def_

    def gradients(self, m, s):
        """
        Compute the gradients of the Gaussian likelihood function.
        This is a convenience method that will call automatically
        the correct version of the gradient.

        :param m: marginal means m(t), (dim_n x dim_d).

        :param s: marginal variances s(t), (dim_n x 1).

        :return: Energy from the observation likelihood, (scalar).
        """
        if self.single_dim:
            return self.gradients_1D(m, s)
        else:
            return self.gradients_nD(m)
        # _end_if_
    # _end_def_

    def gauss_1D(self, m, s):
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
               0.5 * dim_m * (self.log2pi + np.log(self.noise))

        # Energy term from the observations.
        return Eobs
    # _end_def_

    def gauss_nD(self, m, s):
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
        dim_m, dim_o = obs_y.size

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

        # Sum the energy iteratively.
        for n, tn in enumerate(obs_t):
            # Self explained.
            Zn = Z[n]

            # Self explained.
            sn_diag = s[tn].diagonal()

            # Energy term.
            Eobs += Zn.T.dot(Zn) + diag_inv_rn.dot(sn_diag.T)
        # _end_for_

        # Compute the final including the constants.
        Eobs = 0.5 * (Eobs + dim_m * (dim_o * self.log2pi + log_det(self.noise)))

        # Energy from the (noisy) observation set.
        return Eobs
    # _end_def_

    def gradients_1D(self, m, s):
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

    def gradients_nD(self, m):
        """
        nD gradients of Eobs.

        :param m: marginal means m(t), (dim_n x dim_d).

        :return: Energy from the observation likelihood, (scalar).
        """

        # Extract the observation values and times
        # from the parent class (Likelihood).
        obs_y, obs_t = self.values, self.times

        # Dimensions of observations.
        dim_o = obs_y.size[-1]

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
