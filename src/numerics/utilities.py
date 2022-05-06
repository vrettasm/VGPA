import numpy as np
from numba import njit
from numpy.linalg import solve, cholesky, LinAlgError


def finite_diff(fun, x, *args):
    """
    Calculates the approximate derivative of function "fun"
    on a parameter vector "x". A central difference formula
    with step size "h" is used, and the result is returned
    in vector "grad_n".

    :param fun: the objective function that we want to check.

    :param x: the point where we want to check the gradient.

    :param args: additional function parameters.

    :return: the gradient calculated numerically.
    """

    # Make sure input is at least 1-D.
    x = np.atleast_1d(x)

    # Number of input parameters.
    dim_x = x.shape[0]

    # Gradient vector.
    grad_n = np.zeros(dim_x)

    # Step size.
    delta_h = 1.0e-6

    # Unit vector.
    e = np.zeros(dim_x)

    # Iterate over all the dimensions.
    for i in range(dim_x):
        # Turn ON the i-th dimension.
        e[i] = delta_h

        # Move a small way 'x + dh'.
        f_p = fun(x + e, *args)

        # Move a small way 'x - dh'.
        f_m = fun(x - e, *args)

        # Use central difference formula.
        grad_n[i] = 0.5 * (f_p - f_m) / delta_h

        # Turn OFF the i-th dimension.
        e[i] = 0.0
    # _end_for_

    # Return the numerical gradient.
    return grad_n[0] if dim_x == 1 else grad_n
# _end_def_

def log_det(x):
    """
    Returns the log(det(x)), but more stable and accurate.

    :param x: input array (dim_x x dim_x).

    :return: log(det(x)) (dim_x x dim_x).

    Note: if the input is 1D-vector, it will return the
    log(det()) on the diagonal matrix.
    """

    # Make sure input is array.
    x = np.asarray(x)

    # If the input is scalar.
    if x.ndim == 0:
        # Return from here with the log.
        return np.log(x)
    # _end_if_

    # If the input is a 1-D vector.
    if x.ndim == 1:
        # Transform it to diagonal matrix.
        x = np.diag(x)
    else:
        # Get the number of rows/cols.
        rows, cols = x.shape

        # Make sure the array is square.
        if rows != cols:
            raise RuntimeError(" log_det: Rows != Cols.")
        # _end_if_
    # _end_if_

    # More stable than: log(det(x)).
    return 2.0 * np.sum(np.log(cholesky(x).diagonal()))
# _end_def_

@njit(fastmath=True)
def safe_log(x):
    """
    This function prevents the computation of very small,
    or very large values of logarithms that would lead to
    -/+ inf, by setting predefined LOWER and UPPER bounds.

    The bounds are set as follows:

        - LOWER = 1.0E-300
        - UPPER = 1.0E+300

    It is assumed that the input values lie within this range.

    Example:
        >> numpy.log(1.0E-350)
        >> -inf
        >>
        >> safe_log(1.0E-350)
        >> -690.77552789821368

    :param x: input array (dim_n x dim_m).

    :return: the log(x) after the values of x have been
    filtered (dim_n x dim_m).
    """

    # Make sure input is an array.
    x = np.asarray(x)

    # Filter out small and large values.
    x = np.maximum(np.minimum(x, 1.0E+300), 1.0E-300)

    # Return the log() of the filtered input.
    return np.log(x)
# _end_def_

def my_trapz(fx, dx=1.0, obs_t=None):
    """
    This method computes the numerical integral
    of the discrete function values 'fx', with
    space increment dt, using the composite
    trapezoidal rule.

    This code applies the function: numpy.trapz()
    between the times of the observations 'obs_t'. This is
    because the function 'fx' is very rough (it jumps at
    observation times), therefore computing the integral
    incrementally we achieve better numerical results.

    If no 'obs_t' is given, then we call directly trapz().

    NOTE: to allow easy vectorization the input values 'fx',
    assume that the first dimension is the one we are integrating
    over. So:

    1) if 'fx' is scalar (dim_n),
    2) if 'fx' is vector (dim_n x dim_d),
    3) if 'fx' is matrix (dim_n x dim_d x dim_d).
    4) etc.

    :param fx: function values (discrete) (dim_n).

    :param dx: discrete step (time-wise) (scalar).

    :param obs_t: observation times (indexes) - optional.

    :return: definite integral as approximated by trapezoidal rule.
    """

    # Check if there are observation times (indexes).
    if obs_t is None:
        return np.trapz(fx, dx=dx, axis=0)
    # _end_if_

    # Total integral.
    tot_area = 0.0

    # First index.
    f = 0

    # Compute the integral partially.
    for k, l in enumerate(obs_t):
        # Compute the integral incrementally.
        tot_area += np.trapz(fx[f:l+1], dx=dx, axis=0)

        # Set the next first index.
        f = obs_t[k]
    # _end_for_

    # Final interval.
    if f != fx.shape[0] - 1:
        tot_area += np.trapz(fx[f:], dx=dx, axis=0)
    # _end_if_

    # Return the total integral.
    return tot_area
# _end_def_

@njit
def chol_inv_fast(x):
    """
    Helper function implemented with numba.

    :param x: array to be inverted.
    """

    # Invert the Cholesky decomposition.
    c_inv = solve(cholesky(x), np.eye(x.shape[0]))

    # Invert input matrix.
    x_inv = c_inv.T.dot(c_inv)

    return x_inv, c_inv
# _end_def_

def chol_inv(x):
    """
    Inverts an input array (matrix) using Cholesky
    decomposition.

    :param x: input array (dim_d x dim_d)

    :return: inverted 'x' and inverted Cholesky factor.
    """

    # Make sure input is array.
    x = np.asarray(x)

    # Check if the input is scalar.
    if x.ndim == 0:
        return 1.0 / x, 1.0 / np.sqrt(x)
    else:
        return chol_inv_fast(x)
# _end_def_

def ut_approx(fun, x_bar, x_cov, *args):
    """
    This method computes the approximate values for the mean and
    the covariance of a multivariate random variable. To achieve
    that, the "Unscented Transformation" (UT) is used.

    Reference:
    Simon J. Julier (1999). "The Scaled Unscented Transformation".
    Conference Proceedings, pp.4555--4559

    :param fun: function of the nonlinear transformation.

    :param x_bar: current mean of the state vector (dim_d x 1).

    :param x_cov: current covariance of the state vector (dim_d x dim_d).

    :param args: additional parameter for the "fun" function.

    :return: 1) y_bar : estimated mean after nonlinear transformation (dim_k x 1)
             2) y_cov : estimated covariance after nonlinear transformation (dim_k x dim_k).
    """

    # Make sure input is arrays.
    x_bar = np.asarray(x_bar)
    x_cov = np.asarray(x_cov)

    # Get the dimensions of the state vector.
    dim_d = x_bar.size

    # Total number of sigma points.
    dim_m = int(2 * dim_d + 1)

    # Scaling factor.
    k = 1.05 * dim_d

    # Use Cholesky to get the lower triangular matrix.
    try:
        sPxx = cholesky((dim_d + k) * x_cov).T
    except LinAlgError:
        # If the original fails, apply only to
        # the diagonal elements of the covariance.
        sPxx = cholesky(x_cov * np.eye(dim_d)).T
    # _end_if_

    # Replicate the array.
    x_mat = np.tile(x_bar, (dim_d, 1))

    # Put all sigma points together.
    chi = np.concatenate((x_bar[np.newaxis, :],
                          (x_mat + sPxx),
                          (x_mat - sPxx)))
    # Compute the weights.
    w_list = [k / (dim_d + k)]
    w_list.extend([1.0 / (2.0 * (dim_d + k))] * (dim_m - 1))
    weights = np.reshape(np.array(w_list), (1, dim_m))

    # Propagate the new points through
    # the non-linear transformation.
    y = fun(chi, *args)

    # Compute the approximate mean.
    y_bar = weights.dot(y).ravel()

    # Compute the approximate covariance.
    w_m = np.eye(dim_m) - np.tile(weights, (dim_m, 1))
    Q = w_m.dot(np.diag(weights.ravel())).dot(w_m.T)

    # Compute the new approximate covariance.
    y_cov = y.T.dot(Q).dot(y)

    # New (mean / covariance).
    return y_bar, y_cov
# _end_def_
