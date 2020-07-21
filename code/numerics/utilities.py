import numpy as np
from numba import njit
from scipy.linalg import cholesky


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

    # Unit vector.
    e0 = np.zeros(dim_x)

    # Gradient vector.
    grad_n = np.zeros(dim_x)

    # Step size.
    h_step = 1.0e-6

    # Iterate over all dimensions of 'x'.
    for i in range(dim_x):
        # Switch ON i-th direction.
        e0[i] = 1.0

        # Move a small way in the i-th direction of '+x'.
        f_p = fun(x + h_step * e0, *args)

        # Move a small way in the i-th direction of '-x'.
        f_m = fun(x - h_step * e0, *args)

        # Use central difference formula.
        grad_n[i] = 0.5 * (f_p - f_m) / h_step

        # Switch OFF i-th direction.
        e0[i] = 0.0
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
        # Transform it to diagonal
        # (square) matrix.
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
    This (helper) function prevents the computation of very small, or very large
    values of logarithms that would lead to -/+ inf, by setting predefined LOWER
    and UPPER bounds. The bounds are set as follows:

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

    :return: the log(x) after the values of x have been filtered (dim_n x dim_m).
    """

    # Make sure input is an array.
    x = np.asarray(x)

    # Filter out small and large values.
    x = np.maximum(np.minimum(x, 1.0E+300), 1.0E-300)

    # Return the log() of the filtered input.
    return np.log(x)
# _end_def_
