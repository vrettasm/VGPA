import numpy as np
from numpy.linalg import cholesky

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

    # Preallocate array.
    grad_n = np.zeros(dim_x)

    # Unit vector.
    e = np.zeros(dim_x)

    # Step size.
    h = 1.0e-6

    # Check all directions (coordinates of x).
    for i in range(dim_x):
        # Switch ON i-th direction.
        e[i] = 1.0

        # Move a small way in the i-th direction of x.
        f_plus = fun(x + h * e, *args)
        f_minus = fun(x - h * e, *args)

        # Use central difference formula.
        grad_n[i] = 0.5 * (f_plus - f_minus) / h

        # Switch OFF i-th direction
        e[i] = 0.0
    # _end_for_

    # Return the numerical gradient.
    return grad_n[0] if dim_x == 1 else grad_n
# _end_def_

def log_det(x=None):
    """
    Returns the log(det(x)), but more stable and accurate.

    :param x: input array (dim_x x dim_x).

    :return: log(det(x)) (dim_x x dim_x).

    Note: if the input is 1D-vector, it will return the
    log(det()) on the diagonal matrix.
    """

    # Check if input is empty.
    if x is None:
        raise ValueError(" log_det: Input array is Empty! ")
    # _end_if_

    # Make sure input is 1-D.
    x = np.atleast_1d(x)

    # If the input is vector.
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
