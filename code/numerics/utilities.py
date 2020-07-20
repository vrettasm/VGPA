import numpy as np

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
    return grad_n[0] if grad_n.ndim == 1 else grad_n
# _end_def_
