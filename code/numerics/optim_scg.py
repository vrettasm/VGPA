import numpy as np
from .utilities import finite_diff

class SCG(object):
    """
    This class creates a Scaled Conjugate Gradient (SCG) optimization
    object. Attempts to find a local minimum of the function f(x).
    Here 'x0' is a column vector and 'f' must returns a scalar value.

    The minimisation process uses also the gradient 'df' (i.e. df(x)/dx).
    The point at which 'f' has a local minimum is returned as 'x'. The
    function value at that point (the minimum) is returned in "fx".

    Reference Book:
    Ian T. Nabney (2001). "Netlab: Algorithms for Pattern Recognition."
    Advances in Pattern Recognition, Springer.
    """

    __slots__ = ("f", "df", "nit", "x_tol", "f_tol", "display", "stats")

    def __init__(self, f, df=None, *args):
        """

        :param f:

        :param df:

        :param args:
        """

        # Function handle.
        self.f = f

        # Derivative function.
        if df is None:
            self.df = lambda x: finite_diff(self.f, x)
        else:
            self.df = df
        # _end_if_

        # Maximum number of iterations.
        if "max_it" in args:
            self.nit = args["max_it"]
        else:
            self.nit = 150
        # _end_if_

        # Error tolerance in 'x'.
        if "x_tol" in args:
            self.x_tol = args["x_tol"]
        else:
            self.x_tol = 1.0e-6
        # _end_if_

        # Error tolerance in 'fx'.
        if "f_tol" in args:
            self.f_tol = args["f_tol"]
        else:
            self.f_tol = 1.0e-8
        # _end_if_

        # Display statistics flag.
        if "display" in args:
            self.display = args["display"]
        else:
            self.display = False
        # _end_if_

        # Statistics dictionary.
        self.stats = {"MaxIt": self.nit, "fx": np.zeros(self.nit),
                      "dfx": np.zeros(self.nit), "f_eval": 0.0,
                      "df_eval": 0.0, "beta": np.zeros(self.nit)}
    # _end_def_

    def __call__(self, x0, *args):
        """
        The call of the object itself will enable the optimization.

        :param x0: Initial search point.

        :param args: additional function / gradient parameters.

        :return: 1)  x: the point where the minimum was found,
                 2) fx: the function value (at the minimum point).
        """

        # Initialization.
        x = x0.copy()

        # Make input 1-D.
        x = np.atleast_1d(x)

        # Number of input parameters.
        dim_x = x.shape[0]

        # Initial sigma value.
        sigma0 = 1.0e-3

        # Initial function/gradients value.
        fnow = self.f(x, *args)
        gradnew = self.df(x, *args)

        # Increase function / gradient evaluations by one.
        self.stats['f_eval'] += 1
        self.stats['df_eval'] += 1

        # Store the current values.
        fold = fnow

        # Store the current gradient.
        gradold = gradnew

        # Setup the initial search direction.
        d = -gradnew

        # Force calculation of directional derivatives.
        success = 1

        # Counts the number of successes.
        nsuccess = 0

        # Initial scale parameter.
        beta = 1.0

        # Lower & Upper bounds on scale (beta).
        beta_min, beta_max = 1.0e-15, 1.0e+100

        # Initialization of parameters.
        kappa, theta, mu = 0.0, 0.0, 0.0

        # Get the machine precision constant.
        eps_float = np.finfo(float).eps

        # Main optimization loop.
        for j in range(self.nit):

            # Calculate first and second directional derivatives.
            if success == 1:
                # Inner-product.
                mu = d.T.dot(gradnew)

                if mu >= 0.0:
                    d = -gradnew
                    mu = d.T.dot(gradnew)
                # _end_if_

                # Compute kappa.
                kappa = d.T.dot(d)

                # And check for termination.
                if kappa < eps_float:
                    fx = fnow
                    self.stats['MaxIt'] = j

                    # Exit from here.
                    return x, fx
                # _end_if_

                # Update sigma and check the gradient on a new direction.
                sigma = sigma0 / np.sqrt(kappa)
                xplus = x + sigma * d

                # We evaluate the df(xplus).
                gplus = self.df(xplus, *args)

                # Increase function/gradients evaluations by one.
                self.stats['f_eval'] += 1
                self.stats['df_eval'] += 1

                # Compute theta.
                theta = (d.T.dot(gplus - gradnew)) / sigma
            # _end_if_

            # Increase effective curvature and evaluate step size alpha.
            delta = theta + beta * kappa
            if delta <= 0.0:
                delta = beta * kappa
                beta = beta - (theta / kappa)
            # _end_if_

            alpha = -(mu / delta)

            # Evaluate the function at a new point.
            xnew = x + alpha * d
            fnew = self.f(xnew, *args)
            self.stats['f_eval'] += 1

            # Calculate the new comparison ratio.
            Delta = 2.0 * (fnew - fold) / (alpha * mu)
            if Delta >= 0.0:
                success = 1
                nsuccess += 1
                x, fnow, gnow = xnew, fnew, gradnew
            else:
                success = 0
                fnow, gnow = fold, gradold
            # _end_if_

            # Total gradient.
            tot_grad = np.math.fsum(np.abs(gnow))

            # Store statistics.
            self.stats["fx"][j] = fnow
            self.stats["beta"][j] = beta
            self.stats["dfx"][j] = tot_grad

            # Used in debugging mode.
            if self.display:
                print(" {0}:\tfx={1:.3f}\tsum(gx)={2:.3f}".format(j, fnow, tot_grad))
            # _end_if_

            # TBD:
            if success == 1:
                # Check for termination.
                if (np.abs(alpha * d).max() <= self.x_tol) and (np.abs(fnew - fold) <= self.f_tol):
                    fx = fnew
                    self.stats["MaxIt"] = j
                    return x, fx
                else:
                    # Update variables for new position.
                    fold = fnew.copy()
                    gradold = gradnew.copy()

                    # Evaluate function/gradient at the new point.
                    fnow = self.f(x, *args)
                    gradnew = self.df(x, *args)

                    #  Increase function/gradients evaluations by one.
                    self.stats['f_eval'] += 1
                    self.stats['df_eval'] += 1

                    # If the gradient is zero then we are done.
                    if gradnew.T.dot(gradnew) == 0.0:
                        fx = fnow
                        self.stats["MaxIt"] = j
                        return x, fx
                # _end_if_
            # _end_if_

            # Adjust beta according to comparison ratio.
            if Delta < 0.25:
                beta = np.minimum(4.0 * beta, beta_max)
            # _end_if_

            if Delta > 0.75:
                beta = np.maximum(0.5 * beta, beta_min)
            # _end_if_

            # Update search direction using Polak-Ribiere formula,
            # or re-start in direction of negative gradient after
            # 'dim_x' steps.
            if nsuccess == dim_x:
                d = -gradnew
                nsuccess = 0
            else:
                if success == 1:
                    gamma = np.maximum(gradnew.T.dot(gradold - gradnew) / mu, 0.0)
                    d = gamma * d - gradnew
                # _end_if_
            # _end_if_
        # _end_for_

        # Display a final (warning) to the user.
        print(" SGC: Maximum number of iterations has been reached.")

        # Here we have reached the maximum number of iterations.
        fx = fold

        # Exit from here.
        return x, fx
    # _end_def_

    @property
    def statistics(self):
        """
        Accessor method.

        :return: the statistics dictionary.
        """
        return self.stats
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(), along with its fields values.

        :return: a string representation of a SCG object.
        """

        return " SCG Id({0}):" \
               " Function={1}, Gradient={2}, Max-It={3}," \
               " x_tol={4}, f_tol={5}".format(id(self), self.f, self.df, self.nit,
                                              self.x_tol, self.f_tol)
    # _end_def_

# _end_class_
