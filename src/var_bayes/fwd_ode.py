from src.numerics.utilities import num_integration

class FwdOde(object):
    """
    Forward ODE integration methods for the Var.GP.Approximation algorithm.

    This class implements a set of forward ode integration methods for the
    computation of the marginal posterior moments "m(t)" and "s(t)" of the
    variational GP algorithm.
    """

    __slots__ = ("dt", "method", "solver")

    def __init__(self, dt: float, method: str, single_dim: bool = True) -> None:
        """
        Default constructor of forwards ode solver.

        :param dt: discrete time step.

        :param method: of integration.

        :param single_dim: flags the ode as 1D or nD.
        """
        # Check if time step is positive.
        if dt <= 0.0:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Discrete time step should be strictly positive -> {dt}.")
        self.dt = dt

        # Convert method to lower-case.
        method_str = str(method).lower()

        try:
            # Create the solver object.
            self.solver = num_integration[method_str](dt, single_dim)
        except KeyError:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Integration method is unknown -> {method}.")
        # _end_try_

        # Copy of the method name (for the __str__).
        self.method = method
    # _end_def_

    def __call__(self, at, bt, m0, s0, sigma):
        """
        Call the fwd solve method of the solver object.
        This is the uniform interface af all methods.

        :param at: Linear (variational) parameters.

        :param bt: Offset (variational) parameters.

        :param m0: Initial marginal mean (at t=0).

        :param s0: Initial marginal variance (at t=0).

        :param sigma: System noise coefficient.

        :return: the result of the solver (marginal moments).
        """

        # Return the solution of the fwd-ode.
        return self.solver.solve_fwd(at, bt, m0, s0, sigma)
    # _end_def_

    # Auxiliary.
    def __str__(self) -> str:
        """
        Override to print a readable string presentation of
        the object. This will include its id(), along with
        its fields values.

        :return: a string representation of a FwdOde object.
        """
        return f" FwdOde Id({id(self)}):"\
               f" dt={self.dt}, method={self.method}"
    # _end_def_

# _end_class_
