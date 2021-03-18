from ..numerics.heun import Heun
from ..numerics.euler import Euler
from ..numerics.runge_kutta2 import RungeKutta2
from ..numerics.runge_kutta4 import RungeKutta4


class FwdOde(object):
    """
    Forward ODE integration methods for the Var.GP.Approximation algorithm.

    This class implements a set of forward ode integration methods for the
    computation of the marginal posterior moments "m(t)" and "s(t)" of the
    variational GP algorithm.
    """

    __slots__ = ("dt", "method", "solver")

    def __init__(self, dt, method, single_dim=True):
        """
        Default constructor of forwards ode solver.

        :param dt: discrete time step.

        :param method: of integration.

        :param single_dim: flags the ode as 1D or nD.
        """

        # Check if time step is positive.
        if dt > 0.0:
            self.dt = dt
        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Discrete time step should be strictly positive -> {dt}.")
        # _end_if_

        # Convert method to lower-case.
        method_str = str(method).lower()

        # Create the solver object.
        if method_str == "euler":

            self.solver = Euler(dt, single_dim)

        elif method_str == "heun":

            self.solver = Heun(dt, single_dim)

        elif method_str == "rk2":

            self.solver = RungeKutta2(dt, single_dim)

        elif method_str == "rk4":

            self.solver = RungeKutta4(dt, single_dim)
        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Integration method is unknown -> {method}.")
        # _end_if_

        # Get a copy of the method name (for the __str__).
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
    def __str__(self):
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
