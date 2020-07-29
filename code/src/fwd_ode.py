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
            raise ValueError(" {0}: Discrete time step should be"
                             " strictly positive -> {1}.".format(self.__class__.__name__, dt))
        # _end_if_

        # Check if the method is included.
        if method in ["euler", "heun", "rk2", "rk4"]:
            # Get a copy of the method name.
            self.method = method

            # Create the solver object.
            if str.lower(method) == "euler":

                self.solver = Euler(dt, single_dim)

            elif str.lower(method) == "heun":

                self.solver = Heun(dt, single_dim)

            elif str.lower(method) == "rk2":

                self.solver = RungeKutta2(dt, single_dim)

            elif str.lower(method) == "rk4":

                self.solver = RungeKutta4(dt, single_dim)
            # _end_if_
        else:
            raise ValueError(" {0}: Integration method is unknown"
                             " -> {1}.".format(self.__class__.__name__, method))
        # _end_if_
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
        Override to print a readable string presentation of the object.
        This will include its id(), along with its fields values.

        :return: a string representation of a WaterContent object.
        """
        return " FwdOde Id({0}):"\
               " dt={1}, method={2}".format(id(self), self.dt, self.method)
    # _end_def_

# _end_class_
