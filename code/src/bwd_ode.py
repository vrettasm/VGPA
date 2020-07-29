from ..numerics.heun import Heun
from ..numerics.euler import Euler
from ..numerics.runge_kutta2 import RungeKutta2
from ..numerics.runge_kutta4 import RungeKutta4


class BwdOde(object):
    """
    Backward ODE integration methods for the Var.GP.Approximation algorithm.

    This class implements a set of  backward ode integration methods for
    the computation of the Lagrange multipliers "lam(t)" and "psi(t)" of
    the VGPA algorithm.
    """

    __slots__ = ("dt", "method", "solver")

    def __init__(self, dt, method, single_dim=True):
        """
        Default constructor of backwards ode solver.

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
            raise ValueError(" {0}: Integration method is unknown"
                             " -> {1}.".format(self.__class__.__name__, method))
        # _end_if_

        # Get a copy of the method name (for the __str__).
        self.method = method
    # _end_def_

    def __call__(self, at, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds):
        """
        Call the bwd solve method of the solver object.
        This is the uniform interface af all methods.

        :param at: Linear variational parameters.

        :param dEsde_dm: Derivative of Esde w.r.t. m(t).

        :param dEsde_ds: Derivative of Esde w.r.t. s(t).

        :param dEobs_dm: Derivative of Eobs w.r.t. m(t).

        :param dEobs_ds: Derivative of Eobs w.r.t. s(t).

        :return: the result of the solver (Lagrange multipliers).
        """
        # Return the solution of the fwd-ode.
        return self.solver.solve_bwd(at, dEsde_dm, dEsde_ds, dEobs_dm, dEobs_ds)
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(), along with its fields values.

        :return: a string representation of a WaterContent object.
        """
        return " BwdOde Id({0}):"\
               " dt={1}, method={2}".format(id(self), self.dt, self.method)
    # _end_def_

# _end_class_
