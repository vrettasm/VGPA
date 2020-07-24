from code.numerics.heun import Heun
from code.numerics.euler import Euler
from code.numerics.runge_kutta2 import RungeKutta2
from code.numerics.runge_kutta4 import RungeKutta4


class FwdOde(object):
    """
    Forward ODE integration methods for the Var.GP.Approximation algorithm.

    This class implements a set of forward ode integration methods for the
    computation of the marginal posterior moments "m(t)" and "s(t)" of the
    variational GP algorithm.  To make it easier and more adaptable to its
    input the algorithm detects the dimensions of the state vector (dim_d)
    and calls either the -1D or -nD version of the selected algorithm.
    This is due to significant performance gains when writing the code in
    -1D rather than -nD.
    """

    __slots__ = ("dt", "method", "solver")

    def __init__(self, dt, method):
        """
        Default constructor of forwards ode solver.

        :param dt: discrete time step.

        :param method: of integration.
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

                self.solver = Euler(dt)

            elif str.lower(method) == "heun":

                self.solver = Heun(dt)

            elif str.lower(method) == "rk2":

                self.solver = RungeKutta2(dt)

            elif str.lower(method) == "rk4":

                self.solver = RungeKutta4(dt)
            # _end_if_
        else:
            raise ValueError(" {0}: Integration method is unknown"
                             " -> {1}.".format(self.__class__.__name__, method))
        # _end_if_
    # _end_def_

    def __call__(self, *args):
        """
        Call the solver method.

        :param args: dictionary with variational parameters.

        :return: marginal means and co-variances.
        """
        return self.solver(*args)
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
