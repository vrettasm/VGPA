from src.numerics.utilities import num_integration

class BwdOde(object):
    """
    Backward ODE integration methods for the Var.GP.Approximation algorithm.

    This class implements a set of  backward ode integration methods for the
    computation of the Lagrange multipliers "lam(t)" and "psi(t)" of the VGPA
    algorithm.
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
        Override to print a readable string presentation of
        the object. This will include its id(), along with
        its fields values.

        :return: a string representation of a BwdOde object.
        """
        return f" BwdOde Id({id(self)}):"\
               f" dt={self.dt}, method={self.method}"
    # _end_def_

# _end_class_
