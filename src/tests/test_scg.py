import unittest
import numpy as np
from src.numerics.optim_scg import SCG
from src.numerics.utilities import finite_diff

#
# NOTE: THESE TESTS WILL FAIL UNLESS WE CHANGE THE FUNCTION CALL
# IN OPTIM_SCG.PY (LINE=167)
#
# FROM: g_plus = self.df(x_plus, eval_fun=True) <--
#   TO: g_plus = self.df(x_plus) <--
#
# THE REASON IS THAT WE TWEAKED THE OPTIMIZER TO RUN ON THE VGPA
# ALGORITHM.
#

class TestSCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestSCG - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestSCG - FINISH -", end="\n\n")
    # _end_def_

    def test_sphere_fun(self):
        """
        Test the Sphere Function (n=2).

            f(x) = \Sum_{i=1}^{n} x_i^2

        the global minimum is found at: f(0,0,...,0) = 0.

        :return: None.
        """
        # Sphere function.
        fun = lambda x: np.sum(x ** 2)

        # Gradient (numerical).
        gradf = lambda x: finite_diff(fun, x)

        # Initial random point.
        x0 = np.random.randn(4)

        # Create the SCG.
        optim_fun = SCG(fun, gradf)

        # Run the optimization with default parameters.
        x_opt, fx_opt = optim_fun(x0)

        # The global minimum should be zero.
        self.assertTrue(np.allclose(fx_opt, 0.0))

        # Also, the position 'x' should be zero.
        self.assertTrue(np.allclose(x_opt, 0.0, atol=1.0e-4))
    # _end_def_

    def test_rosenbrock_fun(self):
        """
        Test the Rosenbrock Function (n=2).

            f(x) = 100*(x1 - x0^2)^2 + (1 - x0)^2

        the global minimum is found at: f(1,1) = 0.

        :return: None.
        """
        # Sphere function.
        fun = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        # Gradient (numerical).
        gradf = lambda x: finite_diff(fun, x)

        # Initial random point.
        x0 = np.random.randn(2)

        # Create the SCG, with optional parameters.
        optim_fun = SCG(fun, gradf, {"max_it": 50, "display": True})

        # Run the optimization with default parameters.
        x_opt, fx_opt = optim_fun(x0)

        # The global minimum should be zero.
        self.assertTrue(np.allclose(fx_opt, 0.0))

        # The position 'x' should be one.
        self.assertTrue(np.all(np.abs(x_opt - 1.0) <= 1.0e-4))
    # _end_def_


if __name__ == '__main__':
    unittest.main()
