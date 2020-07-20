import unittest
import numpy as np
from math import isclose
from code.numerics.utilities import finite_diff

class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestUtilities - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestUtilities - FINISH -")
    # _end_def_

    def test_finite_diff(self):
        # Test point:
        x0 = 0.0

        # The result should be zero.
        self.assertTrue(isclose(finite_diff(np.cos, x0), 0.0))

        # The result should be one.
        self.assertTrue(isclose(finite_diff(np.exp, x0), 1.0))

        # More complex function.
        fun_1 = lambda x: np.sin(5 * x)

        # Test point.
        x0 = 1.0

        # True value.
        f_true = 5.0 * np.cos(5 * x0)

        # Test.
        self.assertTrue(isclose(finite_diff(fun_1, x0), f_true))

        # Even more complex function.
        fun_2 = lambda x: np.exp(np.sin(x))

        # Test point.
        x0 = 1.0

        # True value.
        f_true = np.cos(x0)*np.exp(np.sin(x0))

        # Test.
        self.assertTrue(isclose(finite_diff(fun_2, x0), f_true))
    # _end_def_


if __name__ == '__main__':
    unittest.main()
