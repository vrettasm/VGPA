import unittest
import numpy as np
from math import isclose
from numpy.linalg import det
from code.numerics.utilities import finite_diff, log_det

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

    def test_log_det(self):
        # Test scalar input:
        x0 = 3.9

        # Log(Det(x0)).
        y0 = log_det(x0)

        # In this case is log(x).
        self.assertEqual(y0, np.log(x0))

        # Test 1D array.
        x1 = np.random.rand(3)

        # Log(Det(x1)).
        y1 = log_det(x1)

        # In this case is log(det(diag(x1))).
        self.assertEqual(y1, np.log(det(np.diag(x1))))

        # Test 2D array.
        x2 = np.array([[0.95276191, 0.43397299, 0.14783354],
                       [0.41602952, 0.68498452, 0.30897903],
                       [0.72359495, 0.29639497, 0.61168427]])

        # Log(Det(x2)).
        y2 = log_det(x2)

        # Compare to log(det(x2)).
        self.assertTrue(isclose(y2, -3.525679097590135))
    # _end_def_


if __name__ == '__main__':
    unittest.main()
