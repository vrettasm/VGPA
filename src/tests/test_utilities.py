import unittest
import numpy as np
from numpy.linalg import det
from src.numerics.utilities import finite_diff, log_det, safe_log, my_trapz, chol_inv

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
        # Information.
        print(" Testing 'finite_diff' ... ")

        # Test point:
        x0 = 0.0

        # The result should be zero.
        self.assertTrue(np.allclose(finite_diff(np.cos, x0), 0.0))

        # The result should be one.
        self.assertTrue(np.allclose(finite_diff(np.exp, x0), 1.0))

        # More complex function.
        fun_1 = lambda x: np.sin(5 * x)

        # Test point.
        x0 = 1.0

        # True value.
        f_true = 5.0 * np.cos(5 * x0)

        # Test.
        self.assertTrue(np.allclose(finite_diff(fun_1, x0), f_true))

        # Even more complex function.
        fun_2 = lambda x: np.sin(x[0]**2 * x[1]**2)

        # Test point.
        z = np.array([1.5, 2.0])

        # True value.
        f_true = [np.cos(z[0] ** 2 * z[1] ** 2) * 2.0 * z[0] * z[1] ** 2,
                  np.cos(z[0] ** 2 * z[1] ** 2) * 2.0 * z[0] ** 2 * z[1]]
        # Test.
        self.assertTrue(np.allclose(finite_diff(fun_2, z), f_true))
    # _end_def_

    def test_safe_log(self):
        # Information.
        print(" Testing 'safe_log' ... ")

        # Test scalar input:
        x1 = 3.9457

        # Safe log.
        y1 = safe_log(x1)

        # In this case is log(x).
        self.assertTrue(np.allclose(y1, np.log(x1)))

        # Test Lower limit:
        x2 = 1.0E-330
        y2 = safe_log(x2)
        self.assertEqual(y2, np.log(1.0E-300))

        # Test Upper limit:
        x3 = 1.0E+330
        y3 = safe_log(x3)
        self.assertEqual(y3, np.log(1.0E+300))
    # _end_def_

    def test_log_det(self):
        # Information.
        print(" Testing 'log_det' ... ")

        # Test scalar input:
        x0 = 3.9457

        # Log(Det(x0)).
        y0 = log_det(x0)

        # In this case is log(x).
        self.assertTrue(np.allclose(y0, np.log(x0)))

        # Test 1D array.
        x1 = np.random.rand(3)

        # Log(Det(x1)).
        y1 = log_det(x1)

        # In this case is log(det(diag(x1))).
        self.assertTrue(np.allclose(y1, np.log(det(np.diag(x1)))))
    # _end_def_

    def test_my_trapz(self):
        from scipy.integrate import trapz as sp_trapz

        # Information.
        print(" Testing 'my_trapz' ... ")

        # Three function types (scalar, vector, matrix):
        fxs = np.random.rand(1000)
        fxv = np.random.rand(1000, 2)
        fxm = np.random.rand(1000, 2, 2)

        # Discrete step.
        dx = 0.01

        # Test all types.
        for x_val in [fxs, fxv, fxm]:
            # Integrals.
            tot_1 = my_trapz(x_val, dx=dx)
            tot_2 = sp_trapz(x_val, dx=dx, axis=0)

            # Test how close they are.
            self.assertTrue(np.all(np.abs(tot_1 - tot_2) <= 1.0e-8))
        # _end_for_

        # Test with observation times.
        fxo = np.random.rand(1000)

        # Observation times (indexes).
        obs_t = [10, 20, 30, 40, 50, 60, 500]

        # Integrals.
        tot_1 = my_trapz(fxo, dx=dx, obs_t=obs_t)
        tot_2 = sp_trapz(fxo, dx=dx, axis=0)

        # Test how close they are.
        self.assertTrue(np.all(np.abs(tot_1 - tot_2) <= 1.0e-6))
    # _end_def_

    def test_chol_inv(self):
        # Information.
        print(" Testing 'chol_inv' ... ")

        # Scalar value test.
        x = 0.2

        # Invert the input value.
        x_inv, _ = chol_inv(x)

        # Test how close they are.
        self.assertTrue(np.allclose(x * x_inv, 1.0))

        # Diagonal matrix test.
        x = np.diag(np.random.rand(3))

        # Invert the input value.
        x_inv, _ = chol_inv(x)

        # Test how close they are.
        self.assertTrue(np.allclose(x.dot(x_inv), np.eye(3)))

        # Positive definite matrix test.
        N = 10
        x = np.linspace(0.5, 1.5, N * N).reshape(N, N)
        x = 0.5 * (x + x.T) + np.eye(N) * N

        # Invert the input value.
        x_inv, _ = chol_inv(x)

        # Test inverted.
        test_eye = x_inv.dot(x)

        # This is not the right test, but we check how close
        # these two matrices are on average.
        self.assertTrue(np.mean(test_eye - np.eye(N)) <= 1.0e-5)
    # _end_def_


if __name__ == '__main__':
    unittest.main()
