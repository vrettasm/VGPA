import unittest
import numpy as np

from var_bayes.gaussian_moments import GaussianMoments

class TestGaussianMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestGaussianMoments - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestGaussianMoments - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates a test object with the random input parameters.

        :return: None.
        """

        # Array size.
        self.n_size = 100

        # Random mean and variance vectors.
        self.v_arr = np.random.rand(self.n_size)
        self.m_arr = np.random.randn(self.n_size)

        # Make sure v_arr is > 0.
        self.v_arr[self.v_arr == 0.0] = 0.1

        # Create the object.
        self.test_obj = GaussianMoments(self.m_arr, self.v_arr)
    # _end_def_

    def test_call(self):
        """
        Test the __call__ method.

        :return: None
        """

        # Zero order moments.
        x_out_0 = self.test_obj(0)

        # True output of 0-th order.
        t_out_0 = np.ones(self.n_size)

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_0 - t_out_0))

        # First order moments.
        x_out_1 = self.test_obj(1)

        # True output of 1-th order.
        t_out_1 = self.m_arr

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_1 - t_out_1))

        # The accepted values are [0-8].
        with self.assertRaises(ValueError):
            self.test_obj(9)
        # _end_with_

    # _end_def_

    def test_dM(self):
        """
        Test the dM() method.

        :return: None
        """

        # First order derivative.
        x_out_1 = self.test_obj.dM(1)

        # True output of 1-th order dM.
        t_out_1 = np.ones(self.n_size)

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_1 - t_out_1))

        # Second order derivative.
        x_out_2 = self.test_obj.dM(2)

        # True output of 2-nd order dM.
        t_out_2 = 2 * self.m_arr

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_2 - t_out_2))

        # The accepted values are [1-8].
        with self.assertRaises(ValueError):
            self.test_obj.dM(0)
        # _end_with_
    # _end_def_

    def test_dS(self):
        """
        Test the dS() method.

        :return: None
        """

        # First order derivative.
        x_out_1 = self.test_obj.dS(1)

        # True output of 1-th order dM.
        t_out_1 = np.zeros(self.n_size)

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_1 - t_out_1))

        # Second order derivative.
        x_out_2 = self.test_obj.dS(2)

        # True output of 2-nd order dM.
        t_out_2 = np.ones(self.n_size)

        # The difference should be zero.
        self.assertEqual(0.0, np.sum(x_out_2 - t_out_2))

        # The accepted values are [1-8].
        with self.assertRaises(ValueError):
            self.test_obj.dS(0)
        # _end_with_
    # _end_def_


if __name__ == '__main__':
    unittest.main()
