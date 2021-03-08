import math
import unittest
from math import cos, sin

import numpy as np

from functions import hat, vee, adj_hat, expm, inv_pose


class TestMath(unittest.TestCase):
    def test_vee_is_inverse_of_hat(self):
        x = np.arange(6) + 1
        result = vee(hat(x))
        self.assertTrue(np.alltrue(result == x))

    def test_adj_hat_is_expected(self):
        x = np.arange(6) + 1
        expected = np.array([
            [0, -6, 5, 0, -3, 2],
            [6, 0, -4, 3, 0, -1],
            [-5, 4, 0, -2, 1, 0],
            [0, 0, 0, 0, -6, 5],
            [0, 0, 0, 6, 0, -4],
            [0, 0, 0, -5, 4, 0]
        ])
        result = adj_hat(x)
        self.assertTrue(np.alltrue(result == expected))

    def test_inv_pose_matches_np_inv(self):
        pose = expm(hat(np.random.random(6)))
        expected = np.linalg.inv(pose)
        actual = inv_pose(pose)
        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
