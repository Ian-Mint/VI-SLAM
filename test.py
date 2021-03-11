import unittest

import numpy as np

from functions import hat, vee, adj_hat, expm, inv_pose, d_pi_dx, pi


def estimate_grad(dq, dq1, q):
    return (pi(q + dq1[:, None]) - pi(q)) / dq


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

    def test_d_pi_dx_broadcasting_works(self):
        x = np.ones(4)
        x[:3] = np.random.random(3)

        expected = (1 / x[2]) * np.array([
            [1, 0, -x[0] / x[2], 0],
            [0, 1, -x[1] / x[2], 0],
            [0, 0, 0, 0],
            [0, 0, -x[3] / x[2], 1]
        ])
        actual = d_pi_dx(x)
        self.assertTrue(np.allclose(expected, actual))

    def test_pi_grad_check(self):
        q = np.random.random((4, 1000)) * 2000 - 1000
        dq = 1e-6
        dq1 = np.array([dq, 0, 0, 0])
        dq2 = np.array([0, dq, 0, 0])
        dq3 = np.array([0, 0, dq, 0])
        dq4 = np.array([0, 0, 0, dq])

        result = d_pi_dx(q)

        estimated_d1 = estimate_grad(dq, dq1, q)
        d1 = result[:, 0]
        self.assertTrue(np.allclose(estimated_d1, d1))

        estimated_d2 = estimate_grad(dq, dq2, q)
        d2 = result[:, 1]
        self.assertTrue(np.allclose(estimated_d2, d2))

        estimated_d3 = estimate_grad(dq, dq3, q)
        d3 = result[:, 2]
        self.assertTrue(np.allclose(estimated_d3, d3))

        estimated_d4 = estimate_grad(dq, dq4, q)
        d4 = result[:, 3]
        self.assertTrue(np.allclose(estimated_d4, d4))


if __name__ == '__main__':
    unittest.main()
