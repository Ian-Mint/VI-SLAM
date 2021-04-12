import unittest

import numpy as np

from sensors import *
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


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        with np.load('data/features.npz') as data:
            k = data['K']
            b = data['b']
            features = data.get('features', None)  # for when we make our own features
            linear_velocity = data['linear_velocity']
            angular_velocity = data['angular_velocity']
            imu_T_cam = data['imu_T_cam']
            timestamps = data['time_stamps']

        time_steps = np.diff(timestamps).squeeze()
        linear_velocity = linear_velocity[:, :-1]
        angular_velocity = angular_velocity[:, :-1]
        features = features[:, :-1]
        n_points = features.shape[1]
        n_samples = len(time_steps)

        accel_var = 1e-4
        gyro_var = 1e-6
        imu_variance = np.array([accel_var, accel_var, accel_var, gyro_var, gyro_var, gyro_var])
        self.runner = Runner(Camera(features, time_steps, k, b, imu_T_cam, depth_threshold=50),
                             Imu(linear_velocity, angular_velocity, time_steps, imu_variance),
                             Map(n_points),
                             n_samples, plot_interval=5000)

    @unittest.skip
    def test_points_to_observations_is_inverse_of_observations_to_points(self):
        # note that this is likely to produce impossible observations
        observation = np.random.random((4, 100)) * 100
        points_from_obs = self.runner.observation_to_points(observation)
        _, _, obs_again = self.runner.points_to_observations(points_from_obs)

        self.assertTrue(np.allclose(observation, obs_again))

    def test_observations_to_points_is_inverse_of_points_to_observations(self):
        points = np.random.random((3, 100)) * 1000
        _, _, obs_from_points = self.runner.points_to_observations(points)
        points_again = self.runner.observation_to_points(obs_from_points)
        _, _, obs_from_points_again = self.runner.points_to_observations(points_again)

        self.assertTrue(np.allclose(points, points_again))
        self.assertTrue(np.allclose(obs_from_points, obs_from_points_again))


if __name__ == '__main__':
    unittest.main()
