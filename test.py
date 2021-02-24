from functools import lru_cache
import unittest
from unittest.mock import patch, MagicMock
import math
from math import cos, sin
import time

import numpy as np

import sensors
from sensors import *


# noinspection PyPep8Naming
def Rz(yaw: float):
    yaw = math.radians(yaw)
    return np.array([[cos(yaw), -sin(yaw), 0],
                     [sin(yaw), cos(yaw), 0],
                     [0, 0, 1]])


# noinspection PyPep8Naming
def Ry(pitch: float):
    pitch = math.radians(pitch)
    return np.array([[cos(pitch), 0, sin(pitch)],
                     [0, 1, 0],
                     [-sin(pitch), 0, cos(pitch)]])


# noinspection PyPep8Naming
def Rx(roll: float):
    roll = math.radians(roll)
    return np.array([[1, 0, 0],
                     [0, cos(roll), -sin(roll)],
                     [0, sin(roll), cos(roll)]])


# noinspection PyPep8Naming
def R_from_RPY(roll, pitch, yaw):
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


@lru_cache()
def get_lidar():
    return Lidar()


class TestLidar(unittest.TestCase):
    def setUp(self) -> None:
        self.lidar = get_lidar()

    def test_time_is_np_array(self):
        self.assertIsInstance(self.lidar.time, np.ndarray)

    def test_time_is_floats(self):
        self.assertEqual(self.lidar.time.dtype, np.float)

    def test_time_has_one_dim(self):
        self.assertEqual(self.lidar.time.ndim, 1)

    def test_rotation_matrix_is_consistent_with_rpy(self):
        R = R_from_RPY(*self.lidar._rpy)
        self.assertTrue(np.allclose(R, self.lidar._rotation))

    def test_all_timestamps_unique(self):
        self.assertEqual(len(np.unique(self.lidar.time)), len(self.lidar.time))


@unittest.skip
class TestEncoderUpdate(unittest.TestCase):
    def test_get_update_function(self):
        n_samples = 100
        theta_max = 0.24
        L = 2  # wheel base
        self.wheel_base = L

        theta = np.random.uniform(-theta_max, theta_max, n_samples)
        R = np.random.uniform(0, 100, n_samples)
        left_turn = np.random.random(n_samples) > 0.5
        theta_world = theta * (-1) ** left_turn
        v = theta_world * (R - L / 2)
        x = v * np.cos(theta_world)
        y = v * np.sin(theta_world)
        expected = np.stack([x, y, theta_world], axis=1)

        # swap these if we have a left turn
        cl = R * theta
        cr = (R - L) * theta

        distance = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if left_turn[i]:
                distance[i, :] = np.array([cl[i], cr[i]])
            else:
                distance[i, :] = np.array([cr[i], cl[i]])

        result = sensors._get_update(distance)
        # x
        self.assertTrue(np.allclose(expected[:, 0], result[:, 0], atol=1))
        # y
        self.assertTrue(np.allclose(expected[:, 1], result[:, 1]))
        # theta
        self.assertTrue(np.allclose(expected[:, 2], result[:, 2]))


# noinspection PyPep8Naming
class TestRunner(unittest.TestCase):
    def test_lidar_single_line(self):
        runner = self.run_lidar_single_line()
        runner.map.show_likelihood('test_lidar_single_line - likelihood')
        runner.map.show_map('test_lidar_single_line - map')

    @staticmethod
    @patch('sensors.Encoder')
    @patch('sensors.Gyro')
    def run_lidar_single_line(MockEncoder, MockGyro):
        runner = Runner(
            MockEncoder(),
            MockGyro(),
            Lidar(data_file='data/test/lidar.csv'),
            Car(n_particles=100, v_var=1e-3, omega_var=1e-6),
            Map(),
        )
        runner.step_gyro = MagicMock(runner.step_gyro)
        runner.step_encoder = MagicMock(runner.step_encoder)
        runner.run()
        return runner

    # @unittest.skip
    @patch('sensors.Encoder')
    @patch('sensors.Gyro')
    def test_lidar_time(self, MockEncoder, MockGyro):
        runner = Runner(
            MockEncoder(),
            MockGyro(),
            Lidar(data_file='data/sensor_data/lidar.csv'),
            Car(n_particles=100, v_var=1e-3, omega_var=1e-6),
            Map(),
        )
        runner.step_gyro = MagicMock(runner.step_gyro)
        runner.step_encoder = MagicMock(runner.step_encoder)

        lidar_len = len(runner.lidar)
        start = time.time()
        runner.run()
        iteration_duration = (time.time() - start) / lidar_len
        print(f'single scan time {iteration_duration :04f}')

    def test_dead_reckoning(self):
        runner = self.run_dead_reckoning()
        # todo: visualize dead reckoning

    @staticmethod
    @patch('sensors.Lidar')
    def run_dead_reckoning(MockLidar):
        runner = Runner(
            Encoder(data_file='data/test/encoder.csv'),
            Gyro(data_file='data/test/gyro.csv'),
            MockLidar(data_file='data/test/lidar.csv'),
            Car(n_particles=100, v_var=1e-3, omega_var=1e-6),
            Map(),
        )
        runner.step_lidar = MagicMock(runner.step_lidar)
        runner.run()
        return runner


if __name__ == '__main__':
    unittest.main()
