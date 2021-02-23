from functools import lru_cache
import unittest
from unittest.mock import patch, MagicMock
import math
from math import cos, sin
import time

import numpy as np

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


# noinspection PyPep8Naming
class TestRunner(unittest.TestCase):
    @patch('sensors.Encoder')
    @patch('sensors.Gyro')
    def test_lidar_single_line(self, MockEncoder, MockGyro):
        runner = Runner(
            MockEncoder(),
            MockGyro(),
            Lidar(data_file='data/test/lidar.csv'),
            Car(n_particles=100),
            Map(),
        )
        runner.step_gyro = MagicMock(runner.step_gyro)
        runner.step_encoder = MagicMock(runner.step_encoder)

        runner.run()
        runner.map.show('test_lidar_single_line')

    @patch('sensors.Encoder')
    @patch('sensors.Gyro')
    def test_lidar_time(self, MockEncoder, MockGyro):
        runner = Runner(
            MockEncoder(),
            MockGyro(),
            Lidar(data_file='data/sensor_data/lidar.csv'),
            Car(n_particles=100),
            Map(),
        )
        runner.step_gyro = MagicMock(runner.step_gyro)
        runner.step_encoder = MagicMock(runner.step_encoder)

        lidar_len = len(runner.lidar)
        start = time.time()
        runner.run()
        iteration_duration = (time.time() - start) / lidar_len
        print(f'single scan time {iteration_duration :04f}')


if __name__ == '__main__':
    unittest.main()
