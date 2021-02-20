import unittest
import math
from math import cos, sin

import numpy as np

import sensors


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


class TestLidar(unittest.TestCase):
    def setUp(self) -> None:
        self.lidar = sensors.Lidar()

    def test_time_is_np_array(self):
        self.assertIsInstance(self.lidar.time, np.ndarray)

    def test_time_is_floats(self):
        self.assertEqual(self.lidar.time.dtype, np.float)

    def test_time_has_one_dim(self):
        self.assertEqual(self.lidar.time.ndim, 1)

    def test_rotation_matrix_is_consistent_with_rpy(self):
        R = R_from_RPY(*self.lidar.rpy)
        self.assertTrue(np.allclose(R, self.lidar.rotation))

    def test_all_timestamps_unique(self):
        self.assertEqual(len(np.unique(self.lidar.time)), len(self.lidar.time))

if __name__ == '__main__':
    unittest.main()
