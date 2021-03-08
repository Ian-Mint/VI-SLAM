import math
import unittest
from math import cos, sin

import numpy as np

from sensors import vee, hat, adj_hat


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


class TestMath(unittest.TestCase):
    def test_vee_is_inverse_of_hat(self):
        x = np.arange(6) + 1
        result = vee(hat(x))
        self.assertTrue(np.alltrue(result == x))

    def test_adj_hat_is_expected(self):
        x = np.arange(6) + 1
        expected = np.array([
            [ 0, -6,  5,  0, -3,  2],
            [ 6,  0, -4,  3,  0, -1],
            [-5,  4,  0, -2,  1,  0],
            [ 0,  0,  0,  0, -6,  5],
            [ 0,  0,  0,  6,  0, -4],
            [ 0,  0,  0, -5,  4,  0]
        ])
        result = adj_hat(x)
        self.assertTrue(np.alltrue(result == expected))


if __name__ == '__main__':
    unittest.main()
