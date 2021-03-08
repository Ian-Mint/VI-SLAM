import itertools
import time
from abc import ABC
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.linalg import expm

__all__ = ['Camera', 'Imu', 'Map', 'Vehicle', 'Runner']

np.seterr(divide='raise', invalid='ignore')  # raise an error on divide by zero


@numba.njit()
def _hat_map(x, out):
    """
    The 3x3 hat map

    Args:
        x: 3-vector
        out: 3x3 array

    Returns:
        3x3 x^
    """
    out[0, 1] = -x[2]
    out[0, 2] = x[1]
    out[1, 0] = x[2]
    out[1, 2] = -x[0]
    out[2, 0] = -x[1]
    out[2, 1] = x[0]


@numba.njit()
def _vee_map(x, out):
    """
    The 3x3 vee map
    Args:
        out: 3-vector
        x: 3x3 array

    Returns:
        Inverse of the 3x3 hat map
    """
    out[0] = x[2, 1]
    out[1] = x[0, 2]
    out[2] = x[1, 0]


@numba.njit()
def hat(x):
    x_hat = np.zeros((4, 4))

    x_hat[:3, -1] = x[:3]
    _hat_map(x[3:], x_hat)
    return x_hat


@numba.njit()
def vee(x_hat):
    x = np.zeros(6)

    x[:3] = x_hat[:3, 3]
    _vee_map(x_hat, x[3:])
    return x


@numba.njit()
def adj_hat(x):
    x_hat = np.zeros((6, 6))

    _hat_map(x[3:], x_hat)
    _hat_map(x[3:], x_hat[3:, 3:])
    _hat_map(x[:3], x_hat[:3, 3:])
    return x_hat


# noinspection PyUnresolvedReferences
class Sensor(ABC):
    def __len__(self):
        return len(self._time)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Time delta and sensor output
        """
        if item >= self.__len__():
            raise StopIteration
        return self._time[item], self._data[:, item, ...]


class Imu(Sensor):
    def __init__(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray, time_steps: np.ndarray):
        assert angular_velocity.shape == linear_velocity.shape
        self._time = np.squeeze(time_steps)
        self._data = np.concatenate([linear_velocity, angular_velocity], axis=0)

        self.pose = expm(np.zeros((4, 4)))
        self.cv = np.eye(6)  # todo: initialize covariance

        self.trail = np.zeros((2, len(self._time)))

    def update(self, idx):
        time_delta, twist_rate = self[idx]
        self.pose = self.pose @ expm(time_delta * hat(twist_rate))  # todo: consider preprocessing the second term
        # s = expm(-time * adj_hat(twist_rate))
        # self.cv = s @ self.cv @ s.T + noise

        self.trail[:, idx] = self.xy_coords

    @property
    def xy_coords(self):
        return self.pose[:2, 3]


class Camera(Sensor):
    def __init__(self, features: np.ndarray, time_steps: np.ndarray):
        self._data = features
        self._time = time_steps


class Vehicle:
    def __init__(self):
        pass

    def predict(self, yaw: float, time_step: int):
        v_noise = np.random.normal(loc=0, scale=self.v_var, size=len(self))
        omega_noise = np.random.normal(loc=0, scale=self.omega_var, size=len(self))

        translation = time_step * (self.velocity + v_noise)
        dtheta = yaw + time_step * omega_noise
        self.yaw += dtheta

        dx = translation * np.cos(self.yaw)
        dy = translation * np.sin(self.yaw)

        self.position[:, 0] += dx
        self.position[:, 1] += dy

    def transform_ml(self, x):
        """
        Transform x into the world frame, using the pose of the maximum likelihood particle

        Args:
            x: a set of coordinates (samples, 3)

        Returns:
            x transformed into the world frame
        """
        return self.ml_pose.transform(x)

    def transform_all(self, x):
        """
        Transform x into the world frame, broadcasting into all particles.

        Args:
            x: a set of coordinates (samples, 3)

        Returns:
            x transformed into the world frame
        """
        return self.pose.transform(x)

    def update(self, new_weights: np.ndarray):
        self.weights[:] *= new_weights
        self.weights[:] /= np.sum(self.weights)
        if self.n_eff < self.resample_threshold:
            self.resample()


@numba.njit()
def coord_to_cell(point: np.ndarray, minima: np.ndarray, resolution: float) -> np.ndarray:
    """
    Discretize according to the map layout.
    Convert from xy (m) coordinates to ij (px) coordinates.

    Args:
        minima:
        resolution:
        point: a point or set of points in meters

    Returns:
        point or set of points in coordinate indices
    """
    return np.ceil((point - minima) / resolution).astype(np.int16) - 1


# @numba.njit()
def get_coords(disparity) -> np.ndarray:
    """
    Return xyz coordinates in the camera frame associated with each pixel

    Args:
        disparity: the disparity image

    Returns:
        h*w x 3 array
    """
    b = 475.143600050775 / 1000
    fsu = 7.7537235550066748e+02
    cu = 6.1947309112548828e+02
    cv = 2.5718049049377441e+02

    height, width = disparity.shape

    u = np.arange(width)
    v = np.arange(height)

    z = fsu * b / disparity
    y = np.expand_dims((v - cv) / fsu, axis=1) * z
    x = np.expand_dims((u - cu) / fsu, axis=0) * z
    return np.stack([x, y, z], axis=2)


class Map:
    def __init__(self):
        pass


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, camera: Camera, imu: Imu, vehicle: Vehicle, map_: Map,
                 downsample: int = 1, plot_interval=0):
        """
        Args:
            camera: Camera object
            imu: Gyro object
            vehicle: Car object
            map_: Map object
            downsample: Factor by which to downsample. default=1, i.e. no down-sampling
            plot_interval: Interval at which to update the map plot
        """

        assert isinstance(downsample, int)
        self.plot_interval = plot_interval
        self.downsample = downsample

        self.camera = camera
        self.imu = imu
        self.map = map_

        # plotting
        self._figure = None
        self._ax = None
        self._animation = None
        self._fig_handle = None

        self._plot_number = 0
        self.plot = self.plot

    def run(self):
        print("Run starting")
        report_iterations = int(1e5)

        start = time.time()
        try:
            for i in itertools.count():
                self._step(i)
                if (i + 1) % report_iterations == 0:
                    print(f'Sample {(i + 1) // 1000} thousand in {time.time() - start: 02f}s')
                    start = time.time()
        except StopIteration:
            pass
        self.plot()

    def _step(self, idx):
        self.imu.update(idx)

    def plot(self):
        plt.scatter(self.imu.trail[0], self.imu.trail[1], s=1)
        plt.title("Map")
        plt.xlabel("x distance from start (m)")
        plt.ylabel("y distance from start (m)")
        plt.savefig(f'results/map{self._plot_number}.png')
        plt.show()

    def step_camera(self, idx):
        time_delta, body_coords = self.camera[idx]

        pose = self.imu.pose
        world_coords = pose.transform(body_coords)
        # todo: update map...

    def step_gyro(self, idx):
        time_step, x = self.imu[idx]
        # todo: update position
