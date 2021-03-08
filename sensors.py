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
def hat(x):
    x_hat = np.zeros((4, 4))

    x_hat[:3, -1] = x[:3]
    x_hat[0, 1] = -x[5]
    x_hat[0, 2] = x[4]
    x_hat[1, 0] = x[5]
    x_hat[1, 2] = -x[3]
    x_hat[2, 0] = -x[4]
    x_hat[2, 1] = x[3]
    return x_hat


@numba.njit()
def vee(x_hat):
    x = np.zeros(6)

    x[:3] = x_hat[:3, 3]
    x[3] = x_hat[2, 1]
    x[4] = x_hat[0, 2]
    x[5] = x_hat[1, 0]
    return x


class Sensor(ABC):
    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Time delta and sensor output
        """
        # noinspection PyUnresolvedReferences
        return self._time[item], self._data[:, item, ...]


class Imu(Sensor):
    def __init__(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray, time_steps: np.ndarray):
        assert angular_velocity.shape == linear_velocity.shape
        self._time = time_steps
        self._data = np.concatenate([linear_velocity, angular_velocity], axis=0)

        self.pose = expm(np.zeros(6))

    def update(self, idx):
        time_delta, twist_rate = self[idx]


class Camera(Sensor):
    def __init__(self, features: np.ndarray, time_steps: np.ndarray):
        self._data = features
        self._time = time_steps

        self.pose = self._get_pose()

    @staticmethod
    def _get_pose():
        """
        Stereo camera (based on left camera) extrinsic calibration parameter from vehicle
        RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
        RPY: -90.878 0.0132 -90.3899
        R: -0.00680499 -0.0153215 0.99985 -0.999977 0.000334627 -0.00680066 -0.000230383 -0.999883 -0.0153234
        T: 1.64239 0.247401 1.58411
        """
        rotation = np.array([[-0.00680499, -0.0153215, 0.99985],
                             [-0.999977, 0.000334627, -0.00680066],
                             [-0.000230383, -0.999883, -0.0153234]])
        position = np.array([1.64239, 0.247401, 1.58411])
        return Pose(rotation, position)


class Vehicle:
    def __init__(self):
        self.weights = self.uniform_prior()
        self.velocity = 0.

    def re_init(self):
        self.yaw = np.zeros_like(self.yaw)
        self.position = np.zeros_like(self.position)
        self.weights = self.uniform_prior()

        self.velocity = 0.

    def uniform_prior(self):
        n_particles = self.n_particles
        return np.ones((n_particles,)) / n_particles

    def __len__(self):
        return self.n_particles

    def resample(self):
        particle_indices = np.arange(0, self.n_particles)
        resampled_indices = np.random.choice(particle_indices, size=self.n_particles, replace=True, p=self.weights)

        self.yaw[:] = self.yaw[resampled_indices]
        self.position[:] = self.position[resampled_indices]
        self.weights = self.uniform_prior()

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
        self._map = np.zeros(self._shape, dtype=np.float)
        self.texture = np.zeros(shape=self._shape, dtype=np.uint8) + 127

        self.update_count = 0

    @property
    def minima(self):
        """x_min, y_min"""
        return self._range[:, 0]

    @property
    def maxima(self):
        """x_max, y_max"""
        return self._range[:, 1]

    @property
    def x_range(self):
        """x_min, x_max"""
        return self._range[0, :]

    @property
    def y_range(self):
        """y_min, y_max"""
        return self._range[1, :]

    @property
    def likelihood(self):
        return np.exp(self._map) / (1 + np.exp(self._map))

    def color(self, cells, pixels):
        self.texture[cells[:, 0], cells[:, 1]] = pixels

    def show_likelihood(self, title):
        plt.imshow(self.likelihood, cmap='gray')
        plt.title(title)
        plt.show()

    def show_map(self, title):
        img = self.ml_map_for_plot
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

    @property
    def ml_map_for_plot(self):
        ml_map = np.ones_like(self._map)
        ml_map[self._map == 0] = 0.5
        ml_map[self._map > 0] = 0
        return ml_map

    @property
    def ml_map(self):
        ml_map = np.zeros_like(self._map, dtype=bool)
        ml_map[self._map > 0] = True
        return ml_map

    @property
    def shape(self):
        return tuple(self._shape)

    def update(self, scan: np.ndarray, origin: Union[List, Tuple, np.ndarray]) -> None:
        """
        Update the map according to the results of a lidar scan

        Args:
            scan: a set of scan points in the world frame
            origin: Origin of the scan. i.e., the ML location of the car

        Returns:
            None
        """
        self.update_count += 1

    def coord_to_cell(self, position):
        return coord_to_cell(position, self.minima, self.resolution)


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
        self.plot = self.plot_occupancy

    def run(self):
        print("Run starting")
        report_iterations = int(1e5)

        start = time.time()
        for i in itertools.count():
            self._step(i)
            if (i + 1) % report_iterations == 0:
                print(f'Sample {(i + 1) // 1000} thousand in {time.time() - start: 02f}s')
                start = time.time()
        self.plot()

    def _step(self, idx):
        self.imu.update(idx)

    def plot_continuous(self):
        map_ = self.get_map_with_particles()

        if self._figure is None:
            self._figure = plt.figure()
            self._ax = self._figure.gca()
            self._fig_handle = self._ax.imshow(map_)
            self._figure.show()
        else:
            self._fig_handle.set_data(map_)
        self._figure.canvas.draw()

    def plot_occupancy(self):
        self._plot_number += 1
        map_ = self.get_map_with_particles()

        plt.imshow(map_, origin='lower', extent=[*self.map.x_range, *self.map.y_range])
        plt.title("Map")
        plt.xlabel("x distance from start (m)")
        plt.ylabel("y distance from start (m)")
        plt.savefig(f'results/map{self._plot_number}.png')
        plt.show()

    def get_map_with_particles(self):
        max_value = 255
        red = np.array([max_value, 0, 0])
        map_ = (self.map.ml_map_for_plot * max_value).astype(np.int)
        map_ = np.stack([map_] * 3, axis=2)
        particles = self.map.coord_to_cell(self.vehicle.position)
        map_[particles[:, 0], particles[:, 1], :] = red
        return np.transpose(map_, (1, 0, 2))

    def step_camera(self, idx):
        time_delta, body_coords = self.camera[idx]

        pose = self.imu.pose
        world_coords = pose.transform(body_coords)
        # todo: update map...

    def step_gyro(self, idx):
        time_step, x = self.imu[idx]
        # todo: update position
