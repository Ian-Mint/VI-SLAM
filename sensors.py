import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from functions import hat, homo_mul, expm, img_to_camera_frame, inv_pose, pi, d_pi_dx

__all__ = ['Camera', 'Imu', 'Map', 'Runner']

np.seterr(divide='raise', invalid='ignore')  # raise an error on divide by zero


class Imu:
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


class Camera:
    def __init__(self, features: np.ndarray, time_steps: np.ndarray, calibration: np.ndarray, base: np.ndarray,
                 pose: np.ndarray):
        self.pose = inv_pose(pose)
        self._data = self._pre_process(features)
        self._time = time_steps.squeeze()

        self.M = self._get_stereo_calibration(calibration, base)

        n_points = len(self._time)
        prior_covariance = 1
        prior_variance = 2
        _measurement_cv = np.zeros((3, 3)) + prior_covariance + np.diag([prior_variance] * 3)
        self.noise = scipy.stats.multivariate_normal(cov=_measurement_cv)

    def img_to_camera_frame(self, observation):
        return img_to_camera_frame(observation, self._fsu, self._fsv, self._cu, self._cv, self._b)

    def _get_stereo_calibration(self, k, b):
        self._fsu = k[0, 0]
        self._fsv = k[1, 1]
        self._cu = k[0, 2]
        self._cv = k[1, 2]
        self._b = b

        return np.array([
            [self._fsu, 0, self._cu, 0],
            [0, self._fsv, self._cv, 0],
            [self._fsu, 0, self._cu, -self._fsu * self._b],
            [0, self._fsv, self._cv, 0]
        ])

    @staticmethod
    def _pre_process(features):
        valid_indices = []
        valid_features = []
        for t in range(features.shape[-1]):
            zt = features[..., t]
            indices = np.argwhere(zt[0] != -1).squeeze()
            valid_indices.append(indices)
            valid_features.append(zt[:, indices])
        return list(zip(valid_indices, valid_features))

    def __len__(self):
        return len(self._time)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Time delta and sensor output
        """
        if item >= self.__len__():
            raise StopIteration
        return self._time[item], self._data[item]


class Map:
    def __init__(self, n_points: int):
        prior_covariance = 1e-5
        prior_variance = 0.25
        self.cv = np.zeros((3, 3, n_points)) + prior_covariance + np.diag([prior_variance] * 3)[..., None]
        self.points = np.zeros((3, n_points))


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, camera: Camera, imu: Imu, map_: Map, n_samples: int, plot_interval=1):
        """
        Args:
            n_samples: number of time steps
            camera: Camera object
            imu: Gyro object
            map_: Map object
            plot_interval: Interval at which to update the map plot
        """

        self.plot_interval = plot_interval
        self.n_samples = n_samples

        self.camera = camera
        self.imu = imu
        self.map = map_

        self._plot_number = 0
        self.plot = self.plot

    def run(self):
        print("Run starting")
        report_iterations = int(1e5)

        start = time.time()
        for i in range(self.n_samples):
            self._step(i)
            if (i + 1) % report_iterations == 0:
                print(f'Sample {(i + 1) // 1000} thousand in {time.time() - start: 02f}s')
                start = time.time()
        self.plot()

    def _step(self, idx):
        self._imu_update(idx)
        self._map_update(idx)

    def _imu_update(self, idx):
        self.imu.update(idx)

    def _map_update(self, idx):
        """
        Use EKF to update the map points
        """
        time_delta, (indices, observations) = self.camera[idx]

        noise = self.camera.noise.rvs(len(indices)).T
        cv = self.map.cv[..., indices]
        mu = self.map.points[:, indices]
        m = self.camera.M

        cam_t_map = self.camera.pose @ inv_pose(self.imu.pose)
        mu_camera_ = homo_mul(cam_t_map, mu)

        dx = d_pi_dx(mu_camera_)
        h = (m @ dx @ cam_t_map)[:3]
        kt, _, _, _ = np.linalg.lstsq((h @ cv @ h.T + noise).T, (cv @ h.T).T)
        k = kt.T

        mu[:] = mu + k @ (observations - m @ pi(mu_camera_))
        cv[:] = (np.eye(4) - k @ h) @ cv

    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.map.points[0], self.map.points[1], s=1, label='landmarks')
        ax.scatter(self.imu.trail[0], self.imu.trail[1], s=1, label='path')
        ax.set_title("Map")
        ax.set_xlabel("x distance from start (m)")
        ax.set_ylabel("y distance from start (m)")
        plt.legend()
        plt.show()


