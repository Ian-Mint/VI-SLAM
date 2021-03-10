import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from functions import hat, homo_mul, expm, img_to_camera_frame, inv_pose, pi, d_pi_dx, lstsq_broadcast, expand_dim

__all__ = ['Camera', 'Imu', 'Map', 'Runner']

np.seterr(divide='raise', invalid='ignore')  # raise an error on divide by zero


class FakeNoise:
    def __init__(self, size):
        self.size = size

    def rvs(self, k=1):
        return np.zeros((k, self.size)).squeeze()


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

        covariance = 0.5
        variance = 3  # 4-5 recommended
        dim = 4
        _measurement_cv = np.zeros((dim, dim)) + covariance + np.diag([variance] * dim)
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
        prior_covariance = 0.1
        prior_variance = 0.5
        self.cv = np.zeros((3, 3, n_points)) + prior_covariance + np.diag([prior_variance] * 3)[..., None]
        self.points = np.zeros((3, n_points))
        self.points[:2] = np.nan

    def update_points(self, indices, update):
        self.points[:, indices] = update[:]


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, camera: Camera, imu: Imu, map_: Map, n_samples: int, plot_interval=1):
        """
        Args:
            n_samples: number of time steps
            camera: Camera object
            imu: Imu object
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

    def _step(self, idx):
        self._imu_update(idx)
        self._map_update(idx)

    def _imu_update(self, idx):
        self.imu.update(idx)

    def _map_update(self, idx):
        """
        Use EKF to update the map points
        """
        # todo: filter out improbable updates
        time_delta, (indices, observations) = self.camera[idx]

        mu = self.map.points[:, indices]

        new_points = np.argwhere(np.isnan(mu[0])).squeeze()
        update_points = np.argwhere(np.logical_not(np.isnan(mu[0]))).squeeze()
        self._initialize_map_points(indices[new_points], observations[..., new_points])

        update_indices = indices[update_points]
        observations = observations[..., update_points]
        mu = mu[:, update_points]
        cv = self.map.cv[..., update_indices]
        if len(update_indices) == 1:
            noise = self.camera.noise.rvs()
            observations, mu, cv, noise = expand_dim([observations, mu, cv, noise], -1)
        elif len(update_indices) > 1:
            noise = self.camera.noise.rvs(len(update_points)).T
        else:
            return
        noise_mat = np.eye(4) * noise.T[..., None]  # diagonalize and broadcast the noise

        cam_t_map, jacobian, predicted_observations = self.points_to_observations(mu)

        h = (self.camera.M @ jacobian.transpose([2, 0, 1]) @ cam_t_map)[..., :3]

        cv_ht = cv.transpose([2, 0, 1]) @ h.transpose([0, 2, 1])
        a = (h @ cv_ht + noise_mat).transpose([0, 2, 1])
        b = cv_ht.transpose([0, 2, 1])

        kt = lstsq_broadcast(a, b)
        k = kt.transpose([0, 2, 1])

        innovation = observations - predicted_observations
        # assert np.all(np.linalg.norm(innovation, axis=0) < 10), \
        #     f"Innovation is very large {np.linalg.norm(innovation, axis=0)}"

        self.map.update_points(update_indices, mu + (k @ innovation.T[..., None]).squeeze().T)
        self.map.cv[..., update_indices] = ((np.eye(3)[None, ...] - k @ h) @ cv.transpose([2, 0, 1])).transpose(
            [1, 2, 0])
        return

    def points_to_observations(self, mu):
        cam_t_map = self.camera.pose @ inv_pose(self.imu.pose)  # pose converting from world to camera
        mu_camera_ = homo_mul(cam_t_map, mu)  # mu in the camera frame
        jacobian = d_pi_dx(mu_camera_)  # derivative of the camera model evaluated at mu in the cam frame
        predicted_observations = self.camera.M @ pi(mu_camera_)
        return cam_t_map, jacobian, predicted_observations

    def _initialize_map_points(self, indices, observations):
        """
        Initialize map points
        Args:
            indices:
            observations:

        Returns:

        """
        map_frame = self.observation_to_world(observations)
        self.map.update_points(indices, map_frame)
        self.map.points[:, indices] = map_frame

    def observation_to_world(self, observations):
        """
        Converts valid stereo image observations to xyz points in the world

        Args:
            observations: (xL, yL, xR, yR)

        Returns:
            (x, y, z)
        """
        camera_frame = self.camera.img_to_camera_frame(observations)
        map_frame = homo_mul(self.imu.pose @ inv_pose(self.camera.pose), camera_frame)[:3]
        return map_frame

    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.map.points[0], self.map.points[1], s=1, label='landmarks')
        ax.scatter(self.imu.trail[0], self.imu.trail[1], s=1, label='path')
        ax.set_title("Map")
        ax.set_xlabel("x distance from start (m)")
        ax.set_ylabel("y distance from start (m)")
        plt.legend()
        plt.show()
