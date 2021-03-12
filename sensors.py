import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.sparse as sparse

from functions import *

__all__ = ['Camera', 'Imu', 'Map', 'Runner']

np.seterr(divide='raise', invalid='ignore')  # raise an error on divide by zero


class FakeNoise:
    def __init__(self, size):
        self.size = size

    def rvs(self, k=1):
        return np.zeros((k, self.size)).squeeze()


class Imu:
    def __init__(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray, time_steps: np.ndarray,
                 variance: np.ndarray):
        assert angular_velocity.shape == linear_velocity.shape
        self._time = np.squeeze(time_steps)
        self._data = np.concatenate([linear_velocity, angular_velocity], axis=0)
        self._variance = np.diag(variance)

        self.pose = expm(np.zeros((4, 4)))
        self.cv = np.eye(6) * 0.25

        self.trail = np.zeros((2, len(self._time)))
        self.noise = self._get_noise()

    def _get_noise(self):
        dim = 6
        _measurement_cv = np.zeros((dim, dim)) + self._variance
        return scipy.stats.multivariate_normal(cov=_measurement_cv)

    def predict(self, idx):
        time_delta, twist_rate = self[idx]
        noise = self.noise.rvs()

        self.update_pose(time_delta * twist_rate)
        s = expm(-time_delta * adj_hat(twist_rate))
        self.cv = s @ self.cv @ s.T + noise

        self.trail[:, idx] = self.xy_coords

    def update_pose(self, twist: np.ndarray):
        """
        Apply hat map and matrix exponential to twist, then matmul with the original pose

        Args:
            twist: 6-long numpy array
        """
        self.pose = self.pose @ expm(hat(twist))

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
                 pose: np.ndarray, depth_threshold=100):
        """

        Args:
            features:
            time_steps:
            calibration:
            base:
            pose:
            depth_threshold:
        """
        self.M = self._get_stereo_calibration(calibration, base)
        self.min_depth = 0.5
        self.max_depth = depth_threshold
        self.inv_pose = pose
        self.pose = inv_pose(pose)
        self._data = self._pre_process(features)
        self._time = time_steps.squeeze()

        self.noise = self._get_noise()

    @staticmethod
    def _get_noise():
        covariance = 0.5
        variance = 3  # 4-5 recommended
        dim = 4
        _measurement_cv = np.zeros((dim, dim)) + covariance + np.diag([variance] * dim)
        return scipy.stats.multivariate_normal(cov=_measurement_cv)

    def img_to_camera_frame(self, observation):
        return img_to_camera_frame(observation, self._fsu, self._fsv, self._cu, self._cv, self._b)

    def img_to_imu_frame(self, observation):
        # todo: untested
        camera_obs = img_to_camera_frame(observation, self._fsu, self._fsv, self._cu, self._cv, self._b)
        imu_obs = homo_mul(self.inv_pose, camera_obs)[:3]
        return imu_obs

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

    def _pre_process(self, features):
        valid_indices = []
        valid_features = []
        for t in range(features.shape[-1]):
            zt = features[..., t]
            indices = np.argwhere(zt[0] != -1).squeeze()
            disparity = zt[0, indices] - zt[2, indices]
            depth = self._fsu * self._b / disparity
            indices = indices[np.logical_and(depth > self.min_depth, depth < self.max_depth)]

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
        self.cv = self._get_cv(n_points)
        self.points = np.zeros((3, n_points))
        self.points[:] = np.nan

        self._update_count = np.zeros(n_points)

    @staticmethod
    def _get_cv(n_points):
        prior_covariance = 0.1
        prior_variance = 0.5
        cv_blocks = np.zeros((n_points, 3, 3)) + prior_covariance + np.diag([prior_variance] * 3)[None, ...]
        cv = sparse.bsr_matrix((cv_blocks, np.arange(n_points), np.arange(n_points + 1)),
                               blocksize=(3, 3),
                               shape=(3 * n_points, 3 * n_points))
        return cv

    def __len__(self):
        return self.points.shape[1]

    def update_points(self, indices, innovation, k):
        self._update_count[indices] += 1
        innovation = innovation.T.flatten()
        update = (k @ innovation).reshape((len(self), 3)).T
        self.points[:, indices] += update[:, indices]

    def update_cv(self, k: np.ndarray, h: sparse.bsr_matrix):
        k_bsr = sparse.bsr_matrix(k)
        update = (sparse.eye(len(self) * 3) - k_bsr @ h) @ self.cv
        self.cv = update


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

        # for debugging
        self._innovation_record = np.zeros((4, len(map_)))

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
        self.imu.predict(idx)

        # map update
        time_delta, (indices, observations) = self.camera[idx]

        mu_m = self.map.points[:, indices]

        new_points = np.argwhere(np.isnan(mu_m[0])).squeeze()
        update_points = np.argwhere(np.logical_not(np.isnan(mu_m[0]))).squeeze()
        self._initialize_map_points(indices[new_points], observations[..., new_points])

        update_indices = indices[update_points]
        observations = observations[..., update_points]
        mu_m = self.map.points[:, update_indices]
        cv_t = self.imu.cv
        if isinstance(update_indices, np.int64):
            noise_m = self.camera.noise.rvs()
            observations, mu_m, noise_m = expand_dim([observations, mu_m, noise_m], -1)
        elif len(update_indices) > 1:
            noise_m = self.camera.noise.rvs(len(update_points)).T
        else:
            return  # we got no observations
        noise_mat_m = vector_to_bsr(noise_m)  # diagonalize and broadcast the noise

        cam_t_map, dpi_dx_at_mu, predicted_observations = self.points_to_observations(mu_m)

        m_times_dpi_dx_at_mu = self.camera.M @ dpi_dx_at_mu.transpose([2, 0, 1])
        h_t = -m_times_dpi_dx_at_mu @ self.camera.pose @ \
              o_dot(homo_mul(inv_pose(self.imu.pose), mu_m)).transpose([2, 0, 1])
        h_m = (m_times_dpi_dx_at_mu @ cam_t_map)[..., :3]

        Hm = sparse.bsr_matrix((h_m, update_indices, np.arange(len(update_indices) + 1)),
                                    blocksize=(4, 3),
                                    shape=(4 * len(update_indices), 3 * len(self.map)))
        self._validate_sparse_construction(Hm, h_m, update_indices)  # todo: remove when finished debugging

        k_m = bsr_kalman_gain(self.map.cv, Hm, noise_mat_m)
        # k_t = kalman_gain(cv_t, h_t, noise_mat_m)

        # k_t = k_t.transpose([1, 2, 0]).reshape(6, -1)  # 6 x 4Nt
        # h_t = h_t.reshape(-1, 6)  # 4Nt x 6

        innovation = (observations - predicted_observations)
        # assert np.all(np.linalg.norm(innovation, axis=0) < 10), \
        #     f"Innovation is very large {np.linalg.norm(innovation, axis=0)}"
        self._update_innovation_record(innovation, update_indices)

        self.map.update_points(update_indices, innovation, k_m)
        self.map.update_cv(k_m, Hm)

        # self.imu.update_pose(k_t @ innovation.flatten())
        # self.imu.cv = (np.eye(6) - k_t @ h_t) @ self.imu.cv
        return

    def _validate_sparse_construction(self, Hm, h_m, update_indices):
        Hm = Hm.toarray()
        rand_idx = np.random.randint(0, len(h_m))
        h_m_slice = h_m[rand_idx]
        Hm_slice = Hm[rand_idx * 4: (rand_idx + 1) * 4,
                   update_indices[rand_idx] * 3: (update_indices[rand_idx] + 1) * 3]
        assert np.all(h_m_slice == Hm_slice)

    def _update_innovation_record(self, innovation, update_indices):
        innovation = innovation.squeeze()
        self._innovation_record[:, update_indices] = innovation

    def points_to_observations(self, mu):
        cam_t_map = self.camera.pose @ inv_pose(self.imu.pose)  # pose converting from world to camera
        mu_camera_ = homo_mul(cam_t_map, mu)  # mu in the camera frame
        dpi_dx_at_mu = d_pi_dx(mu_camera_)  # derivative of the camera model evaluated at mu in the cam frame
        predicted_observations = self.camera.M @ pi(mu_camera_)
        return cam_t_map, dpi_dx_at_mu, predicted_observations

    def _initialize_map_points(self, indices, observations):
        """
        Initialize map points
        Args:
            indices:
            observations:

        Returns:

        """
        map_frame = self.observation_to_points(observations)
        self.map.points[:, indices] = map_frame

    def observation_to_points(self, observations):
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

    def plot(self, blocking=True):
        fig, ax = plt.subplots()
        ax.scatter(self.map.points[0], self.map.points[1], s=1, label='landmarks')
        ax.scatter(self.imu.trail[0], self.imu.trail[1], s=1, label='path')
        ax.set_title("Map")
        ax.set_xlabel("x distance from start (m)")
        ax.set_ylabel("y distance from start (m)")
        plt.legend()
        plt.show(block=blocking)
