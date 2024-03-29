import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.sparse as sparse

from functions import *
from functions import expm, adj_hat
from utils import visualize_trajectory_2d

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

        self.pose_trail = np.zeros((4, 4, len(self._time)))
        self.trail = np.zeros((2, len(self._time)))
        self.noise = self._get_noise()

    def _get_noise(self):
        dim = 6
        self.measurement_cv = np.zeros((dim, dim)) + self._variance
        return scipy.stats.multivariate_normal(cov=self.measurement_cv)

    def predict(self, idx):
        time_delta, twist_rate = self[idx]

        self.update_pose(time_delta * twist_rate)
        s = expm(-time_delta * adj_hat(twist_rate))
        self.cv = s @ self.cv @ s.T + self.measurement_cv

        self.pose_trail[:, :, idx] = self.pose
        self.trail[:, idx] = self.xy_coords

    def update_pose(self, twist: np.ndarray):
        """
        Apply hat map and matrix exponential to twist, then matmul with the original pose

        Args:
            twist: 6-long numpy array
        """
        update = expm(hat(twist))
        self.pose = self.pose @ update
        return

    @property
    def coords(self):
        return self.pose[:3, 3]

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
            depth_threshold: ignore observations beyond this depth
        """
        self.M = self._get_stereo_calibration(calibration, base)
        self.min_depth = 0.5
        self.max_depth = depth_threshold
        self.inv_pose = pose
        self.pose = inv_pose(pose)

        self._data = self._pre_process(features)
        self._time = time_steps.squeeze()

        self.noise = self._get_noise()

    def _get_noise(self):
        covariance = 0.1
        variance = 4  # 4-5 recommended
        dim = 4
        self.measurement_cv = np.zeros((dim, dim)) + covariance + np.diag([variance] * dim)
        return scipy.stats.multivariate_normal(cov=self.measurement_cv)

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
    def __init__(self, n_points: int, max_update: float):
        self.max_update = max_update
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
        return cv.tocsc()

    def __len__(self):
        return self.points.shape[1]

    def update_points(self, indices, innovation, k, validate=True):
        self._update_count[indices] += 1
        innovation = innovation.T.flatten()
        update = (k @ innovation).reshape((len(self), 3)).T

        if validate:
            update_norm = np.linalg.norm(update[:, indices] - self.points[:, indices], ord=2, axis=0)
            valid = update_norm < self.max_update
            indices = indices[valid]

        self.points[:, indices] += update[:, indices]

    def update_cv(self, k: np.ndarray, h: sparse.bsr_matrix):
        # todo: look into validating cv
        k_bsr = sparse.bsr_matrix(k)
        update = (sparse.eye(len(self) * 3) - k_bsr @ h) @ self.cv
        self.cv = update


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, camera: Camera, imu: Imu, map_: Map, n_samples: int, distance_threshold=30, plot_interval=1):
        """
        Args:
            n_samples: number of time steps
            camera: Camera object
            imu: Imu object
            map_: Map object
            distance_threshold: If the map point is farther away than this, don't try to update it
            plot_interval: Interval at which to update the map plot
        """

        self.d_thresh = distance_threshold
        self.plot_interval = plot_interval
        self.n_samples = n_samples

        self.camera = camera
        self.imu = imu
        self.map = map_

        self.cv = self._get_cv()

        self._plot_number = 0
        self.plot = self.plot

        # for debugging
        self._innovation_record = np.zeros((4, len(map_)))

    def _get_cv(self) -> sparse.csc_matrix:
        size = self.map.cv.shape[0] + 6
        out = sparse.lil_matrix((size, size))
        out[:6, :6] = self.imu.cv[:]
        out[6:, 6:] = self.map.cv[:]
        return out.tocsc()

    def run(self):
        print("Run starting")
        for i in range(self.n_samples):
            self._step(i)
            if i % 100 == 0:
                print(f"step: {i}, density: {self.cv.nnz / np.prod(self.cv.shape)}")
            # visualize_trajectory_2d(self.imu.pose_trail[..., :i + 1])

    def _step(self, idx):
        imu = self.imu
        delta, twist_rate = imu[idx]
        imu.update_pose(delta * twist_rate)
        s = expm(-delta * adj_hat(twist_rate))
        self.cv[:6, :6] = s @ self.cv[:6, :6] @ s.T + imu.measurement_cv
        imu.pose_trail[:, :, idx] = imu.pose
        imu.trail[:, idx] = imu.xy_coords

        time_delta, (indices, observations) = self.camera[idx]

        close_enough = np.argwhere(
            np.logical_or(
                np.linalg.norm(self.map.points[:, indices] - self.imu.coords[:, None], ord=2, axis=0) < self.d_thresh,
                np.isnan(self.map.points[0, indices]),
            )).squeeze()
        indices = indices[close_enough]
        observations = observations[..., close_enough]
        points = self.map.points[:, indices]

        self._init_map(indices, points, observations)

        update_points = np.argwhere(np.logical_not(np.isnan(points[0]))).squeeze()
        if update_points.size > 0:
            update_indices = indices[update_points]
            observations = observations[..., update_points]
            points = points[:, update_points]
        else:
            return

        if isinstance(update_indices, np.int64):
            update_indices = np.array([update_indices])
            noise = self.camera.noise.rvs()
            observations, points, noise = expand_dim([observations, points, noise], -1)
        elif len(update_indices) > 1:
            noise = self.camera.noise.rvs(len(update_indices)).T
        else:
            return  # we got no observations
        bsr_noise = block_to_bsr(self.camera.measurement_cv, len(update_indices))  # diagonalize and broadcast the noise

        cam_t_map, dpi_dx_at_mu, predicted_observations = self.points_to_observations(points)

        m_times_dpi_dx_at_mu = self.camera.M @ dpi_dx_at_mu.transpose([2, 0, 1])
        h_t = -m_times_dpi_dx_at_mu @ self.camera.pose @ o_dot(homo_mul(inv_pose(self.imu.pose), points)).transpose(
            [2, 0, 1])
        h_m = (m_times_dpi_dx_at_mu @ cam_t_map)[..., :3]

        Hm_bsr = sparse.bsr_matrix((h_m, update_indices, np.arange(len(update_indices) + 1)),
                                   blocksize=(4, 3),
                                   shape=(4 * len(update_indices), 3 * len(self.map)))
        Ht = h_t.reshape((h_t.shape[0] * h_t.shape[1], h_t.shape[2]))
        H = sparse.hstack([sparse.csc_matrix(Ht), Hm_bsr.tocsc()])

        # Km = kalman_gain(self.map.cv, Hm_bsr, bsr_noise)
        # Kt = kalman_gain(self.imu.cv, Ht, bsr_noise.toarray())
        K = kalman_gain(self.cv, H, bsr_noise.tocsc())

        innovation = (observations - (predicted_observations + noise))
        self._update_innovation_record(innovation, update_indices)

        self.map.update_points(update_indices, innovation, K[6:])
        # self.map.update_cv(K, H)

        self.imu.update_pose(K[:6] @ innovation.T.flatten())
        # self.imu.cv = (np.eye(6) - K @ H) @ self.imu.cv
        self.cv = (sparse.eye(K.shape[0]) - sparse.csc_matrix(K) @ H) @ self.cv
        return

    def _init_map(self, indices, mu_m, observations):
        new_points = np.argwhere(np.isnan(mu_m[0])).squeeze()
        if new_points.size > 0:
            self._initialize_map_points(indices[new_points], observations[..., new_points])

    @staticmethod
    def _validate_sparse_construction(Hm, h_m, update_indices):
        Hm = Hm.toarray()
        rand_idx = np.random.randint(0, len(h_m))
        h_m_slice = h_m[rand_idx]
        Hm_slice = Hm[rand_idx * 4: (rand_idx + 1) * 4,
                   update_indices[rand_idx] * 3: (update_indices[rand_idx] + 1) * 3]
        assert np.all(h_m_slice == Hm_slice)

    def _update_innovation_record(self, innovation, update_indices):
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
