from abc import ABC
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numba

import pr2_utils as utils

__all__ = ['Encoder', 'Gyro', 'Lidar', 'Map', 'Car', 'Runner']


@numba.njit()
def r_2d(theta: np.ndarray):
    rotations = np.zeros((len(theta), 2, 2))
    for i, t in enumerate(theta):
        rotations[i, 0, 0] = np.cos(t)
        rotations[i, 0, 1] = -np.sin(t)
        rotations[i, 1, 0] = np.sin(t)
        rotations[i, 1, 1] = np.cos(t)
    return rotations


class Pose:
    def __init__(self, rotation: np.ndarray, position: np.ndarray):
        if rotation.ndim > 1:
            assert (rotation.shape[-2], rotation.shape[-1]) == (3, 3)
            self.rotation = rotation
        else:
            self.rotation = r_2d(rotation)
        self.position = position
        assert rotation.ndim - 1 == position.ndim
        assert rotation.ndim in {2, 3}
        assert len(rotation) == len(position)

    def transform(self, x):
        """
        multiplies by rotation and adds position
        Args:
            x: coordinates must be in last dimension

        Returns:
            transformed coordinates. in dimensions (n_x, n_r, d_r)
        """
        t = x @ self.rotation.T + self.position
        if self.rotation.ndim > 2:
            return np.transpose(t, axes=(1, 2, 0))
        else:
            return t

    def __matmul__(self, other):
        rotation = self.rotation @ other.rotation
        position = self.position + self.transform(other.position)
        return Pose(rotation, position)


class Sensor(ABC):
    def __getitem__(self, item):
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Sensor output
        """
        # noinspection PyUnresolvedReferences
        return self._data.pop(item)


class Lidar(Sensor):
    """
    FOV: 190 (degree)
    Start angle: -5 (degree)
    End angle: 185 (degree)
    Angular resolution: 0.666 (degree)
    Max range: 80 (meter)

    * LiDAR rays with value 0.0 represent infinite range observations.
    """

    def __init__(self, data_file='data/sensor_data/lidar.csv'):
        self.time, scans = utils.read_data_from_csv(data_file)
        self._max_range = 75.
        self._min_range = 2.

        body_xyz_scans = self._pre_process(scans)
        self._data = dict(zip(self.time, body_xyz_scans))

    def __len__(self):
        return len(self._data)

    def _pre_process(self, scans) -> np.ndarray:
        """
        Pre-process the dataset into xy coordinates in the body frame

        Lidar sensor (LMS511) extrinsic calibration parameter from vehicle
        RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
        RPY: 142.759 0.0584636 89.9254
        R: 0.00130201 0.796097 0.605167 0.999999 -0.000419027 -0.00160026 -0.00102038 0.605169 -0.796097
        T: 0.8349 -0.0126869 1.76416p

        Returns:
            n x d x 3 numpy array
        """
        self._rpy = (142.759, 0.0584636, 89.9254)
        self._rotation = np.array([[0.00130201, 0.796097, 0.605167],
                                   [0.999999, -0.000419027, -0.00160026],
                                   [-0.00102038, 0.605169, -0.796097]])
        self._position = np.array([0.8349, -0.0126869, 1.76416])
        pose = Pose(self._rotation, self._position)

        angles = np.deg2rad(np.linspace(-5, 185, scans.shape[1]))
        assert np.allclose(np.linspace(-5, 185, 286) / 180 * np.pi, angles), "angles are calculated wrong"

        x_scale = np.cos(angles)
        y_scale = np.sin(angles)

        out_of_range_mask = np.logical_or(scans < self._min_range, scans > self._max_range)
        scans[out_of_range_mask] = np.nan

        x = scans * x_scale
        y = scans * y_scale

        xyz_scan_sensor = np.stack([x, y, np.zeros_like(x)], axis=2)
        xyz_scan_body = pose.transform(xyz_scan_sensor)

        assert np.allclose(self._rotation @ xyz_scan_sensor[0, 0] + self._position, xyz_scan_body[0, 0]), \
            "Broadcasting did not work as expected"
        return xyz_scan_body


class Gyro(Sensor):
    def __init__(self, data_file='data/sensor_data/gyro.csv'):
        self.time, omega_sensor = utils.read_data_from_csv(data_file)
        yaw = omega_sensor[:, 2]
        data = self._pre_process(yaw)
        self._data = dict(zip(self.time[1:], data))

    def _pre_process(self, yaw) -> np.ndarray:
        """
        Pre-process measurements into the body frame

        FOG (Fiber Optic Gyro) extrinsic calibration parameter from vehicle
        RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix, meter)
        RPY: 0 0 0
        R: 1 0 0 0 1 0 0 0 1
        T: -0.335 -0.035 0.78

        * The sensor measurements are stored as [timestamp, delta roll, delta pitch, delta yaw] in radians.

        Returns:

        """
        time_delta = np.diff(self.time)
        return np.stack((time_delta, yaw), axis=1)


def _get_update(distance) -> np.ndarray:
    """
    Compute the x, y, and angular translation over each time step
    Uses the Euler approximation (treats movements as linear, instead of as an arc)

    
    Args:
        distance: distance traveled by each wheel in each time step

    Returns:
        translated distance of the center of the vehicle
    """
    # difference between right and left count in a time step
    distance_diff = np.diff(distance, axis=1).squeeze()
    center_arc = distance_diff / 2
    return center_arc


class Encoder(Sensor):
    def __init__(self, data_file='data/sensor_data/encoder.csv'):
        self.time, counts = utils.read_data_from_csv(data_file)
        deltas = self._pre_process(counts)
        self._data = dict(zip(self.time, deltas))

    def _pre_process(self, counts) -> np.ndarray:
        """
        Encoder calibrated parameter
        Encoder resolution: 4096
        Encoder left wheel diameter: 0.623479
        Encoder right wheel diameter: 0.622806
        Encoder wheel base: 1.52439

        * The encoder data is stored as [timestamp, left count, right count].

        Returns:
            [dt, dx, dy, dtheta]
        """
        self._resolution = 4096
        self._diameter = (0.623479, 0.623479)
        self._wheel_base = 1.52439

        # calibration constants, (m/count)
        cal = np.pi * np.array(self._diameter) / self._resolution

        counts_delta = np.diff(counts, axis=0)
        time_delta = np.diff(self.time)
        assert counts_delta.shape == (len(counts) - 1, 2)
        assert len(counts_delta) == len(time_delta)

        v = _get_update(counts_delta * cal) / time_delta
        return v


class Car:
    def __init__(self, n_particles, v_var, omega_var):
        self.v_var = v_var
        self.omega_var = omega_var
        self.n_particles = n_particles

        n_dims = 2
        self.yaw = np.zeros(n_particles)
        self.position = np.zeros((n_particles, n_dims))
        self.weights = np.ones((n_particles,)) / n_particles

        self.velocity = 0.

    def __len__(self):
        return self.n_particles

    def resample(self):
        pass

    def predict(self, yaw, time_step):
        v_noise = np.random.normal(loc=0, scale=self.v_var, size=len(self))
        omega_noise = np.random.normal(loc=0, scale=self.omega_var, size=len(self))

        translation = time_step * (self.velocity + v_noise)
        dx = translation * np.cos(yaw)
        dy = translation * np.sin(yaw)
        dtheta = yaw + time_step * omega_noise

        self.position[:, 0] += dx
        self.position[:, 1] += dy
        self.yaw += dtheta

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

    @property
    def ml_pose(self) -> Pose:
        """
        Pose object for the particle with the largest weight

        Returns:
            a Pose object
        """
        idx = np.argmax(self.weights)
        return Pose(self.yaw[idx], self.position[idx])

    @property
    def pose(self) -> Pose:
        """
        Pose object for the particle with the largest weight

        Returns:
            # a Pose object
        """
        return Pose(self.yaw, self.position)


@numba.njit(parallel=True, fastmath=True)
def _negative_update(scan_cells, origin_cell, map_, decrement, lambda_min) -> None:
    """
    Decrement the likelihood of cells where no object was detected

    Args:
        scan_cells: Indices of cells where an object was detected
        origin_cell: The origin of the trace rays (the vehicle location)
        map_: A 2d numpy array containing cell log-likelihoods
        decrement: How much to decrement each cell encountered during Bresenham trace
        lambda_min: Clip log-likelihood below this value
    """
    # 0.002024s per iteration (241s total) without JIT (and without the inner for loop)
    # 0.001472s per iteration (172s total) with JIT
    # 0.000646s per iteration (81s total) parallelized with JIT (four cores)
    sx, sy = origin_cell
    for i in numba.prange(len(scan_cells)):
        ex, ey = scan_cells[i]
        trace = utils.bresenham2D(sx, sy, ex, ey)
        # use a loop for numba compatibility
        for tx, ty in trace[:-1]:  # the last element of trace is (ex, ey)
            map_[tx, ty] -= decrement
            # clip minimum
            if map_[tx, ty] < lambda_min:
                map_[tx, ty] = lambda_min


@numba.njit(parallel=True, fastmath=True)
def _positive_update(scan_cells, map_, increment, lambda_max) -> None:
    """
    Increment the likelihood of cells where we detected an object
    map_[scan_cells[:, 0], scan_cells[:, 1]] += increment

    Args:
        scan_cells: Indices of cells where an object was detected
        map_: The map to be updated according to scan_cells
        increment: How much to increment each cell by
        lambda_max: The clipping limit
    """
    for i in numba.prange(len(scan_cells)):
        x, y = scan_cells[i]
        map_[x, y] += increment
        # clip upper limit
        if map_[x, y] > lambda_max:
            map_[x, y] = lambda_max


class Map:
    def __init__(self, resolution=0.1, x_range=(-50, 50), y_range=(-50, 50), lambda_max_factor=100, plt_interval=1000):
        assert isinstance(resolution, float)
        self.resolution = resolution
        self._range = np.array([x_range, y_range])

        self._increment = np.log(16)
        self._decrement = np.log(4)
        self._lambda_max = self._increment * lambda_max_factor
        self._lambda_min = -self._decrement * lambda_max_factor
        self._shape = np.array([int(np.ceil(np.diff(x_range) / resolution + 1)),
                                int(np.ceil(np.diff(y_range)) / resolution + 1)])
        self._map = np.zeros(self._shape, dtype=np.float)

        self.update_count = 0
        self.plt_interval = plt_interval  # plot the map every 1000 scans

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

    def show_likelihood(self, title):
        plt.imshow(self.likelihood, cmap='gray')
        plt.title(title)
        plt.show()

    def show_map(self, title):
        img = -self.ml_map + 1  # invert the map
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

    @property
    def ml_map(self):
        ml_map = np.zeros_like(self._map)
        ml_map[self._map == 0] = 0.5
        ml_map[self._map > 0] = 1
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
        origin_cell = self.coord_to_cell(origin)
        assert self.is_in_map(origin_cell), "origin is outside the map"
        valid_scan_cells = self.get_valid_scan_cells(scan)

        _positive_update(valid_scan_cells, self._map, self._increment, self._lambda_max)
        _negative_update(valid_scan_cells, origin_cell, self._map, self._decrement, self._lambda_min)

    def get_valid_scan_cells(self, scan):
        assert scan.shape[-1] == 2, "drop the z-dimension for mapping"
        scan_cells = self.coord_to_cell(scan)
        scan_valid = self.valid_scan(scan_cells)
        valid_scan_cells = scan_cells[scan_valid]
        assert valid_scan_cells.ndim == 2
        return valid_scan_cells

    def valid_scan(self, cells):
        """
        Tests each point of the scan for validity.
        Invalid if a dimension is less than two cells or greater than the dimensions of the map.

        Args:
            cells: numpy array of cell coordinates

        Returns:
            1-d boolean array, where valid cells are set to True
        """
        gt_one_cell = cells > 1
        lt_map_size = cells < self._shape
        valid = np.logical_and.reduce([gt_one_cell[:, 0], gt_one_cell[:, 1], lt_map_size[:, 0], lt_map_size[:, 1]])
        assert valid.ndim == 1
        return valid

    def is_in_map(self, origin_cell: np.ndarray) -> bool:
        """
        Test if a cell is in the map
        Args:
            origin_cell: one or more cell coordinates

        Returns:
            True if the cell is in the map
        """
        return np.all(origin_cell > 0) and np.all(origin_cell < self._shape)

    def coord_to_cell(self, point) -> np.ndarray:
        """
        Discretize according to the map layout.
        Convert from xy (m) coordinates to ij (px) coordinates.

        Args:
            point: a point or set of points in meters

        Returns:
            point or set of points in coordinate indices
        """
        if isinstance(point, (tuple, list)):
            point = np.array(point)
        else:
            assert isinstance(point, np.ndarray)
        point_is = np.ceil((point - self.minima) / self.resolution).astype(np.int16) - 1
        return point_is

    def correlation(self, valid_scan_cells):
        return np.sum(self.ml_map[valid_scan_cells])


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, encoder: Encoder, gyro: Gyro, lidar: Lidar, car: Car, map_: Map):
        self.encoder = encoder
        self.gyro = gyro
        self.lidar = lidar
        self.car = car
        self.map = map_

        self.execution_seq = self.get_execution_sequence()

    def __len__(self):
        return len(self.execution_seq)

    def run(self):
        for timestamp, executor in self.execution_seq:
            executor(timestamp)

    def get_execution_sequence(self) -> np.ndarray:
        """
        Create a numpy array consisting of (timestamp, executor) pairs, sorted by timestamp.
        """
        timestamps = np.concatenate((self.encoder.time, self.gyro.time, self.lidar.time), axis=0)
        assert timestamps.ndim == 1
        executors = np.concatenate(([self.step_encoder] * len(self.encoder.time),
                                    [self.step_gyro] * len(self.gyro.time),
                                    [self.step_lidar] * len(self.lidar.time)), axis=0)
        execution_sequence = np.stack((timestamps, executors), axis=1)
        execution_sequence = execution_sequence[execution_sequence[:, 0].argsort()]
        assert execution_sequence.shape == (len(timestamps), 2)
        return execution_sequence

    def step_gyro(self, timestamp):
        time_step, yaw = self.gyro[timestamp]
        self.car.predict(time_step, yaw)

    def step_encoder(self, timestamp):
        """
        Only use the velocity data from the encoder.
        Do no state updates other than the car velocity.

        Args:
            timestamp: the encoder timestamp
        """
        v = self.encoder[timestamp]
        self.car.velocity = v

    def step_lidar(self, timestamp):
        scan_body = self.lidar[timestamp]
        if self.map.update_count > 0:  # ensure there is a map
            correlation = self._get_correlation(scan_body)

        # Update map according to the maximum likelihood pose of the car
        car_pose = self.car.ml_pose
        scan_world = car_pose.transform(scan_body)
        self.map.update(scan=scan_world[:, :2], origin=car_pose.position[:2])

    def _get_correlation(self, scan_body) -> np.ndarray:
        scan_world = self.car.transform_ml()