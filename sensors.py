import time
from abc import ABC
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numba
import numpy as np

import pr2_utils as utils

__all__ = ['Encoder', 'Gyro', 'Lidar', 'Map', 'Car', 'Runner']

np.seterr(divide='raise', invalid='ignore')  # raise an error on divide by zero


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax function, normalizing over the last axis

    Args:
        x: numpy array with features in the last dimension

    Returns:
        shape x.shape[:-1]
    """
    return x / np.nansum(x, axis=x.ndim - 1, keepdims=True)


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
        """

        Args:
            rotation: rotation matrix or yaw angles
            position: positions
        """
        if rotation.ndim > 1:
            assert (rotation.shape[-2], rotation.shape[-1]) == (3, 3)
            self.rotation = rotation
        else:
            if isinstance(rotation, np.ndarray):
                self.rotation = r_2d(rotation)
            else:
                self.rotation = r_2d(np.array([rotation])).squeeze()
        self.position = position
        assert self.rotation.ndim - 1 == self.position.ndim
        assert self.rotation.ndim in {2, 3}
        assert len(self.rotation) == len(self.position)

    def transform(self, x) -> np.ndarray:
        """
        multiplies by rotation and adds position
        Args:
            x: coordinates must be in last dimension

        Returns:
            transformed coordinates. in dimensions (n_particles, n_points, 2)
        """
        rotated = x @ self.rotation.T
        if self.rotation.ndim > 2:
            rotated = rotated.T
        if self.rotation.ndim > 2 and x.ndim > 1:
            return rotated + np.expand_dims(self.position, axis=1)
        else:
            return rotated + self.position

    def __repr__(self):
        return f'position: {self.position.__repr__()}\nrotation: {self.rotation.__repr__()}'

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

        body_xy_scans = self._pre_process(scans)
        self._data = dict(zip(self.time, body_xy_scans))

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
            n x d x 2 numpy array
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
        return xyz_scan_body[..., :2]


class Gyro(Sensor):
    def __init__(self, data_file='data/sensor_data/gyro.csv'):
        self.time, omega_sensor = utils.read_data_from_csv(data_file)
        yaw = omega_sensor[:, 2]
        data = self._pre_process(yaw)
        self.time = self.time[1:]
        self._data = dict(zip(self.time, data))

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
            time-delta and yaw. The first yaw sample is dropped because there is no corresponding time delta.
        """
        time_delta = np.diff(self.time)
        assert np.all(time_delta > 0)
        return np.stack((time_delta, yaw[1:]), axis=1)


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
    avg_distance = np.sum(distance, axis=1).squeeze()
    return avg_distance / 2


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
    def __init__(self, n_particles, v_var, omega_var, resample_threshold):
        self.resample_threshold = resample_threshold * n_particles
        self.v_var = v_var
        self.omega_var = omega_var
        self.n_particles = n_particles

        n_dims = 2
        self.yaw = np.zeros(n_particles)
        self.position = np.zeros((n_particles, n_dims))
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
        self.weights[:] = new_weights
        if self.n_eff < self.resample_threshold:
            self.resample()

    @property
    def n_eff(self) -> float:
        """
        Number of effective samples

        Returns:
            A value in the range [0, n_samples]
        """
        return 1 / np.sum(self.weights ** 2)

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


class Map:
    def __init__(self, resolution=0.1, x_range=(-50, 50), y_range=(-50, 50), lambda_max_factor=100):
        resolution = float(resolution)
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
        origin_cell = coord_to_cell(origin, self.minima, self.resolution)
        assert self.is_in_map(origin_cell), "origin is outside the map"
        valid_scan_cells = self.get_valid_scan_cells(scan)

        _positive_update(valid_scan_cells, self._map, self._increment, self._lambda_max)
        _negative_update(valid_scan_cells, origin_cell, self._map, self._decrement, self._lambda_min)

    def coord_to_cell(self, position):
        return coord_to_cell(position, self.minima, self.resolution)

    def get_valid_scan_cells(self, scan):
        scan_cells = coord_to_cell(scan, self.minima, self.resolution)
        scan_valid = self.valid_scan(scan_cells)
        valid_scan_cells = scan_cells[scan_valid]
        return valid_scan_cells

    def valid_scan(self, cells):
        """
        Tests each point of the scan for validity.
        Invalid if a dimension is less than two cells or greater than the dimensions of the map.

        Args:
            cells: numpy array of cell coordinates

        Returns:
            Valid cells are set to True. Drops the last (x,y) dimension of cells
        """
        gt_one_cell = cells > 1
        lt_map_size = cells < self._shape
        valid = np.logical_and.reduce(
            [gt_one_cell[..., 0], gt_one_cell[..., 1], lt_map_size[..., 0], lt_map_size[..., 1]])
        assert valid.shape == cells.shape[:-1]
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

    def correlation(self, scan) -> np.ndarray:
        """
        Compute the correlation
        Args:
            scan: shape (n_particles, n_scans, 2)

        Returns:
            shape (n_particles)
        """
        ml_map = self.ml_map
        corr = np.zeros(len(scan))
        for i, particle_scan in enumerate(scan):
            valid_scan_cells = self.get_valid_scan_cells(scan)
            selected_map_cells = ml_map[valid_scan_cells[..., 0], valid_scan_cells[..., 1]]
            corr[i] = np.sum(selected_map_cells)
        assert len(corr) == len(scan)
        assert corr.ndim == 1
        return corr


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, encoder: Encoder, gyro: Gyro, lidar: Lidar, car: Car, map_: Map, downsample: int = 1,
                 plot_interval=0):
        """
        Args:
            encoder: Encoder object
            gyro: Gyro object
            lidar: Lidar object
            car: Car object
            map_: Map object
            downsample: Factor by which to downsample. default=1, i.e. no down-sampling
            plot_interval: Interval at which to update the map plot
        """
        assert isinstance(downsample, int)
        self.plot_interval = plot_interval
        self.downsample = downsample
        self.encoder = encoder
        self.gyro = gyro
        self.lidar = lidar
        self.car = car
        self.map = map_
        self._figure = None
        self._ax = None
        self._animation = None
        self._fig_handle = None

        self.execution_seq = self.get_execution_sequence()

    def __len__(self):
        return len(self.execution_seq)

    def run(self):
        print("Run starting")
        report_iterations = int(1e6)

        start = time.time()
        for i, (timestamp, executor) in enumerate(self.execution_seq):
            executor(timestamp)
            if (i + 1) % report_iterations == 0:
                print(f'Sample {i // report_iterations} million in {time.time() - start: 02f}s')
                start = time.time()

    def plot(self):
        max_value = 255
        red = np.array([max_value, 0, 0])
        map_ = (self.map.ml_map_for_plot * max_value).astype(np.int)
        map_ = np.stack([map_] * 3, axis=2)

        particles = self.map.coord_to_cell(self.car.position)
        map_[particles[:, 0], particles[:, 1], :] = red

        if self._figure is None:
            self._figure = plt.figure()
            self._ax = self._figure.gca()
            self._fig_handle = self._ax.imshow(map_)
            self._figure.show()
        else:
            self._fig_handle.set_data(map_)
        self._ax.set_title(f"map")
        self._figure.canvas.draw()

    def get_execution_sequence(self) -> np.ndarray:
        """
        Create a numpy array consisting of (timestamp, executor) pairs, sorted by timestamp.
        """
        gyro_time = self.gyro.time
        encoder_time = self.encoder.time
        lidar_time = self.lidar.time

        if self.downsample > 1:
            gyro_time = gyro_time[0::self.downsample]
            lidar_time = gyro_time[0::self.downsample]
            encoder_time = encoder_time[0::self.downsample]

        timestamps = np.concatenate((encoder_time, gyro_time, lidar_time), axis=0)
        assert timestamps.ndim == 1
        executors = np.concatenate(([self.step_encoder] * len(encoder_time),
                                    [self.step_gyro] * len(gyro_time),
                                    [self.step_lidar] * len(lidar_time)), axis=0)
        execution_sequence = np.stack((timestamps, executors), axis=1)
        execution_sequence = execution_sequence[execution_sequence[:, 0].argsort()]
        assert np.all(execution_sequence[1:, 0] >= execution_sequence[:-1, 0])
        assert execution_sequence.shape == (len(timestamps), 2)
        return execution_sequence

    def step_gyro(self, timestamp):
        time_step, yaw = self.gyro[timestamp]
        self.car.predict(yaw, time_step)

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
            self.car.update(softmax(correlation))

        # Update map according to the maximum likelihood pose of the car
        car_pose = self.car.ml_pose
        scan_world = car_pose.transform(scan_body)
        self.map.update(scan=scan_world[:, :2], origin=car_pose.position)

        if (self.map.update_count + 1) % self.plot_interval == 0:
            self.plot()

    def _get_correlation(self, scan_body) -> np.ndarray:
        scan_world = self.car.transform_all(scan_body)
        assert scan_world.shape == (self.car.n_particles, *scan_body.shape)
        return self.map.correlation(scan_world)
