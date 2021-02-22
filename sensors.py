from abc import ABC
from typing import List, Tuple, Union

import cv2
import numpy as np

import pr2_utils as utils


class Pose:
    def __init__(self, rotation: np.ndarray, position: np.ndarray):
        self.rotation = rotation
        self.position = position
        assert rotation.ndim - 1 == self.position.ndim
        assert rotation.ndim in {2, 3}

    def transform(self, x):
        """
        multiplies by rotation and adds position
        Args:
            x: coordinates must be in last dimension

        Returns:
            transformed coordinates
        """
        if self.rotation.ndim == 2:
            assert x.ndim <= 2
            return x @ self.rotation.T + self.position
        if self.rotation.ndim == 3:
            assert x.ndim == 1, "only one vector can be transformed at a time when using tensor pose"
            raise NotImplementedError("Tensor pose not yet implemented")


class Sensor(ABC):
    def __getitem__(self, item):
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Sensor output
        """
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

        angles = np.deg2rad(np.linspace(-5, 185, scans._shape[1]))
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
        omega_body = self._pre_process(omega_sensor)
        self._data = dict(zip(self.time, omega_body))

    def _pre_process(self, omega_sensor) -> np.ndarray:
        """
        Pre-process measurements into the body frame

        FOG (Fiber Optic Gyro) extrinsic calibration parameter from vehicle
        RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix, meter)
        RPY: 0 0 0
        R: 1 0 0 0 1 0 0 0 1
        T: -0.335 -0.035 0.78

        * The sensor measurements are stored as [timestamp, delta roll, delta pitch, delta yaw] in radians.
        """
        self._rpy = (0, 0, 0)
        self._rotation = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        self._position = np.array([-0.335, -0.035, 0.78])
        pose = Pose(self._rotation, self._position)

        return pose.transform(omega_sensor)


class Encoder(Sensor):
    def __init__(self, data_file='data/sensor_data/encoder.csv'):
        self.time, counts = utils.read_data_from_csv(data_file)
        some_data = self._pre_process(counts)
        self._data = dict(zip(self.time, some_data))

    def _pre_process(self, counts) -> np.ndarray:
        """
        Encoder calibrated parameter
        Encoder resolution: 4096
        Encoder left wheel diameter: 0.623479
        Encoder right wheel diameter: 0.622806
        Encoder wheel base: 1.52439

        * The encoder data is stored as [timestamp, left count, right count].

        Returns:
            The vehicle's velocity and angular velocity
        """
        return None


class Car:
    def __init__(self, n_particles):
        n_dims = 3
        self.rotation = np.stack([np.eye(n_dims)] * n_particles, axis=0)
        self.position = np.zeros((n_particles, n_dims))
        self.weights = np.ones((n_particles,)) / n_particles

    def transform(self, x):
        """
        Transform x into the world frame, using the pose of the maximum likelihood particle

        Args:
            x: a set of coordinates (samples, 3)

        Returns:
            x transformed into the world frame
        """
        # todo: also need a method transforming a single vector into multiple positions
        return self.ml_pose.transform(x)

    @property
    def ml_pose(self) -> Pose:
        """
        Pose object for the particle with the largest weight

        Returns:
            a Pose object
        """
        idx = np.argmax(self.weights)
        return Pose(self.rotation[idx], self.position[idx])

    @property
    def pose(self) -> Pose:
        """
        Pose object for the particle with the largest weight

        Returns:
            a Pose object
        """
        return Pose(self.rotation, self.position)


class Map:
    def __init__(self, resolution=0.1, x_range=(-50, 50), y_range=(-50, 50), lambda_max_factor=100):
        assert isinstance(self.resolution, float)
        self.resolution = resolution
        self.range = np.array([x_range, y_range])
        # x_min, y_min = self.range[:, 0]
        # x_max, y_max = self.range[:, 1]
        # x_min, x_max = self.range[0, :]
        # y_min, y_max = self.range[1, :]

        self._inc = np.log(4)
        self._lambda_max = self._inc * lambda_max_factor
        self._shape = np.array([int(np.ceil(np.diff(x_range) / resolution + 1)),
                                int(np.ceil(np.diff(y_range)) / resolution + 1)])
        self._map = np.zeros(self._shape, dtype=np.int8)

    @property
    def shape(self):
        return tuple(self._shape)

    def process_lidar(self, scan: np.ndarray, origin: Union[List, Tuple, np.ndarray]) -> bool:
        """
        Update the map according to the results of a lidar scan

        Args:
            scan: a set of scan points in the world frame
            origin: Origin of the scan. i.e., the ML location of the car

        Returns:
            True for a successful update
        """
        scan_cells = self.coord_to_cell(scan)
        origin_cell = self.coord_to_cell(origin)
        assert self.is_in_map(origin_cell), "origin is outside the map"
        scan_valid = self.valid_scan(scan_cells)

        valid_scan_cells = scan_cells[scan_valid]
        assert valid_scan_cells.ndim == 2

        self._positive_update(valid_scan_cells)
        self._negative_update(valid_scan_cells, origin)
        return False

    def _negative_update(self, scan_cells, origin) -> None:
        """
        Decrement the likelihood of cells where no object was detected

        Args:
            scan_cells:  Indices of cells where an object was detected
        """
        # todo: look into JITing this with numba
        for x, y in scan_cells:
            utils.bresenham2D(x, y, *origin)


    def _positive_update(self, scan_cells) -> None:
        """
        Increment the likelihood of cells where we detected an object
        
        Args:
            scan_cells: Indices of cells where an object was detected
        """
        self._map[scan_cells[:, 0], scan_cells[:, 1]] += self._inc

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
        point_is = np.ceil((point - self.range[:, 0]) / self.resolution).astype(np.int16) - 1
        return point_is


class Runner:
    """
    A runner class that processes sensor inputs and appropriately updates the car and map objects.
    """

    def __init__(self, n_particles=100):
        self.encoder = Encoder()
        self.gyro = Gyro()
        self.lidar = Lidar()
        self.car = Car(n_particles)
        self.map = Map()

        self.execution_seq = self.get_execution_sequence()

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
        assert execution_sequence._shape == (len(timestamps), 2)
        return execution_sequence

    def step_gyro(self, timestamp):
        pass

    def step_encoder(self, timestamp):
        pass

    def step_lidar(self, timestamp):
        scan_body = self.lidar[timestamp]
        scan_world = self.car.ml_pose.transform(scan_body)
