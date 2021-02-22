from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import pr2_utils as utils


class Pose:
    def __init__(self, rotation, position):
        self.rotation = rotation
        self.position = position

    def transform(self, x):
        """
        multiplies by rotation and adds position
        Args:
            x: coordinates must be in last dimension

        Returns:
            transformed coordinates
        """
        return x @ self.rotation.T + self.position


class Lidar:
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

    def __getitem__(self, item):
        """
        Get the measurement associated with the timestamp

        Args:
            item: a timestamp

        Returns:
            Array of xy coordinates corresponding to a scan in the body frame
        """
        return self._data.pop(item)

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


class Gyro:
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


class Encoder:
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
        self.rotation = np.zeros((n_particles, 2, 2))
        self.position = np.zeros((n_particles, 2))
        self.weights = np.ones((n_particles,)) / n_particles

    @property
    def pose(self) -> Pose:
        """
        Pose object for the particle with the largest weight

        Returns:
            a Pose object
        """
        idx = np.argmax(self.weights)
        return Pose(self.rotation[idx], self.position[idx])


class Map:
    def __init__(self, resolution=0.1, x_range=(-50, 50), y_range=(-50, 50)):
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range

        self.shape = (int(np.ceil(np.diff(x_range) / resolution + 1)),
                      int(np.ceil(np.diff(y_range)) / resolution + 1))
        self._map = np.zeros(self.shape, dtype=np.int8)


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
        assert execution_sequence.shape == (len(timestamps), 2)
        return execution_sequence

    def step_gyro(self, timestamp):
        pass

    def step_encoder(self, timestamp):
        pass

    def step_lidar(self, timestamp):
        pass
