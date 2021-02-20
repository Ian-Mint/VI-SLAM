from collections import OrderedDict

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


class Lidar:
    """
    FOV: 190 (degree)
    Start angle: -5 (degree)
    End angle: 185 (degree)
    Angular resolution: 0.666 (degree)
    Max range: 80 (meter)

    * LiDAR rays with value 0.0 represent infinite range observations.
    """

    def __init__(self):
        self.time, self._scans = utils.read_data_from_csv('data/sensor_data/lidar.csv')
        self._max_range = 75.
        self._min_range = 2.

        body_xy_scans = self._pre_process()
        self.data = OrderedDict(zip(self.time, body_xy_scans))

    def update(self, timestamp, car: Car):
        """
        computes the measurement in the world frame, given the car object using only the particle with the largest
        weight.
        Args:
            timestamp: a timestamp corresponding to a measurement
            car: a Car object

        Returns:

        """

    def _pre_process(self) -> np.ndarray:
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
        self.rpy = (142.759, 0.0584636, 89.9254)
        self.rotation = np.array([[0.00130201, 0.796097, 0.605167],
                                  [0.999999, -0.000419027, -0.00160026],
                                  [-0.00102038, 0.605169, -0.796097]])
        self.position = np.array([0.8349, -0.0126869, 1.76416])
        self.pose = Pose(self.rotation, self.position)

        angles = np.linspace(-5, 185, self._scans.shape[1])
        x_scale = np.sin(angles)
        y_scale = -np.cos(angles)

        out_of_range_mask = np.logical_or(self._scans < self._min_range, self._scans > self._max_range)
        self._scans[out_of_range_mask] = np.nan

        x = self._scans * x_scale
        y = self._scans * y_scale

        xyz_scan_sensor = np.stack([x, y, np.zeros_like(x)], axis=2)
        xyz_scan_body = self.pose.transform(xyz_scan_sensor)

        # verify broadcasting worked as expected
        assert np.allclose(self.rotation @ xyz_scan_sensor[0, 0] + self.position, xyz_scan_body[0, 0])
        return xyz_scan_body[:, :, :2]
