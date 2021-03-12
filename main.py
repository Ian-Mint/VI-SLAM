import time

import numpy as np

from sensors import *


if __name__ == '__main__':

    with np.load('data/features.npz') as data:
        k = data['K']
        b = data['b']
        features = data.get('features', None)  # for when we make our own features
        linear_velocity = data['linear_velocity']
        angular_velocity = data['angular_velocity']
        imu_T_cam = data['imu_T_cam']
        timestamps = data['time_stamps']

    time_steps = np.diff(timestamps).squeeze()
    linear_velocity = linear_velocity[:, :-1]
    angular_velocity = angular_velocity[:, :-1]
    features = features[:, :, :-1]

    downsample_by = 10
    features = features[..., ::downsample_by, :]
    n_points = features.shape[1]
    n_samples = len(time_steps)

    # todo: move on to full slam
    accel_var = 1e-4
    gyro_var = 1e-6
    imu_variance = np.array([accel_var, accel_var, accel_var, gyro_var, gyro_var, gyro_var])
    runner = Runner(Camera(features, time_steps, k, b, imu_T_cam, depth_threshold=50),
                    Imu(linear_velocity, angular_velocity, time_steps, imu_variance),
                    Map(n_points),
                    n_samples, plot_interval=5000)
    start = time.time()
    runner.run()
    print(f'complete in {time.time() - start:02f} seconds')

    runner.plot(blocking=True)

    # debugging
    from functions import *
    print(pose_to_axis(runner.imu.pose))
    print(pose_to_angle(runner.imu.pose))
    print(runner.imu.pose[:3, -1])

    bad_points = np.argwhere(np.logical_or.reduce([
        runner.map.points[0] > 250,
        runner.map.points[0] < -1050,
        runner.map.points[1] > 150,
        runner.map.points[1] < -450,
    ]))

    exit()
