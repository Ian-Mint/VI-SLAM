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

    time_steps = np.diff(timestamps)
    linear_velocity = linear_velocity[:, :-1]
    angular_velocity = angular_velocity[:, :-1]
    features = features[:, :-1]

    runner = Runner(
        Camera(features, time_steps),
        Imu(linear_velocity, angular_velocity, time_steps),
        Vehicle(),
        Map(),
        downsample=1,
        plot_interval=5000,
    )
    start = time.time()
    runner.run()
    print(f'complete in {time.time() - start:02f} seconds')
    runner.plot()
