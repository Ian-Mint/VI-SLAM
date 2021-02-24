"""
All timestamps should be put together and sorted, with the associated sensor tagged. Execution will proceed
sequentially through the timestamps, calling the appropriate handlers.
"""

from sensors import *

if __name__ == '__main__':
    runner = Runner(
        Encoder(),
        Gyro(),
        Lidar(),
        Car(n_particles=4, v_var=0, omega_var=0, resample_threshold=0.5),  # variance in m/nanosecond
        Map(resolution=1., x_range=(-500, 500), y_range=(-500, 500), lambda_max_factor=100),
        downsample=1,
        plot_interval=1000
    )
    runner.run()
