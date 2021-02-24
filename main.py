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
        Car(n_particles=100, v_var=1e-3, omega_var=1e-6, resample_threshold=0.5),
        Map(resolution=0.1, x_range=(-50, 50), y_range=(-50, 50), lambda_max_factor=100),
        downsample=1,
        plot_interval=1
    )
    runner.run()
