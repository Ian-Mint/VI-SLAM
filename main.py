"""
All timestamps should be put together and sorted, with the associated sensor tagged. Execution will proceed
sequentially through the timestamps, calling the appropriate handlers.
"""

from sensors import *


if __name__ == '__main__':
    runner = Runner(
        Encoder(),
        Gyro(),
        Lidar(downsample=1),
        Car(n_particles=10, v_var=1e-9, omega_var=1e-10, resample_threshold=0.5),  # variance in m/nanosecond
        Map(resolution=2., x_range=(-200, 1300), y_range=(-1300, 200), lambda_max_factor=100, increment=16,
            decrement=4),
        downsample=1,
        plot_interval=1000000000000
    )
    runner.run()
    runner.plot()
    input("Press any key to exit...")
