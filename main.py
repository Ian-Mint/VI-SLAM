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
        Car(n_particles=100, v_var=1e-9, omega_var=1e-10, resample_threshold=0.5),  # variance in m/nanosecond
        Map(resolution=1., x_range=(-100, 1300), y_range=(-1200, 200), lambda_max_factor=100),
        downsample=1,
        plot_interval=1000
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        import matplotlib.pyplot as plt
        plt.show()
