import time

from sensors import *


if __name__ == '__main__':
    runner = Runner(
        Camera(),
        Encoder(),
        Gyro(),
        Lidar(downsample=1),
        Car(n_particles=20, v_var=1e-9, omega_var=1e-11, resample_threshold=0.8),  # variance in m/nanosecond
        Map(resolution=1., x_range=(-100, 1400), y_range=(-1200, 100), lambda_max_factor=100, increment=16,
            decrement=4),
        downsample=1,
        plot_interval=5000,
        plot_type='texture',
    )
    start = time.time()
    runner.run()
    print(f'complete in {time.time() - start:02f} seconds')
    runner.plot()
