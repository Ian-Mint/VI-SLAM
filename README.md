# ECE 276A PR3

```
.
├── data
│   ├── sensor_data
│   │   ├── encoder.csv
│   │   ├── gyro.csv
│   │   └── lidar.csv
│   ├── stereo_images
│   │   ├── stereo_left
│   │   │   ├── *.png
│   │   └── stereo_right
│   │   │   ├── *.png
│   └── test
│       ├── encoder.csv
│       ├── gyro.csv
│       └── lidar.csv
└── results
├── environment.yml
├── main.py
├── pr2_utils.py
├── README.md
├── requirements.txt
├── sensors.py
└── test.py
```

## Installation

Required a conda environment for compatibility with `numba`.

set up environment with:

```
conda env create -f environment.yml -n ece276a
```

Store data as shown in the directory structure above.

## Run

Run with:

```python main.py```

There are several options that can be adjusted within `main.py`.
Run options are available in the constructor for `Runner`.

