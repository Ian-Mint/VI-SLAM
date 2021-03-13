# ECE 276A PR3

```
.
├── data
│   ├── features.npz
│   ├── images
│   │   ├── image_00
│   │   │   ├── data
│   │   │   │   ├── .png
│   │   │   └── timestamps.txt
│   │   └── image_01
│   │       ├── data
│   │       │   ├── *.png
│   │       └── timestamps.txt
│   └── raw.npz
├── environment.yml
├── functions.py
├── main.py
├── README.md
├── requirements.txt
├── sensors.py
├── test.py
└── utils.py
```

## Installation

Requires a conda environment for compatibility with `numba`.

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

