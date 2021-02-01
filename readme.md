# Recycling bin image segmentation
```
.
├── bin_detection
│   ├── bin_detector.py
│   ├── data
│   │   ├── bin_labels
│   │   │   ├── ...
│   │   ├── masks
│   │   │   └── training
│   │   │       ├── black
│   │   │       │   ├── ...
│   │   │       ├── blue
│   │   │       │   ├── ...
│   │   │       ├── green
│   │   │       │   ├── ...
│   │   │       ├── not-blue
│   │   │       │   ├── ...
│   │   │       └── white
│   │   │           ├── ...
│   │   ├── training
│   │   │   ├── ...
│   │   └── validation
│   │       ├── ...
│   ├── data_loader.py
│   ├── mean.pkl
│   ├── pre_processing
│   │   ├── create_dataset.py
│   │   └── merge_masks.py
│   ├── requirements.txt
│   ├── roipoly
│   │   ├── __init__.py
│   │   ├── roipoly.py
│   │   └── version.py
│   ├── std.pkl
│   ├── test_bin_detector.py
│   ├── test.py
│   ├── test_roipoly.py
│   ├── train.py
│   └── weights.pkl
├── environment.yml
├── pixel_classification
│   ├── data
│   │   ├── training
│   │   │   ├── blue
│   │   │   │   ├── ...
│   │   │   ├── green
│   │   │   │   ├── ...
│   │   │   └── red
│   │   │       ├── ...
│   │   └── validation
│   │       ├── blue
│   │       │   ├── ...
│   │       ├── green
│   │       │   ├── ...
│   │       └── red
│   │           ├── ...
│   ├── data_loader.py
│   ├── generate_rgb_data.py
│   ├── pixel_classifier.py
│   ├── requirements.txt
│   ├── test_pixel_classifier.py
│   ├── test.py
│   ├── train.py
│   └── weights.pkl
├── readme.md
└── regression.py
```

## Installation

`conda create env`

`conda activate ece276a-hw1`

## Usage

### Pixel classification

First, train the model and save the weights in weights.pkl:

`python pixel_classification/train.py`

To run unit tests:

`python pixel_classification/test.py `

To run validation test:

`python pixel_classification/test_pixel_classifier.py`

### Bin detection

Create the image masks, selecting the color you want to label:

`python bin_detection/pre_processing/create_dataset.py --color blue`

There is also a utility script, `merge_masks.py` that can be used to merge two or more masks.

With trained data organized in directory structure outlined at the top of this document, we can start training:

`python bin_detection/train.py `

This will save the weights, which can then be used for color classification. Finally, unittest and validation tests can
be run using the following commands respectively:

`python bin_detection/test.py`

`python bin_detection/test_bin_detector.py `