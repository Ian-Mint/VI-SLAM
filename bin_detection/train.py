#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from bin_detection.data_loader import DataLoader
from regression import Regression

TRAIN_DATA_DIR = 'bin_detection/data/training'
TRAIN_MASK_DIR = 'bin_detection/data/masks/training'
VAL_DATA_DIR = 'bin_detection/data/validation'
VAL_MASK_DIR = 'bin_detection/data/masks/validation'

if __name__ == '__main__':
    train_data = DataLoader(TRAIN_DATA_DIR, TRAIN_MASK_DIR)
    train_data.dump_stats()
    classifier = Regression(
        [train_data.data],
        [train_data.labels],
        learning_rate=1,
        epochs=3000,
        cross_validation=False,
    )

    train_data.normalize(train_data.data)

    classifier.train()
    classifier.plot()
    classifier.dump_weights('bin_detection/weights.pkl')
