from collections import namedtuple

import numpy as np
from typing import List

from generate_rgb_data import read_pixels

BLUE_DIR = 'data/training/blue'
GREEN_DIR = 'data/training/green'
RED_DIR = 'data/training/red'

CLASSES = ('blue', 'green', 'red')
Classes = namedtuple('ClassDirs', CLASSES)
CLASS_DIRS = Classes(BLUE_DIR, GREEN_DIR, RED_DIR)


class DataLoader:
    def __init__(self, n_splits=5, resample=True):
        self.k = n_splits
        self.data = []
        self._load_data()
        if resample:
            self._resample()

    def _resample(self):
        max_len = max(len(d) for d in self.data)

        for label in range(len(self.data)):
            d = self.data[label]
            missing = max_len - len(d)
            if missing == 0:
                continue

            mean = np.mean(d, 0)
            cov = np.cov(d.T)
            samples = np.random.multivariate_normal(mean, cov, size=missing)

            self.data[label] = np.concatenate((d, samples), axis=0)
            np.random.shuffle(self.data[label])

    def _load_data(self):
        for d in CLASS_DIRS:
            self.data.append(read_pixels(d))

    def get_splits(self):
        """
        Splits the data and labels into k splits.

        Returns: data splits, label splits

        """
        num_of_classes = len(self.data)

        data_np_classes_split = []
        labels_classes_split = []

        for label in range(num_of_classes):
            data_copy = self.data[label].copy()

            data_np_split = np.array_split(data_copy, self.k)  # split the array into k partitions
            labels_split = []

            for data_split in data_np_split:  # generate the labels for each of the data splits
                length = len(data_split)
                label_add = [label] * length
                labels_split.append(label_add.copy())

            # adds the splits to the corresponding class index
            data_np_classes_split.append(data_np_split.copy())
            labels_classes_split.append(labels_split.copy())

        final_data_np_split = []
        final_labels_split = []
        for split_index in range(self.k):  # this rearranges the data so that the splits are right [split]
            data_class = None
            label_class = None
            for label in range(num_of_classes):
                if data_class is None:  # if this is the first one
                    data_class = data_np_classes_split[label][split_index].copy()
                    label_class = labels_classes_split[label][split_index].copy()
                else:  # concatenate the other classes into this split
                    data_class = np.concatenate((data_class, data_np_classes_split[label][split_index].copy()),
                                                axis=0)
                    label_class = np.concatenate((label_class, labels_classes_split[label][split_index].copy()),
                                                 axis=0)
            # append the split with all the classes
            final_data_np_split.append(data_class.copy())
            final_labels_split.append(label_class.copy())

        return final_data_np_split, final_labels_split
