from collections import namedtuple

import numpy as np
from typing import List, Tuple

from generate_rgb_data import read_pixels

CLASSES = ('blue', 'green', 'red')
Classes = namedtuple('ClassDirs', CLASSES)

TRAIN_DIRS = Classes('data/training/blue', 'data/training/green', 'data/training/red')
VAL_DIRS = Classes('data/validation/blue', 'data/validation/green', 'data/validation/red')


class DataLoader:
    def __init__(self, n_splits=5, cross_validation=True, resample=True):
        self.cv = cross_validation
        self.k = n_splits
        self.train_data = []
        self._load_data(TRAIN_DIRS, self.train_data)

        if not cross_validation:
            self.val_data = []
            self._load_data(VAL_DIRS, self.val_data)

        if resample:
            self._resample(self.train_data)

    @staticmethod
    def _resample(data: List[np.ndarray]) -> None:
        """
        Creates samples for data classes that have fewer than the maximum number of samples. Re-sampling is done
        assuming a Gaussian data distribution.

        Args:
            data: Data to resample (n x d)
        """
        max_len = max(len(d) for d in data)

        for label in range(len(data)):
            d = data[label]
            missing = max_len - len(d)
            if missing == 0:
                continue

            mean = np.mean(d, 0)
            cov = np.cov(d.T)
            samples = np.random.multivariate_normal(mean, cov, size=missing)

            data[label] = np.concatenate((d, samples), axis=0)
            np.random.shuffle(data[label])

    @staticmethod
    def _load_data(dirs: Classes, out: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Load data into out from the specified directories
        Args:
            dirs: directories stored in a Classes object
            out: list in which to store loaded data

        Returns:
            out
        """
        if out is None:
            out = []
        for d in dirs:
            out.append(read_pixels(d))
        return out

    def get_splits(self):
        """
        Splits the data and labels into k splits.

        Returns: data splits, label splits

        """
        if not self.cv:
            return self._get_no_cv_splits()

        num_of_classes = len(self.train_data)

        data_np_classes_split = []
        labels_classes_split = []

        for label in range(num_of_classes):
            data_copy = self.train_data[label].copy()

            data_np_split = np.array_split(data_copy, self.k)  # split the array into k partitions
            labels_split = []

            for data_split in data_np_split:  # generate the labels for each of the data splits
                label_add = [label] * len(data_split)
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

    def _get_no_cv_splits(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Just return two splits, one for train and one for validation. For when you don't use cross-validation
        """
        train_data = np.concatenate(self.train_data)
        val_data = np.concatenate(self.val_data)
        train_labels = self._get_labels_given_data(self.train_data)
        val_labels = self._get_labels_given_data(self.val_data)
        return [train_data, val_data], [train_labels, val_labels]

    @staticmethod
    def _get_labels_given_data(data: List[np.ndarray]) -> np.ndarray:
        """
        Labels in the same order as the class-separated data. Labels are assigned by the index of the data in the list.
        Args:
            data: A list of the data separated by class
        """
        assert isinstance(data, list)
        assert isinstance(data[0], np.ndarray)

        return np.concatenate(
            [np.array([n] * len(d), dtype=int) for n, d in enumerate(data)],
            axis=0
        )
