import os
import pickle
from typing import List, Tuple
from collections import Counter

import cv2
import numpy as np
import skimage.transform

classes = ('blue', 'black', 'green', 'not-blue', 'white')


class DataLoader:
    def __init__(self, data_dir: str, mask_dir: str, resample=True, load=True):
        if load:
            self.img_shape = (100, 100)

            data = []

            for img_file in os.listdir(data_dir):
                data.append([])
                img_path = os.path.join(os.path.abspath(os.curdir), data_dir, img_file)
                assert os.path.exists(img_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
                img = img.reshape((np.prod(self.img_shape), 3))

                for class_id, class_name in enumerate(classes):
                    mask_path = os.path.join(os.path.abspath(os.curdir), mask_dir, class_name, f'{img_file}_mask.pkl')
                    assert os.path.exists(mask_path)
                    with open(mask_path, 'rb') as f:
                        mask: np.ndarray = pickle.load(f)
                    mask = skimage.transform.resize(mask, self.img_shape) > 0.5
                    mask = mask.flatten()

                    masked_img = img[mask, :].astype(float)
                    assert len(masked_img) == np.sum(mask)
                    data[-1].append(masked_img)

            # resort the list to (class x image)
            data = list(zip(*data))
            data = [np.concatenate(d, axis=0) for d in data]
            if resample:
                self._resample(data)
            labels = [np.zeros((len(d),), dtype=int) + class_id for class_id, d in enumerate(data)]

            self.data = np.concatenate(data, axis=0)
            self.labels = np.concatenate(labels, axis=0)

            self.mean = self.data.mean(axis=0)
            self.std = self.data.std(axis=0)

            print(classes)
            print(Counter(self.labels))

    def dump_stats(self):
        with open('bin_detection/mean.pkl', 'wb') as f:
            pickle.dump(self.mean, f)

        with open('bin_detection/std.pkl', 'wb') as f:
            pickle.dump(self.std, f)

    def load_stats(self):
        with open('bin_detection/mean.pkl', 'rb') as f:
            self.mean = pickle.load(f)

        with open('bin_detection/std.pkl', 'rb') as f:
            self.std = pickle.load(f)

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
    def load_img(data_dir, img_file):
        img_path = os.path.join(os.path.abspath(os.curdir), data_dir, img_file)
        img = cv2.imread(img_path)
        return img

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize according to the mean and standard deviation of this data.
        Args:
            data: data to normalize

        Returns:
            transformed data
        """
        data[:] = (data - self.mean) / self.std
        return data

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        """
        inverse of normalize
        Args:
            data: data to restore

        Returns:
            original data
        """
        data[:] = data * self.std + self.mean
        assert np.max(data) <= 255
        assert np.min(data) >= 0
        return data.astype(int)
