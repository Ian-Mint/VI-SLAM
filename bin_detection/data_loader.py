import os
import pickle

import cv2
import numpy as np
import skimage.transform


class DataLoader:
    def __init__(self, data_dir: str, mask_dir: str):
        data = []
        labels = []

        for img_file in os.listdir(data_dir):
            mask_path = os.path.join(os.path.abspath(os.curdir), mask_dir, f'{img_file}_mask.pkl')

            img_path = os.path.join(os.path.abspath(os.curdir), data_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img_shape = (500, 500)
            img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
            data.append(img.reshape((np.prod(self.img_shape), 3)))

            with open(mask_path, 'rb') as f:
                mask: np.ndarray = pickle.load(f)
            mask = skimage.transform.resize(mask, self.img_shape).astype(int)
            labels.append(mask.flatten())

        self.data = np.concatenate(data, axis=0)
        self.labels = np.concatenate(labels, axis=0)

        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)

        self.data = self.normalize(self.data)

    def normalize(self, data: np.ndarray):
        """
        Normalize according to the mean and standard deviation of this data.
        Args:
            data:

        Returns:

        """
        data = (data - self.mean) / self.std
        return data
