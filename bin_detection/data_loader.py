import os
import pickle
from collections import Counter

import cv2
import numpy as np
import skimage.transform

classes = ('blue', 'black', 'green', 'not-blue', 'white')


class DataLoader:
    def __init__(self, data_dir: str, mask_dir: str):
        self.img_shape = (100, 100)

        data = []
        labels = []

        for img_file in os.listdir(data_dir):
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
                data.append(masked_img)
                labels.append(np.zeros((len(masked_img)), dtype=int) + class_id)

        self.data = np.concatenate(data, axis=0)
        self.labels = np.concatenate(labels, axis=0)

        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)

        print(classes)
        print(Counter(self.labels))

    def normalize(self, data: np.ndarray):
        """
        Normalize according to the mean and standard deviation of this data.
        Args:
            data:

        Returns:

        """
        data[:] = (data - self.mean) / self.std
        return data
