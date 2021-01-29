import os
import unittest
from typing import Union, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bin_detection.data_loader import DataLoader
from bin_detection.bin_detector import BinDetector

TRAIN_DATA_DIR = 'bin_detection/data/training'
TRAIN_MASK_DIR = 'bin_detection/data/masks/training'
VAL_DATA_DIR = 'bin_detection/data/validation'
VAL_MASK_DIR = 'bin_detection/data/masks/validation'


class TestSegmentImages(unittest.TestCase):
    def setUp(self) -> None:
        self.bd = BinDetector()
        self.images = [
            DataLoader.load_img(VAL_DATA_DIR, img_file) for img_file in os.listdir(VAL_DATA_DIR)
        ]

    def test_segment_images(self):
        for img in self.images:
            mask = self.bd.segment_image(img)
            self.plot(img, mask)

    def plot(self, img: Union[np.ndarray, int], mask: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(mask)
        plt.show()


if __name__ == '__main__':
    unittest.main()
