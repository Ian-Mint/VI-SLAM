import os
import unittest
from typing import Union, Tuple, List

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from bin_detection.data_loader import DataLoader
from bin_detection.bin_detector import BinDetector

TRAIN_DATA_DIR = 'bin_detection/data/training'
TRAIN_MASK_DIR = 'bin_detection/data/masks/training'
VAL_DATA_DIR = 'bin_detection/data/validation'
VAL_MASK_DIR = 'bin_detection/data/masks/validation'


@unittest.skip
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


class TestBinDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.bd = BinDetector()
        self.images = [
            DataLoader.load_img(VAL_DATA_DIR, img_file) for img_file in os.listdir(VAL_DATA_DIR)
        ]

    def test_bound_bin(self):
        for img in self.images:
            mask = self.bd.segment_image(img)
            bounding_boxes = self.bd.get_bounding_boxes(mask)
            print(bounding_boxes)
            self.assertIsInstance(bounding_boxes, list)
            for box in bounding_boxes:
                self.assertIsInstance(box, tuple)
                for element in box:
                    self.assertIsInstance(element, int)

            self.plot(img, mask, bounding_boxes)

    def plot(self, img: Union[np.ndarray, int], mask: np.ndarray, bounding_boxes: List[List[int]]):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(mask)

        for ax in axs:
            for box in bounding_boxes:
                rect = self.get_rect(box)
                ax.add_patch(rect)

        plt.show()

    @staticmethod
    def get_rect(box: List[int]):
        width = abs(box[1] - box[3])
        height = abs(box[0] - box[2])
        anchor = (box[1], box[2] - height)
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        return rect


if __name__ == '__main__':
    unittest.main()
