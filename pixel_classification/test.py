#!/usr/bin/env python
import os
import sys
import unittest
sys.path.append(os.path.abspath(os.curdir))

import numpy as np

# noinspection PyUnresolvedReferences
from generate_rgb_data import read_pixels
from data_loader import DataLoader
from pixel_classifier import PixelClassifier
from regression import Regression


class TestDataLoaderWithCV(unittest.TestCase):
    def setUp(self) -> None:
        self.n_splits = 5
        self.resample = True
        self.loader = DataLoader(self.n_splits, resample=self.resample, cross_validation=True)

    def test_we_get_3_data_classes(self):
        self.assertEqual(3, len(self.loader.train_data))

    def test_each_class_is_a_numpy_array(self):
        for d in self.loader.train_data:
            self.assertIsInstance(d, np.ndarray)

    def test_each_class_has_second_dim_equal_to_3(self):
        for d in self.loader.train_data:
            self.assertEqual(3, d.shape[1])

    def test_splits(self):
        data, labels = self.loader.get_splits()
        self.assertEqual(self.n_splits, len(data))
        self.assertEqual(self.n_splits, len(labels))

    def test_resampling(self):
        if self.resample:
            shape = self.loader.train_data[0].shape
            for d in self.loader.train_data:
                self.assertEqual(shape, d.shape)


class TestDataLoaderWithoutCV(unittest.TestCase):
    def setUp(self) -> None:
        self.resample = True
        self.loader = DataLoader(0, resample=self.resample, cross_validation=False)

    def test_we_get_3_data_classes(self):
        self.assertEqual(3, len(self.loader.train_data))

    def test_each_class_is_a_numpy_array(self):
        for d in self.loader.train_data:
            self.assertIsInstance(d, np.ndarray)

    def test_each_class_has_second_dim_equal_to_3(self):
        for d in self.loader.train_data:
            self.assertEqual(3, d.shape[1])

    def test_splits(self):
        data, labels = self.loader.get_splits()
        self.assertEqual(2, len(data))
        self.assertEqual(2, len(labels))

    def test_resampling(self):
        if self.resample:
            shape = self.loader.train_data[0].shape
            for d in self.loader.train_data:
                self.assertEqual(shape, d.shape)


class TestRegression(unittest.TestCase):
    @staticmethod
    def loss(x, y):
        exp = np.exp(x)
        log_softmax = np.log(exp / np.sum(exp))

    def setUp(self) -> None:
        data_loader = DataLoader(n_splits=0, resample=True, cross_validation=False)
        self.data, self.labels = data_loader.get_splits()

        self.learner = Regression(self.data, self.labels, learning_rate=10, epochs=5000, cross_validation=False)

    @unittest.skip  # incomplete test
    def test_grad(self):
        epsilon = np.logspace(-3, -1, 100)
        labels_one_hot = Regression._encode_one_hot(self.labels[0], 3)
        weights = self.learner.weights[0]

        grad = self.learner._grad(weights, self.data[0], self.labels[0])

    def test_debug(self):
        self.learner.train()
        self.learner.plot()


class TestPixelClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = PixelClassifier()

    def test_blue_precision_is_above_50(self):
        folder = 'pixel_classification/data/training/blue'
        x = read_pixels(folder)
        y = self.classifier.classify(x)

        self.assertGreater(sum(y == 3) / len(y), 0.5)

    def test_green_precision_is_above_50(self):
        folder = 'pixel_classification/data/training/green'
        x = read_pixels(folder)
        y = self.classifier.classify(x)

        self.assertGreater(sum(y == 2) / len(y), 0.5)

    def test_red_precision_is_above_50(self):
        folder = 'pixel_classification/data/training/red'
        x = read_pixels(folder)
        y = self.classifier.classify(x)

        self.assertGreater(sum(y == 1) / len(y), 0.5)


if __name__ == '__main__':
    unittest.main()
