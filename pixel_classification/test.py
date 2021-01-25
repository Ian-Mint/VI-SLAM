import unittest

import numpy as np

# noinspection PyUnresolvedReferences
from generate_rgb_data import read_pixels
from data_loader import DataLoader
from pixel_classifier import PixelClassifier


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.n_splits = 5
        self.resample = True
        self.loader = DataLoader(self.n_splits, resample=self.resample)

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


class TestPixelClassifier(unittest.TestCase):
    def test_basic(self):
        folder = 'data/training/blue'

        X = read_pixels(folder)
        myPixelClassifier = PixelClassifier()
        y = myPixelClassifier.classify(X)

        print('Precision: %f' % (sum(y == 1) / y.shape[0]))


if __name__ == '__main__':
    unittest.main()
