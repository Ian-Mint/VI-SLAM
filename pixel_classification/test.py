import unittest

import numpy as np

# noinspection PyUnresolvedReferences
from generate_rgb_data import read_pixels
from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.n_splits = 5
        self.resample = True
        self.loader = DataLoader(self.n_splits, self.resample)

    def test_we_get_3_data_classes(self):
        self.assertEqual(3, len(self.loader.data))

    def test_each_class_is_a_numpy_array(self):
        for d in self.loader.data:
            self.assertIsInstance(d, np.ndarray)

    def test_each_class_has_second_dim_equal_to_3(self):
        for d in self.loader.data:
            self.assertEqual(3, d.shape[1])

    def test_splits(self):
        data, labels = self.loader.split()
        self.assertEqual(self.n_splits, len(data))
        self.assertEqual(self.n_splits, len(labels))

    def test_resampling(self):
        if self.resample:
            shape = self.loader.data[0].shape
            for d in self.loader.data:
                self.assertEqual(shape, d.shape)


if __name__ == '__main__':
    unittest.main()
