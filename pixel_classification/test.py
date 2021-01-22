import unittest

import numpy as np

# noinspection PyUnresolvedReferences
from generate_rgb_data import read_pixels
from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DataLoader()

    def test_we_get_3_data_classes(self):
        self.assertEqual(3, len(self.loader.data))

    def test_each_class_is_a_numpy_array(self):
        for d in self.loader.data:
            self.assertIsInstance(d, np.ndarray)


if __name__ == '__main__':
    unittest.main()
