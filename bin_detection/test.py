import unittest

import numpy as np

from bin_detection.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DataLoader()

    def test_data_has_second_dim_equal_3(self):
        self.assertEqual(3, self.loader.data.shape[1])

    def test_data_length_matches_label_length(self):
        self.assertEqual(len(self.loader.data), len(self.loader.labels))


if __name__ == '__main__':
    unittest.main()
