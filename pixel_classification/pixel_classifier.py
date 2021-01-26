"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""

import numpy as np

from data_loader import DataLoader
from regression import Regression

BLUE_DIR = 'data/training/blue'
GREEN_DIR = 'data/training/green'
RED_DIR = 'data/training/red'


class PixelClassifier:
    def __init__(self):
        """
        Initialize your classifier with any parameters and attributes you need
        """
        data_loader = DataLoader(n_splits=0, resample=True, cross_validation=False)
        data, labels = data_loader.get_splits()

        self.learner = Regression(data, labels, learning_rate=10, epochs=5000, cross_validation=False)
        self.learner.train()

    def classify(self, X):
        """
        Classify a set of pixels into red, green, or blue

        Inputs:
          X: n x 3 matrix of RGB values
        Outputs:
          y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        """
        real = 1 + self.learner.classify(X)
        return real
