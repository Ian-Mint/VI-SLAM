"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""
import pickle

from pixel_classification.data_loader import DataLoader
from pixel_classification.train import WEIGHTS_FILE
from regression import Regression

BLUE_DIR = 'data/training/blue'
GREEN_DIR = 'data/training/green'
RED_DIR = 'data/training/red'


class PixelClassifier:
    def __init__(self):
        """
        Initialize your classifier with any parameters and attributes you need
        """
        with open(WEIGHTS_FILE, 'rb') as f:
            self.weights = pickle.load(f)[0]

    def classify(self, X):
        """
        Classify a set of pixels into red, green, or blue

        Inputs:
          X: n x 3 matrix of RGB values
        Outputs:
          y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        """
        real = 1 + Regression.classify(X, self.weights)[0]
        return real
