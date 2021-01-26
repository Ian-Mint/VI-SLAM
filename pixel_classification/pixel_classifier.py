"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""
import pickle

from regression import Regression


class PixelClassifier:
    def __init__(self):
        """
        Initialize your classifier with any parameters and attributes you need
        """
        with open('pixel_classification/weights.pkl', 'rb') as f:
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
