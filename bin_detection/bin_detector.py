"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""
import pickle

import numpy as np
import cv2
from skimage.measure import label, regionprops

from regression import Regression
from bin_detection.data_loader import DataLoader


class BinDetector:
    def __init__(self):
        """
        Initialize your recycling bin detector with the attributes you need,
        e.g., parameters of your classifier
        """
        self.data_loader = DataLoader('', '', load=False)
        self.data_loader.load_stats()
        with open('bin_detection/weights.pkl', 'rb') as f:
            self.weights = pickle.load(f)[0]

    def segment_image(self, img):
        """
        Obtain a segmented image using a color classifier,
        e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
        call other functions in this class if needed
        
        Args:
            img - original image
        Returns:
            mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = img.shape
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2])).astype(float)
        self.data_loader.normalize(img)

        mask = Regression.classify(img, self.weights)[0]
        mask = mask.reshape((shape[0], shape[1])) == 0
        return mask

    def get_bounding_boxes(self, img: np.ndarray):
        """
        Find the bounding boxes of the recycling bins
        call other functions in this class if needed
        
        Args:
            img - Binary image
        Returns:
            boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
            where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        """
        # YOUR CODE HERE
        x = np.sort(np.random.randint(img.shape[0], size=2)).tolist()
        y = np.sort(np.random.randint(img.shape[1], size=2)).tolist()
        boxes = [[x[0], y[0], x[1], y[1]]]
        boxes = [[182, 101, 313, 295]]
        return boxes
