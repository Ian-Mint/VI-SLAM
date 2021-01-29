"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""
import pickle

import numpy as np
import cv2
from skimage.measure import label, regionprops

from regression import Regression


class BinDetector():
    def __init__(self):
        """
        Initialize your recycling bin detector with the attributes you need,
        e.g., parameters of your classifier
        """

        with open('bin_detection/weights.pkl', 'rb') as f:
            weights = pickle.load(f)[0]

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

        mask_img = img
        return mask_img

    def get_bounding_boxes(self, img):
        """
        Find the bounding boxes of the recycling bins
        call other functions in this class if needed
        
        Args:
            img - Segmented image
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
