'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    # YOUR CODE HERE
    # Just a random classifier for now
    # Replace this with your own approach 
    y = 1 + np.random.randint(3, size=X.shape[0])
    return y

