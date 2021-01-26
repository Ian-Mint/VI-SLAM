'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier

if __name__ == '__main__':
  # test the classifier
  
  folder = 'data/validation/blue'
  
  X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)
  
  print('Precision: %f' % (sum(y==1)/y.shape[0]))

  folder = 'data/validation/green'

  X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)

  print('Precision: %f' % (sum(y==2)/y.shape[0]))

  folder = 'data/validation/red'

  X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)

  print('Precision: %f' % (sum(y==3)/y.shape[0]))

  
