'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2

def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values 
  '''  
  n = len(next(os.walk(folder))[2]) # number of files
  X = np.empty([n, 3])
  i = 0
  
  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
  
  for filename in os.listdir(folder):  
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1
    
    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()

  return X


if __name__ == '__main__':
  folder = 'data/training'
  X1 = read_pixels(folder+'/red', verbose = True)
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))

  

