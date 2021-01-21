'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
from bin_detector import BinDetector
import yaml


def iou(box1,box2):
  '''
    Computes the intersection over union of two bounding boxes box = [x1,y1,x2,y2]
    where (x1, y1) and (x2, y2) are the top left and bottom right coordinates respectively
  '''
  x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
  x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
  inter_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
  union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) - inter_area
  return inter_area/union_area


def compare_boxes(true_boxes, estm_boxes):
  '''
    Compares the intersection over union of two bounding box lists.
    The iou is computed for every box in true_boxes sequentially with respect to each box in estm_boxes.
    If any estm_box achieves an iou of more than 0.5, then the true box is considered accurately detected.
  '''
  num_true_boxes = len(true_boxes)
  if num_true_boxes == 0:
    return float(len(estm_boxes) == 0)

  accuracy = 0.0
  for box1 in true_boxes:
    for box2 in estm_boxes:
      if iou(box1,box2) >= 0.5:
        accuracy += 1.0
        break
  return accuracy / num_true_boxes



if __name__ == '__main__':
  folder = "data/validation"
  my_detector = BinDetector()
  for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
      # read one test image
      img = cv2.imread(os.path.join(folder,filename))

      # load ground truth label
      with open(os.path.join(folder,os.path.splitext(filename)[0]+'.txt'), 'r') as stream:
        true_boxes = yaml.safe_load(stream)
      
      # show image
      for box in true_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
      cv2.imshow('image', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      # convert from BGR (opencv convention) to RGB (everyone's convention)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # segment the image
      mask_img = my_detector.segment_image(img)

      # detect recycling bins
      estm_boxes = my_detector.get_bounding_boxes(mask_img)
      
      # The autograder checks your answers to the functions segment_image() and get_bounding_box()
      
      # measure accuracy      
      accuracy = compare_boxes(true_boxes, estm_boxes)
      
      print('The accuracy for %s is %f %%.'%(filename,accuracy*100))



