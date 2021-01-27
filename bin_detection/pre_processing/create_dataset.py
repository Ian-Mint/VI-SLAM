import os
import pickle

import cv2

from bin_detection.roipoly import RoiPoly
from matplotlib import pyplot as plt

DATA_DIR = 'data/validation'

if __name__ == '__main__':
    mask_list = []
    # for fn in os.listdir(DATA_DIR):
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.txt'):
            continue
        path = os.path.join(os.path.abspath(os.curdir), DATA_DIR, fn)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots()
        ax.imshow(img)
        roi = RoiPoly(fig=fig, ax=ax, color='r')
        mask = roi.get_mask(img)
        with open(f'data/masks/validation/{fn}_mask.pkl', 'wb') as f:
            pickle.dump(mask, f)
        mask_list.append(mask)

