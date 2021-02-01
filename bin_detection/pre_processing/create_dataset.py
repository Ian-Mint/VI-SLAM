#!/usr/bin/env python
import argparse
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.curdir))

import cv2

from bin_detection.roipoly import RoiPoly
from matplotlib import pyplot as plt

DATA_DIR = 'bin_detection/data/training'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', required=True)
    args = parser.parse_args()

    mask_list = []
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

        mask_dir = f'bin_detection/data/masks/training/{args.color}'
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        with open(f'{mask_dir}/{fn}_mask.pkl', 'wb') as f:
            pickle.dump(mask, f)
        mask_list.append(mask)

