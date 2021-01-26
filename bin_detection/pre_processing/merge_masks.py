#!/usr/bin/env python

import argparse
import pickle

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', nargs='+')
    parser.add_argument('--outfile')
    args = parser.parse_args()

    masks = []
    for fn in args.masks:
        with open(fn, 'rb') as f:
            masks.append(pickle.load(f))

    output = np.logical_or.reduce(masks)

    with open(args.outfile, 'wb') as f:
        pickle.dump(output, f)

