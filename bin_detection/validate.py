import os

import matplotlib.pyplot as plt

from common import *

if __name__ == '__main__':
    train_data.normalize(train_data.data)

    weights = classifier.load_weights('weights.pkl')[0]

    for img_file in os.listdir(VAL_DATA_DIR):
        img = DataLoader.load_img(VAL_DATA_DIR, img_file)
        shape = img.shape
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2])).astype(float)
        train_data.normalize(img)

        mask = classifier.classify(img, weights)[0]
        mask = mask.reshape((shape[0], shape[1]))
        img = train_data.unnormalize(img)
        img = img.reshape(shape)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(mask)
        plt.show()
