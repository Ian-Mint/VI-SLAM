from bin_detection.common import *

if __name__ == '__main__':
    train_data.normalize(train_data.data)

    classifier.train()
    classifier.create_plots()
    classifier.dump_weights('weights.pkl')
