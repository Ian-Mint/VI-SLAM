from regression import Regression
from pixel_classification.data_loader import DataLoader

WEIGHTS_FILE = 'weights.pkl'

if __name__ == '__main__':
    data_loader = DataLoader(n_splits=0, resample=True, cross_validation=False)
    data, labels = data_loader.get_splits()

    learner = Regression(data, labels, learning_rate=10, epochs=5000, cross_validation=False)
    learner.train()
    learner.create_plots()
    learner.dump_weights(WEIGHTS_FILE)
