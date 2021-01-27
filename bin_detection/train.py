from bin_detection.data_loader import DataLoader
from regression import Regression

TRAIN_DATA_DIR = 'data/training'
TRAIN_MASK_DIR = 'data/masks/training'

VAL_DATA_DIR = 'data/validation'
VAL_MASK_DIR = 'data/masks/validation'

if __name__ == '__main__':
    train_data = DataLoader(TRAIN_DATA_DIR, TRAIN_MASK_DIR)
    val_data = DataLoader(VAL_DATA_DIR, VAL_MASK_DIR)

    learner = Regression(
        [train_data.data, val_data.data],
        [train_data.labels, val_data.labels],
        learning_rate=1,
        epochs=100,
        cross_validation=False,
    )
    learner.train()
    learner.create_plots()
    learner.dump_best_weights('weights.pkl')
