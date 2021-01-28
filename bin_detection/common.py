from bin_detection.data_loader import DataLoader
from regression import Regression

TRAIN_DATA_DIR = 'data/training'
TRAIN_MASK_DIR = 'data/masks/training'
VAL_DATA_DIR = 'data/validation'
VAL_MASK_DIR = 'data/masks/validation'

train_data = DataLoader(TRAIN_DATA_DIR, TRAIN_MASK_DIR)
classifier = Regression(
    [train_data.data],
    [train_data.labels],
    learning_rate=1,
    epochs=300,
    cross_validation=False,
)
