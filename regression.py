import pickle
from typing import Tuple, List

import numpy as np

VALIDATION = 'validation'
TRAIN = 'train'
STOP_TRAINING = True


def rotate(x: list, n: int):
    """
    rotates list x by distance n

    Args:
        x: a list
        n: distance to rotate

    Returns:
        A rotated copy of the list
    """
    return x[-n:] + x[:-n]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The sigmoid function

    Args:
        x: numpy vector

    Returns:
        numpy vector
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x_exp = np.exp(x)
    y = x_exp / np.sum(x_exp, axis=0)
    assert np.all(np.isclose(np.sum(y, axis=0), 1)), print(x)
    return y


class Regression:
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5, epochs=300,
                 cross_validation=True):
        """
        Base class for regression. If cross_validation==False, label_splits and data_splits should be of length 2 with
        the first element the training data and the second element the validation data. Otherwise, they should be of
        length equal to the number of cross-validation splits.

        Args:
            cross_validation:
            data_splits: List of data matrices.
                         Matrices must be of shape==(n,d) where n is the number of samples and d is the dimension.
            label_splits: List of label vectors corresponding to data matrices
            learning_rate: The optimization learning rate
            epochs: The maximum number of epochs over which to train
            cross_validation: if True, use cross-validation
        """
        self.cv = cross_validation
        self.n_classes = self._validate_labels(label_splits)
        assert self.n_classes >= 2

        self.data_dim = data_splits[0].shape[1]
        self.lr = learning_rate
        assert len(data_splits) == len(label_splits)
        self.n_splits = len(data_splits) if cross_validation else 1
        self.epochs = epochs
        self.validation = len(data_splits) >= 2

        self.label_splits = label_splits
        self.data_splits = data_splits

        # overall training and validation loss (split, epoch) is dimension
        self.overall_train_loss: np.ndarray = np.zeros((self.n_splits, self.epochs))
        self.overall_val_loss: np.ndarray = np.zeros((self.n_splits, self.epochs))
        # overall training and validation accuracy (split, epoch) is dimension
        self.overall_train_accuracy: np.ndarray = np.zeros((self.n_splits, self.epochs))
        self.overall_val_accuracy: np.ndarray = np.zeros((self.n_splits, self.epochs))

        self.min_val_loss = np.inf
        self.min_epoch = 0

        self.weights = self._init_weights()
        self.best_weights: List[np.ndarray] = [
            np.zeros((self.n_classes, self.data_dim)) for _ in range(self.n_splits)
        ]
        self.min_loss: List[float] = [np.inf] * self.n_splits

    def dump_weights(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filename: str):
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)
        self.best_weights = self.weights
        return self.weights

    @staticmethod
    def _validate_labels(label_splits) -> int:
        """
        Enforces that labels must start from 0 and be sequential
        """
        labels = np.concatenate(label_splits, 0)
        unique = np.unique(labels)
        sorted(unique)
        n_unique = len(unique)
        assert tuple(range(n_unique)) == tuple(unique)
        assert np.min(unique) == 0
        return n_unique

    def _init_weights(self) -> List[np.ndarray]:
        """
        Returns a list of gaussian initialized weight vectors.
        """
        data_dim = self.data_dim
        initializer_scale = 1e-2 / data_dim

        mean = np.zeros(data_dim)
        cov = initializer_scale * np.identity(data_dim)
        weights = [
            np.random.multivariate_normal(mean, cov, size=self.n_classes)
            for _ in range(self.n_splits)
        ]
        return weights

    def one_augment(self, x: np.ndarray):
        """
        Augments x with ones along the data dimension (n x d)
        """
        assert np.ndim(x) == 2
        assert x.shape[1] == self.data_dim

        one_vector = np.ones((len(x), 1))
        return np.concatenate((x, one_vector), axis=1)

    def get_data_split(self, i: int, split_type: str) -> np.ndarray:
        """
        Returns the i_th data split
        Args:
            split_type: one of 'test', 'validation', 'holdout'
            i: Split number

        Returns:
            split data
        """
        return self._get_split(i, split_type, self.data_splits)

    def get_label_split(self, i: int, split_type: str) -> np.ndarray:
        """
        Returns the i_th labels split
        Args:
            split_type: one of 'train', 'validation', 'holdout'
            i: Split number

        Returns:
            split labels
        """
        return self._get_split(i, split_type, self.label_splits)

    def _get_split(self, i: int, split_type: str, split_list: list):
        """
        Method for returning data split or label split
        """
        if self.cv:
            if split_type == TRAIN:
                split_list = rotate(split_list, i)
                split = np.concatenate(split_list[0:-1], axis=0)
            elif split_type == VALIDATION:
                split = split_list[self.n_splits - i - 1]
            else:
                raise ValueError(f"{split_type} is an invalid split type")
        else:
            if split_type == TRAIN:
                split = split_list[0]
            elif split_type == VALIDATION:
                split = split_list[1]
            else:
                raise ValueError(f"{split_type} is an invalid split type")
        return split

    def train(self):
        """
        Train using cross-validation. Won't use cross-validation if there are only 2 splits.
        """
        for e in range(self.epochs):
            stop_training = True
            for split_number in range(self.n_splits):
                stop_training &= self._optimize(split_number, e)
            # early stopping
            if stop_training:
                return

    @staticmethod
    def _encode_one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
        """
        The label vector to encode

        Args:
            labels: The labels to convert in a 1d array
            n_classes: The number of classes

        Returns:
            (n x n_classes) numpy array
        """
        one_hot = np.zeros((len(labels), n_classes), dtype=int)
        one_hot[np.arange(labels.size), labels] = 1
        assert (len(labels), n_classes) == one_hot.shape
        assert np.all(one_hot.sum(axis=1) == 1)
        return one_hot

    def _optimize(self, split_number: int, epoch: int):
        """
        Optimizes the specific weights for a split.

        Args:
            split_number: The index of the split to optimize
            epoch: The epoch the training is at

        Returns:
            None

        """
        train_data = self.get_data_split(split_number, TRAIN)
        train_label = self.get_label_split(split_number, TRAIN)
        if self.validation:
            val_data = self.get_data_split(split_number, VALIDATION)
            val_label = self.get_label_split(split_number, VALIDATION)

        # Get the weights
        weights = self.weights[split_number]

        # convert the labels into one hot encodings
        one_hot_enc_train_label = self._encode_one_hot(train_label, self.n_classes)

        # weight update
        grad = self._grad(weights, train_data, one_hot_enc_train_label)
        weights[:, :] = weights + self.lr * grad

        # update stats
        train_loss, train_accuracy = self._loss_and_accuracy(weights, train_data, train_label)
        self.overall_train_loss[split_number, epoch] = train_loss
        self.overall_train_accuracy[split_number, epoch] = train_accuracy

        if self.validation:
            val_loss, val_accuracy = self._loss_and_accuracy(weights, val_data, val_label)
            self.overall_val_loss[split_number, epoch] = val_loss
            self.overall_val_accuracy[split_number, epoch] = val_accuracy

        if self.validation:
            if val_loss < self.min_loss[split_number]:
                self.min_loss[split_number] = val_loss
                self.best_weights[split_number][:, :] = weights
                self.min_epoch = epoch

            if epoch - self.min_epoch > 10:
                return STOP_TRAINING
            else:
                return not STOP_TRAINING
        else:
            return not STOP_TRAINING

    def plot(self) -> None:
        """
        Generate training and validation plots for loss and accuracy
        """
        import matplotlib.pyplot as plt  # so that we don't have dependency issues in the autograder

        avg_train_loss = np.average(self.overall_train_loss, axis=0)
        avg_val_loss = np.average(self.overall_val_loss, axis=0)
        avg_train_accuracy = np.average(self.overall_train_accuracy, axis=0)
        avg_val_accuracy = np.average(self.overall_val_accuracy, axis=0)

        blue_color = 'tab:blue'
        fig, ax1 = plt.subplots(figsize=(4, 3))
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss', color=blue_color)
        ax1.tick_params(axis='y', labelcolor=blue_color)
        ax1.plot(avg_train_loss, 'k-', label='Train')
        # ax1.plot(avg_val_loss, 'k--', label='Validation')
        # ax1.legend(loc='center right')
        ax1.plot(avg_train_loss, '-', color=blue_color, label='Train')
        # ax1.plot(avg_val_loss, '--', color=blue_color, label='Validation')

        ax2 = ax1.twinx()

        orange_color = 'tab:orange'
        ax2.set_ylabel('accuracy', color=orange_color)
        ax2.tick_params(axis='y', labelcolor=orange_color)
        ax2.plot(avg_train_accuracy, '-', color=orange_color, label='Train')
        # ax2.plot(avg_val_accuracy, '--', color=orange_color, label='Validation')

        plt.show()

    @staticmethod
    def _grad(weights: np.ndarray, data, label) -> np.ndarray:
        """
        Calculate the negative gradient of the softmax with cross-entropy loss

        weights = size(number of classes, number of components)
        data = size(number of examples, number of components)
        x = size(number of classes, number of examples)
        a_matrix_max = size(number of examples,)
        label = size(number of examples, number of classes)
        difference = size(number of classes, number of examples)
        update = size(number of classes, number of components)

        Args:

            weights: the weights of the model
            data: dataset
            label: the labels

        Returns:
            update: the gradient that needs to be used to update the weights
        """
        y = softmax(weights @ data.T)
        update = (label.T - y) @ data
        update /= len(data)
        return update

    def _loss_and_accuracy(self, weights: np.ndarray, data, label) -> (float, float):
        """
        Get the accuracy and loss based on the weights, data, and labels provided

        Args:
            weights: the weights of the model
            data: the data set being used
            label: the labels being used

        Returns:
            (loss, accuracy): The loss and accuracy of the model
        """
        predicted, log_y = self.classify(data, weights)
        accuracy = np.sum(predicted == label) / len(label)
        one_hot_label = self._encode_one_hot(label, self.n_classes)
        loss = -np.sum(one_hot_label.T * log_y) / len(data)
        return loss, accuracy

    @staticmethod
    def classify(data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classifies the data given the weights

        Args:
            data: data (n x d)
            weights: weights (d x n_classes)

        Returns:
            predicted labels, log of the softmax
        """
        y = softmax(weights @ data.T)
        predicted = y.argmax(axis=0)
        log_y = np.log(y)
        return predicted, log_y
