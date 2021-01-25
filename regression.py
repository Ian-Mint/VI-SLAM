from abc import ABC, abstractmethod
from typing import Tuple, List

import matplotlib.pyplot as plt
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
    sum_x_exp = np.sum(x_exp, axis=0)
    y = x_exp / sum_x_exp
    assert np.all(np.isclose(np.sum(y, axis=0), 1))
    return y


class Regression(ABC):
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

        self.data_dim = 3
        self.lr = learning_rate
        assert len(data_splits) == len(label_splits)
        assert len(data_splits) >= 2
        self.n_splits = len(data_splits) if cross_validation else 1
        self.epochs = epochs

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
        n_classes = self.n_classes
        train_split_length = len(self.data_splits[0]) * self.n_splits
        initializer_scale = 1 / train_split_length

        mean = np.zeros(n_classes)
        cov = initializer_scale * np.identity(n_classes)
        weights = [
            np.random.multivariate_normal(mean, cov, size=self.data_dim)
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
        if self.n_splits >= 2:
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
                stop_training &= self.optimize(split_number, e)
            # stop training if we get a stop_training signal from all splits
            if stop_training:
                return

    @abstractmethod
    def optimize(self, split_number: int, epoch: int) -> None:
        ...

    @abstractmethod
    def _grad(self, weights: np.ndarray, data, label) -> np.ndarray:
        ...

    @abstractmethod
    def loss_and_accuracy(self, new_weights, data, label) -> Tuple[float, float]:
        ...


class SoftMaxRegression(Regression):
    """
    Softmax Regression classifier
    """

    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5, epochs=300):
        super().__init__(data_splits, label_splits, learning_rate, epochs)
        self.weights = self._init_weights()
        self.best_weights: List[np.ndarray] = [
            np.zeros((self.n_classes, self.data_dim)) for _ in range(self.n_splits)
        ]
        self.min_loss: List[float] = [-np.inf] * self.n_splits

    def _encode_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """
        The label vector to encode

        Returns:
            (n x n_classes) numpy array
        """
        one_hot = np.zeros((len(labels), self.n_classes), dtype=int)
        one_hot[np.arange(labels.size), labels] = 1
        assert (len(labels), self.n_classes) == one_hot.shape
        assert np.all(one_hot.sum(axis=1) == 1)
        return one_hot

    def optimize(self, split_number: int, epoch: int):
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
        val_data = self.get_data_split(split_number, VALIDATION)
        val_label = self.get_label_split(split_number, VALIDATION)

        # Get the weights
        weights = self.weights[split_number]

        # convert the labels into one hot encodings
        one_hot_enc_train_label = self._encode_one_hot(train_label)
        one_hot_enc_val_label = self._encode_one_hot(val_label)

        # weight update
        grad = self._grad(weights, train_data, one_hot_enc_train_label)
        weights[:, :] = weights + self.lr * grad

        # calculate the training/validation loss and accuracy
        train_loss, train_accuracy = self.loss_and_accuracy(weights, train_data, one_hot_enc_train_label)
        val_loss, val_accuracy = self.loss_and_accuracy(weights, val_data, one_hot_enc_val_label)

        # record training, val loss and accuracy
        self.overall_train_loss[split_number, epoch] = train_loss
        self.overall_val_loss[split_number, epoch] = val_loss
        self.overall_train_accuracy[split_number, epoch] = train_accuracy
        self.overall_val_accuracy[split_number, epoch] = val_accuracy

        if val_loss < self.min_loss[split_number]:
            self.min_loss[split_number] = val_loss
            self.best_weights[split_number][:, :] = weights
            self.min_epoch = epoch

        if epoch - self.min_epoch > 10:
            return STOP_TRAINING
        else:
            return not STOP_TRAINING

    def create_plots_softmax(self):
        """
        Generate training and validation plots for loss and accuracy

        Returns:
            None
        """
        train_loss_matrix = self.overall_train_loss
        val_loss_matrix = self.overall_val_loss
        train_accuracy_matrix = self.overall_train_accuracy
        val_accuracy_matrix = self.overall_val_accuracy

        avg_train_loss = np.average(train_loss_matrix, axis=0)
        avg_val_loss = np.average(val_loss_matrix, axis=0)
        avg_train_accuracy = np.average(train_accuracy_matrix, axis=0)
        avg_val_accuracy = np.average(val_accuracy_matrix, axis=0)

        std_train_loss = np.std(train_loss_matrix, axis=0)
        std_val_loss = np.std(val_loss_matrix, axis=0)
        std_train_accuracy = np.std(train_accuracy_matrix, axis=0)
        std_val_accuracy = np.std(val_accuracy_matrix, axis=0)

        x_axis = [(i + 1) for i in range(self.epochs)]

        error_bar_loss_train = []
        error_bar_loss_val = []
        error_bar_accuracy_train = []
        error_bar_accuracy_val = []
        for epoch in range(self.epochs):
            if (epoch + 1) % 50 == 0:
                error_bar_loss_train.append(std_train_loss[epoch])
                error_bar_loss_val.append(std_val_loss[epoch])
                error_bar_accuracy_train.append(std_train_accuracy[epoch])
                error_bar_accuracy_val.append(std_val_accuracy[epoch])
            else:
                error_bar_loss_train.append(np.nan)
                error_bar_loss_val.append(np.nan)
                error_bar_accuracy_train.append(np.nan)
                error_bar_accuracy_val.append(np.nan)

        plt.errorbar(x_axis, avg_train_loss,
                     yerr=error_bar_loss_train,
                     fmt='b-',
                     label="training")

        plt.errorbar(x_axis, avg_val_loss,
                     yerr=error_bar_loss_val,
                     fmt='r-',
                     label="validation")

        plt.legend(loc="upper right")
        plt.xlabel('epochs')
        plt.ylabel('loss error')
        plt.title('Softmax Regression Loss Error')
        plt.show()

        plt.errorbar(x_axis, avg_train_accuracy,
                     yerr=error_bar_accuracy_train,
                     fmt='b-',
                     label="training")

        plt.errorbar(x_axis, avg_val_accuracy,
                     yerr=error_bar_accuracy_val,
                     fmt='r-',
                     label="validation")

        plt.legend(loc="upper right")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Softmax Regression Accuracy')
        plt.show()

    def _grad(self, weights: np.ndarray, data, label) -> np.ndarray:
        """
        Calculate the negative gradient descent using stable softmax

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
        update /= (self.n_classes * len(data))
        return update

    def loss_and_accuracy(self, weights: np.ndarray, data, label) -> (float, float):
        """
        Get the accuracy and loss based on the weights, data, and labels provided

        Args:
            weights: the weights of the model
            data: the data set being used
            label: the labels being used

        Returns:
            (loss, accuracy): The loss and accuracy of the model
        """
        predicted, log_y = self._classify(data, weights)

        predicted_one_hot = self._encode_one_hot(predicted)

        n_correct = np.sum(np.sum(predicted_one_hot == label, axis=1) == self.n_classes)
        accuracy = n_correct / len(log_y)

        loss = -np.sum(label.T * log_y) / (self.n_classes * len(data))
        return loss, accuracy

    def classify(self, data):
        self._classify(data, self.best_weights)

    @staticmethod
    def _classify(data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
