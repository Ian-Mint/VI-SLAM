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
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5, epochs=300):
        """
        Base class for regression

        Args:
            data_splits: List of data matrices.
                         Matrices must be of shape==(n,d) where n is the number of samples and d is the dimension.
            label_splits: List of label vectors corresponding to data matrices
            learning_rate: The optimization learning rate
            epochs: The maximum number of epochs over which to train
        """
        self.n_classes = self._validate_labels(label_splits)

        self.data_dim = 3
        self.lr = learning_rate
        assert len(data_splits) == len(label_splits)
        self.n_splits = len(data_splits)
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
        train_split_length = len(self.data_splits[0]) * (self.n_splits - 1)
        initializer_scale = 1 / train_split_length

        if n_classes == 1:
            weights = [
                np.random.normal(loc=0, scale=initializer_scale, size=self.data_dim)
                for _ in range(self.n_splits)
            ]
        else:
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
        if split_type == TRAIN:
            split_list = rotate(split_list, i)
            split = np.concatenate(split_list[0:-1], axis=0)
        elif split_type == VALIDATION:
            split = split_list[self.n_splits - i - 1]
        else:
            raise ValueError(f"{split_type} is an invalid split type")

        return split

    def cv_train(self):
        """
        Train using cross-validation
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


class LogisticRegression(Regression):
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5, epochs=300):
        for label in label_splits:
            assert len(np.unique(label)) == 2, "LogisticRegression only handles two classes at a time"
            assert sorted(np.unique(label)) == [0, 1], "Class labels must be 0 or 1"
        super().__init__(data_splits, label_splits, learning_rate, epochs)
        self.n_classes = 2
        self.weight_splits = self._init_weights()
        self.best_weight_splits: List[np.ndarray] = [np.zeros(self.data_dim)] * self.n_splits

    def loss_and_accuracy(self, weights, data, label) -> Tuple[float, float]:
        loss = self._get_loss(data, label, weights)
        accuracy = self._get_accuracy(data, label, weights)
        return loss, accuracy

    def _get_accuracy(self, data, label, weights) -> float:
        return (np.sum(self.predict(data, weights) == label) / len(label)).item()

    def _grad(self, weights: np.ndarray, data, label) -> np.ndarray:
        return data.T @ (sigmoid(data @ weights) - label)

    @staticmethod
    def _get_loss(data: np.ndarray, label: np.ndarray, weights: np.ndarray) -> float:
        n = len(label)
        s = sigmoid(data @ weights)
        return (-label.T @ np.log(s) - (1 - label.T) @ np.log(1 - s)).item() / n

    @staticmethod
    def predict(samples: np.ndarray, weights) -> np.ndarray:
        """
        Predict the class of an array of samples

        Args:
            weights: the logistic regression weights
            samples: numpy array with `data.shape==(n, d)`. `n` is the number of samples and `d` is the data dimension.

        Returns:
            numpy array of predicted classes
        """
        result = (np.log(sigmoid(samples @ weights)) - np.log(sigmoid(-samples @ weights))) > 0
        return result.astype(int)

    def optimize(self, split_number: int, epoch: int) -> None:
        """
        Run parameter optimization

        Args:
            epoch:
            split_number: The index of the split to optimize
        """
        data = self.get_data_split(split_number, TRAIN)
        label = self.get_label_split(split_number, TRAIN)
        weights = self.weight_splits[split_number]

        self._optimize(data, label, split_number, weights)

        for split_type in (TRAIN, VALIDATION):
            self._update_stats(epoch, split_number, split_type)

        val_loss = self.overall_val_loss[split_number, epoch]
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.best_weight_splits[split_number][:] = self.weight_splits[split_number]

    def final_results(self, split_number: int, split_type: str) -> Tuple[float, float]:
        """
        Compute the loss and accuracy given the holdout dataset

        Args:
            split_type: one of 'train', 'validation', 'holdout'
            split_number: The split number to report

        Returns:
            loss, accuracy
        """
        weights = self.best_weight_splits[split_number]
        label = self.get_label_split(split_number, split_type)
        data = self.get_data_split(split_number, split_type)

        loss = self._get_loss(data, label, weights)
        accuracy = self._get_accuracy(data, label, weights)

        return loss, accuracy

    def _update_stats(self, epoch: int, split_number: int, split_type: str):
        weights = self.weight_splits[split_number]
        label = self.get_label_split(split_number, split_type)
        data = self.get_data_split(split_number, split_type)
        if split_type == TRAIN:
            overall_loss = self.overall_train_loss
            overall_accuracy = self.overall_train_accuracy
        elif split_type == VALIDATION:
            overall_loss = self.overall_val_loss
            overall_accuracy = self.overall_val_accuracy
        else:
            raise ValueError(f"{split_type} is not a valid split type")

        loss, accuracy = self.loss_and_accuracy(weights, data, label)
        overall_loss[split_number, epoch] = loss
        overall_accuracy[split_number, epoch] = accuracy

    def _optimize(self, data, label, split_number, weights):
        """
        Perform the weight update

        Args:
            data: The data matrix for a split
            label: The labels for a split
            split_number: The split number
            weights: The weights for a split

        Returns:
            None
        """
        self.weight_splits[split_number] = weights - (self.lr / len(label)) * self._grad(weights, data, label)

    def create_plots(self, plot_params: dict = None, cross_validation=True):
        """
        Generate training and validation plots for loss and accuracy

        Args:
            plot_params: See code for defaults
            cross_validation: Set to True if this should run cross validation
        Returns:
            None
        """
        if plot_params is None:
            plot_params = {
                'title': 'Softmax Regression Loss Error',
            }

        (avg_train_accuracy, avg_train_loss, avg_val_accuracy, avg_val_loss, std_train_accuracy, std_train_loss,
         std_val_accuracy, std_val_loss) = self._get_plot_data(cross_validation)

        (error_bar_accuracy_train, error_bar_accuracy_val, error_bar_loss_train, error_bar_loss_val) = \
            self._generate_error_bars(std_train_accuracy, std_train_loss, std_val_accuracy, std_val_loss)

        x_axis = list(range(1, self.epochs + 1))

        plt.errorbar(x_axis, avg_train_loss,
                     yerr=error_bar_loss_train,
                     fmt='b-',
                     label="training",
                     solid_capstyle='projecting',
                     capsize=3)
        plt.errorbar(x_axis, avg_val_loss,
                     yerr=error_bar_loss_val,
                     fmt='r-',
                     label="validation",
                     solid_capstyle='projecting',
                     capsize=3)

        plt.legend(loc='upper right')
        plt.xlabel("epoch")
        plt.ylabel("loss error")
        plt.title(plot_params['title'])
        plt.show()

        plt.errorbar(x_axis, avg_train_accuracy,
                     yerr=error_bar_accuracy_train,
                     fmt='b-',
                     label="training",
                     solid_capstyle='projecting',
                     capsize=3)

        plt.errorbar(x_axis, avg_val_accuracy,
                     yerr=error_bar_accuracy_val,
                     fmt='r-',
                     label="validation",
                     solid_capstyle='projecting',
                     capsize=3)

        plt.legend(loc="upper right")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(plot_params['title'])
        plt.show()

    def _generate_error_bars(self, std_train_accuracy, std_train_loss, std_val_accuracy, std_val_loss):
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
        return error_bar_accuracy_train, error_bar_accuracy_val, error_bar_loss_train, error_bar_loss_val

    def _get_plot_data(self, cross_validation):
        train_loss_matrix = self.overall_train_loss
        val_loss_matrix = self.overall_val_loss
        train_accuracy_matrix = self.overall_train_accuracy
        val_accuracy_matrix = self.overall_val_accuracy
        if cross_validation:
            avg_train_loss = np.average(train_loss_matrix, axis=0)
            avg_val_loss = np.average(val_loss_matrix, axis=0)
            avg_train_accuracy = np.average(train_accuracy_matrix, axis=0)
            avg_val_accuracy = np.average(val_accuracy_matrix, axis=0)

            std_train_loss = np.std(train_loss_matrix, axis=0)
            std_val_loss = np.std(val_loss_matrix, axis=0)
            std_train_accuracy = np.std(train_accuracy_matrix, axis=0)
            std_val_accuracy = np.std(val_accuracy_matrix, axis=0)
        else:
            avg_train_loss = train_loss_matrix[0, :]
            avg_val_loss = val_loss_matrix[0, :]
            avg_train_accuracy = train_accuracy_matrix[0, :]
            avg_val_accuracy = val_accuracy_matrix[0, :]

            std_train_loss = np.zeros_like(avg_train_loss)
            std_val_loss = np.zeros_like(avg_val_loss)
            std_train_accuracy = np.zeros_like(avg_train_accuracy)
            std_val_accuracy = np.zeros_like(avg_val_accuracy)
        return avg_train_accuracy, avg_train_loss, avg_val_accuracy, avg_val_loss, std_train_accuracy, std_train_loss, std_val_accuracy, std_val_loss


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
        y = softmax(weights @ data.T)
        predicted = y.argmax(axis=0)
        log_y = np.log(y)

        predicted_one_hot = self._encode_one_hot(predicted)

        n_correct = np.sum(np.sum(predicted_one_hot == label, axis=1) == self.n_classes)
        accuracy = n_correct / len(y)

        loss = -np.sum(label.T * log_y) / (self.n_classes * len(data))
        return loss, accuracy
