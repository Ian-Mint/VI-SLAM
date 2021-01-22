from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from collections import defaultdict

HOLDOUT = 'holdout'
VALIDATION = 'validation'
TRAIN = 'train'

PCData = namedtuple('PCData', ('mean', 'singular_values', 'eigenvectors'))


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


class Regression(ABC):
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5,
                 n_principal_components=10, epochs=300):
        """
        Base class for regression

        Args:
            data_splits: List of data matrices.
                         Matrices must be of shape==(n,d) where n is the number of samples and d is the dimension.
            label_splits: List of label vectors corresponding to data matrices
            learning_rate: The optimization learning rate
            n_principal_components: The number of principal components to use when reducing dimensionality
        """
        self.n_pcs = n_principal_components
        self.lr = learning_rate
        assert len(data_splits) == len(label_splits)
        self.n_splits = len(data_splits)
        self.epochs = epochs

        self.label_splits = label_splits
        self.data_splits = data_splits

        # Training splits and the pca data are all calculated and stored at the start of training. Other splits can
        # be generated dynamically from self.label_splits and self.data_splits using the get_data_split and
        # get_label_split methods.
        self.train_splits: List[np.ndarray] = []
        self.holdout_splits: List[np.ndarray] = []
        self.val_splits: List[np.ndarray] = []
        self.pca_data_splits: List[PCData] = []
        self._init_pca()

        assert len(self.holdout_splits) == len(self.val_splits) == len(self.train_splits)
        # Storing the weights for each of the splits specified, initializing them to random values
        self.weight_splits: List[np.ndarray] = [
            stats.norm(loc=0, scale=1 / len(self.train_splits[0])).rvs(self.n_pcs + 1) for i in range(self.n_splits)
        ]
        self.best_weight_splits: List[np.ndarray] = [np.zeros(self.n_pcs + 1)] * self.n_splits

        # overall training and validation loss (split, epoch) is dimension
        self.overall_train_loss: np.ndarray = np.zeros((self.n_splits, self.epochs))
        self.overall_val_loss: np.ndarray = np.zeros((self.n_splits, self.epochs))
        # overall training and validation accuracy (split, epoch) is dimension
        self.overall_train_accuracy: np.ndarray = np.zeros((self.n_splits, self.epochs))
        self.overall_val_accuracy: np.ndarray = np.zeros((self.n_splits, self.epochs))
        # test loss and accuracy on the best model for each split
        self.test_loss: np.ndarray = np.zeros(self.n_splits)
        self.test_accuracy: np.ndarray = np.zeros(self.n_splits)
        # overall test loss and accuracy on the best model for each split
        self.overall_test_loss: np.ndarray = np.zeros((self.n_splits, self.epochs))
        self.overall_test_accuracy: np.ndarray = np.zeros((self.n_splits, self.epochs))

        self.min_val_loss = np.inf

    def _init_pca(self):
        """
        Populate the PCA outputs for each split.

        Returns:
            None
        """
        for i in range(self.n_splits):
            train_split = self.get_data_split(i, TRAIN)
            val_split = self.get_data_split(i, VALIDATION)
            holdout_split = self.get_data_split(i, HOLDOUT)

            # If you do not want to use PCA
            projected, mean, singular_values, eigenvectors = PCA(train_split, self.n_pcs)

            assert np.all(np.isclose(np.mean(projected, axis=0), 0))
            assert np.all(np.isclose(np.std(projected, axis=0) * np.sqrt(len(projected)), 1))

            # adding in the bias term for the datasets
            size_of_projected = len(projected)
            one_vector = np.ones((size_of_projected,1))
            new_projected = np.concatenate((projected, one_vector), axis=1)

            self.train_splits.append(new_projected)
            self.pca_data_splits.append(PCData(mean, singular_values, eigenvectors))

            val_projected = ((val_split - mean) @ eigenvectors) / singular_values
            size_of_projected = len(val_projected)
            one_vector = np.ones((size_of_projected, 1))
            new_val_projected = np.concatenate((val_projected, one_vector), axis=1)
            self.val_splits.append(new_val_projected)

            holdout_projected = ((holdout_split - mean) @ eigenvectors) / singular_values
            size_of_projected = len(holdout_projected)
            one_vector = np.ones((size_of_projected, 1))
            new_holdout_projected = np.concatenate((holdout_projected, one_vector), axis=1)
            self.holdout_splits.append(new_holdout_projected)

            assert new_projected.shape[1] == new_val_projected.shape[1] == new_holdout_projected.shape[1]

    def plot_pcs(self, split_number: int, n_pcs: int = 4):
        eigenvectors = self.pca_data_splits[split_number].eigenvectors
        for i in range(n_pcs):
            plt.title(f"Principal component {i}")
            img = eigenvectors[:, i].reshape(200, 300)
            plt.imshow(img)
            plt.colorbar()
            plt.show()

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
            split = np.concatenate(split_list[0:-2], axis=0)
        elif split_type == VALIDATION:
            split = split_list[self.n_splits - i - 2]
        elif split_type == HOLDOUT:
            split = split_list[self.n_splits - i - 1]
        else:
            raise ValueError(f"{split_type} is an invalid split type")

        return split

    def cv_train(self):
        """
        Train using cross-validation

        Returns:
            None
        """
        for e in range(self.epochs):
            for split_number in range(self.n_splits):
                self.optimize(split_number, e)

    def train(self, split_number: int):
        """
        Train using a single split

        Args:
            split_number: the index of the split

        Returns:
            None
        """
        for e in range(self.epochs):
            self.optimize(split_number, e)

    @abstractmethod
    def optimize(self, split_number: int, epoch: int) -> None: ...

    @abstractmethod
    def _grad(self, weights: np.ndarray, data, label) -> np.ndarray: ...

    @abstractmethod
    def loss_and_accuracy(self, new_weights, data, label) -> Tuple[float, float]: ...


class LogisticRegression(Regression):
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray], learning_rate=1e-5,
                 n_principal_components=10, epochs=300):
        for label in label_splits:
            assert len(np.unique(label)) == 2, "LogisticRegression only handles two classes at a time"
            assert sorted(np.unique(label)) == [0, 1], "Class labels must be 0 or 1"
        super().__init__(data_splits, label_splits, learning_rate, n_principal_components, epochs)

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
        data = self.train_splits[split_number]
        label = self.get_label_split(split_number, TRAIN)
        weights = self.weight_splits[split_number]

        self._optimize(data, label, split_number, weights)

        for split_type in (TRAIN, VALIDATION, HOLDOUT):
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
        if split_type == TRAIN:
            data = self.train_splits[split_number]
        elif split_type == VALIDATION:
            data = self.val_splits[split_number]
        elif split_type == HOLDOUT:
            data = self.holdout_splits[split_number]
        else:
            raise ValueError(f"{split_type} is an invalid split type")

        loss = self._get_loss(data, label, weights)
        accuracy = self._get_accuracy(data, label, weights)

        return loss, accuracy

    def _update_stats(self, epoch: int, split_number: int, split_type: str):
        weights = self.weight_splits[split_number]
        label = self.get_label_split(split_number, split_type)
        if split_type == TRAIN:
            data = self.train_splits[split_number]
            overall_loss = self.overall_train_loss
            overall_accuracy = self.overall_train_accuracy
        elif split_type == VALIDATION:
            data = self.val_splits[split_number]
            overall_loss = self.overall_val_loss
            overall_accuracy = self.overall_val_accuracy
        elif split_type == HOLDOUT:
            data = self.holdout_splits[split_number]
            overall_loss = self.overall_test_loss
            overall_accuracy = self.overall_test_accuracy
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

    Args:
        same as Regression, however has two more parameter
        labels: the label dictionary to use for one-hot-encoding
        gradient_descent: determine whether to do batch or stochastic gradient descent
    """
    def __init__(self, data_splits: List[np.ndarray], label_splits: List[np.ndarray],
                 labels: defaultdict, gradient_descent: str = "batch", learning_rate=1e-5,
                 n_principal_components=10, epochs=300):
        # initializing the base class
        Regression.__init__(self, data_splits, label_splits, learning_rate, n_principal_components, epochs)
        self.number_of_classes = len(labels)
        # generating weight matrix for each split
        self.softmax_weight_splits: List[np.ndarray] = [np.random.rand(self.number_of_classes, self.n_pcs + 1) for i in range(self.n_splits)]
        self.softmax_best_weight_splits: List[np.ndarray] = [np.random.rand(self.number_of_classes, self.n_pcs + 1) for i in range(self.n_splits)]
        self.softmax_best_loss_splits: List[float] = [-1.0] * self.n_splits
        self.label_dict = labels
        self.label_to_index_dict = defaultdict()
        self.index_to_label_dict = defaultdict()
        self.one_hot_enc_dict = defaultdict(np.ndarray)
        self.init_one_hot_encoding()
        self.gradient_descent = gradient_descent

    def init_one_hot_encoding(self):
        """
        Creates the one hot encodings dictionary based on the label dictionary.
        Sets the class variable of the one hot encoding for future use

        Returns:
            None
        """
        index = 0
        for key in self.label_dict:
            self.label_to_index_dict[key] = index
            self.index_to_label_dict[index] = key
            array = np.zeros((1, self.number_of_classes))
            array[0][index] = 1
            self.one_hot_enc_dict[key] = np.copy(array)
            index += 1

    def generate_one_hot_encoding(self, label: np.ndarray) -> np.ndarray:
        """
        Creates the one hot encodings for the labels provided

        Args:
            label: The labels for the dataset at hand (training, validation, holdout)

        Returns:
            one_hot_enc: The labels transformed into the one hot encodings

        """
        one_hot_enc = None
        for index in range(len(label)):
            specific_encoding = self.one_hot_enc_dict[label[index]]  # get one hot encoding for label value
            if one_hot_enc is None:
                one_hot_enc = specific_encoding
            else:
                one_hot_enc = np.concatenate((one_hot_enc, specific_encoding), axis=0)

        return one_hot_enc

    def optimize(self, split_number: int, epoch: int):
        """
        Optimizes the specific weights for a split.

        Args:
            split_number: The index of the split to optimize
            epoch: The epoch the training is at

        Returns:
            None

        """
        #Get the training and validation data
        train_data = self.train_splits[split_number]
        train_label = self.get_label_split(split_number, "train")
        val_data = self.val_splits[split_number]
        val_label = self.get_label_split(split_number, "validation")

        # Get the weights
        weights = self.softmax_weight_splits[split_number]

        # convert the labels into one hot encodings
        one_hot_enc_train_label = self.generate_one_hot_encoding(train_label)
        one_hot_enc_val_label = self.generate_one_hot_encoding(val_label)
        # calculate the new weights based on training data
        if self.gradient_descent == "batch":
            new_weights = weights + self.lr * self._grad(weights, train_data, one_hot_enc_train_label)
        elif self.gradient_descent == "weighted":
            new_weights = weights + self.lr * self._grad_weighted(weights, train_data, one_hot_enc_train_label)
        elif self.gradient_descent == "stochastic":
            new_weights = np.copy(self._stochastic_grad(weights, train_data, one_hot_enc_train_label, self.lr))
        else:
            raise Exception("ERROR: Gradient Descent Input Not Recognized")
        # update the weights
        self.softmax_weight_splits[split_number] = new_weights

        # calculate the training/validation loss and accuracy
        train_loss, train_accuracy = self.loss_and_accuracy(new_weights, train_data, one_hot_enc_train_label)
        val_loss, val_accuracy = self.loss_and_accuracy(new_weights, val_data, one_hot_enc_val_label)

        # record training, val loss and accuracy
        self.overall_train_loss[split_number][epoch] = train_loss
        self.overall_val_loss[split_number][epoch] = val_loss
        self.overall_train_accuracy[split_number][epoch] = train_accuracy
        self.overall_val_accuracy[split_number][epoch] = val_accuracy

        # early-stopping implementation
        if self.softmax_best_loss_splits[split_number] == -1:
            self.softmax_best_loss_splits[split_number] = val_loss
            self.softmax_best_weight_splits[split_number] = new_weights
        elif val_loss < self.softmax_best_loss_splits[split_number]:
            self.softmax_best_loss_splits[split_number] = val_loss
            self.softmax_best_weight_splits[split_number] = new_weights

    def test_set_accuracy_and_loss(self) -> Tuple[int, int, pd.core.frame.DataFrame]:
        """
        Get the test set accuracy and loss based on the best models for each split

        Returns:
            (avg_loss, avg_accuracy, confusion_matrix_df): The average loss and accuracy of the best models on the test
            set. Also includes the confusion matrix of the test set
        """
        # use the best model for each split to record accuracy and loss on test set and also create confusion matrix
        confusion_matrix = np.zeros((self.number_of_classes, self.number_of_classes), int)
        number_of_training = 0
        number_right_total = 0
        for split_number in range(self.n_splits):
            softmax_best_model_weights = self.softmax_best_weight_splits[split_number]

            # get the data and the labels
            holdout_data = self.holdout_splits[split_number]
            holdout_label = self.get_label_split(split_number, "holdout")
            number_of_training += len(holdout_label)
            # convert the labels into one hot encodings
            one_hot_enc_holdout_label = self.generate_one_hot_encoding(holdout_label)

            test_loss, test_accuracy = self.loss_and_accuracy(softmax_best_model_weights, holdout_data, one_hot_enc_holdout_label)
            confusion_matrix_split = self.gen_confusion_matrix(softmax_best_model_weights, holdout_data, holdout_label)
            number_right_total += test_accuracy * len(holdout_data)

            self.test_loss[split_number] = test_loss
            self.test_accuracy[split_number] = test_accuracy
            confusion_matrix = confusion_matrix + confusion_matrix_split

        # average test loss and accuracy from above
        avg_loss = np.average(self.test_loss)
        avg_accuracy = np.average(self.test_accuracy)

        assert number_of_training == np.sum(confusion_matrix)
        assert number_right_total == np.trace(confusion_matrix)

        confusion_matrix_df = pd.DataFrame(confusion_matrix)
        column_dict = defaultdict(str)
        row_dict = defaultdict(str)
        for index in range(self.number_of_classes):
            label_value = self.index_to_label_dict[index]
            label_name = self.label_dict[label_value]
            column_dict[index] = label_name
            row_dict[index] = label_name
        confusion_matrix_df.rename(columns=column_dict,
                                   index=row_dict,
                                   inplace=True)

        return avg_loss, avg_accuracy, confusion_matrix_df

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

        x_axis = [(i+1) for i in range(self.epochs)]

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

        Args:

            weights: the weights of the model
            data: dataset
            label: the labels

        Returns:
            update: the gradient that needs to be used to update the weights
        """

        """
        weights = size(number of classes, number of components)
        data = size(number of examples, number of components)
        a_matrix = size(number of classes, number of examples)
        a_matrix_max = size(number of examples,)
        label = size(number of examples, number of classes)
        difference = size(number of classes, number of examples)
        update = size(number of classes, number of components)
        """
        a_matrix = weights @ data.transpose()
        a_matrix_exp = np.exp(a_matrix)

        sum_of_a_matrix_exp = np.sum(a_matrix_exp, axis=0)
        y_matrix = a_matrix_exp / sum_of_a_matrix_exp.flatten()

        difference = label.transpose() - y_matrix
        update = difference @ data

        assert update.shape == weights.shape

        length_of_class = len(a_matrix)
        length_of_examples = len(data)
        update = 1.0/length_of_class * 1.0/length_of_examples * update

        return update

    def _grad_weighted(self, weights: np.ndarray, data, label) -> np.ndarray:
        """
        Calculate the negative gradient descent using stable softmax

        Args:

            weights: the weights of the model
            data: dataset
            label: the labels

        Returns:
            update: the gradient that needs to be used to update the weights
        """

        """
        weights = size(number of classes, number of components)
        data = size(number of examples, number of components)
        a_matrix = size(number of classes, number of examples)
        a_matrix_max = size(number of examples,)
        label = size(number of examples, number of classes)
        difference = size(number of classes, number of examples)
        update = size(number of classes, number of components)
        """
        a_matrix = weights @ data.transpose()
        a_matrix_exp = np.exp(a_matrix)

        sum_of_a_matrix_exp = np.sum(a_matrix_exp, axis=0)
        y_matrix = a_matrix_exp / sum_of_a_matrix_exp.flatten()

        difference = label.transpose() - y_matrix
        update = difference @ data

        assert update.shape == weights.shape

        sum_each = np.sum(label, axis=0)
        sum_total = np.sum(sum_each)
        weight_balance = sum_total / (sum_each)
        weighted_update = (update.transpose() * weight_balance.flatten()).transpose()

        length_of_class = len(a_matrix)
        length_of_examples = len(data)

        assert weighted_update.shape == update.shape

        return weighted_update * 1.0 / length_of_examples * 1.0 / length_of_class

    def _stochastic_grad(self, weights: np.ndarray, data, label, learning_rate):
        """
        Calculate the negative gradient descent for stochastic descent

        Args:

            weights: the weights of the model
            data: dataset
            label: the labels
            learning_rate: the learning rate

        Returns:
            update: the updated weights after going through all training examples
        """

        """
                weights = size(number of classes, number of components)
                data_singular = size(number of components, 1)
                a_matrix = size(number of classes, 1)
                label_singular = size(number of examples, 1)
                difference = size(number of classes, 1)
                update = size(number of classes, number of components)
                """

        # shuffle the order of the learning
        index_list = [i for i in range(len(data))]
        np.random.shuffle(index_list)

        updated_weights = np.copy(weights)
        # update the weights per training example
        for index in index_list:

            data_singular = np.reshape(data[index], (len(data[index]), 1))
            a_matrix = updated_weights @ data_singular
            a_matrix_exp = np.exp(a_matrix)
            sum_of_a_matrix_exp = np.sum(a_matrix_exp)
            y_matrix = a_matrix_exp / sum_of_a_matrix_exp

            label_singular = np.reshape(label[index], (len(label[index]), 1))
            difference = label_singular - y_matrix
            update = difference @ data_singular.transpose()
            assert update.shape == updated_weights.shape

            length_of_class = len(a_matrix)

            updated_weights = updated_weights + learning_rate * 1.0/length_of_class * update

        return updated_weights

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
        a_matrix = weights @ data.transpose()
        a_matrix_exp = np.exp(a_matrix)
        sum_of_a_matrix_exp = np.sum(a_matrix_exp, axis=0)
        y_matrix = a_matrix_exp / sum_of_a_matrix_exp.flatten()
        y_matrix_log = np.log(y_matrix)

        non_enc_pred = y_matrix.argmax(axis=0)

        assert np.all(np.isclose(np.sum(y_matrix, axis=0),1))

        # Need to convert to one hot encoding, so that can compare the predictions and labels
        one_hot_enc_pred = None
        for index in range(len(non_enc_pred)):
            specific_encoding = np.zeros((1,self.number_of_classes))
            specific_encoding[0][non_enc_pred[index]] = 1
            if one_hot_enc_pred is None:
                one_hot_enc_pred = specific_encoding
            else:
                one_hot_enc_pred = np.concatenate((one_hot_enc_pred, specific_encoding), axis=0)

        total_correct = 0.0
        for index in range(len(one_hot_enc_pred)):
            if np.array_equal(one_hot_enc_pred[index],label[index]):
                total_correct += 1
        accuracy = total_correct / len(one_hot_enc_pred)

        length_of_class = len(a_matrix)
        length_of_examples = len(data)

        if self.gradient_descent == "weighted":
            sum_each = np.sum(label, axis=0)
            sum_total = np.sum(sum_each)
            weight_balance = sum_total / (sum_each)
            y_matrix_log = (y_matrix_log.transpose() * weight_balance.flatten()).transpose()
            loss = -1 * 1.0/length_of_examples * np.sum(np.multiply(label.transpose(), y_matrix_log))
        else:
            loss = -1 * 1.0/length_of_class * 1.0/length_of_examples * np.sum(np.multiply(label.transpose(), y_matrix_log))
        return loss, accuracy

    def gen_confusion_matrix(self, weights: np.ndarray, data, label) -> np.ndarray:
        """
        Get the confusion matrix based on the weights, data, and labels provided

        Args:
            weights: the weights of the model
            data: the data set being used
            label: the labels being used

        Returns:
            confusion_matrix: The confusion matrix for the dataset
        """
        a_matrix = weights @ data.transpose()
        a_matrix_exp = np.exp(a_matrix)
        sum_of_a_matrix_exp = np.sum(a_matrix_exp, axis=0)
        y_matrix = a_matrix_exp / sum_of_a_matrix_exp.flatten()

        non_enc_pred = y_matrix.argmax(axis=0)

        # converting the true label value to the index value generated by softmax
        converted_true_value = []
        for true_value in label:
            converted_true_value.append(self.label_to_index_dict[true_value])

        confusion_matrix = np.zeros((self.number_of_classes, self.number_of_classes), int)
        for index in range(len(non_enc_pred)):
            i_true_value = converted_true_value[index]
            j_pred = non_enc_pred[index]
            confusion_matrix[i_true_value][j_pred] += 1

        return confusion_matrix

    def stable_soft_max(self, data):
        max = np.max(data, axis=0)
        output = data - max.flatten()
        assert data.shape == output.shape
        return output