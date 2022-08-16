import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

# constants
# name of the environments
FLAPPY_BASE_NAME = "FlappyBird-v0"
FLAPPY_IMAGE_NAME = "FlappyBird-rgb-v0"
# input shape base model
BASE_SHAPE = (1, 8, 3)
# shape of the preprocessed image
IMAGE_SHAPE = (84, 84)
# maximum value of a pixel
MAX_PIXEL_VALUE = 255
# length of the timeseries of the base model
SERIES_LENGTH = 8
# number of channels for the stack of the cnn model
NUM_CHANNELS = 4
# name of the default pretrained model
BASE = "trained_base"
CNN = "trained_cnn"
ENTROPY = "trained_entropy"


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return list of mean tensors
    """
    acc = [tf.zeros_like(x) for x in tensors[0]]
    num_tensors = len(tensors[0])
    num_elem = len(tensors)
    for i in range(num_tensors):
        for j in range(num_elem):
            acc[i] = tf.add(acc[i], tensors[j][i])
        acc[i] = tf.divide(acc[i], tf.constant(num_elem, dtype='float32'))
    return acc


def log2(x):
    """
    Compute logarithm in base 2, this functionality is not supported in tensorflow
    :param x: argument of the logarithm
    :return log in base 2 of x
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def save_series(series, path):
    """
    Store a timeseries in a csv file
    :param series: numpy array
    :param path: location to store the series
    """
    pd_series = pd.Series(series)
    pd_series.to_csv(path, header=False, index=False)


def load_series(path):
    """
    Load a series from a specified location
    :param path: location where the series is stored
    :return numpy series
    """
    pd_series = pd.read_csv(path, header=False)
    series = pd_series.to_numpy().flatten()
    return series


def plot_graph(series_list, series_labels, markers, title, x_label, y_label, grid=True, save=False, path=None):
    """
    Compare different timeseries
    :param series_list: list of numpy arrays
    :param series_labels: labels of each series
    :param markers: markers of each series
    :param title: title of the plot
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param grid: flag to enable grid
    :param save: flag to save the plot
    :param path: location where to save the image
    """
    plt.grid(grid)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for (series, label, marker) in zip(series_list, series_labels, markers):
        plt.plot(series, marker, label=label)
    plt.legend(loc='upper left')
    # saving the image
    if save:
        assert path is not None
        plt.savefig(path)
    plt.show()


def compare_boxplot(dictionary, save=False, path=None):
    """
    Compare boxplots of different samples
    :param dictionary: dictionary having the xtick labels as keys and data as values
    :param save: flag to save the plot
    :param path: location where to save the image
    """
    fig, ax = plt.subplots()
    ax.boxplot(dictionary.values())
    ax.set_xticklabels(dictionary.keys())
    if save:
        assert path is not None
        plt.savefig(path)
    plt.show()


if __name__ == "__main__":

    data = {}
    data["A"] = np.array([1,1,2,3,4,5])
    data["B"] = np.array([4,6,7,8,9,9,5,6])

    compare_boxplot(data)
