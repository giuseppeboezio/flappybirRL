import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

# constants
# input shape base model
BASE_SHAPE = (1, 5, 3)
# dot size scatter plot
DOT_SIZE = 7


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return: list of mean tensors
    """
    acc = tf.zeros_like(tensors[0])
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


if __name__ == "__main__":

    data = range(1,11)
    log = np.log(data)
    exp = np.exp(data)
    pow = np.power(data, 2)
    list = [log, exp, pow]
    labels = ["log", "exp", "pow"]
    markers = ["-b", "-y", "-r"]
    title = "Example"
    xtitle = "x axis"
    ytitle = "y axis"
    plot_graph(list,labels,markers,title,xtitle,ytitle,True,True,"example.png")
