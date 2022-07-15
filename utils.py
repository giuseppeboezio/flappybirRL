import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

# Environment constant

FLAPPY_BIRD_ENV = "FlappyBird-rgb-v0"


def extract_spaces(env, decompose=False):
    """
    Extract observation and action spaces in Gym format or in python data structure
    :param env: gym environment
    :param decompose: whether to return gym object or not, default is False
    """
    obs_space = env.observation_space
    act_space = env.action_space
    if decompose:
        return obs_space.shape, act_space.n
    else:
        return obs_space, act_space


def preprocess_obs(observation):
    """
    Preprocess the image before passing it to the agent
    :param observation: observation of the environment
    """
    new_obs = observation.astype('float64')
    new_obs /= 255
    shape = list(new_obs.shape)
    shape = [1] + shape
    new_obs = new_obs.reshape(tuple(shape))
    return new_obs


def initialize_acc(weights):
    """
    Initialize all weights of the net to 0
    :param weights: weights of a network
    :return: gradient 0
    """
    acc = []
    for weight in weights:
        acc.append(tf.zeros(weight.shape))
    return acc


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return: list of mean tensors
    """
    acc = initialize_acc(tensors[0])
    num_tensors = len(tensors[0])
    num_elem = len(tensors)
    for i in range(num_tensors):
        for j in range(num_elem):
            acc[i] = tf.add(acc[i], tensors[j][i])
        acc[i] = tf.divide(acc[i], tf.constant(num_elem, dtype='float32'))
    return acc


class SeriesManager:

    def __init__(self, series_names):
        """
        Create an object to store temporal series
        :param series_names: names of temporal series
        """
        self.data = {}
        for name in series_names:
            self.data[name] = []

    def add_empty_series(self, name):
        """
        Add an empty temporal series
        :param name: name of the temporal series
        """
        self.data[name] = []

    def add_series(self, name, series):
        """
        Add a series to the store
        :param name: name of the series
        :param series: series list
        """
        self.data[name] = series

    def add_data(self, series_name, data):
        """
        Add a data to a specified series
        :param series_name: index of the series
        :param data: data
        """
        self.data[series_name].append(data)

    def save_series(self, series_name, path):
        """
        Save a series in the specified path
        :param series_name: name of the series
        :param path: location to save the series
        """
        target = self.data[series_name]
        pd_series = pd.Series(target)
        pd_series.to_csv(path, header=False, index=False)

    def load_series(self, series_name, path):
        """
        Load a series in the specified path
        :param series_name: name of the series
        :param path: location to load the series from
        """
        pd_series = pd.read_csv(path)
        target = pd_series.to_numpy().flatten()
        self.add_series(series_name, target)

    def plot_graph(self, title, x_label, y_label, grid=True, path=None, save=False):
        """
        Plot comparison among temporal series
        :param title: title of the plot
        :param x_label: label axis-x
        :param y_label: label axis-y
        :param grid: flag to enable the grid on the plot, default True
        :param path: path to optionally save the image, used only when save is True
        :param save: flag to save the image in path
        """
        keys = self.data.keys()
        plt.grid(grid)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        for key in keys:
            plt.plot(self.data[key], label=key)
        plt.legend(loc='upper right')
        # saving the image
        if save:
            plt.savefig(path)
        plt.show()


if __name__ == "__main__":

    first_tensor = tf.fill((3,2),3.0)
    second_tensor = tf.fill((3,2),7.0)
    third_tensor = tf.fill((5,3,4), 8.0)
    fourth_tensor = tf.fill((5,3,4), 10.0)
    tensors = [[first_tensor, third_tensor], [second_tensor, fourth_tensor]]
    average = mean_tensors(tensors)
    print(average)
