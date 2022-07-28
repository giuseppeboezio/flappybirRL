import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd


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
