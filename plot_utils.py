from matplotlib import pyplot as plt


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
    Compare boxplot of different samples
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
