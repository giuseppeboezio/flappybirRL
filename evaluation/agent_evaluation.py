from utils import load_series
from constants import BASE, CNN, ENTROPY
from plot_utils import compare_boxplot

if __name__ == "__main__":

    # load evaluation data to compare agents
    directory = "./data"
    base = load_series(f"{directory}/{BASE}.csv")
    cnn = load_series(f"{directory}/{CNN}.csv")
    entropy = load_series(f"{directory}/{ENTROPY}.csv")

    # generate boxplot for comparison
    dictionary = {"base": base, "entropy": entropy, "cnn": cnn}
    compare_boxplot(dictionary, save=True, path="boxplot.png")
