from utils import load_series, plot_graph

if __name__ == "__main__":

    directory = "evaluation/data"
    base = load_series(f"{directory}/trained_base.csv")
    cnn = load_series(f"{directory}/trained_cnn.csv")
    entropy = load_series(f"{directory}/trained_entropy.csv")

    plot_graph(
        [base, cnn, entropy],
        ["base model", "cnn model", "entropy model"],
        ["-b", "-y", "-g"],
        "Evaluation",
        "Games",
        "Score",
        True,
        True,
        "evaluation.png"
    )
