from utils import load_series, plot_graph

if __name__ == "__main__":

    directory = "evaluation/data"
    base = load_series(f"{directory}/base_model.csv")
    cnn = load_series(f"{directory}/cnn_model.csv")
    entropy = load_series(f"{directory}/entropy_model.csv")

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
