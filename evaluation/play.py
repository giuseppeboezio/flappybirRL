import argparse
from base_evaluation import evaluate_agent as evaluate_base
from cnn_evaluation import evaluate_agent as evaluate_cnn

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description="play with a Flappy Bird agent")
    #parser.add_argument("agent", type=str, help="Player agent", choices=["base", "cnn", "entropy"])
    #args = parser.parse_args()
    agent = "cnn"
    if agent == "base":
        evaluate_base("trained_base", num_games=1, human_mode=True)
    elif agent == "cnn":
        evaluate_cnn("trained_cnn", num_games=1, human_mode=True)
    else:
        evaluate_base("trained_entropy", num_games=1, human_mode=True)
