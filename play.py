import argparse

from evaluation.base_evaluation import evaluate_agent as evaluate_base
from evaluation.cnn_evaluation import evaluate_agent as evaluate_cnn
from constants import BASE, CNN, ENTROPY

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="play with a Flappy Bird agent")
    parser.add_argument("agent", type=str, help="Player agent", choices=["base", "cnn", "entropy"])
    args = parser.parse_args()
    agent = args.agent
    if agent == "base":
        evaluate_base(BASE, num_games=1, human_mode=True)
    elif agent == "cnn":
        evaluate_cnn(CNN, num_games=1, human_mode=True)
    else:
        evaluate_base(ENTROPY, num_games=1, human_mode=True)
