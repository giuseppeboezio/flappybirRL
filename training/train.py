import flappy_bird_gym
from agents.networks import *
from utils import BASE_SHAPE, IMAGE_SHAPE
from training.loss_estimator import A2CLossEstimator, A2CEntropyLossEstimator
from training import train_base, train_cnn
from tensorflow.keras.optimizers import RMSprop
from train_utils import train
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train a Flappy Bird agent")
    parser.add_argument("agent", type=str, help="Name of the agent to train", choices=["base", "cnn", "entropy"])
    parser.add_argument("num_episodes", type=int, help="Number of episodes")
    parser.add_argument("num_processes", type=int, help="Number of processes")
    parser.add_argument("discount_rate", type=float, help="Discount rate to computed expected return")
    parser.add_argument("learning_rate", type=float, help="Learning rate of the optimizer")

    args = parser.parse_args()

    if args.agent == "base":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-v0"),
            ActorCriticBase,
            BASE_SHAPE,
            100000,
            args.discount_rate,
            A2CLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "base_model"
        )
    elif args.agent == "cnn":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-rgb-v0"),
            ActorCriticCNN,
            (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 4),
            100000,
            args.discount_rate,
            A2CLossEstimator(),
            train_cnn.episode,
            RMSprop(learning_rate=args.learning_rate),
            "cnn_model"
        )
    else:
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-v0"),
            ActorCriticBase,
            BASE_SHAPE,
            100000,
            args.discount_rate,
            A2CEntropyLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "entropy_model"
        )
