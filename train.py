from tensorflow.keras.optimizers import RMSprop
import flappy_bird_gym

import argparse

from agents.networks import *
from constants import BASE_SHAPE, IMAGE_SHAPE, FLAPPY_BASE_NAME, FLAPPY_IMAGE_NAME
from training.loss_estimator import A2CLossEstimator, A2CEntropyLossEstimator
from training import train_base, train_cnn
from training.train_utils import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train a Flappy Bird agent")
    parser.add_argument("agent", type=str, help="Name of the agent to train", choices=["base", "cnn", "entropy"])
    parser.add_argument("num_episodes", type=int, help="Number of episodes")
    parser.add_argument("num_processes", type=int, help="Number of processes")
    parser.add_argument("discount_rate", type=float, help="Discount rate to computed expected return")
    parser.add_argument("learning_rate", type=float, help="Learning rate of the optimizer")

    args = parser.parse_args()
    agent = args.agent
    max_steps = 100000

    if agent == "base":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make(FLAPPY_BASE_NAME),
            ActorCriticBase,
            BASE_SHAPE,
            max_steps,
            args.discount_rate,
            A2CLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "base_model"
        )
    elif agent == "cnn":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make(FLAPPY_IMAGE_NAME),
            ActorCriticCNN,
            (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 4),
            max_steps,
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
            flappy_bird_gym.make(FLAPPY_BASE_NAME),
            ActorCriticBase,
            BASE_SHAPE,
            max_steps,
            args.discount_rate,
            A2CEntropyLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "entropy_model"
        )
