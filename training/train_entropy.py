from tensorflow.keras.optimizers import RMSprop
import flappy_bird_gym

from agents.networks import ActorCriticBase
from loss_estimator import A2CEntropyLossEstimator
from utils import BASE_SHAPE, FLAPPY_BASE_NAME
from train_base import episode
from train_utils import train


if __name__ == "__main__":
    num_episodes = 3000
    num_threads = 3
    env = flappy_bird_gym.make(FLAPPY_BASE_NAME)
    max_steps = 100000
    gamma = 0.90
    estimator = A2CEntropyLossEstimator()
    optimizer = RMSprop(learning_rate=0.01)

    train(
        num_episodes,
        num_threads,
        env,
        ActorCriticBase,
        BASE_SHAPE,
        max_steps,
        gamma,
        estimator,
        episode,
        optimizer,
        model_name="base_model"
    )