import tensorflow as tf
import flappy_bird_gym

import time

from agents.actor_critic_agent import ActorCriticAgent
from agents.networks import ActorCriticBase
from training.train_base import update_series
from utils import run_step, get_base_initial_state, save_series
from constants import BASE_SHAPE, FLAPPY_BASE_NAME, BASE, DIR_MODELS


def evaluate_agent(model_name, num_games, human_mode=False):
    """
    Evaluate the performance of an agent for a specified number of games
    :param model_name: name of the pretrained model
    :param num_games: number of games
    :param human_mode: flag to enable the window of the game
    :return save scores of num_games
    """
    scores = []
    env = flappy_bird_gym.make(FLAPPY_BASE_NAME)
    agent = ActorCriticAgent(ActorCriticBase, BASE_SHAPE, env.action_space.n)
    agent.load_weights(f"{DIR_MODELS}/{model_name}/{model_name}")

    for _ in range(num_games):

        # initial state
        obs = tf.constant(env.reset())
        state_series = get_base_initial_state(obs)

        done = False

        while not done:
            _, _, _, state, _, done, info = run_step(agent, env, state_series)
            # display the window
            if human_mode:
                env.render()
                time.sleep(1 / 30)
            state_series = update_series(state_series, state)

        scores.append(info["score"])

    save_series(scores, f"data/{model_name}.csv")


if __name__ == "__main__":

    evaluate_agent(BASE, num_games=10)
