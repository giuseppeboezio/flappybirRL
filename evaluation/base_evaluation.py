import tensorflow as tf
import flappy_bird_gym
from agents.actor_critic_agent import ActorCriticAgent
from agents.networks import ActorCriticBase
from training.train_base import update_series
from utils import BASE_SHAPE, save_series
import numpy as np
import time


def evaluate_agent(model_name, num_games, human_mode=False):
    """
    Evaluate the performance of an agent for a certain number of games
    :param model_name: name of the pretrained model
    :param num_games: number of games
    :param human_mode: flag to enable the window of the game
    :return save scores of num_games
    """
    scores = []
    env = flappy_bird_gym.make("FlappyBird-v0")
    agent = ActorCriticAgent(ActorCriticBase, BASE_SHAPE, env.action_space.n)
    agent.load_weights(f"../training/saved_models/{model_name}/{model_name}")

    for _ in range(num_games):

        obs = tf.constant(env.reset())
        SERIES_LENGTH = 8
        initial_state_reshaped = tf.reshape(obs, (1, obs.shape[0]))
        state_series = tf.repeat(initial_state_reshaped, SERIES_LENGTH, axis=0)
        state_series = tf.expand_dims(state_series, 0)

        done = False

        while not done:
            # agent's behaviour
            action_probs_step, value = agent.act(state_series)
            # action choice
            action = np.random.choice(range(agent.num_actions), p=action_probs_step.numpy().flatten())
            # update the environment
            state, reward, done, info = env.step(action)
            # display the window
            if human_mode:
                env.render()
                time.sleep(1 / 30)
            state_series = update_series(state_series, state)

        # storing score
        scores.append(info["score"])

    save_series(scores, f"data/{model_name}.csv")


if __name__ == "__main__":

    evaluate_agent("trained_base", num_games=10)
