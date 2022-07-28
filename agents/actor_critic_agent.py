from queue import Queue
from threading import Thread
import numpy as np
from .base_agent import BaseAgent
import time
import flappy_bird_gym
import utils


class ActorCriticAgent(BaseAgent):

    def __init__(self, network_class, num_actions):
        """
        Creation of the agent
        :param network_class: neural network class used by the agent
        :param num_actions: number of actions to interact with the environment
        """
        super().__init__()
        self.net_class = network_class
        self.num_actions = num_actions
        self.network = network_class(num_actions)

    def act(self, observation):
        """
        Corresponds to the selection of an action given a probability mass function
        :param observation: observable part of the environment
        :return: action to perform
        """
        return self.network(observation)

    def save_weights(self, path):
        """
        Save the policy model
        :param path: path of the model's weights
        :return: save the model
        """
        self.network.save_weights(path)

    def load_weights(self, path):
        """
        Load the policy from a stored model
        :param path: path of the model's weights
        """
        self.network.load_weights(path)

    def copy(self):
        """
        Copy of the agent
        :return: copy of the agent with the same weights of the current one
        """
        new_agent = ActorCriticAgent(self.net_class, self.num_actions)
        new_agent.network.set_weights(self.network.get_weights())
        return new_agent


if __name__ == "__main__":

    env = flappy_bird_gym.make(utils.FLAPPY_BIRD_ENV)
    obs_shape, num_actions = utils.extract_spaces(env, decompose=True)
    agent = ActorCriticAgent(obs_shape, num_actions)

    obs = env.reset()
    obs = utils.preprocess_obs(obs)

    while True:

        action = agent.act(obs)

        # Processing:
        obs, reward, done, info = env.step(action)
        obs = utils.preprocess_obs(obs)

        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()

