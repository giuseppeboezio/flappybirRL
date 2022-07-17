from .base_agent import BaseAgent
import numpy as np
from keras.models import save_model, load_model
from .networks import get_a2c_network
import time
import flappy_bird_gym
import utils


class ActorCriticAgent(BaseAgent):

    def __init__(self, obs_shape, num_actions):
        """
        Creation of the agent
        :param obs_shape: shape of the input
        :param num_actions: number of different actions
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.actor, self.critic = get_a2c_network(self.obs_shape, self.num_actions)

    def act(self, observation):
        """
        Corresponds to the selection of an action given a probability mass function
        :param observation: observable part of the environment
        :return: action to perform
        """
        predictions = self.actor.predict(observation)
        action = np.random.choice(self.num_actions, p=predictions.flatten())
        return action

    def save_model(self, path):
        """
        Save the policy model
        :param path: path of the model
        :return: save the model
        """
        save_model(self.actor,f"{path}_actor")
        save_model(self.critic, f"{path}_critic")

    def load_model(self, path):
        """
        Load the policy from a stored model
        :param folder: folder where the model is contained
        :param name: name of the model
        :return: load the model
        """
        self.actor = load_model(f"{path}_actor")
        self.critic = load_model(f"{path}_critic")

    def copy(self):
        """
        Copy of the agent
        :return: copy of the agent with the same weights of the current one
        """
        new_agent = ActorCriticAgent(self.obs_shape, self.num_actions)
        new_agent.actor.set_weights(self.actor.get_weights())
        new_agent.critic.set_weights(self.critic.get_weights())
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

