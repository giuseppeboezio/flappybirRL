from base_agent import BaseAgent
import numpy as np
from keras.models import save_model, load_model
from networks import get_a2c_network


class ActorCriticAgent(BaseAgent):

    def __init__(self, obs_space, action_space):
        """
        Creation of the agent
        :param obs_space: shape of the input
        :param action_space: number of different actions
        """
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor, self.critic = get_a2c_network(obs_space, action_space)

    def act(self, observation):
        """
        Corresponds to the selection of an action given a probability mass function
        :param observation: observable part of the environment
        :return: action to perform
        """
        predictions = self.actor.predict(observation)
        action = np.random.choice(self.action_space, p=predictions.flatten())
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
        new_agent = ActorCriticAgent(self.obs_space, self.action_space)
        new_agent.actor.set_weights(self.actor.get_weights())
        new_agent.critic.set_weights(self.critic.get_weights())
        return new_agent
