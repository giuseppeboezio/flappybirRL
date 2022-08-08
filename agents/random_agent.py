from base_agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def act(self, observation):
        action = np.random.randint(self.num_actions)
        return action
