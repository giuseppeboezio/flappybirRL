from agents.base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):

    def __init__(self, network_class, input_shape, num_actions):
        """
        Creation of the agent
        :param network_class: neural network class used by the agent
        :param num_actions: number of actions to interact with the environment
        """
        super().__init__()
        self.net_class = network_class
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.network = network_class(num_actions)
        self.network.build(input_shape)

    def act(self, observation):
        """
        Corresponds to the probability distribution of actions given an observation and the value of the state
        :param observation: observable part of the environment
        :return: actions probability distribution
        """
        return self.network(observation)

    def save_weights(self, path):
        """
        Save the weights of the model
        :param path: path to store the weights
        """
        self.network.save_weights(path)

    def load_weights(self, path):
        """
        Load the weights of the model
        :param path: path where the model's weights are stored
        """
        self.network.load_weights(path)

    def copy(self):
        """
        Copy of the agent
        :return: copy of the agent with the same weights of the current one
        """
        new_agent = ActorCriticAgent(self.net_class, self.input_shape, self.num_actions)
        new_agent.network.set_weights(self.network.get_weights())
        return new_agent
