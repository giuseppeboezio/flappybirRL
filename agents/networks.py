from keras import layers
from tensorflow import keras
import flappy_bird_gym
import utils


def get_a2c_network(obs_shape, num_actions):
    """
    Generate the actor-critic network
    :param obs_shape: shape of gym observation space
    :param num_actions: number of actions in gym action space of the environment
    :return actor network and critic network
    """

    input = layers.Input(shape=obs_shape)
    x = layers.Conv2D(16, (3, 3), kernel_regularizer='l2', activation='relu')(input)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), kernel_regularizer='l2', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), kernel_regularizer='l2', activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, kernel_regularizer='l2', activation='relu')(x)
    x = layers.Dense(32, kernel_regularizer='l2', activation='relu')(x)
    actions = layers.Dense(num_actions, activation='softmax')(x)
    value = layers.Dense(1, activation='relu')(x)
    # actor-critic network
    actor = keras.Model(input, actions, name="actor")
    critic = keras.Model(input, value, name="critic")
    return actor, critic


if __name__ == "__main__":

    env = flappy_bird_gym.make(utils.FLAPPY_BIRD_ENV)
    obs_shape, num_actions = utils.extract_spaces(env, decompose=True)

    actor, critic = get_a2c_network(obs_shape, num_actions)

    actor.summary()
    critic.summary()
