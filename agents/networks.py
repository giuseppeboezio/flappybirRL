from keras import layers
from tensorflow import keras
import flappy_bird_gym
import utils


def get_a2c_network(obs_space, action_space):
    """
    generate the actor-critic network
    Parameters:
        - obs_space: gym observation space of the environment
        - action_space: gym action space of the environment
    Returns:
        - actor network and critic network
    """
    input_shape = obs_space.shape
    output_shape = action_space.n

    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu')(input)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    actions = layers.Dense(output_shape, activation='softmax')(x)
    value = layers.Dense(1, activation='relu')(x)
    # actor-critic network
    actor = keras.Model(input, actions, name="actor")
    critic = keras.Model(input, value, name="critic")
    # visual_keras_model = keras.Model(input, [actions, value])
    return actor, critic


if __name__ == "__main__":

    env = flappy_bird_gym.make(utils.FLAPPY_BIRD_ENV)
    obs_space = env.observation_space
    act_space = env.action_space

    actor, critic = get_a2c_network(obs_space, act_space)

    actor.summary()
    critic.summary()
