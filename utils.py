import flappy_bird_gym
from PIL import Image as im
import tensorflow as tf


# Environment constant

FLAPPY_BIRD_ENV = "FlappyBird-rgb-v0"


def extract_spaces(env, decompose=False):
    """
    Extract observation and action spaces in Gym format or in python data structure
    :param env: gym environment
    :param decompose: whether to return gym object or not, default is False
    """
    obs_space = env.observation_space
    act_space = env.action_space
    if decompose:
        return obs_space.shape, act_space.n
    else:
        return obs_space, act_space


def preprocess_obs(observation):
    """
    Preprocess the image before passing it to the agent
    :param observation: observation of the environment
    """
    new_obs = observation.astype('float64')
    new_obs /= 255
    shape = list(new_obs.shape)
    shape = [1] + shape
    new_obs = new_obs.reshape(tuple(shape))
    return new_obs


def initialize_acc(weights):
    """
    Initialize all weights of the net to 0
    :param weights: weights of a network
    :return: gradient 0
    """
    acc = []
    for weight in weights:
        acc.append(tf.zeros(weight.shape))
    return acc


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return: list of mean tensors
    """
    acc = initialize_acc(tensors[0])
    num_tensors = len(tensors[0])
    num_elem = len(tensors)
    for i in range(num_tensors):
        for j in range(num_elem):
            acc[i] = tf.add(acc[i], tensors[j][i])
        acc[i] = tf.divide(acc[i], tf.constant(num_elem, dtype='float32'))
    return acc


if __name__ == "__main__":

    first_tensor = tf.fill((3,2),3.0)
    second_tensor = tf.fill((3,2),7.0)
    third_tensor = tf.fill((5,3,4), 8.0)
    fourth_tensor = tf.fill((5,3,4), 10.0)
    tensors = [[first_tensor, third_tensor], [second_tensor, fourth_tensor]]
    average = mean_tensors(tensors)
    print(average)
