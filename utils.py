import numpy as np
import tensorflow as tf
import pandas as pd
from constants import SERIES_LENGTH, NUM_CHANNELS


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return list of mean tensors
    """
    acc = [tf.zeros_like(x) for x in tensors[0]]
    num_tensors = len(tensors[0])
    num_elem = len(tensors)
    for i in range(num_tensors):
        for j in range(num_elem):
            acc[i] = tf.add(acc[i], tensors[j][i])
        acc[i] = tf.divide(acc[i], tf.constant(num_elem, dtype='float32'))
    return acc


def get_base_initial_state(obs):
    """
    Get initial state for the FlappyBird-v0 environment
    :param obs: initial observation of the environment
    :return initial state of the environment
    """
    # batch size = 1
    initial_state_reshaped = tf.reshape(obs, (1, obs.shape[0]))
    # initial state for the neural network
    state_series = tf.repeat(initial_state_reshaped, SERIES_LENGTH, axis=0)
    state_series = tf.expand_dims(state_series, 0)
    return state_series


def get_cnn_initial_state(obs, functions, preprocessing_fun):
    """
    Get initial state for the FlappyBird-rgb-v0 environment
    :param obs: initial observation of the environment
    :param functions: functions to preprocess the image
    :param preprocessing_fun: functions which applies functions
    :return initial state of the environment
    """
    processed_image = preprocessing_fun(functions, obs.numpy())
    h = processed_image.shape[0]
    w = processed_image.shape[1]
    # state of the environment
    processed_image = np.reshape(processed_image, (h, w, 1))
    stack = tf.repeat(processed_image, NUM_CHANNELS, axis=2)
    stack = tf.expand_dims(stack, 0)
    return stack


def run_step(agent, env, obs):
    """
    Run a step of interaction with the environment
    :param agent: player of the game
    :param env: OpenAI Gym environment
    :param obs: observation of the environment
    :return action: action chosen by the agent
    :return action_probs_step: action probability distribution
    :return value: value of the observation
    :return state: new state of the environment
    :return reward: reward after the interaction
    :return done: flag to know whether state is terminal or not
    :return info: score obtained by the agent
    """
    action_probs_step, value = agent.act(obs)
    # action choice
    action = np.random.choice(range(agent.num_actions), p=action_probs_step.numpy().flatten())
    # update the environment
    state, reward, done, info = env.step(action)
    return action, action_probs_step, value, state, reward, done, info


def log2(x):
    """
    Compute logarithm in base 2, this functionality is not supported in tensorflow
    :param x: argument of the logarithm
    :return log in base 2 of x
    """
    # change of basis formula
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def save_series(series, path):
    """
    Store a timeseries in a csv file
    :param series: numpy array
    :param path: location to store the series
    """
    pd_series = pd.Series(series)
    pd_series.to_csv(path, header=False, index=False)


def load_series(path):
    """
    Load a series from a specified location
    :param path: location where the series is stored
    :return numpy series
    """
    pd_series = pd.read_csv(path, header=None)
    series = pd_series.to_numpy().flatten()
    return series
