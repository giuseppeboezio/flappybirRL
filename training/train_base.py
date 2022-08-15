import tensorflow as tf
from agents.networks import ActorCriticBase
import numpy as np
from agents.actor_critic_agent import ActorCriticAgent
import flappy_bird_gym
from tensorflow.keras.optimizers import RMSprop
from train_a2c import train_step
from loss_estimator import A2CLossEstimator
from utils import BASE_SHAPE, save_series, plot_graph
from train import train


def update_series(series, obs):
    """
    Update the time series
    :param series: timeseries
    :param: current observation
    :return updated timeseries
    """
    new_series = series.numpy()
    new_series[:, :-1, :] = new_series[:, 1:, :]
    new_series[:, -1, :] = obs
    return tf.constant(new_series)


def episode(agent, env, max_steps):
    """
    Run an episode of the environment
    :param agent: player of the game
    :param env: environment
    :param max_steps: maximum amount of steps to interact with the environment
    :return values: state-value of each state of the trajectory
    :return action_probs: action probability for each action performed in the episode
    :return rewards: rewards of the episode plus the value of the last state
    """

    values = []
    action_probs = []
    rewards = []

    obs = tf.constant(env.reset())
    SERIES_LENGTH = 8
    initial_state_reshaped = tf.reshape(obs, (1, obs.shape[0]))
    state_series = tf.repeat(initial_state_reshaped, SERIES_LENGTH, axis=0)
    state_series = tf.expand_dims(state_series, 0)

    step = 1
    done = False

    while step <= max_steps and not done:
        # agent's behaviour
        action_probs_step, value = agent.act(state_series)
        # action choice
        action = np.random.choice(range(agent.num_actions), p=action_probs_step.numpy().flatten())
        # storing value and action
        values.append(value[0,0])
        action_probs.append(action_probs_step[0, action])
        state, reward, done, _ = env.step(action)
        state_series = update_series(state_series, state)
        # storing reward
        rewards.append(reward)

        step += 1

    # check exit condition
    value = 0
    if not done:
        # the reward of the last state is estimated with the state-value function
        action_probs_step, value = agent.act(state_series)
        value = value[0, 0]

    rewards.append(value)
    return values, action_probs, rewards


if __name__ == "__main__":

    num_episodes = 5000
    num_threads = 3
    env = flappy_bird_gym.make("FlappyBird-v0")
    max_steps = 100000
    gamma = 0.90
    estimator = A2CLossEstimator()
    optimizer = RMSprop(learning_rate=0.01)

    train(
        num_episodes,
        num_threads,
        env,
        ActorCriticBase,
        BASE_SHAPE,
        max_steps,
        gamma,
        estimator,
        episode,
        optimizer,
        "base_model"
    )
