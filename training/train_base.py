import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import flappy_bird_gym

from agents.networks import ActorCriticBase
from training.loss_estimator import A2CLossEstimator
from utils import get_base_initial_state, run_step
from constants import BASE_SHAPE, FLAPPY_BASE_NAME
from training.train_utils import train


def update_series(series, obs):
    """
    Update the time series
    :param series: timeseries
    :param obs: current observation
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
    :param env: OpenAI Gym environment
    :param max_steps: maximum amount of steps to interact with the environment
    :return values: state-value of each state of the trajectory
    :return action_probs: action probability for each action performed in the episode
    :return rewards: rewards of the episode plus the value of the last state
    """

    values = []
    action_probs = []
    rewards = []

    obs = tf.constant(env.reset())
    state_series = get_base_initial_state(obs)

    step = 1
    done = False

    while step <= max_steps and not done:
        action, action_probs_step, value, state, reward, done, _ = run_step(agent, env, state_series)
        # storing value and action
        values.append(value[0,0])
        action_probs.append(action_probs_step[0, action])
        # storing reward
        rewards.append(reward)
        state_series = update_series(state_series, state)

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

    num_episodes = 2000
    num_threads = 3
    env = flappy_bird_gym.make(FLAPPY_BASE_NAME)
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
        model_name="base_model"
    )
