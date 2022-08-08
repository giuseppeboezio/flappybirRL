from queue import Queue
from threading import Thread
from utils import mean_tensors
import numpy as np
import tensorflow as tf


def get_expected_returns(rewards, gamma):
    """
    Returns expected returns of an episode
    :param rewards: rewards of an episode plus the value of the last state
    :param gamma: discount rate to compute expected returns
    :return expected returns
    """
    # epsilon for stabilizing division operation
    eps = np.finfo(np.float32).eps.item()
    # number of rewards excluded the last state value
    t = len(rewards) - 1
    expected_returns = np.zeros(len(rewards), dtype='float32')
    # the last position is used just to set the last state value as starting value
    expected_returns[-1] = rewards[-1]
    for i in range(t-1, -1, -1):
        expected_returns[i] = rewards[i] + gamma * expected_returns[i + 1]
    # standardization for a more stable outcome
    # (expected_returns - mean(expected_returns)) / (std(expected_returns + eps)
    # eps is added to avoid division by 0
    expected_returns = ((expected_returns - tf.math.reduce_mean(expected_returns)) /
               (tf.math.reduce_std(expected_returns) + eps))
    # last reward was just a way to store the value of the last state
    return expected_returns[:-1]


def train_agent(agent, env, run_episode, max_steps, gamma, loss_estimator, queue_gradients, queue_history):
    """
    Compute the gradient of the loss wrt the network's weights for a specific thread
    :param agent: player of the game
    :param env: OpenAI Gym environment
    :param run_episode: function to run a single episode,
    its parameters are the agent, the environment and the maximum number of steps per episode
    :param max_steps: upper limit of interactions with the environment in a single episode
    :param gamma: discount rate to compute expected returns
    :param loss_estimator: estimator of the loss function
    :param queue_gradients: queue to store gradients of different threads
    :param queue_history: queue to store cumulative rewards of different threads
    """
    with tf.GradientTape() as tape:

        # run an episode
        values, action_probs, rewards = run_episode(agent, env, max_steps)
        # get the discounted rewards
        expected_returns = get_expected_returns(rewards, gamma)
        # compute loss
        loss = loss_estimator.compute_loss(values, expected_returns, action_probs)

    grads = tape.gradient(loss, agent.network.trainable_variables)
    queue_gradients.put(grads)
    queue_history.put(np.sum(rewards))


def train_step(num_threads, agent, env_class, run_episode, max_steps, gamma, loss_estimator, optimizer):
    """
    Train the agent for one episode averaging the gradients of each thread
    :param num_threads: number of different environment interaction in an episode
    :param agent: player of the game
    :param env_class: openAI Gym environment class
    :param run_episode: function to run a single thread episode,
    its parameters are the agent, the environment and the maximum number of steps per episode
    :param max_steps: upper limit of interactions with the environment in a single episode
    :param gamma: discount rate to compute expected returns
    :param loss_estimator: estimator of the loss function
    :param optimizer: optimizer to update network's weights
    :return mean of cumulative rewards
    :return standard deviation of cumulative rewards
    """
    # queue used to store the gradients of each thread
    queue_gradients = Queue()
    # queue to store the cumulative reward of each thread
    queue_rewards = Queue()

    threads = []
    for _ in range(num_threads):
        tuple_process = (
            agent.copy(),
            env_class(),
            run_episode,
            max_steps,
            gamma,
            loss_estimator,
            queue_gradients,
            queue_rewards
        )
        threads.append(Thread(target=train_agent, args=tuple_process))

    # start the threads
    for thread in threads:
        thread.start()

    # terminate the threads
    for thread in threads:
        thread.join()

    # get the gradients of each thread
    grads = []
    while not queue_gradients.empty():
        grads.append(queue_gradients.get())

    cum_rewards = []
    while not queue_rewards.empty():
        cum_rewards.append(queue_rewards.get())

    network_weights = agent.network.trainable_variables
    average_grad = mean_tensors(grads)
    optimizer.apply_gradients(zip(average_grad, network_weights))

    return np.mean(cum_rewards), np.std(cum_rewards)
