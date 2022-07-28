from queue import Queue
from threading import Thread
from train_utils import mean_tensors, update_series
import numpy as np
import tensorflow as tf


def train_step(
        num_threads,
        agent,
        env_class,
        run_episode,
        max_steps,
        gamma,
        optimizer,
        path
):
    # queue used to store the gradients of each process
    queue_gradients = Queue()
    # queue to store the cumulative reward of each thread
    queue_history = Queue()

    threads = []
    for _ in range(num_threads):
        tuple_process = (
            agent.copy(),
            env_class(),
            run_episode,
            gamma,
            max_steps,
            queue_gradients,
            queue_history
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
    while not queue_history.empty():
        cum_rewards.append(queue_history.get())

    network_weights = agent.trainable_variables
    average_grad = mean_tensors(grads)
    optimizer.apply_gradients(zip(average_grad, network_weights))

    agent.save_weights(path)

    return np.mean(cum_rewards)


def train_agent(
        agent,
        env,
        run_episode,
        gamma,
        max_steps,
        queue_gradients,
        queue_history
):
    with tf.GradientTape() as tape:

        # run an episode
        values, action_probs, rewards = run_episode(agent, env, max_steps)

        # get the discounted rewards
        discounted_rewards = get_discounted_rewards(rewards, gamma)


def episode(agent, env, max_steps):
    values = []
    action_probs = []
    rewards = []

    obs = tf.constant(env.reset())
    SERIES_LENGTH = 5
    initial_state_reshaped = tf.reshape(obs, (1, 3))
    state_series = tf.repeat(initial_state_reshaped, SERIES_LENGTH, axis=0)
    state_series = tf.expand_dims(state_series, 0)

    step = 1
    done = False

    while step < max_steps and not done:
        # agent's behaviour
        action_probs_step, value = agent.act(state_series)

        # action choice
        action = np.random.choice(range(1), p=action_probs_step.flatten())

        # storing value and action
        values.append(value)
        action_probs.append(action_probs_step[0, action])

        state, reward, done = env.step(action)

        state_series = update_series(state_series, state)

        # storing reward
        rewards.append(reward)

    # check exit condition
    if done:
        last_reward = 0
    else:
        action_probs_step, last_reward = agent.act(state_series)

    rewards.append(last_reward)

    return values, action_probs, rewards


def get_discounted_rewards(rewards, gamma):

    # epsilon for stabilizing division
    eps = np.finfo(np.float32).eps.item()
    # number of rewards excluded the last one
    t = len(rewards) - 1
    discounted_rewards = np.zeros(len(rewards))
    discounted_rewards[-1] = rewards[-1]
    for i in range(t-1,-1,-1):
        discounted_rewards[i] = rewards[i] + gamma * discounted_rewards[i + 1]
    # standardization
    discounted_rewards = ((discounted_rewards - tf.math.reduce_mean(discounted_rewards)) /
               (tf.math.reduce_std(discounted_rewards) + eps))
    return discounted_rewards



