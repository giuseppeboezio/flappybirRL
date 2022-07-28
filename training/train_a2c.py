from queue import Queue
from threading import Thread
from train_utils import mean_tensors
import numpy as np
import tensorflow as tf


def get_discounted_rewards(rewards, gamma):

    # epsilon for stabilizing division
    eps = np.finfo(np.float32).eps.item()
    # number of rewards excluded the last one
    t = len(rewards) - 1
    discounted_rewards = np.zeros(len(rewards), dtype='float32')
    discounted_rewards[-1] = rewards[-1]
    for i in range(t-1, -1, -1):
        discounted_rewards[i] = rewards[i] + gamma * discounted_rewards[i + 1]
    # standardization
    discounted_rewards = ((discounted_rewards - tf.math.reduce_mean(discounted_rewards)) /
               (tf.math.reduce_std(discounted_rewards) + eps))
    # last reward was just a way to store the value of the last state
    return discounted_rewards[:-1]


def compute_loss(values, discounted_rewards, action_probs):

    advantages = tf.constant(discounted_rewards - values)
    log_probs = tf.math.log(action_probs)
    actor_loss = - tf.math.reduce_sum(log_probs * advantages)
    critic_loss = tf.math.reduce_sum(tf.math.pow(advantages, tf.constant(2.0)))
    loss = actor_loss + critic_loss
    return loss


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
        # compute loss
        loss = compute_loss(values, discounted_rewards, action_probs)

    grads = tape.gradient(loss, agent.network.trainable_variables)
    queue_gradients.put(grads)
    queue_history.put(np.sum(rewards))


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
    queue_rewards = Queue()

    threads = []
    for _ in range(num_threads):
        tuple_process = (
            agent.copy(),
            env_class(),
            run_episode,
            gamma,
            max_steps,
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

    agent.save_weights(path)

    return np.mean(cum_rewards)
