from threading import Thread
from queue import Queue
import numpy as np
import utils
from utils import initialize_acc, mean_tensors, run_episode
import tensorflow as tf
import flappy_bird_gym
from tensorflow.keras.optimizers import RMSprop
from agents.actor_critic_agent import ActorCriticAgent


def compute_gradients_actor(actor, critic, obs, act, num_actions, cumulative):
    """
    Generate the gradient update for the actor in A2C algorithm
    :param actor: policy network
    :param critic: sate-value network
    :param obs: observation at a certain timestep
    :param act: action at a certain timestep
    :param num_actions: number of possible actions which can be chosen by the policy network
    :param cumulative: cumulative reward at a certain timestep
    :return: gradients of the actor
    """
    mask = np.zeros(num_actions)
    mask[act] = 1
    weights = actor.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(weights)
        predictions = tf.reshape(actor(obs), [-1])
        loss = tf.keras.backend.log(predictions) * mask
    grads = tape.gradient(loss, weights)
    value = tf.reshape(critic(obs), [-1])
    delta = float((cumulative - value).numpy()[0])
    num_weights = len(weights)
    for j in range(num_weights):
        grads[j] = tf.math.scalar_mul(- delta, grads[j])

    return grads


def compute_gradients_actor_with_entropy(actor, critic, obs, act, num_actions, cumulative, beta=0.3):
    """
    Generate the gradient update for the actor in A2C algorithm with the entropy variant
    :param actor: policy network
    :param critic: sate-value network
    :param obs: observation at a certain timestep
    :param act: action at a certain timestep
    :param num_actions: number of possible actions which can be chosen by the policy network
    :param cumulative: cumulative reward at a certain timestep
    :param beta: strength of entropy regularization
    :return: gradients of the actor
    """
    grads = compute_gradients_actor(actor, critic, obs, act, num_actions, cumulative)
    # compute entropy of policy and corresponding gradients
    weights = actor.trainable_variables
    num_weights = len(weights)
    with tf.GradientTape() as tape:
        tape.watch(weights)
        predictions = tf.reshape(actor(obs), [-1])
        # computation of the entropy
        log = tf.keras.backend.log(predictions)
        num_axes = len(predictions.shape)
        entropy = - tf.tensordot(predictions, log, num_axes)
    grads_entropy = tape.gradient(entropy, weights)
    # strength of entropy regularization
    for i in range(num_weights):
        grads_entropy[i] = tf.math.scalar_mul(beta, grads_entropy[i])
    # sum gradients of two component of loss function
    for i in range(num_weights):
        grads[i] = tf.add(grads[i], grads_entropy[i])

    return grads


def compute_gradients_critic(critic, obs, cumulative):
    """
    Generate gradients of the critic wrt the corresponding loss of A2C algorithm
    :param critic: state-value network
    :param obs: observation of the environment
    :param cumulative: cumulative reward
    :return: gradients of the critic
    """
    weights = critic.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(weights)
        value = tf.reshape(critic(obs), [-1])
        delta = cumulative - value
        loss = tf.pow(delta, tf.constant(2, dtype='float32'))
    grads = tape.gradient(loss, weights)

    return grads


def train_a2c_single_agent(agent, env, gamma, max_steps, queue_actor, queue_critic, queue_history):
    """
    Update the gradients of a single agent using A2C method for an episode putting them in respective queues
    :param agent: agent to train
    :param env: gym environment
    :param gamma: discount rate
    :param max_steps: maximum number of steps if a final step is not reached
    :param queue_actor: shared queue of gradients for updating policy network of the main agent
    :param queue_critic: shared queue of gradients for updating value network of the main agent
    :param queue_history: shared queue of cumulative rewards
    :return:
    """
    # networks information
    single_policy = agent.actor
    single_val_net = agent.critic
    num_actions = agent.num_actions

    num_weights_actor = len(single_policy.trainable_variables)
    num_weights_val_net = len(single_val_net.trainable_variables)

    history_obs, history_actions, history_rewards, done = run_episode(env, agent, max_steps)

    # compute the gradients for each timestep

    r = 0 if done else single_val_net(history_obs[-1])
    # one less step because array index start from 0
    t = len(history_obs) - 1
    # accumulators of gradients
    acc_actor = initialize_acc(single_policy.trainable_variables)
    acc_critic = initialize_acc(single_val_net.trainable_variables)

    for i in range(t - 1, -1, -1):

        r = history_rewards[i] + gamma * r
        # gradients of actor
        grads_actor = compute_gradients_actor(
            single_policy,
            single_val_net,
            history_obs[i],
            history_actions[i],
            num_actions,
            r
        )

        # accumulate gradients actor
        for j in range(num_weights_actor):
            acc_actor[j] = tf.add(acc_actor[j], grads_actor[j])

        # gradients of critic
        grads_critic = compute_gradients_critic(single_val_net, history_obs[i], r)
        # accumulate gradients critic
        for j in range(num_weights_val_net):
            acc_critic[j] = tf.add(acc_critic[j], grads_critic[j])

    # load the gradients and the cumulative reward to the queue for the parent thread
    queue_actor.put(acc_actor)
    queue_critic.put(acc_critic)
    queue_history.put(sum(history_rewards))


def train_a2c_agent(
        agent_class,
        env,
        optimizer,
        max_steps=100000,
        num_episodes=3000,
        gamma=0.90,
        num_threads=1
):
    """
    Train an agent using the A2C algorithm
    :param agent_class: agent class
    :param env: gym environment instance
    :param gamma: discount rate
    :param optimizer: optimization algorithm to update the weights
    :param max_steps: maximum number of timesteps per episode
    :param num_episodes: number of episodes
    :param num_threads: number of threads which interact with the environment
    :return: history of training
    """
    # Environment settings
    env_class = env.__class__
    obs_shape, num_actions = utils.extract_spaces(env, decompose=True)
    # Agent settings
    agent = agent_class(obs_shape, num_actions)
    policy = agent.actor
    value_net = agent.critic
    # average cumulative reward for each episode
    history = []

    for episode in range(num_episodes):

        print(f"Episode {episode + 1}")

        # queue used to store the gradients of each process
        queue_actor = Queue()
        queue_critic = Queue()
        # queue to store the cumulative reward of each thread
        queue_history = Queue()

        threads = []
        for _ in range(num_threads):

            tuple_process = (
                agent.copy(),
                env_class(),
                gamma,
                max_steps,
                queue_actor,
                queue_critic,
                queue_history
            )
            threads.append(Thread(target=train_a2c_single_agent, args=tuple_process))

        print(f"Start threads episode {episode + 1}")

        # start the threads
        for thread in threads:
            thread.start()

        # terminate the threads
        for thread in threads:
            thread.join()

        # get the gradients of each thread
        grads_actor = []
        while not queue_actor.empty():
            grads_actor.append(queue_actor.get())

        grads_critic = []
        while not queue_critic.empty():
            grads_critic.append(queue_critic.get())

        cum_rewards = []
        while not queue_history.empty():
            cum_rewards.append(queue_history.get())

        # updating weights of the neural networks
        actor_weights = policy.trainable_variables
        critic_weights = value_net.trainable_variables

        # the update is done using the average gradients of threads
        average_grad_actor = mean_tensors(grads_actor)
        average_grad_critic = mean_tensors(grads_critic)

        optimizer.apply_gradients(zip(average_grad_actor, actor_weights))
        optimizer.apply_gradients(zip(average_grad_critic, critic_weights))

        # save the model
        agent.save_model("saved_models/a2c")

        # compute the average cumulative reward of the episode
        history.append(np.mean(cum_rewards))

    return history


if __name__ == "__main__":

    env = flappy_bird_gym.make(utils.FLAPPY_BIRD_ENV)

    history = train_a2c_agent(
        ActorCriticAgent,
        env,
        RMSprop()
    )
    manager = utils.SeriesManager(["A2C"])
    manager.add_series("A2C", history)
    manager.plot_graph(
        "A2C Training",
        "Episode",
        "Average cumulative reward",
        path=r"plot/A2C.png",
        save=True
    )
    manager.save_series("A2C", r"data/A2C.csv")
