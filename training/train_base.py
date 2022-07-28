import tensorflow as tf
from agents.networks import ActorCriticBase
import numpy as np
from agents.actor_critic_agent import ActorCriticAgent
import flappy_bird_gym
from tensorflow.keras.optimizers import RMSprop
from train_a2c import train_step
from utils import SeriesManager, DOT_SIZE, BASE_SHAPE


def update_series(series, obs):
    new_series = series.numpy()
    new_series[:, :-1, :] = new_series[:, 1:, :]
    new_series[:, -1, :] = obs
    return tf.constant(new_series)


def episode(agent, env, max_steps):

    values = []
    action_probs = []
    rewards = []

    obs = tf.constant(env.reset())
    SERIES_LENGTH = 5
    # (1,3) because the state is (x_distance, y_distance, y_velocity)
    initial_state_reshaped = tf.reshape(obs, (1, 3))
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

    # check exit condition
    if done:
        value = 0
    else:
        action_probs_step, value = agent.act(state_series)

    rewards.append(value)

    return values, action_probs, rewards


if __name__ == "__main__":

    # Initialization
    num_episodes = 1000
    num_threads = 1
    env = flappy_bird_gym.make("FlappyBird-v0")
    num_actions = env.action_space.n
    agent = ActorCriticAgent(ActorCriticBase, BASE_SHAPE, num_actions)
    max_steps = 100000
    gamma = 0.99
    optimizer = RMSprop()
    path = "saved_models/base/base"

    history_rewards = []

    for i in range(num_episodes):

        episode_reward = train_step(
            num_threads,
            agent,
            env.__class__,
            episode,
            max_steps,
            gamma,
            optimizer,
            path
        )

        print(f"Episode {i+1}, Reward: {episode_reward}")

        history_rewards.append(episode_reward)

    manager = SeriesManager(["base_model"])
    manager.add_series("base_model", history_rewards)
    manager.save_series("base_model", "plot/base_model.csv")
    manager.plot_scatter(
        "base_model",
        "Base Model",
        "Episode",
        "Average Reward",
        DOT_SIZE,
        "plot/base.png",
        save=True
    )
