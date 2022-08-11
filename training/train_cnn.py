import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import flappy_bird_gym
from agents.actor_critic_agent import ActorCriticAgent
from agents.networks import ActorCriticCNN
from training.loss_estimator import A2CLossEstimator
from training.train_a2c import train_step
from utils import IMAGE_SHAPE, MAX_PIXEL_VALUE, CNN_SHAPE, save_series, plot_graph


def luminance(image):
    """
    Convert a 3d image to a 2d extracting luminance from rgb image
    :param image: 3d image
    :return 2d grayscale image
    """
    WEIGHT_R = 0.3
    WEIGHT_G = 0.59
    WEIGHT_B = 0.11
    return WEIGHT_R * image[:,:,0] + WEIGHT_G * image[:,:,1] + WEIGHT_B * image[:,:,2]


def rescale(image):
    """
    Rescale the image to a given size
    :param image: image to rescale
    :return rescaled image
    """
    return cv2.resize(image, IMAGE_SHAPE, interpolation = cv2.INTER_AREA)


def normalize(image):
    """
    Normalize each pixel value between 0 and 1
    :param image: image to normalize
    :return normalized image
    """
    return image / MAX_PIXEL_VALUE


# preprocessing of the screenshot
def preprocess_image(functions_list, image):
    """
    Preprocess an image applying in sequence functions provided as parameters
    :param functions_list: sequence of functions to apply to the image
    :param image: image to be processed
    :return preprocessed image
    """
    output = image
    for function in functions_list:
        output = function(output)

    return tf.constant(output)


def update_stack(stack, image):
    """
    Update input of the convolutional neural network
    :param stack: stack of the last preprocessed images
    :param image: current preprocessed image
    :return updated stack with last image equals to the current one
    """
    new_series = stack.numpy()
    new_series[:, :, :, :-1] = new_series[:, :, :, 1:]
    new_series[:, :, :, -1] = image
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
    functions = [luminance, rescale, normalize]
    processed_image = preprocess_image(functions, obs.numpy())
    SERIES_LENGTH = 4
    h = processed_image.shape[0]
    w = processed_image.shape[1]
    processed_image = np.reshape(processed_image, (h, w, 1))
    stack = tf.repeat(processed_image, SERIES_LENGTH, axis=2)
    stack = tf.expand_dims(stack, 0)

    step = 1
    done = False

    while step <= max_steps and not done:
        # agent's behaviour
        action_probs_step, value = agent.act(stack)
        # action choice
        action = np.random.choice(range(agent.num_actions), p=action_probs_step.numpy().flatten())
        # storing value and action
        values.append(value[0,0])
        action_probs.append(action_probs_step[0, action])
        state, reward, done, _ = env.step(action)
        processed_image = preprocess_image(functions, state)
        stack = update_stack(stack, processed_image)
        # storing reward
        rewards.append(reward)

        step += 1

    # check exit condition
    value = 0
    if not done:
        # the reward of the last state is estimated with the state-value function
        action_probs_step, value = agent.act(stack)
        value = value[0, 0]

    rewards.append(value)
    return values, action_probs, rewards


if __name__ == "__main__":
    # Initialization
    num_episodes = 500
    num_threads = 3
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    num_actions = env.action_space.n
    agent = ActorCriticAgent(ActorCriticCNN, CNN_SHAPE, num_actions)
    max_steps = 100000
    gamma = 0.99
    estimator = A2CLossEstimator()
    optimizer = RMSprop(learning_rate=0.01)

    mean_rewards = []
    std_rewards = []

    for i in range(num_episodes):
        mean, std = train_step(
            num_threads,
            agent,
            env.__class__,
            episode,
            max_steps,
            gamma,
            estimator,
            optimizer
        )

        agent.save_weights("saved_models/image/image")

        print(f"Episode {i + 1}, Mean: {mean} Std: {std}")

        mean_rewards.append(mean)
        std_rewards.append(std)

    # save the results
    save_series(mean_rewards, "data/image/image_mean.csv")
    save_series(std_rewards, "data/image/image_std.csv")
    plot_graph(
        [mean_rewards, std_rewards],
        ["Mean", "Std"],
        ["-b", "-y"],
        "",
        "Training Episode",
        "",
        True,
        True,
        "plot/image.png"
    )
