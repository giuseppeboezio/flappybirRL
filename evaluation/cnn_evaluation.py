import tensorflow as tf
import flappy_bird_gym
from agents.actor_critic_agent import ActorCriticAgent
from agents.networks import ActorCriticCNN
from training.train_cnn import luminance, rescale, normalize, update_stack, preprocess_image
from utils import IMAGE_SHAPE, save_series
import numpy as np


def evaluate_agent(model_name, num_games):
    """
    Evaluate the performance of an agent for a certain number of games
    :param model_name: name of the pretrained model
    :param num_games: number of games
    :return save scores of num_games
    """
    scores = []
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    agent = ActorCriticAgent(ActorCriticCNN, IMAGE_SHAPE, env.action_space.n)
    agent.load_weights(f"training/saved_models/{model_name}/{model_name}")

    for _ in range(num_games):

        obs = tf.constant(env.reset())
        functions = [luminance, rescale, normalize]
        processed_image = preprocess_image(functions, obs.numpy())
        SERIES_LENGTH = 4
        h = processed_image.shape[0]
        w = processed_image.shape[1]
        processed_image = np.reshape(processed_image, (h, w, 1))
        stack = tf.repeat(processed_image, SERIES_LENGTH, axis=2)
        stack = tf.expand_dims(stack, 0)

        done = False

        while not done:
            # agent's behaviour
            action_probs_step, value = agent.act(stack)
            # action choice
            action = np.random.choice(range(agent.num_actions), p=action_probs_step.numpy().flatten())
            # updating the environment
            state, reward, done, info = env.step(action)
            processed_image = preprocess_image(functions, state)
            stack = update_stack(stack, processed_image)

        # storing score
        scores.append(int(info["score"]))

    save_series(scores, f"data/{model_name}.csv")


if __name__ == "__main__":

    evaluate_agent("trained_cnn", num_games=10)
