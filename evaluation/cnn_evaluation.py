import tensorflow as tf
import flappy_bird_gym

import time

from agents.actor_critic_agent import ActorCriticAgent
from agents.networks import ActorCriticCNN
from training.train_cnn import luminance, rescale, normalize, update_stack, preprocess_image
from utils import run_step, save_series, get_cnn_initial_state
from constants import IMAGE_SHAPE, FLAPPY_IMAGE_NAME, CNN, DIR_MODELS


def evaluate_agent(model_name, num_games, human_mode=False):
    """
    Evaluate the performance of an agent for a specified number of games
    :param model_name: name of the pretrained model
    :param num_games: number of games
    :param human_mode: flag to enable the window of the game
    :return save scores of num_games
    """
    scores = []
    env = flappy_bird_gym.make(FLAPPY_IMAGE_NAME)
    input_shape = (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 4)
    agent = ActorCriticAgent(ActorCriticCNN, input_shape, env.action_space.n)
    agent.load_weights(f"{DIR_MODELS}/{model_name}/{model_name}")

    for _ in range(num_games):

        # initial state
        obs = tf.constant(env.reset())
        # preprocessing functions
        functions = [luminance, rescale, normalize]
        stack = get_cnn_initial_state(obs, functions, preprocess_image)

        done = False

        while not done:
            _, _, _, state, _, done, info = run_step(agent, env, stack)
            # display the window
            if human_mode:
                env.render()
                time.sleep(1 / 30)
            processed_image = preprocess_image(functions, state)
            stack = update_stack(stack, processed_image)

        scores.append(info["score"])

    save_series(scores, f"data/{model_name}.csv")


if __name__ == "__main__":

    evaluate_agent(CNN, num_games=10)
