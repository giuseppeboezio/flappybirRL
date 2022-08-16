from agents.actor_critic_agent import ActorCriticAgent
from training.train_a2c import train_step
from utils import save_series, plot_graph
import flappy_bird_gym
from agents.networks import *
from utils import BASE_SHAPE, IMAGE_SHAPE
from training.loss_estimator import A2CLossEstimator, A2CEntropyLossEstimator
from training import train_base, train_cnn
from tensorflow.keras.optimizers import RMSprop
import argparse


def train(
        num_episodes,
        num_threads,
        env,
        network_class,
        input_shape,
        max_steps,
        gamma,
        loss,
        episode,
        optimizer,
        model_name
):
    """
    Train an agent
    :param num_episodes: number of episodes
    :param num_threads: different parallel executions
    :param env: environment
    :param network_class: class of the neural network
    :param input_shape: input shape of the neural network
    :param max_steps: maximum number of steps per episode
    :param gamma: discount rate for expected return
    :param loss: loss function to minimize
    :param episode: function to run an episode
    :param optimizer: optimization algorithm
    :param model_name: name of the model to save
    """
    # Initialization
    num_actions = env.action_space.n
    agent = ActorCriticAgent(network_class, input_shape, num_actions)

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
            loss,
            optimizer
        )

        agent.save_weights(f"saved_models/{model_name}/{model_name}")

        print(f"Episode {i + 1}, Mean: {mean} Std: {std}")

        mean_rewards.append(mean)
        std_rewards.append(std)

    # save the results
    save_series(mean_rewards, f"data/{model_name}/{model_name}_mean.csv")
    save_series(std_rewards, f"data/{model_name}/{model_name}_std.csv")
    plot_graph(
        [mean_rewards, std_rewards],
        ["Mean", "Std"],
        ["+b", "+y"],
        "",
        "Training Episode",
        "",
        True,
        True,
        f"plot/{model_name}.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train a Flappy Bird agent")
    parser.add_argument("agent", type=str, help="Name of the agent to train", choices=["base", "cnn", "entropy"])
    parser.add_argument("num_episodes", type=int, help="Number of episodes")
    parser.add_argument("num_processes", type=int, help="Number of processes")
    parser.add_argument("discount_rate", type=float, help="Discount rate to computed expected return")
    parser.add_argument("learning_rate", type=float, help="Learning rate of the optimizer")

    args = parser.parse_args()

    if args.agent == "base":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-v0"),
            ActorCriticBase,
            BASE_SHAPE,
            100000,
            args.discount_rate,
            A2CLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "base_model"
        )
    elif args.agent == "cnn":
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-rgb-v0"),
            ActorCriticCNN,
            (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 4),
            100000,
            args.discount_rate,
            A2CLossEstimator(),
            train_cnn.episode,
            RMSprop(learning_rate=args.learning_rate),
            "cnn_model"
        )
    else:
        train(
            args.num_episodes,
            args.num_processes,
            flappy_bird_gym.make("FlappyBird-v0"),
            ActorCriticBase,
            BASE_SHAPE,
            100000,
            args.discount_rate,
            A2CEntropyLossEstimator(),
            train_base.episode,
            RMSprop(learning_rate=args.learning_rate),
            "entropy_model"
        )
