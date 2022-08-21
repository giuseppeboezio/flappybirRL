import os

from agents.actor_critic_agent import ActorCriticAgent
from training.train_a2c import train_step
from utils import save_series
from constants import DIR_MODELS, DIR_DATA, DIR_PLOT
from plot_utils import plot_graph


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
    :param num_threads: number of different parallel executions
    :param env: OpenAI Gym environment
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

        agent.save_weights(f"{DIR_MODELS}/{model_name}/{model_name}")

        print(f"Episode {i + 1}, Mean: {mean} Std: {std}")

        mean_rewards.append(mean)
        std_rewards.append(std)

    # save the results
    # check whether the folder already exists
    if not os.path.isdir(f"{DIR_DATA}/{model_name}"):
        os.mkdir(f"{DIR_DATA}/{model_name}")
    # save data
    save_series(mean_rewards, f"{DIR_DATA}/{model_name}/{model_name}_mean.csv")
    save_series(std_rewards, f"{DIR_DATA}/{model_name}/{model_name}_std.csv")
    plot_graph(
        [mean_rewards, std_rewards],
        ["Mean", "Std"],
        ["+b", "+y"],
        "",
        "Training Episode",
        "",
        grid=True,
        save=True,
        path=f"{DIR_PLOT}/{model_name}.png"
    )
