from base_agent import BaseAgent
import numpy as np
import utils
import time
import flappy_bird_gym


class RandomAgent(BaseAgent):
    def __init__(self, act_space):
        super().__init__()
        self.num_actions = act_space.n

    def act(self, observation):
        action = np.random.randint(self.num_actions)
        return action


if __name__ == "__main__":

    env = flappy_bird_gym.make(utils.FLAPPY_BIRD_ENV)
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    while True:

        action = agent.act(obs)

        # Processing:
        obs, reward, done, info = env.step(action)

        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()
