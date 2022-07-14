import flappy_bird_gym
from PIL import Image as im


# Environment constant

FLAPPY_BIRD_ENV = "FlappyBird-rgb-v0"


def extract_spaces(env, decompose=False):
    """
    Extract observation and action spaces in Gym format or in python data structure
    :param env: gym environment
    :param decompose: whether to return gym object or not, default is False
    """
    obs_space = env.observation_space
    act_space = env.action_space
    if decompose:
        return obs_space.shape, act_space.n
    else:
        return obs_space, act_space


def preprocess_obs(observation):
    """
    Preprocess the image before passing it to the agent
    :param observation: observation of the environment
    """
    new_obs = observation.astype('float64')
    new_obs /= 255
    shape = list(new_obs.shape)
    shape = [1] + shape
    new_obs = new_obs.reshape(tuple(shape))
    return new_obs


if __name__ == "__main__":

    env = flappy_bird_gym.make(FLAPPY_BIRD_ENV)
    obs = env.reset()
    print(preprocess_obs(obs))
    data = im.fromarray(obs)
    data.save('observatiion.png')