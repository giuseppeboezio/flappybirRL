import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple
import flappy_bird_gym
import time


class ActorCriticBase(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_actions: int):
        """Initialize."""
        super().__init__()

        self.gru1 = layers.GRU(5)
        self.dense1 = layers.Dense(4, activation='relu')
        self.dense2 = layers.Dense(4, activation='relu')
        self.dense3 = layers.Dense(2, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1, activation='relu')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.gru1(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.actor(x), self.critic(x)

def update_series(series, obs):
  new_series = series.numpy()
  new_series[:, :-1, :] = new_series[:, 1:, :]
  new_series[:, -1, :] = obs
  return tf.constant(new_series)


if __name__ == "__main__":
    env = flappy_bird_gym.make("FlappyBird-v0")
    num_actions = env.action_space.n
    model = ActorCriticBase(num_actions)
    model.load_weights("baseline/baseline")

    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    SERIES_LENGTH = 5
    initial_state_reshaped = tf.reshape(initial_state, (1, 3))
    state_series = tf.repeat(initial_state_reshaped, SERIES_LENGTH, axis=0)
    state_series = tf.expand_dims(state_series, 0)

    while True:
        # Next action:

        action_probs_t, value = model(state_series)
        action = tf.random.categorical(tf.math.log(action_probs_t), 1)[0, 0]

        # Processing:
        obs, reward, done, info = env.step(action)
        print(reward)
        state_series = update_series(state_series, obs)

        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()


