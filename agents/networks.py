from keras import layers
import tensorflow as tf


# Neural network for FlappyBird-v0 environment
class ActorCriticBase(tf.keras.Model):

    def __init__(
            self,
            num_actions: int):
        super().__init__()

        self.gru1 = layers.GRU(5)
        self.dense1 = layers.Dense(4, activation='relu')
        self.dense2 = layers.Dense(4, activation='relu')
        self.dense3 = layers.Dense(2, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1, activation='relu')

    def call(self, inputs: tf.Tensor):
        x = self.gru1(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.actor(x), self.critic(x)


# Neural network for FlappyBird-rb-v0 environment
class ActorCriticCNN(tf.keras.Model):

    def __init__(
            self,
            num_actions: int):
        super().__init__()

        self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        return self.actor(x), self.critic(x)


if __name__ == "__main__":

    model = ActorCriticBase(2)
    model.build((1,5,3))
    model.summary()
