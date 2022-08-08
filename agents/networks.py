from keras import layers
import tensorflow as tf


# Neural network for FlappyBird-v0 environment
class ActorCriticBase(tf.keras.Model):

    def __init__(self, num_actions):

        super().__init__()

        self.gru = layers.GRU(8)
        self.dense1 = layers.Dense(4, activation='relu')
        self.dense2 = layers.Dense(4, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.actor(x), self.critic(x)


# Neural network for FlappyBird-rgb-v0 environment
class ActorCriticCNN(tf.keras.Model):

    def __init__(self, num_actions):

        super().__init__()

        self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten1 = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense(x)
        return self.actor(x), self.critic(x)
