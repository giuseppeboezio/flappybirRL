import tensorflow as tf
from utils import log2


# Base class for computing loss
class LossEstimator:

    def __init__(self):
        pass

    def compute_loss(self, values, expected_returns, action_probs):
        pass


# Class to compute the loss described in the paper "Asynchronous methods for deep reinforcement learning"
class A2CLossEstimator(LossEstimator):

    def __init__(self):
        super().__init__()

    def compute_loss(self, values, expected_returns, action_probs):
        advantages = tf.constant(expected_returns - values)
        log_probs = tf.math.log(action_probs)
        actor_loss = - tf.math.reduce_sum(log_probs * advantages)
        critic_loss = tf.math.reduce_sum(tf.math.pow(advantages, tf.constant(2.0)))
        loss = actor_loss + critic_loss
        return loss


# Class to compute a variant of the loss described in "Asynchronous methods for deep reinforcement learning"
class A2CEntropyLossEstimator(A2CLossEstimator):

    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta

    def compute_loss(self, values, expected_returns, action_probs):
        base_loss = super().compute_loss(values, expected_returns, action_probs)
        log_probs = log2(action_probs)
        entropy = - tf.reduce_sum(action_probs * log_probs)
        loss = base_loss + self.beta * entropy
        return loss
