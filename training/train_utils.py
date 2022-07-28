import tensorflow as tf


def initialize_acc(weights):
    """
    Initialize all weights of the net to 0
    :param weights: weights of a network
    :return: gradient 0
    """
    acc = []
    for weight in weights:
        acc.append(tf.zeros(weight.shape))
    return acc


def mean_tensors(tensors):
    """
    Compute the mean of tensors contained in different lists
    :param tensors: nested list of tensors
    :return: list of mean tensors
    """
    acc = initialize_acc(tensors[0])
    num_tensors = len(tensors[0])
    num_elem = len(tensors)
    for i in range(num_tensors):
        for j in range(num_elem):
            acc[i] = tf.add(acc[i], tensors[j][i])
        acc[i] = tf.divide(acc[i], tf.constant(num_elem, dtype='float32'))
    return acc

def update_series(series, obs):
  new_series = series.numpy()
  new_series[:, :-1, :] = new_series[:, 1:, :]
  new_series[:, -1, :] = obs
  return tf.constant(new_series)
