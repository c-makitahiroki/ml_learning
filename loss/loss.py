import tensorflow as tf


def cross_entropy(Y, net):
    loss = -tf.reduce_sum(Y * tf.log(net))
    return loss
