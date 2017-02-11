import tensorflow as tf
import numpy as np

class LinearRegression(object):
    def __init__(self, dim_size):
        self.x = tf.placeholder(tf.float64, [None, dim_size], name='x')
        self.y = tf.placeholder(tf.float64, [None], name='y')
        W = tf.Variable(tf.random_normal([dim_size, 1], dtype=tf.float64, stddev=0.5), name='weight')
        self.output = tf.matmul(self.x, W)
        self.loss = tf.reduce_mean(tf.square(self.output - self.y))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)