import tensorflow as tf
import numpy as np

class DNN(object):
    weight = {
        'fc1': tf.Variable(tf.random_uniform([4, 3], 0, 1.0), name='fc1')
    }
    bias = {
        'fc1': tf.Variable(tf.constant(0.1, shape=[3]))
    }
    
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 4], name='inputs')
        self.tags = tf.placeholder(tf.float32, [None, 3], name='tags')
        hidden = tf.add(tf.matmul(self.inputs, self.weight['fc1']), self.bias['fc1'])
        self.output = tf.nn.softmax(hidden)

        # Loss
        self.loss_part = self.tags * tf.log(self.output)
        self.loss = -tf.reduce_sum(self.tags * tf.log(self.output))

        # Acc
        correct_num = tf.equal(tf.arg_max(self.output, 1), tf.arg_max(self.tags, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_num, "float"))