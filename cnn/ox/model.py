import tensorflow as tf

class OXNet(object):
    weight = {
        'conv1': tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.5), name='kernel1'),
        'conv2': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=1.0), name='kernel2'),
        'fc': tf.Variable(tf.random_normal([3200, 2]), name='fc')
    }
    bias = {
        'fc': tf.Variable(tf.constant(0.5, shape=[1, 2]), name='fc')
    }

    def __init__(self, img_height, img_weight):
        # placeHolder
        self.input = tf.placeholder(tf.float32, [None, img_height, img_weight, 3], name='input')
        self.tag = tf.placeholder(tf.float32, [None, 2], name='tag')

        # network
        map1 = tf.nn.conv2d(self.input, self.weight['conv1'], [1, 2, 2, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(map1, name='relu1')
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool1')

        map2 = tf.nn.conv2d(map1, self.weight['conv2'], [1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(map2)
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool2')

        flat = tf.reshape(pool2, [-1, 3200])
        self.output = tf.add(tf.matmul(flat, self.weight['fc']), self.bias['fc'])

        # loss and acc
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.tag))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast( \
            tf.equal(tf.argmax(self.output, 1), tf.argmax(self.tag, 1)), "float"), name='accuracy')      