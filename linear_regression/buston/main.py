from model import LinearRegression
from utils import *

import matplotlib.pyplot as plt
import tensorflow as tf
import plotin1 as pi1
import numpy as np

train_epoch = 200    # variable

def load():
    all = datasets.load_boston()
    return all['data'], all['target']

# Load data and pre-process
data, tags = load()
data, tags = shuffleDataAndLabel(data, tags)
train_data, train_tags, test_data, test_tags = tenCrossValid(data, tags, 0)

if __name__ == "__main__":
    loss_list = []
    model = LinearRegression(np.shape(data)[1])
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())

        # Train
        for i in range(train_epoch):
            _loss, _ = sess.run([model.loss, model.optimizer], feed_dict= \
                {model.x: train_data, model.y: train_tags})
            print "epoch: ", i, "\tloss: ", _loss
            loss_list.append(_loss)

        # Testing
        print "test loss: ", sess.run([model.loss, model.optimizer], feed_dict= \
                {model.x: test_data, model.y: test_tags})[0]
        
        # Draw
        plt = pi1.LineChart(range(len(loss_list)), loss_list, ["loss"], dont_show_value=True)
        plt.show()