from sklearn import datasets
from model import DNN
import tensorflow as tf
import numpy as np

def raiseDim(_arr):
    _res = np.zeros([len(_arr), 3], dtype=np.float)
    for i in range(len(_arr)):
        _res[i][_arr[i]] = 1
    return _res

data = datasets.load_iris()
x = data['data'].astype(np.float32)[-5:]
y = data['target'].astype(np.float32)[-5:]

model = DNN()

sess = tf.Session()
with sess.as_default():
    step = tf.Variable(0, False)
    optimizer = tf.train.AdamOptimizer(0.001)
    # grad_vars = optimizer.compute_gradients(model.loss_revised)
    # train_op = optimizer.apply_gradients(grad_vars)
    train_step = optimizer.minimize(model.loss)

    loss = tf.scalar_summary('loss', model.loss)
    write_op = tf.merge_summary([loss])
    writer = tf.train.SummaryWriter('./res', sess.graph)

    sess.run(tf.initialize_all_variables())
    for i in range(300):
        feed = {
            model.inputs: x,
            model.tags: raiseDim(y)
        }
        #_loss, _acc, _, __, _step = sess.run([model.loss, model.acc, train_op, write_op, step], feed_dict=feed)

        _loss, _acc, _, _summary, _step = sess.run([model.loss, model.acc, train_step, write_op, step], feed)
        current_step = tf.train.global_step(sess, step)
        writer.add_summary(_summary, current_step)

        #print "loss part: ", _output_part
        print "loss: ", _loss, '\tacc: ', _acc
