from utils import *
from model import DNN
import tensorflow as tf
import numpy as np

# Parameters
epoch_num = 200
batch_size = 10

model = DNN()
sess = tf.Session()
with sess.as_default():
    # (1)(2) Create optimizer and step tensor
    step = tf.Variable(0, False)
    optimizer = tf.train.AdamOptimizer(0.001)
    grad_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grad_vars, step)

    # (3)(4) Merge operators and create summary object
    loss = tf.scalar_summary('loss', model.loss)
    accu = tf.scalar_summary('acc', model.acc)
    write_op = tf.merge_summary([loss, accu])
    writer = tf.summary.FileWriter('./res', sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(epoch_num):
        loss_sum = 0
        acc_sum = 0
        for j in range(len(train_x)/batch_size):
            feed = {
                model.inputs: train_x[j*batch_size:j*batch_size+batch_size],
                model.tags: raiseDim(train_y[j*batch_size:j*batch_size+batch_size])
            }
            # (5)(6) Optimize weight and record the process
            _loss, _acc, _, _summary, _step = sess.run([model.loss, model.acc, train_op, write_op, step], feed)
            current_step = tf.train.global_step(sess, step)
            writer.add_summary(_summary, current_step)

            # The rest
            loss_sum += _loss
            acc_sum += _acc
        print "epoch: ", i, "\tloss: ", _loss, '\tacc: ', _acc

    # Evaluation
    print "< testing >"
    feed = {
        model.inputs: test_x,
        model.tags: raiseDim(test_y)
    }
    _loss, _acc = sess.run([model.loss, model.acc], feed)
    print "loss: ", _loss, '\tacc: ', _acc