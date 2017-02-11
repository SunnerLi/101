import tensorflow as tf
import model
import utils

train_epoch = 20    # Variable

imgs, tags = utils.load()
train_imgs, train_tags, test_imgs, test_tags = utils.tenCrossValid(imgs, tags, 0)
net = model.OXNet(30, 30)

sess = tf.Session()
with sess.as_default():
    sess.run(tf.initialize_all_variables())

    for i in range(train_epoch):
        _loss, _acc, _ = sess.run([net.loss, net.accuracy, net.optimizer], 
            feed_dict={ net.input: train_imgs, net.tag: utils.raiseDim(train_tags)})
        print 'epoch: ', i, '\tloss: ', _loss, '\tacc: ', _acc

    # Testing
    _loss, _acc, _ = sess.run([net.loss, net.accuracy, net.optimizer], \
        feed_dict={net.input: test_imgs, net.tag: utils.raiseDim(test_tags)})
    print 'eval ->> loss: ', _loss, '\tacc: ', _acc