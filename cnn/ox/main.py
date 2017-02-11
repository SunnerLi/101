import tensorflow as tf
import model
import utils

imgs, tags = utils.load()
train_imgs, train_tags, test_imgs, test_tags = utils.tenCrossValid(imgs, tags, 0)
net = model.OXNet(30, 30)

sess = tf.Session()
with sess.as_default():
    sess.run(tf.initialize_all_variables())

    for i in range(20):
        loss_sum = 0
        acc_sum = 0
        for j in range(len(train_imgs)):
            feed_dict = {
                net.input: [train_imgs[j]],
                net.tag: utils.raiseDim(train_tags[j])
            }
            _loss, _acc, _ = sess.run([net.loss, net.accuracy, net.optimizer], feed_dict=feed_dict)
            loss_sum += _loss
            acc_sum = acc_sum + _acc / len(train_imgs)
        print 'epoch: ', i, '\tloss: ', loss_sum, '\tacc: ', acc_sum

    # Testing
    loss_sum = 0
    acc_sum = 0
    for j in range(len(test_imgs)):
        feed_dict = {
            net.input: [test_imgs[j]],
            net.tag: utils.raiseDim(test_tags[j])
        }
        _loss, _acc, _ = sess.run([net.loss, net.accuracy, net.optimizer], feed_dict=feed_dict)
        loss_sum += _loss
        acc_sum = acc_sum + _acc / len(test_imgs)
    print 'eval ->> loss: ', loss_sum, '\tacc: ', acc_sum