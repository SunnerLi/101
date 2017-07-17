from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model import DNN
import data_loader
import numpy as np

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_loader.load()

    """
    # SVM model
    clf = SVC()
    clf.fit(train_x, np.reshape(train_y, [-1]))
    y_ = clf.predict(test_x)
    print('<< SVM >>')
    print('tag    : ', np.reshape(test_y, [-1]))
    print('predict: ', y_)
    print('acc    : ', np.sum(np.equal(np.reshape(test_y, [-1]), y_)) / len(y_))

    # Random forest model
    clf = RandomForestClassifier()
    clf.fit(train_x, np.reshape(train_y, [-1]))
    y_ = clf.predict(test_x)
    print('<< RF >>')
    print('tag    : ', np.reshape(test_y, [-1]))
    print('predict: ', y_)
    print('acc    : ', np.sum(np.equal(np.reshape(test_y, [-1]), y_)) / len(y_))
    """

    clf = DNN()
    clf.fit(train_x, train_y)
    y_ = np.round(clf.predict(test_x))

    
    print('<< DNN >>')
    print('tag    : ', np.reshape(test_y, [-1]))
    print('predict: ', np.reshape(y_, [-1]))
    print('acc    : ', np.sum(np.equal(np.reshape(test_y, [-1]), y_)) / len(y_))