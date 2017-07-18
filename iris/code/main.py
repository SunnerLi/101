from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model import DNN
import data_loader
import numpy as np
import gc

def show(title, tag_arr, predict_arr):
    """
        Show the predict result

        Arg:    title       - The string of title
                tag_arr     - The tag array
                predict_arr - The predict array
    """
    print(title)
    print('tag    : ', np.reshape(tag_arr, [-1]))
    print('predict: ', predict_arr)
    print('acc    : ', np.sum(np.equal(np.reshape(tag_arr, [-1]), predict_arr)) / len(predict_arr))

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_loader.load()

    # SVM model
    clf = SVC()
    clf.fit(train_x, np.reshape(train_y, [-1]))
    svm_predict_y = clf.predict(test_x)

    # Random forest model
    clf = RandomForestClassifier()
    clf.fit(train_x, np.reshape(train_y, [-1]))
    rf_predict_y = clf.predict(test_x)
    
    # DNN model
    one_hot_train_y = data_loader.oneHotEncode(train_y)
    clf = DNN(len(data_loader.id_2_spice))
    clf.fit(train_x, one_hot_train_y)
    dnn_predict_y = data_loader.oneHotDecode(np.round(clf.predict(test_x)))

    # Show the predict result
    show('<< SVM >>', test_y, svm_predict_y)
    show('<< RF >>', test_y, rf_predict_y)
    show('<< DNN >>', test_y, dnn_predict_y)

    # End
    clf.close()