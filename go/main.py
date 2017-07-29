from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model import Net
import data_helper
import numpy as np

train_x, test_x, train_y, test_y = data_helper.load()

print(np.shape(train_x), np.shape(train_y))
print(train_x[0])
print(train_y[0])
"""
clf = RandomForestClassifier(verbose=0)
clf.fit(train_x, list(train_y))

"""

_error_list = []
for i in range(3):
    clf = Net()
    clf.fit(train_x, data_helper.oneHotEncode(train_y))

    print('<< train result >>')
    predict_output = data_helper.oneHotDecode(clf.predict(train_x))
    print('err     : ', data_helper.errorSum(train_y, predict_output))
    print('')

    print('<< predict >>')
    print('tag     : ', test_y)
    predict_output = data_helper.oneHotDecode(clf.predict(test_x))
    print(clf.predict(test_x))
    print('predict : ', predict_output)
    print('err     : ', data_helper.errorSum(test_y, predict_output))

    _error_list.append((
        round(data_helper.errorSum(train_y, data_helper.oneHotDecode(clf.predict(train_x))), 3), 
        round(data_helper.errorSum(test_y, data_helper.oneHotDecode(clf.predict(test_x))), 3)
    ))
print(_error_list)