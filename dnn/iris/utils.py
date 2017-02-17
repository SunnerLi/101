from sklearn import cross_validation
from sklearn import datasets
import numpy as np

def raiseDim(_arr):
    _res = np.zeros([len(_arr), 3], dtype=np.float)
    for i in range(len(_arr)):
        _res[i][int(_arr[i])] = 1
    return _res

data = datasets.load_iris()
x = data['data'].astype(np.float32)
y = data['target'].astype(np.float32)
train_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y, test_size=0.1)