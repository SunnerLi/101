from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model import Net
import data_helper
import numpy as np

# -----------------------
# Random forest model
# -----------------------

# Load data
data_frame = data_helper.load()
merge_frame = data_helper.mergeMultipleTypes(data_frame)
train_x, test_x, train_y, test_y = data_helper.splitData(merge_frame)

# Train
clf = RandomForestClassifier(verbose=0)
clf.fit(train_x, list(np.reshape(train_y, [-1])))

# Test
print('<< predict >>')
#print('tag       : ', np.reshape(test_y, [-1]))
predict_output = clf.predict(test_x)
#print('predict   : ', predict_output)
print('match rate: ', data_helper.matchRate(np.reshape(test_y, [-1]), predict_output))


# -----------------------
# DNN model
# -----------------------

# Load data
train_x, test_x, train_y, test_y = data_helper.splitData(data_frame)

# Train
clf = Net()
clf.fit(train_x, data_helper.oneHotEncode(train_y))
print('<< train result >>')
predict_output = data_helper.oneHotDecode(clf.predict(train_x))
print('match rate: ', data_helper.matchRate(train_y, predict_output))
print('')

# Test
print('<< predict >>')
#print('tag       : ', test_y)
predict_output = data_helper.oneHotDecode(clf.predict(test_x))
#print('predict   : ', predict_output)
print('match rate: ', data_helper.matchRate(test_y, predict_output))
clf.save()
clf.close()