from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

x, y = datasets.load_iris()['data'], datasets.load_iris()['target']
train_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y, test_size=0.1)
model = svm.NuSVC()
model.fit(x, y)

# Get the accuracy of the training data
acc_sum = 0.0
for i in range(len(train_x)):
    if train_y[i] == model.predict(train_x[i])[0]:
        acc_sum += 1
print "train acc: ", acc_sum / len(train_x)

# Get the accuracy of the testing data
acc_sum = 0.0
for i in range(len(test_x)):
    if test_y[i] == model.predict(test_x[i])[0]:
        acc_sum += 1
print "test acc: ", acc_sum / len(test_x)