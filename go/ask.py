from model import Net
import data_helper
import numpy as np

# Load data
data_frame = data_helper.load()
train_x, test_x, train_y, test_y = data_helper.splitData(data_frame)

# Ask
clf = Net()
clf.load()
print('\n\n\n')
while True:
    _string = input('Enter CP and HP in order: ')
    if _string == 'd':
        break
    _cp, _hp = _string.split(' ')[0], _string.split(' ')[1]
    input_arr = data_helper.newDataPreprocess(np.asarray([[_cp, _hp]]))
    predict_result = data_helper.oneHotDecode(clf.predict(input_arr))
    print('result: ', data_helper.toTypeName(predict_result[0]))