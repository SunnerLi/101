from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import Model
import data_helper
import numpy as np

class Net():
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        self.model.fit(x, y, epochs=1, batch_size=256)

    def predict(self, x):
        return self.model.predict(x)

class DNNNet(Net):
    def __init__(self):
        input_layer = Input((784,))
        self.network = Dense(units=256, activation='relu')(input_layer)
        self.network = Dense(units=64, activation='relu')(self.network)
        self.output = Dense(units=10, activation='softmax')(self.network)
        self.model = Model(inputs=input_layer, outputs=self.output)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

class CNNNet(Net):
    def __init__(self):
        input_layer = Input((28, 28, 1))

        self.network = Conv2D(8, (2, 2), activation='relu')(input_layer)
        self.network = Conv2D(8, (2, 2), activation='relu')(input_layer)
        self.network = MaxPool2D()(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Conv2D(16, (2, 2), activation='relu')(self.network)
        self.network = Conv2D(16, (2, 2), activation='relu')(self.network)
        self.network = MaxPool2D()(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Flatten()(self.network)
        self.network = Dense(units=32, activation='relu')(self.network)
        self.output = Dense(units=10, activation='softmax')(self.network)
        self.model = Model(inputs=input_layer, outputs=self.output)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

train_x, test_x, train_y, test_y = data_helper.load()
train_x = np.reshape(train_x, [-1, 28, 28, 1])
train_y = to_categorical(train_y)

print(np.shape(test_x))
test_x = np.reshape(test_x, [-1, 28, 28, 1])

net = CNNNet()
net.fit(train_x, train_y)
print('test acc: ', np.sum(np.equal(test_y, np.argmax(net.predict(test_x)))))