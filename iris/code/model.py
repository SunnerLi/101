from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.models import Sequential
from keras import backend as K
import numpy as np

class DNN(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dropout(0.5, input_shape=[4,]))
        self.model.add(Dense(32))
        self.model.add(LeakyReLU())
        self.model.add(Dense(16))
        self.model.add(LeakyReLU())
        self.model.add(Dense(8))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=16, epochs=2000, verbose=1)

    def predict(self, x):
        return self.model.predict(x)