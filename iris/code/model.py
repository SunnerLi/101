from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Sequential
from keras import backend as K
from keras import metrics
import numpy as np

class DNN(object):
    def __init__(self, class_num=3):
        self.model = Sequential()        
        self.model.add(Dense(64, activation='relu', input_shape=[4,]))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.compile(
            loss='mse', 
            optimizer='adam'   
        )

    def fit(self, x, y, epoch=5000):
        self.model.fit(x, y, batch_size=140, epochs=epoch, verbose=1)

    def predict(self, x):
        return self.model.predict(x)

    def close(self):
        K.clear_session()