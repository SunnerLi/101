from keras.layers import Dense, Input, Dropout
from keras.models import Model
import numpy as np

class Net(object):
    def __init__(self):
        input_layer = Input((2,))
        self.network = Dense(units=128, activation='elu')(input_layer)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=64)(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=32)(self.network)
        self.output = Dense(units=17, activation='sigmoid')(self.network)
        self.model = Model(inputs=input_layer, outputs=self.output)
        self.model.compile(
            loss='mse',
            optimizer='adam'
        )

    def fit(self, x, y):
        self.model.fit(x, y, epochs=200, batch_size=200)

    def predict(self, x):
        return self.model.predict(x)

#net = Net()