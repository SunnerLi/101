from keras.layers import Dense, Input, Dropout
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
import numpy as np

"""
[(0.683, 0.667), (0.41, 0.556), (0.731, 0.778)]
[(0.669, 0.667), (0.703, 0.667), (0.699, 0.889)]
"""

class Net(object):
    def __init__(self):
        input_layer = Input((2,))

        # Define auto-encoder structure
        self.encode_layer = Dense(units=128, activation='elu')(input_layer)
        decode_layer = Dense(units=2, activation='elu')(self.encode_layer)
        self.encode_model = Model(inputs=input_layer, outputs=decode_layer)

        # Define DNN structure
        self.network = Dense(units=128, activation='elu')(self.encode_layer)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=128, activation='elu')(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.output = Dense(units=17, activation='sigmoid')(self.network)
        self.dnn_model = Model(inputs=input_layer, outputs=self.output)        

    def fit(self, x, y, epoch=2000, batch_size=32):
        # Train auto-encoder first
        self.encode_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.encode_model.fit(x, x, epochs=epoch, batch_size=200)

        # Train classification model
        for i in range(len(self.encode_model.layers)):
            self.dnn_model.layers[i].trainable = False
        self.dnn_model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.dnn_model.fit(x, y, epochs=epoch, batch_size=batch_size)

    def save(self, model_name='model.h5'):
        self.dnn_model.save(model_name)

    def load(self, model_name='model.h5'):
        self.dnn_model = load_model(model_name)

    def predict(self, x):
        return self.dnn_model.predict(x)

    def close(self):
        K.clear_session()