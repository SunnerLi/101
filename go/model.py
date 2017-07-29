from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import metrics
import numpy as np

class Net(object):
    def __init__(self):
        input_layer = Input((2,))

        # Define auto-encoder structure
        self.encode_layer = Dense(units=128, activation='elu')(input_layer)
        decode_layer = Dense(units=2, activation='elu')(self.encode_layer)
        self.encode_model = Model(inputs=input_layer, outputs=decode_layer)

        # Define DNN structure
        
        self.network = Dense(units=256, activation='elu')(self.encode_layer)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=128, activation='elu')(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.output = Dense(units=17, activation='sigmoid')(self.network)
        self.dnn_model = Model(inputs=input_layer, outputs=self.output)
        """
        self.network = Dense(units=128, activation='elu')(self.encode_layer)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=128, activation='elu')(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.output = Dense(units=17, activation='softmax')(self.network)
        self.dnn_model = Model(inputs=input_layer, outputs=self.output)
        """
        

    def fit(self, x, y):
        # Train auto-encoder first
        self.encode_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.encode_model.fit(x, x, epochs=2000, batch_size=200)

        # Train classification model
        for i in range(len(self.encode_model.layers)):
            self.dnn_model.layers[i].trainable = False
        self.dnn_model.compile(
            loss='categorical_crossentropy',    # categorical_crossentropy
            optimizer='adam',
            metrics=['accuracy']
        )
        #self.dnn_model.fit(x, y, epochs=5000, batch_size=32)
        self.dnn_model.fit(x, y, epochs=1000, batch_size=256)

    def predict(self, x):
        return self.dnn_model.predict(x)

#net = Net()