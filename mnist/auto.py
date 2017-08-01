from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from keras.models import Model, load_model
import data_helper
import numpy as np
import cv2
import os

model_path = 'model.h5'

class CNNAutoEncoder(object):
    def __init__(self):
        input_layer = Input((28, 28, 1))

        # Down
        self.network = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        self.network = MaxPool2D(padding='same')(self.network)
        self.network = Conv2D(8, (3, 3), activation='relu', padding='same')(self.network)
        self.network = MaxPool2D(padding='same')(self.network)
        self.network = Conv2D(8, (3, 3), activation='relu', padding='same')(self.network)
        encoded_layer = MaxPool2D(padding='same')(self.network)

        # Up
        self.network = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_layer)
        self.network = UpSampling2D()(self.network)
        self.network = Conv2D(8, (3, 3), activation='relu', padding='same')(self.network)
        self.network = UpSampling2D()(self.network)
        self.network = Conv2D(16, (3, 3), activation='relu')(self.network)
        self.network = UpSampling2D()(self.network)
        output_layer = Conv2D(1, (3, 3), activation='relu', padding='same')(self.network)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            loss='mse',
            optimizer='adam'
        )

    def fit(self, x):
        self.model.fit(x, x, epochs=100, batch_size=256)
        
    def save(self):
        self.model.save(model_path)

    def load(self):
        self.model = load_model(model_path)

    def predict(self, x):
        return self.model.predict(x)

if __name__ == '__main__':
    # Train
    x, y = data_helper.load()
    x = np.reshape(x, [-1, 28, 28, 1])
    net = CNNAutoEncoder()
    if not os.path.exists(model_path):
        net.fit(x)
        net.save()

    # Test
    net.load()
    predict_y = net.predict(np.asarray(x[:10]))

    # Show
    show_x = np.concatenate((x[0], x[1]), axis=1)
    for i in range(2, 10):
        show_x = np.concatenate((show_x, x[i]), axis=1)
    show_y = np.concatenate((predict_y[0], predict_y[1]), axis=1)
    for i in range(2, 10):
        show_y = np.concatenate((show_y, predict_y[i]), axis=1)
    show_img = np.concatenate((show_x, show_y))
    cv2.imshow('result', show_img)
    cv2.waitKey()