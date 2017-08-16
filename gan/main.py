from keras.layers import Dense, Conv2D, UpSampling2D, MaxPool2D, Reshape, Input, Activation, Flatten, BatchNormalization
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.optimizers import SGD
from PIL import Image
import numpy as np
import math

# The number of image that you want to generate
generate_size = 20

class DCGAN():
    def __init__(self):
        self.Generator()
        self.Discriminator()

    def Generator(self):
        input_layer = Input((100,))
        generator = Dense(units=1024)(input_layer)
        generator = Activation('tanh')(generator)
        generator = Dense(units=7*7*128)(generator)
        generator = BatchNormalization()(generator)
        generator = Activation('tanh')(generator)
        generator = Reshape((7, 7, 128))(generator)
        generator = UpSampling2D()(generator)
        generator = Conv2D(64, (5, 5), padding='same')(generator)
        generator = Activation('tanh')(generator)
        generator = UpSampling2D()(generator)
        generator = Conv2D(1, (5, 5), padding='same')(generator)
        output_layer = Activation('tanh')(generator)
        return Model(inputs=input_layer, outputs=output_layer)

    def Discriminator(self):
        input_layer = Input((28, 28, 1))
        discriminator = Conv2D(64, (5, 5), padding='same')(input_layer)
        discriminator = Activation('tanh')(discriminator)
        discriminator = MaxPool2D()(discriminator)
        discriminator = Conv2D(128, (5, 5), padding='same')(discriminator)
        discriminator = Activation('tanh')(discriminator)
        discriminator = MaxPool2D()(discriminator)
        discriminator = Flatten()(discriminator)
        discriminator = Dense(units=1024)(discriminator)
        discriminator = Activation('tanh')(discriminator)
        discriminator = Dense(units=1)(discriminator)
        output_layer = Activation('sigmoid')(discriminator)
        return Model(inputs=input_layer, outputs=output_layer)

    def fit(self, x, batch_size=32, epoch=1):
        global generate_size

        # Form the structure
        self.generator = self.Generator()
        discriminator = self.Discriminator()
        training_generator = Sequential()
        training_generator.add(self.generator)
        discriminator.trainable = False
        training_generator.add(discriminator)

        # Compile
        generator_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        discriminator_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(
            optimizer="SGD",
            loss="binary_crossentropy"
        )
        training_generator.compile(
            optimizer=generator_optimizer,
            loss="binary_crossentropy"
        )
        discriminator.trainable = True
        discriminator.compile(
            optimizer=discriminator_optimizer,
            loss='binary_crossentropy'
        )

        # Train
        for i in range(epoch):
            batch_num = int(math.ceil(len(x) / batch_size))
            print(batch_num)
            for j in range(batch_num):
                # Get actural image
                _actural_x = x[j*batch_size: j*batch_size + batch_size, :, :, :]

                # Get fake image
                _z_vector = np.random.uniform(-1, 1, [batch_size, 100])
                _fake_image = self.generator.predict(_z_vector)

                # Form as mini-batch data
                _x = np.concatenate((_actural_x, _fake_image))
                _y = [1] * batch_size + [0] * batch_size

                # Alternative training
                discriminator_loss = discriminator.train_on_batch(_x, _y)
                discriminator.trainable = False
                generator_loss = training_generator.train_on_batch(_z_vector, [1] * batch_size)
                discriminator.trainable = True
                print('batch index: ', j, '\tgenerator loss: ', generator_loss, '\tdiscriminator loss: ', discriminator_loss)

                # Save generate result
                _z_vector = np.random.uniform(-1, 1, [generate_size, 100])
                _fake_image = self.generator.predict(_z_vector)
                if j % 50 == 0:
                    save_img = self.mergeBatchImg(_fake_image)
                    Image.fromarray(save_img.astype(np.uint8)).save(str(i) + '_' + str(j) + '.png')


    def generate(self, _size=32):
        return self.generator.predict(np.random.random([_size, 100]))

    def mergeBatchImg(self, imgs):
        imgs = imgs * 127.5 + 127.5
        res = imgs[0, :, :, 0]
        for i in range(1, len(imgs)):
            res = np.concatenate((res, imgs[i, :, :, 0]))
        return res

    def generateAndShow(self, _size=32):
        generate_imgs = self.generate(_size)
        res = self.mergeBatchImg(generate_imgs)
        plt.imshow(res, cmap='gray')
        plt.show()


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = (np.reshape(train_x, [60000, 28, 28, 1]) - 127.5) / 127.5
    net = DCGAN()
    net.fit(train_x, epoch=2)
    net.generateAndShow(_size=generate_size)