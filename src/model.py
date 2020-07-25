from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, LSTMCell, Bidirectional, RNN


class MyModel(Model):
    # model constants
    img_width = 128
    img_height = 32
    batch_size = 50
    max_text_len = 32
    num_chars = 80

    def __init__(self):
        super(MyModel, self).__init__()

        # define all the layers and input

        self.conv2d = Conv2D(input_shape=(MyModel.img_width, MyModel.img_height, 1), padding='same', strides=(1, 1),
                             filters=32,
                             kernel_size=(5, 5), activation='relu')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.batch_norm = BatchNormalization()

        self.conv2d_1 = Conv2D(padding='same', strides=(1, 1), filters=64, kernel_size=(5, 5),
                               activation='relu')
        self.max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.batch_norm_1 = BatchNormalization()

        self.conv2d_2 = Conv2D(padding='same', strides=(1, 1), filters=128, kernel_size=(3, 3), activation='relu')
        self.max_pool_2 = MaxPool2D(pool_size=(1, 5), strides=(1, 1))
        self.batch_norm_2 = BatchNormalization()

        self.conv2d_3 = Conv2D(padding='same', strides=(1, 1), filters=128, kernel_size=(3, 3), activation='relu')
        self.max_pool_3 = MaxPool2D(pool_size=(1, 3), strides=(1, 1))
        self.batch_norm_3 = BatchNormalization()

        self.conv2d_4 = Conv2D(padding='same', strides=(1, 1), filters=256, kernel_size=(3, 3), activation='relu')
        self.max_pool_4 = MaxPool2D(pool_size=(1, 2), strides=(1, 1))
        self.batch_norm_4 = BatchNormalization()

        self.rnn = Bidirectional(RNN([
            LSTMCell(256),
            LSTMCell(256),
        ], return_sequences=True))

        self.atrous_conv2d = Conv2D(padding='same', kernel_size=(1, 1), filters=MyModel.num_chars)

    def call(self, t):
        t = self.conv2d(t)
        t = self.max_pool(t)
        t = self.batch_norm(t)

        t = self.conv2d_1(t)
        t = self.max_pool_1(t)
        t = self.batch_norm_1(t)

        t = self.conv2d_2(t)
        t = self.max_pool_2(t)
        t = self.batch_norm_2(t)

        t = self.conv2d_3(t)
        t = self.max_pool_3(t)
        t = self.batch_norm_3(t)

        t = self.conv2d_4(t)
        t = self.max_pool_4(t)
        t = self.batch_norm_4(t)

        t = tf.squeeze(t, axis=[2])
        t = self.rnn(t)
        t = tf.expand_dims(t, 2)
        t = self.atrous_conv2d(t)
        t = tf.squeeze(t, axis=[2])
        t = tf.transpose(t, [1, 0, 2])  # time, batch, sequence

        return t
