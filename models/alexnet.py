import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Add, Input, ZeroPadding2D, MaxPooling2D, Dropout, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

def AlexNet():
    input = Input(shape=(227, 227, 3))

    x = input

    x = Conv2D(96, (11, 11), strides=(4, 4), padding="valid", activation="relu", name="conv_1")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool_1")(x)

    x = Conv2D(256, (11, 11), padding="valid", activation="relu", name="conv_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool_2")(x)

    x = Conv2D(384, (3, 3), padding="valid", activation="relu", name="conv_3")(x)

    x = Conv2D(384, (3, 3), padding="valid", activation="relu", name="conv_4")(x)

    x = Conv2D(256, (3, 3), padding="valid", activation="relu", name="conv_5")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool_5")(x)

    x = Flatten()(x)

    x = Dense(4096, activation="relu", name="dense_1")(x)
    x = Dropout(0.4)(x)

    x = Dense(4096, activation="relu", name="dense_2")(x)
    x = Dropout(0.4)(x)

    x = Dense(1000, activation="relu", name="dense_3")(x)

    x = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(input, x)

    return model
