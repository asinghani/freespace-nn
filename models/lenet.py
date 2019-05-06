import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Add, Input, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

def LeNet():
    input = Input(shape=(32, 32, 3))

    x = input

    x = Conv2D(6, (3, 3), padding="valid", activation="relu", name="conv_1")(x)
    x = AveragePooling2D((2, 2), name="pool_1")(x)

    x = Conv2D(16, (3, 3), padding="valid", activation="relu", name="conv_2")(x)
    x = AveragePooling2D((2, 2), name="pool_2")(x)

    x = Flatten()(x)

    x = Dense(120, activation="relu", name="dense_1")(x)
    x = Dense(84, activation="relu", name="dense_2")(x)
    x = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(input, x)

    return model
