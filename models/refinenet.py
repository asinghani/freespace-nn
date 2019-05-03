import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def RefineNet(image_shape):
    input = Input(shape=(image_shape[0], image_shape[1], 7)) # scaled up seg, scaled up edges, source edges

    x = input

    x = Conv2D(128, (7, 7), padding="same", use_bias=True, name="conv_1")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_1_BN")(x)

    x = Conv2D(128, (5, 5), padding="same", use_bias=True, name="conv_2")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_2_BN")(x)

    x = Conv2D(128, (3, 3), padding="same", use_bias=True, name="conv_3")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_3_BN")(x)

    x = Conv2D(128, (1, 1), padding="same", use_bias=True, name="conv_4")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_4_BN")(x)

    x = Conv2D(2, (1, 1), padding="same", use_bias=True, name="logits")(x)
    x = Activation("softmax", name="logits_softmax")(x)

    model = Model(input, x)

    return model
