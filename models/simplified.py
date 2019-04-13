import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, Cropping2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Lambda, Concatenate, ReLU, DepthwiseConv2D, Conv2DTranspose, ELU
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv

def SpatialPoolingBlock(dilation_rate, name, bn_epsilon=1e-3, bn_momentum=0.999, dilation_replace=False, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: SpatialPoolingBlock(dilation_rate=dilation_rate, name=name, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, dilation_replace=dilation_replace, input=x))

    x = input
    name = name + "_"

    x = Conv2D(16, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand")(x)
    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"expand_BN")(x)
    x = ReLU(max_value=6.0, name=name+"expand_relu")(x)

    # Depthwise
    if dilation_replace: # Replace with reduced rate
        x = DepthwiseConv2D(kernel_size=(3 + 2 * (rate - 1)), strides=1, activation=None, use_bias=False, padding="same", dilation_rate=(1, 1), name=name+"depthwise")(x)
    else:
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding="same", dilation_rate=(dilation_rate, dilation_rate), name=name+"depthwise")(x)

    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"depthwise_BN")(x)
    x = ReLU(max_value=6.0, name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(16, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"pointwise")(x)
    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"pointwise_BN")(x)

    return Add(name=name+"add")([input, x])

def EncoderBlock(filters, stride, dilation_rate, name, bn_epsilon=1e-3, bn_momentum=0.999, dilation_replace=False, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: EncoderBlock(filters=filters, stride=stride, dilation_rate=dilation_rate, name=name, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, dilation_replace=dilation_replace, input=x))

    x = input
    name = name + "_"

    x = Conv2D(int(input.shape[-1]) * 6, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand")(x)
    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"expand_BN")(x)
    x = ELU(name=name+"expand_relu")(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding="same", dilation_rate=(dilation_rate, dilation_rate), name=name+"depthwise")(x)

    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"depthwise_BN")(x)
    x = ELU(name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"pointwise")(x)
    #x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"pointwise_BN")(x)

    if stride == 1 and filters == input.shape[-1]:
        return Add(name=name+"add")([input, x]) # Residual
    else:
        return x

def SimpleFloorNet(config, tx2_gpu=False, input_layer=False, load_enc_weights=None):
    input = Input(shape=config.input_shape, name="input")
    x = input

    x = Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name="Conv1")(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv1_BN")(x)
    x = ELU(name="Conv1_Relu6")(x)

    rates = [1, 2, 3, 4, 6, 8, 12, 16]

    x = Concatenate(name="spatial_concat")([SpatialPoolingBlock(dilation_rate=rate, name="spatial"+str(rate))(x) for rate in rates])

    x = EncoderBlock(filters=128, stride=1, dilation_rate=1, name="encoder_block1")(x)
    x = Dropout(0.1)(x)

    x = EncoderBlock(filters=64, stride=1, dilation_rate=1, name="encoder_block2")(x)
    x = Dropout(0.1)(x)

    x = EncoderBlock(filters=32, stride=1, dilation_rate=1, name="encoder_block3")(x)
    x = Dropout(0.1)(x)

    x = Conv2D(16, (1, 1), padding="same", name="downsample")(x)
    x = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), use_bias = False, padding="same", name="logits")(x)

    x = Activation("softmax", name="decoder_softmax")(x)

    model = Model(input, x)

    if input_layer:
        return model, input
    else:
        return model
