import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, Cropping2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Lambda, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv

def SpatialPoolingBlock(dilation_rate, name, bn_epsilon=1e-3, bn_momentum=0.999, dilation_replace=False, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: SpatialPoolingBlock(dilation_rate=dilation_rate, name=name, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, dilation_replace=dilation_replace, input=x))

    x = input
    name = name + "_"

    x = Conv2D(16, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand")(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"expand_BN")(x)
    x = ReLU(max_value=6.0, name=name+"expand_relu")(x)

    # Depthwise
    if dilation_replace: # Replace with reduced rate
        x = DepthwiseConv2D(kernel_size=(3 + 2 * (rate - 1)), strides=1, activation=None, use_bias=False, padding="same", dilation_rate=(1, 1), name=name+"depthwise")(x)
    else:
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding="same", dilation_rate=(dilation_rate, dilation_rate), name=name+"depthwise")(x)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"depthwise_BN")(x)
    x = ReLU(max_value=6.0, name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(8, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"pointwise")(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"pointwise_BN")(x)

    return x

def EncoderBlock(filters, stride, dilation_rate, name, bn_epsilon=1e-3, bn_momentum=0.999, dilation_replace=False, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: EncoderBlock(filters=filters, stride=stride, dilation_rate=dilation_rate, name=name, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, dilation_replace=dilation_replace, input=x))

    x = input
    name = name + "_"

    x = Conv2D(int(input.shape[-1]) * 6, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand")(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"expand_BN")(x)
    x = ReLU(max_value=6.0, name=name+"expand_relu")(x)

    # Depthwise
    if dilation_replace: # Replace with reduced rate
        x = DepthwiseConv2D(kernel_size=(3 + 2 * (rate - 1)), strides=stride, activation=None, use_bias=False, padding="same", dilation_rate=(1, 1), name=name+"depthwise")(x)
    else:
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding="same", dilation_rate=(dilation_rate, dilation_rate), name=name+"depthwise")(x)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"depthwise_BN")(x)
    x = ReLU(max_value=6.0, name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"pointwise")(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"pointwise_BN")(x)

    if stride == 1 and filters == input.shape[-1]:
        return Add(name=name+"add")([input, x]) # Residual
    else:
        return x

def FloorNet(config, tx2_gpu=False, input_layer=False, load_enc_weights=None):
    input = Input(shape=config.input_shape, name="input")
    x = input

    x = Conv2D(32, kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name="Conv1")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="Conv1_BN")(x)
    x = ReLU(max_value=6.0, name="Conv1_Relu6")(x)

    # Spatial pyramid
    s1 = Conv2D(8, kernel_size=1, padding="same", use_bias=False, activation=None, name="spatial1_conv")(x)
    s1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name="spatial1_BN")(s1)
    s1 = ReLU(max_value=6.0, name="spatial1_relu")(s1)

    s2 = SpatialPoolingBlock(dilation_rate=2, name="spatial2")(x)
    s3 = SpatialPoolingBlock(dilation_rate=4, name="spatial3")(x)
    s4 = SpatialPoolingBlock(dilation_rate=6, name="spatial4")(x)
    s5 = SpatialPoolingBlock(dilation_rate=8, name="spatial5")(x)

    x = Concatenate(name="spatial_concat")([s1, s2, s3, s4, s5])

    x = EncoderBlock(filters=24, stride=2, dilation_rate=1, name="encoder_block1")(x)
    x = EncoderBlock(filters=24, stride=1, dilation_rate=2, name="encoder_block2")(x)
    x = Dropout(0.3)(x)

    x = EncoderBlock(filters=64, stride=2, dilation_rate=1, name="encoder_block3")(x)
    x = Dropout(0.3)(x)

    x = EncoderBlock(filters=96, stride=1, dilation_rate=2, name="encoder_block4")(x)
    x = EncoderBlock(filters=96, stride=1, dilation_rate=2, name="encoder_block5")(x)

    x = EncoderBlock(filters=160, stride=1, dilation_rate=2, name="encoder_block6")(x)
    x = EncoderBlock(filters=160, stride=1, dilation_rate=2, name="encoder_block7")(x)

    x = EncoderBlock(filters=320, stride=1, dilation_rate=4, name="encoder_block8")(x)

    if load_enc_weights is not None:
        model = Model(input, x)
        model.load_weights(load_enc_weights, by_name=True)

    OS = 8 # Smaller output stride = less params

    # Branch 1 (1x1 Conv with BatchNorm and ReLU)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b1_conv")(x)
    b1 = BatchNormalization(epsilon=1e-5, name="b1_BN")(b1)
    b1 = Activation("relu", name="b1_relu")(b1)

    # Branch 2 (Feature branch)
    b2 = AveragePooling2D(pool_size=(int(np.ceil(config.input_shape[0] / OS)), int(np.ceil(config.input_shape[1] / OS))), name="b2_pool")(x)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = BatchNormalization(epsilon=1e-5, name="b2_BN")(b2)
    b2 = Activation("relu", name="b2_relu")(b2)
    b2 = BilinearUpsampling((int(np.ceil(config.input_shape[0] / OS)), int(np.ceil(config.input_shape[1] / OS))), name="b2_upsampling")(b2)


    x = Concatenate(name="decoder_concat")([b1, b2])

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="decoder_conv")(x)
    x = BatchNormalization(epsilon=1e-5, name="decoder_BN")(x)
    x = Activation("relu", name="decoder_relu")(x)
    x = Dropout(0.1, name="decoder_dropout")(x)

    x = Conv2D(2, (1, 1), padding="same", name="decoder_conv2")(x) # Logits layer
    x = BilinearResize((config.input_shape[0], config.input_shape[1]), name="decoder_resize")(x)

    #x = Activation("softmax", name="decoder_softmax")(x)

    model = Model(input, x)

    if input_layer:
        return model, input
    else:
        return model
