import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def BottleneckBlock(filters, rate, name, batchnorm=False, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: BottleneckBlock(filters, rate, name, batchnorm=batchnorm, input=x))

    x = input

    name = name + "_"

    # Expand
    x = Conv2D(6 * int(input.shape[-1]), kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand")(x)
    if batchnorm:
        x = BatchNormalization(name=name+"expand_BN", momentum=0.1)(x)
    x = ReLU(max_value=6.0, name=name+"expand_relu")(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding="same", dilation_rate=(rate, rate), name=name+"depthwise")(x)
    if batchnorm:
        x = BatchNormalization(name=name+"depthwise_BN", momentum=0.1)(x)
    x = ReLU(max_value=6.0, name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"project")(x)
    if batchnorm:
        x = BatchNormalization(name=name+"project_BN", momentum=0.1)(x)

    if filters == input.shape[-1]:
        x = Add(name=name+"add")([input, x])

    return x

def Upsampling4(init_weights=True, trainable=False, input=None):
    # Conform to functional API
    if input is None:
        return (lambda x: Upsampling4(init_weights=init_weights, trainable=trainable, input=x))

    def initializer(shape, dtype, **kwargs):
        grid = np.ogrid[:8, :8]
        k = (1 - abs(grid[0] - 3.5) / 4) * (1 - abs(grid[1] - 3.5) / 4)

        weights = np.zeros(shape, dtype=np.float32)
        for i in xrange(shape[-1]):
            weights[:, :, i, i] = k

        return K.backend.cast_to_floatx(weights)

    x = input
    if init_weights:
        x = Conv2DTranspose(int(input.shape[-1]), kernel_size=8, strides=4, padding="same", use_bias=False, kernel_initializer=initializer, trainable=trainable)(x)
    else:
        x = Conv2DTranspose(int(input.shape[-1]), kernel_size=8, strides=4, padding="same", use_bias=False)(x)

    return x

def FloorNet(config, load_enc_weights=True, batchnorm=False, aux=True, pyramid=True, upsampling_trainable=True, upsampling_init=True):
    input = Input(shape=config.input_shape)

    mobilenet = MobileNetV2(input_tensor=input, alpha=1.0, include_top=False, weights=("imagenet" if load_enc_weights else None), pooling=None)

    # Don't train BN layers
    for layer in mobilenet.layers:
        if ("_BN" in layer.name) or ("_bn" in layer.name) or ("bn_" in layer.name):
            #layer.trainable = True
            pass

    x = mobilenet.get_layer("block_12_add").output

    x = BottleneckBlock(filters=160, rate=1, name="enc_block_1", batchnorm=batchnorm)(x)
    x = BottleneckBlock(filters=160, rate=1, name="enc_block_2", batchnorm=batchnorm)(x)

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="out__conv")(x)

    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b1_conv")(x)
    if batchnorm:
        b1 = BatchNormalization(epsilon=1e-3, momentum=0.1)(b1)
    b1 = Activation("relu", name="b1_relu")(b1)

    b2 = AveragePooling2D(pool_size=(int(np.ceil(config.input_shape[0] / 16)), int(np.ceil(config.input_shape[1] / 16))), name="b2_pool")(x)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = Activation("relu", name="b2_relu")(b2)
    b2 = BilinearUpsampling((int(np.ceil(config.input_shape[0] / 16)), int(np.ceil(config.input_shape[1] / 16))), name="b2_upsampling")(b2)


    if pyramid:
        b3 = Conv2D(256, (3, 3), dilation_rate=1, padding="same", use_bias=False, name="b3_conv")(x)
        b3 = Activation("relu", name="b3_relu")(b3)

        b4 = Conv2D(256, (3, 3), dilation_rate=2, padding="same", use_bias=False, name="b4_conv")(x)
        b4 = Activation("relu", name="b4_relu")(b4)

        b5 = Conv2D(256, (3, 3), dilation_rate=4, padding="same", use_bias=False, name="b5_conv")(x)
        b5 = Activation("relu", name="b5_relu")(b5)

        x = Concatenate(name="concat")([b1, b3, b4, b5])
    else:
        x = b1

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="decoder_conv")(x)
    if batchnorm:
        x = BatchNormalization(epsilon=1e-3, momentum=0.1)(x)
    x = Activation("relu", name="decoder_relu")(x)

    x = Upsampling4(trainable=upsampling_trainable, init_weights=upsampling_init)(x)

    input2 = Input(shape=(config.input_shape[0] // 2, config.input_shape[0] // 2, 3))
    if aux:
        f = input2
        f = Conv2D(128, (3, 3), strides=2, padding="same", use_bias=False, name="feature_conv")(f)
        f = Activation("relu", name="feature_relu")(f)
        f = Conv2D(128, (1, 1), strides=1, padding="same", use_bias=False, name="feature_conv2")(f)
        f = Activation("relu", name="feature_relu2")(f)
        x = Concatenate()([f, x])

    x = Conv2D(256, (3, 3), padding="same", use_bias=False, name="decoder_conv2")(x)
    if batchnorm:
        x = BatchNormalization(epsilon=1e-3, momentum=0.1)(x)
    x = Activation("relu", name="decoder_relu2")(x)

    x = Conv2D(2, (1, 1), padding="same", name="logits")(x) # Logits layer
    x = Upsampling4(trainable=upsampling_trainable, init_weights=upsampling_init)(x)


    if config.include_softmax:
        x = Activation("softmax", name="decoder_softmax")(x)

    model = Model([input, input2], x)

    return model
