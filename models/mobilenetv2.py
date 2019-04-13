import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dropout, Add, Input, Cropping2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Lambda, Concatenate, ReLU
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv

# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _mobilenet_block(expansion, stride, alpha, filters, block_num, res_connection, rate, bn_epsilon=1e-3, bn_momentum=0.999, freeze=False, tx2_gpu=False, input=None):
    """
    Mobilenet block based on arXiv:1801.04381 [cs.CV] with added support for custom dilation rate
    """

    trainable = not freeze

    # Conform to functional API
    if input is None:
        return (lambda x: _mobilenet_block(expansion, stride, alpha, filters, block_num, res_connection, rate, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, freeze=freeze, tx2_gpu=tx2_gpu, input=x))

    pointwise_filters = _make_divisible(int(filters * alpha), 8)

    x = input

    if block_num != 0:
        name = "expanded_conv_{}_".format(block_num)

        # Expand
        x = Conv2D(expansion * int(input.shape[-1]), kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"expand", trainable=trainable)(x)
        x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"expand_BN", trainable=False)(x)
        x = ReLU(max_value=6.0, name=name+"expand_relu")(x)
    else:
        name = "expanded_conv_"

    # Depthwise
    if tx2_gpu: # Replace with reduced rate
        x = DepthwiseConv2D(kernel_size=(3 + 2 * (rate - 1)), strides=stride, activation=None, use_bias=False, padding="same", dilation_rate=(1, 1), name=name+"depthwise", trainable=trainable)(x)
    else:
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding="same", dilation_rate=(rate, rate), name=name+"depthwise", trainable=trainable)(x)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"depthwise_BN", trainable=False)(x)
    x = ReLU(max_value=6.0, name=name+"depthwise_relu")(x)

    # Pointwise
    x = Conv2D(pointwise_filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=name+"project", trainable=trainable)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name=name+"project_BN", trainable=False)(x)

    # Residual/skip connection
    if res_connection:
        return Add(name=name+"add")([input, x])
    else:
        return x

def mobilenetV2(input, input_shape, alpha, bn_epsilon=1e-3, bn_momentum=1e-5, freeze_first_n=-1, load_weights=None, tx2_gpu=False):
    if alpha <= 0 or alpha % 0.25 != 0:
        raise Exception("MobileNet only supports positive alpha values divisible by 0.25")

    x = input

    x = Conv2D(int(32 * alpha), kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name="Conv", trainable=(not (0 <= freeze_first_n)))(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum, name="Conv_BN", trainable=False)(x)
    x = ReLU(max_value=6.0, name="Conv_Relu6")(x)

    x = _mobilenet_block(filters=16, alpha=alpha, stride=1, expansion=1, block_num=0, rate=1, res_connection=False, freeze=(0 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    x = _mobilenet_block(filters=24, alpha=alpha, stride=2, expansion=6, block_num=1, rate=1, res_connection=False, freeze=(1 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=24, alpha=alpha, stride=1, expansion=6, block_num=2, rate=1, res_connection=True, freeze=(2 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    x = _mobilenet_block(filters=32, alpha=alpha, stride=2, expansion=6, block_num=3, rate=1, res_connection=False, freeze=(3 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=32, alpha=alpha, stride=1, expansion=6, block_num=4, rate=1, res_connection=True, freeze=(4 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=32, alpha=alpha, stride=1, expansion=6, block_num=5, rate=1, res_connection=True, freeze=(5 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    # Stride and rate changed from original to have higher resolution
    x = _mobilenet_block(filters=64, alpha=alpha, stride=1, expansion=6, block_num=6, rate=1, res_connection=False, freeze=(6 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=64, alpha=alpha, stride=1, expansion=6, block_num=7, rate=2, res_connection=True, freeze=(7 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=64, alpha=alpha, stride=1, expansion=6, block_num=8, rate=2, res_connection=True, freeze=(8 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=64, alpha=alpha, stride=1, expansion=6, block_num=9, rate=2, res_connection=True, freeze=(9 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    x = _mobilenet_block(filters=96, alpha=alpha, stride=1, expansion=6, block_num=10, rate=2, res_connection=False, freeze=(10 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=96, alpha=alpha, stride=1, expansion=6, block_num=11, rate=2, res_connection=True, freeze=(11 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=96, alpha=alpha, stride=1, expansion=6, block_num=12, rate=2, res_connection=True, freeze=(12 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    # Stride and rate changed from original to have higher resolution
    x = _mobilenet_block(filters=160, alpha=alpha, stride=1, expansion=6, block_num=13, rate=2, res_connection=False, freeze=(13 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=160, alpha=alpha, stride=1, expansion=6, block_num=14, rate=4, res_connection=True, freeze=(14 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)
    x = _mobilenet_block(filters=160, alpha=alpha, stride=1, expansion=6, block_num=15, rate=4, res_connection=True, freeze=(15 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    x = _mobilenet_block(filters=320, alpha=alpha, stride=1, expansion=6, block_num=16, rate=4, res_connection=False, freeze=(16 <= freeze_first_n), tx2_gpu=tx2_gpu)(x)

    if load_weights is not None:
        model = Model(input, x)
        model.load_weights(load_weights, by_name=True)

    return x
