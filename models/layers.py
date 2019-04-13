import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Layer, Input, InputSpec, DepthwiseConv2D, Conv2D, Activation, BatchNormalization, ZeroPadding2D

from tensorflow.python.keras import backend

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints



def SepConv(filters, stride=1, kernel_size=3, rate=1, bn_epsilon=1e-3, input=None):

    # Conform to functional API
    if input is None:
        return (lambda x: SepConv(filters, stride=stride, kernel_size=kernel_size, rate=rate, bn_epsilon=bn_epsilon, input=x))

    if stride == 1:
        padding = "same"
        x = input
    else:
        pad_total = (kernel_size + ((kernel_size - 1) * (rate - 1))) - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = ZeroPadding2D((pad_left, pad_right))(input)
        padding = "valid"

    # TODO add tx2 support
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=padding, use_bias=False)(x)
    #x = BatchNormalization(epsilon=bn_epsilon)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    #x = BatchNormalization(epsilon=bn_epsilon)(x)
    x = Activation("relu")(x)

    return x

class BilinearUpsampling(Layer):
    """
    Upsampling layer which grows a 4D tensor in the width and height dimensions
    """

    def __init__(self, upsampling, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.upsampling = upsampling

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.upsampling[0] * input_shape[1], self.upsampling[1] * input_shape[2], input_shape[3])

    def call(self, inputs):
        return tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]), align_corners=True)

    def get_config(self):
        return dict(list(super(BilinearUpsampling, self).get_config().items()) + list(({"upsampling": self.upsampling}).items()))


class BilinearResize(Layer):
    """
    Upsampling layer which grows a 4D tensor in the width and height dimensions to a new size
    """

    def __init__(self, output_shape, **kwargs):
        super(BilinearResize, self).__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.output_shape_ = output_shape

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_shape_[0], self.output_shape_[1], input_shape[3])

    def call(self, inputs):
        return tf.image.resize_bilinear(inputs, (self.output_shape_[0], self.output_shape_[1]), align_corners=True)

    def get_config(self):
        return dict(list(super(BilinearResize, self).get_config().items()) + list(({"output_shape": self.output_shape_}).items()))

