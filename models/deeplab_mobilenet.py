import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, Cropping2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

from .layers import BilinearUpsampling, BilinearResize, SepConv
from .mobilenetv2 import mobilenetV2

def DeepLabV3_MobileNetV2(config, tx2_gpu=False):
    """
    Implementation of DeepLabV3+ with MobileNetV2 backend based on arXiv:1706.05587v3 [cs.CV] and arXiv:1801.04381 [cs.CV]
    """
    input = Input(shape=config.input_shape)

    x = mobilenetV2(input, config.input_shape, alpha=config.mobilenet_alpha, freeze_first_n=-1, load_weights=None, tx2_gpu=tx2_gpu)


    # Atrous Spatial Pyramid Pooling

    OS = 8 # Smaller output stride = less params

    # Branch 1 (1x1 Conv with BatchNorm and ReLU)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b1_conv")(x)
    b1 = BatchNormalization(name="b1_BN")(b1)
    b1 = Activation("relu", name="b1_relu")(b1)

    # Branch 2 (Feature branch)
    b2 = AveragePooling2D(pool_size=(int(np.ceil(config.input_shape[0] / OS)), int(np.ceil(config.input_shape[1] / OS))), name="b2_pool")(x)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = BatchNormalization(name="b2_BN")(b2)
    b2 = Activation("relu", name="b2_relu")(b2)
    b2 = BilinearUpsampling((int(np.ceil(config.input_shape[0] / OS)), int(np.ceil(config.input_shape[1] / OS))), name="b2_upsampling")(b2)


    # ASPP Pooling
    #b3 = SepConv(256, rate=1)(x)
    #b4 = SepConv(256, rate=2)(x)
    #b5 = SepConv(256, rate=4)(x)

    x = Concatenate(name="decoder_concat")([b1, b2])

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="decoder_conv")(x)
    x = BatchNormalization(epsilon=1e-5, name="decoder_BN")(x)
    x = Activation("relu", name="decoder_relu")(x)
    x = Dropout(0.1, name="decoder_dropout")(x)

    x = Conv2D(2, (1, 1), padding="same", name="decoder_conv2")(x) # Logits layer
    x = BilinearResize((config.input_shape[0], config.input_shape[1]), name="decoder_resize")(x)

    if config.include_softmax:
        x = Activation("softmax", name="decoder_softmax")(x)

    model = Model(input, x)

    return model
