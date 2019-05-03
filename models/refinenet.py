import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Concatenate, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from .layers import BilinearUpsampling, BilinearResize, SepConv
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def RefineNet(image_shape):
    input = Input(shape=(image_shape[1], image_shape[0], 4)) # scaled up seg, edges

    x = input

    x = Conv2D(4, (9, 9), padding="same", use_bias=True, name="conv_1")(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_1_BN")(x)

    """x = Conv2D(4, (5, 5), padding="same", use_bias=True, name="conv_2")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_2_BN")(x)

    x = Conv2D(4, (3, 3), padding="same", use_bias=True, name="conv_3")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_3_BN")(x)

    x = Conv2D(4, (1, 1), padding="same", use_bias=True, name="conv_4")(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.1, name="conv_4_BN")(x)"""

    x = Conv2D(2, (1, 1), padding="same", use_bias=True, name="logits")(x)
    x = Activation("softmax", name="logits_softmax")(x)

    model = Model(input, x)

    print(get_model_memory_usage(16, model))

    return model
