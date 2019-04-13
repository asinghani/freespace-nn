import numpy as np

from ..config import Config, random_seed
from ..util.data_loader import load_data

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, Cropping2D, ZeroPadding2D, Activation, BatchNormalization, Lambda

from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG16

def crop(tensor, size):
    """
    Crops the given tensor to the size of the second tensor
    """

    dx = int(tensor.shape[2] - size.shape[2])
    dy = int(tensor.shape[1] - size.shape[1])

    crop = Cropping2D(cropping=((0, dy), (0, 0)))(tensor)
    crop = Cropping2D(cropping=((0, 0), (0, dx)))(crop)

    return crop

def FCN8_VGG16(mean, stddev, config):
    print(mean, stddev)
    """
    Implementation of FCN8 with VGG16 backend based on arXiv:1411.4038 [cs.CV]
    """
    input = Input(shape=config.input_shape)

    vgg16_input = ZeroPadding2D(padding=(100, 100))(input)

    vgg16 = VGG16(include_top=False, weights=config.init_weights, input_tensor=vgg16_input, input_shape=config.input_shape, pooling=None)

    vgg16_block3 = vgg16.get_layer(name="block3_pool").output
    vgg16_block4 = vgg16.get_layer(name="block4_pool").output
    vgg16_block5 = vgg16.get_layer(name="block5_pool").output

    # Fully connected from VGG16
    fc1_conv = BatchNormalization()(Conv2D(4096, (7, 7), activation="relu", padding="valid", kernel_regularizer=l2(config.l2_constant))(vgg16_block5))
    dropout1 = Dropout(config.dropout)(fc1_conv)
    fc2_conv = BatchNormalization()(Conv2D(4096, (1, 1), activation="relu", padding="valid", kernel_regularizer=l2(config.l2_constant))(dropout1))
    dropout2 = Dropout(config.dropout)(fc2_conv)

    # 2 = number of classes
    fcn32_conv = BatchNormalization()(Conv2D(2, (1, 1), kernel_regularizer=l2(config.l2_constant))(dropout2))
    fcn32_deconv = BatchNormalization()(Conv2DTranspose(2, kernel_size = (4, 4), strides = (2, 2), use_bias = False, kernel_regularizer=l2(config.l2_constant))(fcn32_conv))

    fcn16_conv = BatchNormalization()(Conv2D(2, (1, 1), kernel_regularizer=l2(config.l2_constant))(vgg16_block4))
    fcn16_crop = crop(fcn16_conv, fcn32_deconv)
    fcn16_add = Add()([fcn32_deconv, fcn16_crop])
    fcn16_deconv = BatchNormalization()(Conv2DTranspose(2, kernel_size = (4, 4), strides = (2, 2), use_bias = False, kernel_regularizer=l2(config.l2_constant))(fcn16_add))

    fcn8_conv = BatchNormalization()(Conv2D(2, (1, 1), kernel_regularizer=l2(config.l2_constant))(vgg16_block3))
    fcn8_crop = crop(fcn8_conv, fcn16_deconv)
    fcn8_add = Add()([fcn16_deconv, fcn8_crop])
    fcn8_deconv = BatchNormalization()(Conv2DTranspose(2, kernel_size = (16, 16), strides = (8, 8), use_bias = False, kernel_regularizer=l2(config.l2_constant))(fcn8_add))

    final = crop(fcn8_deconv, input)

    final = Activation("softmax")(final)

    # Use Z-scores as a form of normalization
    model = Model(input, final)

    return model

"""

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


data = np.random.random((1000, 3002))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 3002))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=1000, batch_size=3002,
                  validation_data=(val_data, val_labels),
                  callbacks=[K.callbacks.TensorBoard(log_dir='/tmp/logs', histogram_freq=0, batch_size=32, write_graph=True) ])
"""
