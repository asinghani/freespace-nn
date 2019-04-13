import numpy as np
import os, sys
import random

from ..config import Config, random_seed
from ..util.data_loader import load_data
from ..util.callbacks import SegCallback, SimpleTensorboardCallback

from .custom import FloorNet

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Add, Input, Cropping2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, Lambda, Concatenate, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from .mobilenetv2 import mobilenetV2

#############################
# Basic configuration setup
############################
if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
    print("PYTHONHASHSEED must be set to 0")
    sys.exit(0)

np.random.seed(random_seed)
tf.set_random_seed(random_seed)
random.seed(random_seed)
config = Config()

seg_model, input = FloorNet(config, input_layer=True)
#input = Input(shape=(224, 224, 3))
#seg_model = mobilenetV2(input, (224, 224, 3), 1.0, bn_epsilon=1e-3, bn_momentum=0.999, freeze_first_n=-1, tx2_gpu=False)

x = seg_model.get_layer("encoder_block8_pointwise_BN").output

x = Conv2D(640, kernel_size=3, strides=2, use_bias=False, name="end_conv1")(x)
x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="end_conv1_BN")(x)
x = ReLU(6., name="end_conv1_relu")(x)

x = Conv2D(640, kernel_size=3, strides=2, use_bias=False, name="end_conv2")(x)
x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="end_conv2_BN")(x)
x = ReLU(6., name="end_conv2_relu")(x)

x = Conv2D(1280, kernel_size=3, strides=2, use_bias=False, name="end_conv3")(x)
x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="end_conv3_BN")(x)
x = ReLU(6., name="end_conv3_relu")(x)

x = Flatten()(x)
x = Dense(1000, activation="relu", name="end_dense1")(x)
x = Dense(1000, activation="relu", name="end_dense2")(x)
x = Dense(100, activation="softmax", name="logits")(x)
model = Model(input, x)

plot_model(model, to_file="/tmp/model.png", show_shapes=True)

labels = np.load("/hdd/datasets/caltech101/labels.npy")
images = np.load("/hdd/datasets/caltech101/images.npy") / 255.0

checkpoint = K.callbacks.ModelCheckpoint(os.path.join("/hdd/models/detection/custom/", "model-{epoch:04d}.h5"), monitor="val_loss", verbose=0, save_best_only=False, mode="auto")

writer = tf.summary.FileWriter("/tmp/logs")
tensorboard = SimpleTensorboardCallback(config, writer)

def mean_iou(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy", mean_iou])

print(images.shape)
print(labels.shape)

model.fit(
    x=images,
    y=labels,
    batch_size=16,
    epochs=1,
    verbose=1,
    callbacks=[checkpoint, tensorboard],
    validation_split=0.15,
    shuffle=True,
)

seg_model.save_weights("/hdd/models/detection/custom/encoder.h5")
