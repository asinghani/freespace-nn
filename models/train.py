import numpy as np
import os

from ..config import Config, random_seed
from ..util.data_loader import load_data
from ..util.callbacks import SegCallback, SimpleTensorboardCallback
from .basic_fcn import FCN8_VGG16

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.optimizers import Adam

np.random.seed(random_seed)
config = Config()

try:
    os.mkdirs(config.weights_save_location)
except: # Errors thrown if folder exists
    pass

train_data, test_data, valid_data, mean, stddev = load_data(config)

model = FCN8_VGG16(mean, stddev, config)

checkpoint = K.callbacks.ModelCheckpoint(os.path.join(config.weights_save_location, "model-{epoch:04d}.h5"), monitor="val_loss", verbose=0, save_best_only=False, mode="auto")

writer = tf.summary.FileWriter("/tmp/logs")
tensorboard = SimpleTensorboardCallback(config, writer)
segCb = SegCallback(valid_data, config, writer)

csvLogger = K.callbacks.CSVLogger(config.training_log_location, append=False, separator=",")

def mean_iou(y_true, y_pred):
    return tf.constant(0) # tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=config.learning_rate), metrics=["accuracy", mean_iou])

with open(config.config_save_location, "w") as f:
    f.write(config.serialize())


model.fit_generator(
    train_data,
    config.samples_per_epoch,
    config.total_epoch,
    validation_data=test_data,
    validation_steps=20,
    callbacks=[checkpoint, tensorboard, segCb, csvLogger],
    verbose=1
)
