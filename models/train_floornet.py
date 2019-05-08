import numpy as np
import os, sys
import random

from ..config import Config, random_seed
from ..util.data_loader import load_data
from ..util.callbacks import SegCallback, SimpleTensorboardCallback, poly_lr
from ..util.generator_thread import GeneratorThread

from .floornet import FloorNet

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam, SGD

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

try:
    os.mkdirs(config.weights_save_location)
except: # Errors thrown if folder exists
    pass

aug = True
train_data1, test_data, valid_data, mean, stddev = load_data(config, aug_data=aug)
train_data2, _1, _2, mean, stddev = load_data(config, aug_data=aug)
train_data3, _1, _2, mean, stddev = load_data(config, aug_data=aug)
train_data4, _1, _2, mean, stddev = load_data(config, aug_data=aug)
train_data5, _1, _2, mean, stddev = load_data(config, aug_data=aug)

train_data = GeneratorThread([train_data1, train_data2, train_data3, train_data4, train_data5], max_storage=500).get_iterator()
test_data = GeneratorThread([test_data], max_storage=200).get_iterator()
valid_data = GeneratorThread([valid_data], max_storage=10).get_iterator()

model = FloorNet(config, batchnorm=False, aux=True, pyramid=True, upsampling_trainable=True, upsampling_init=True)

#save_location = "/hdd/models/final_floorseg/f{}{}{}{}/".format(1 if aux else 0, 1 if pyramid else 0, 1 if upsampling_trainable else 0, 1 if upsampling_init else 0)
save_location = "/hdd/models/isef/endeavor/"

print(save_location)

checkpoint = K.callbacks.ModelCheckpoint(os.path.join(save_location, "model-{epoch:04d}.h5"), monitor="val_loss", verbose=0, save_best_only=False, mode="auto")

writer = tf.summary.FileWriter("/tmp/logs")
tensorboard = SimpleTensorboardCallback(config, writer)
segCb = SegCallback(valid_data, config, writer)

initial_lr = 2.0e-3 # Should be 6.0e-4
epochs = 2000 # Should be 1000

lr = K.callbacks.LearningRateScheduler(poly_lr(initial_lr, epochs, exp=0.9), verbose=1)

csvLogger = K.callbacks.CSVLogger(save_location+"log.csv", append=False, separator=",")

def mean_iou(y_true, y_pred):
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=initial_lr, momentum=0.9, nesterov=True), metrics=["accuracy", mean_iou])

with open(save_location+"config.json", "w") as f:
    f.write(config.serialize())

plot_model(model, to_file=save_location+"model.png", show_shapes=True)

model.fit_generator(
    train_data,
    config.samples_per_epoch,
    epochs,
    validation_data=test_data,
    validation_steps=20,
    callbacks=[checkpoint, tensorboard, segCb, csvLogger, lr],
    verbose=1
)
