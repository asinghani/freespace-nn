import numpy as np
import os, sys
import random

from ..config import Config, random_seed
from ..util.callbacks import SegCallback, SimpleTensorboardCallback, poly_lr
from ..util.generator_thread import GeneratorThread

from ..util.data_loader_refinenet import load_data, load_data_raw

from .refinenet import RefineNet

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam

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

model = RefineNet((1024, 576))

save_location = "/hdd/models/isef/refinenet/"
print(save_location)

checkpoint = K.callbacks.ModelCheckpoint(os.path.join(save_location, "model-{epoch:04d}.h5"), monitor="val_loss", verbose=0, save_best_only=False, mode="auto")

initial_lr = 6.0e-4 # Should be 6.0e-4
epochs = 1000 # Should be 1000

csvLogger = K.callbacks.CSVLogger(save_location+"log.csv", append=False, separator=",")

#train_data, test_data = load_data(config)
#train_data = GeneratorThread([train_data], max_storage=500).get_iterator()
trainX, trainY, testX, testY = load_data_raw(config)


def mean_iou(y_true, y_pred):
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy", mean_iou])

with open(save_location+"config.json", "w") as f:
    f.write(config.serialize())

plot_model(model, to_file=save_location+"model.png", show_shapes=True)

model.fit(
    trainX, trainY,
    batch_size=16,
    #config.samples_per_epoch,
    epochs=epochs,
    validation_data=(testX, testY),
    #validation_steps=20,
    callbacks=[checkpoint, csvLogger],
    verbose=1
)

"""model.fit_generator(
    train_data,
    config.samples_per_epoch,
    epochs,
    validation_data=test_data,
    validation_steps=20,
    callbacks=[checkpoint, csvLogger],
    verbose=1
)"""
