import numpy as np
import cv2
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session

from ..util.vis import view_seg_map, img_stats

from ..config import Config
from .deeplab_mobilenet import DeepLabV3_MobileNetV2

millis_time = lambda: int(round(time.time() * 1000))

config = Config()
config.init_weights = None

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
set_session(session)

model2, model = DeepLabV3_MobileNetV2(config)
model2.load_weights("/home/ubuntu/model.h5")

np.random.seed(42)
image = np.random.rand(512, 512, 3)

start_time = millis_time()
predictions = model.predict(np.array([image]))[0]
print(millis_time() - start_time)

#predictions = predictions.reshape((image.shape[0], image.shape[1], 2)).argmax(axis=2)

#vis = view_seg_map(image, predictions, color=(0, 1, 0)) * 255

print("\033[1m \033[94m")
print(img_stats(predictions))
print("\033[0m")
#cv2.imshow("img", vis)
#cv2.waitKey(500)
