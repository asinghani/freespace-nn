import cv2
import numpy as np
import pandas as pd

import tensorflow as tf

from freespacenn.config import Config, random_seed
from freespacenn.benchmark.data_loader_benchmarking import load_data
from freespacenn.models.floornet import FloorNet

dir = "camvid"

config = Config()

img, aux, label = load_data(config, seed=42, amount=50, resize=True)
input = [img, aux]
y_real = label.argmax(axis=3)

model = FloorNet(config)

last = None

i = 2000

model.load_weights("/hdd/models/isef/{}/model-{}.h5".format(dir, str(i).zfill(4)))
out = model.predict(input)
y_pred = out.argmax(axis=3)

print(i)
print("TP = {}".format(float(np.logical_and(y_real == 1, y_pred == 1).sum()))) # / np.prod(y_real.shape)))
print("TN = {}".format(float(np.logical_and(y_real == 0, y_pred == 0).sum()))) # / np.prod(y_real.shape)))
print("FP = {}".format(float(np.logical_and(y_real == 0, y_pred == 1).sum()))) # / np.prod(y_real.shape)))
print("FN = {}".format(float(np.logical_and(y_real == 1, y_pred == 0).sum()))) # / np.prod(y_real.shape)))
