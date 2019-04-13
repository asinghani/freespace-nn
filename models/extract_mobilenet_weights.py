"""
Extract feature extractor weights from DeepLab v3 Pascal VOC segmentation pretrained model for transfer learning
"""
import numpy as np

import tensorflow as tf
from tensorflow import keras as K

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.utils import get_file

from .mobilenetv2 import mobilenetV2

import tempfile, os
from tqdm import tqdm # Progress bar

input_shape = (512, 512, 3) # Same input shape as pretrained model
alpha = 1.0

TF_CKPT_TAR = "deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz"
TF_CKPT_URL = "http://download.tensorflow.org/models/" + TF_CKPT_TAR
TF_CKPT_DIR = "deeplabv3_mnv2_pascal_trainval"
TF_CKPT_FILE = "model.ckpt-30000"

dir = "/tmp/models"
try:
    os.mkdirs(dir)
except:
    pass

get_file(TF_CKPT_TAR, TF_CKPT_URL, extract=True, cache_subdir="", cache_dir=dir)
ckpt_file = os.path.join(dir, TF_CKPT_DIR, TF_CKPT_FILE)
reader = tf.train.NewCheckpointReader(ckpt_file)

tensors = {}


for key in reader.get_variable_to_shape_map():
    tensor_name = str(key).replace("/", "_").replace("MobilenetV2_", "").replace("BatchNorm", "BN").replace("_weights", "_kernel").replace("_biases", "_bias")

    if "Momentum" not in tensor_name:
        tensors[tensor_name] = reader.get_tensor(key)

for i in tensors:
    print(i)

input = Input(shape=input_shape)
output = mobilenetV2(input, input_shape, alpha=1.0)

model = Model(input, output)

for layer in tqdm(model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(":0", "")
            weights.append(tensors[layer.name + "_" + weight_name])

        layer.set_weights(weights)

model.save_weights("mobilenet_weights.h5")
