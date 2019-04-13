import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session, set_floatx
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.python.framework import graph_util

from ..config import Config
from .deeplab_mobilenet import DeepLabV3_MobileNetV2

from tqdm import tqdm

millis_time = lambda: int(round(time.time() * 1000))

# Transfer weights from src to model while fixing dilated convolutions
def transfer_weights(src, model):
    print([l.name for l in src.layers[122:124]])
    for layer in tqdm(src.layers):
        if isinstance(layer, DepthwiseConv2D) and (layer.dilation_rate[0] > 1):
            print(layer.name)
            weights = layer.get_weights()[0]

            x = 3 + (2 * (layer.dilation_rate[0] - 1))
            new_weights = np.zeros((x, x, weights.shape[2], 1), dtype=weights.dtype)
            new_weights[::(layer.dilation_rate[0]), ::(layer.dilation_rate[0])] = weights
            model.get_layer(layer.name).set_weights([new_weights])
        else:
            try:
                model.get_layer(layer.name).set_weights(layer.get_weights())
            except Exception as e:
                print(e, layer.name)


config = Config()
config.init_weights = None

#set_floatx("float16")

K.set_learning_phase(0)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
set_session(session)

#model1 = DeepLabV3_MobileNetV2(config, tx2_gpu=False)
#model1.load_weights("/home/ubuntu/model.h5")

model = DeepLabV3_MobileNetV2(config, tx2_gpu=True)
model.load_weights("/home/ubuntu/model_tx2.h5")

#transfer_weights(model1, model)

print(model.input)
print(model.output, model.output.name)

output_node_name = model.output.name.replace(":0", "")

with session as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    graph_def = sess.graph.as_graph_def()
    #print([n.name for n in graph_def.node])

    output_graph_def = graph_util.convert_variables_to_constants(
                                                                 sess,
                                                                 sess.graph.as_graph_def(),
                                                                 output_node_name.split(","))
    tf.train.write_graph(output_graph_def,
                         logdir="/tmp/logs",
                         name="/tmp/model.pb",
                         as_text=False)
