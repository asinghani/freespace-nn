import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
import tensorflow.contrib.tensorrt as trt
import tensorflow.keras.backend as K

from ..config import Config
from ..models.floornet import FloorNet

K.clear_session()
K.set_learning_phase(0)

config = Config()
model = FloorNet(config)
model.load_weights("/home/ubuntu/model-floornet.h5")

session = K.get_session()

inputs = [layer.op.name for layer in model.inputs]
outputs = [layer.op.name for layer in model.outputs]

print("Inputs: {}".format(str(inputs)))
print("Outputs: {}".format(str(outputs)))

graph = session.graph

with graph.as_default():
    graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
    frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, [out.op.name for out in model.outputs])

trt_graph = trt.create_inference_graph(
    input_graph_def = frozen,
    outputs = outputs,
    max_batch_size = 1,
    max_workspace_size_bytes = 1 << 25,
    precision_mode = "FP16",
    minimum_segment_size = 50
)

graph_io.write_graph(trt_graph, "/home/ubuntu/", "model-floornet-trt.pb", as_text=False)
