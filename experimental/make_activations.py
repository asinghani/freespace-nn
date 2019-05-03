import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.backend import set_session, set_floatx
#from keract import get_activations

from ..util.vis import view_seg_map

from ..config import Config
from ..models.floornet import FloorNet

config = Config()
config.init_weights = None
config.image_size = (224, 224)
config.input_shape = (config.image_size[0], config.image_size[1], 3)

model = FloorNet(config)
model.load_weights("/hdd/models/final_floorseg/final/model-0999.h5")

img = cv2.imread("/hdd/datasets/floorseg/raw/image00132.png")

# preprocessing
img = cv2.resize(img[int(img.shape[0] * 0.375):], (224, 224))
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
x = cv2.convertScaleAbs(x) / 255.0
x = x.reshape((x.shape[0], x.shape[1], 1))
y = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)
y = cv2.convertScaleAbs(y) / 255.0
y = y.reshape((y.shape[0], y.shape[1], 1))
z = cv2.Laplacian(img2, cv2.CV_32F)
z = cv2.convertScaleAbs(z) / 255.0
z = z.reshape((z.shape[0], z.shape[1], 1))
image2 = np.concatenate((x, y, z), axis=2)
image2 = cv2.resize(image2, (112, 112))
nnInput = np.array(img, dtype=np.float32) / 255.0

nnInput = 2 * (nnInput - 0.5)
image2 = 2 * (image2 - 0.5)

layer_outputs = [layer.output for layer in model.layers if "input" not in layer.name]
layer_names = [layer.name for layer in model.layers if "input" not in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict([nnInput[:, :, ::-1].reshape((1, 224, 224, 3)), image2.reshape((1, 112, 112, 3))])

for x in range(len(activations)):
    acts = activations[x]
    layer_name = layer_names[x]
    print(layer_name, x)
    nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
    ncols = int(math.ceil(acts.shape[-1] / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    fig.suptitle(layer_name)
    for i in range(nrows * ncols):
        if i < acts.shape[-1]:
            img = acts[0, :, :, i]
            hmap = axes.flat[i].imshow(img, cmap=None, interpolation="nearest")
        axes.flat[i].axis('off')
    #fig.subplots_adjust(right=0.8)
    #cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    #fig.colorbar(hmap, cax=cbar)
    plt.savefig('/hdd/temp/layer{}_{}.png'.format(str(x).zfill(3), layer_name), bbox_inches='tight')
    plt.close(fig)

print(activations[15])
print(activations[15].shape)
