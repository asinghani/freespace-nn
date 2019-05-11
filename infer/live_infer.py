import numpy as np
import cv2
import tensorflow as tf

from threading import Thread
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session, set_floatx
from tensorflow.keras.layers import DepthwiseConv2D

from ..freespacenn.config import Config
from ..freespacenn.vis import view_seg_map
from ..freespacenn.models.floornet import FloorNet

class CameraThread:
    def __init__(self, camera, size=None, preprocess_func=lambda x: x):
        self.cap = cv2.VideoCapture(src)

        if size is not None:
            self.cap.set(3, size[0])
            self.cap.set(4, size[1])

        _, self.frame = self.cap.read()

    def start(self):
        Thread(target=self.loop, args=()).start()
        return self

    def loop(self):
        while True:
            _, self.frame = self.cap.read()

    def read(self):
        return self.frame


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
set_session(session)

config = Config()
model = FloorNet(config)
model.load_weights("/home/ubuntu/model-floornet.h5")

cap = CameraThread(2, size=(480, 270))

while True:
    frame = cap.read()

    img = frame
    img = cv2.resize(img, (224, 224))

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

    data = model.predict([nnInput[:, :, ::-1].reshape((1, 224, 224, 3)), image2.reshape((1, 112, 112, 3))])
    seg = data[0].argmax(axis=2).astype(np.float32)
    #seg = resize(seg, (frame.shape[0], frame.shape[1])).reshape((frame.shape[0], frame.shape[1], 1))
    #seg = np.around(seg)

    vis = view_seg_map(img, seg, color=(0, 255, 0), alpha=0.4)

    cv2.imshow("Image", vis)
    cv2.waitKey(10)
