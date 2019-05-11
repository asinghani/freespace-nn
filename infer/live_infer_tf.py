import numpy as np
import cv2
import tensorflow as tf

from threading import Thread
import tensorflow.keras.backend as K

from ..config import Config
from ..util.vis import view_seg_map

import time
millis = lambda: int(round(time.time() * 1000))

class CameraThread:
    def __init__(self, camera, size=None, preprocess_func=lambda x: x):
        self.cap = cv2.VideoCapture(camera)

        if size is not None:
            self.cap.set(3, size[0])
            self.cap.set(4, size[1])

        _, self.frame = self.cap.read()
        print(self.frame.shape)

    def start(self):
        th = Thread(target=self.loop, args=())
        th.dameon = True
        th.start()
        return self

    def loop(self):
        while True:
            success, frame = self.cap.read()
            if success:
                self.frame = frame

    def read(self):
        return self.frame

cap = CameraThread(1, size=(480, 270))
cap.start()


with tf.gfile.FastGFile("/home/ubuntu/model-floornet-trt.pb", "rb") as f:
    trt_graph = tf.GraphDef()
    trt_graph.ParseFromString(f.read())

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name="")

input_tensor_name = "input_1:0"
aux_input_tensor_name = "input_2:0"

output_tensor = sess.graph.get_tensor_by_name("decoder_softmax/truediv:0")

while True:
    t = millis()
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
    nnInput = nnInput[:, :, ::-1].reshape((1, 224, 224, 3))
    image2 = 2 * (image2 - 0.5)
    image2 = image2.reshape((1, 112, 112, 3))

    print(millis() - t)
    t = millis()

    data = sess.run(output_tensor, {input_tensor_name: nnInput, aux_input_tensor_name: image2})
    #data = model.predict([nnInput[:, :, ::-1].reshape((1, 224, 224, 3)), image2.reshape((1, 112, 112, 3))])

    print(millis() - t)
    t = millis()

    seg = data[0].argmax(axis=2).astype(np.float32)
    vis = view_seg_map(cv2.resize(cap.read(), (224, 224)), seg, color=(0, 255, 0), alpha=0.3)
    cv2.imshow("Image", vis)
    cv2.waitKey(5)

    print(millis() - t)
    print()
    print()
