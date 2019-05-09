from sklearn.model_selection import train_test_split
import os, sys
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import random

from ..util.vis import img_stats
from ..config import Config, random_seed

from ..aug.augmentation import augment
from ..aug.patchwise import get_patch

def load_data(config, seed, amount, resize=True):

    random.seed(seed)
    np.random.seed(seed)

    images = [f.split("/")[-1] for f in glob.glob(os.path.join(config.labels_location, "*.png"))]

    X = [os.path.join(config.images_location, f) for f in images]
    Y = [os.path.join(config.labels_location, f) for f in images]

    X, Y = shuffle(X, Y, random_state=seed)

    X = X[:amount]
    Y = Y[:amount]

    if resize:
        X = [mpimg.imread(x) for x in X]
        X = [preprocess_image(cv2.resize(x[:, :, 0:3], config.image_size[::-1]), config = config) for x in X]

        Y = [mpimg.imread(y) for y in Y]
        Y = [preprocess_label(cv2.resize(y[int(0.03*x.shape[0]):], config.image_size[::-1]), config = config) for y in Y]
    else:
        X = [preprocess_image(mpimg.imread(x), config = config) for x in X]
        Y = [preprocess_label(mpimg.imread(y), config = config) for y in Y]

        return X, Y

    print("Read all data")

    images = np.empty([amount, config.input_shape[0], config.input_shape[1], 3])
    images2 = np.empty([amount, 112, 112, 3])
    labels = np.empty([amount, config.input_shape[0], config.input_shape[1], 2])

    # Pick random portion of data 
    for index in range(amount):
        image = X[index]
        label = Y[index]
        image, label = postprocess(image, label, config)

        image, label = augment(image, label)

        newLabel = np.zeros([label.shape[0], label.shape[1], 2], dtype=np.float32)
        newLabel[np.where((label < 0.5).all(axis=2))] = (1, 0)
        newLabel[np.where((label > 0.5).all(axis=2))] = (0, 1)

        img2 = cv2.cvtColor(image * 255.0, cv2.COLOR_RGB2GRAY)
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

        images[index] = 2 * (image - 0.5)
        images2[index] = 2 * (image2 - 0.5)
        labels[index] = newLabel

    return images, images2, labels

def preprocess_image(image, config = Config()):
    #image = (image - 0.4574565) / 0.3043379 # hard-coded
    return image

def reverse_preprocess(image, config = Config()):
    #image = (image * 0.3043379) + 0.4574565 # hard-coded
    return image

def preprocess_label(label, config = Config()):
    return label

def postprocess(image, label, config = Config()):
    # TODO convert data to proper format for neural network (normalize, etc)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    newLabel = np.zeros([config.input_shape[0], config.input_shape[1], 1], dtype=np.float32)

    newLabel[np.where((label < 0.5))] = 0.0
    newLabel[np.where((label > 0.5))] = 1.0

    newImage = np.array(image, dtype=np.float32)

    return newImage, newLabel
