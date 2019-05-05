from sklearn.model_selection import train_test_split
import os, sys
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.feature_extraction.image import extract_patches_2d

from ..util.vis import img_stats
from ..config import Config, random_seed

from ..aug.augmentation import augment
from ..aug.patchwise import get_patch

def load_data(config = Config(), aug_data = True):
    """
    Loads and prepares data. Returns generators for (train, test)
    """
    X_train, X_test, Y_train, Y_test = load_files(config)

    train_generator, mean, stddev = prepare_data(X_train, Y_train, 100, aug_data, config)

    test_generator, _1, _2 = prepare_data(X_test, Y_test, 20, False, config)

    valid_generator, _1, _2 = prepare_data(X_test[::2], Y_test[::2], 1, aug_data, config)

    return train_generator, test_generator, valid_generator, mean, stddev


def load_files(config = Config()):
    images = [f.split("/")[-1] for f in glob.glob(os.path.join(config.labels_location, "*.png"))]

    X = [os.path.join(config.images_location, f) for f in images]
    Y = [os.path.join(config.labels_location, f) for f in images]

    X, Y = shuffle(X, Y)

    # Use 20% of the dataset for testing, 80% for training 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

    return X_train, X_test, Y_train, Y_test


def prepare_data(X, Y, batch_size, augment_data, config = Config()):

    X = [mpimg.imread(x) for x in X]
    X = [preprocess_image(cv2.resize(x[int(x.shape[0]):], config.image_size[::-1]), config = config) for x in X]

    Y = [mpimg.imread(y) for y in Y]
    Y = [preprocess_label(cv2.resize(y[int(y.shape[0]):], config.image_size[::-1]), config = config) for y in Y]

########## NO RESIZE #######################################    
    #X = [preprocess_image(mpimg.imread(x), config = config) for x in X]
    #Y = [preprocess_label(mpimg.imread(y), config = config) for y in Y]

    print("Read all data")


    mean = np.mean(np.array(X, dtype=np.float32).flatten())
    stddev = np.std(np.array(X, dtype=np.float32).flatten())

    def gen():
        # Generate infinite amount of data
        while True:
            i = 0

            images = np.empty([batch_size, config.input_shape[0], config.input_shape[1], 3])
            labels = np.empty([batch_size, config.input_shape[0], config.input_shape[1], 2])

            # Pick random portion of data 
            for index in np.random.permutation(len(X)):

                image = X[index]
                label = Y[index]
                image, label = postprocess(image, label, config)

                if augment_data:
                    image, label = augment(image, label)

                image = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

                image = 2 * (image - 0.5)
                image = image.reshape((image.shape[0], image.shape[1], 1))
                newLabel = newLabel.reshape((newLabel.shape[0], newLabel.shape[1], 1))
                image = np.stack([image, newLabel], axis=-1)
                patches = extract_patches_2d(image, (32, 32), max_patches=20)
                patches_img = patches[:, :, :, 0]
                patches_label = patches[:, :, :, 1]

                for x in range(len(patches)):
                    images[i] = patches_img[x]
                    labels[i] = 1.0 if (patches_label[x].flatten().sum() / len(patches_label[x].flatten())) > 0.5 else 0.0
                    i = i + 1

                if i >= batch_size:
                    break


            yield images, labels

    return gen(), mean, stddev

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
