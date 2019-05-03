from sklearn.model_selection import train_test_split
import os, sys
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from ..util.vis import img_stats
from ..config import Config, random_seed

from ..aug.augmentation import augment
from ..aug.patchwise import get_patch

def load_data(config = Config()):
    """
    Loads and prepares data. Returns generators for (train, test)
    """
    X_train, X_test, Y_train, Y_test = load_files(config)

    train_generator = prepare_data(X_train, Y_train, config.batch_size, True, config)

    test_generator = prepare_data(X_test, Y_test, config.test_batch_size, False, config)

    return train_generator, test_generator

def load_data_raw(config = Config()):
    X_train, X_test, Y_train, Y_test = load_files(config)

    train_generator = prepare_data(X_train, Y_train, 110, True, config)
    trainX, trainY = next(train_generator)

    test_generator = prepare_data(X_test, Y_test, 10, False, config)
    testX, testY = next(test_generator)

    return trainX, trainY, testX, testY

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

    Y = [mpimg.imread(y) for y in Y]
    Y_small = [cv2.resize(y[int(y.shape[0] * 0.375):], config.image_size[::-1]) for y in Y]

    print("Read all data")

    def gen():
        # Generate infinite amount of data
        while True:
            i = 0

            images = np.empty([batch_size, 576, 1024, 4])
            labels = np.empty([batch_size, 576, 1024, 2])

            # Pick random portion of data 
            for index in np.random.permutation(len(X)):

                image = np.array(X[index], dtype=np.float32)

                label_small = Y_small[index]
                label_small = cv2.cvtColor(label_small, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                label_small_mid = cv2.resize(label_small, (1024, 360))
                label_small_mid = label_small_mid.reshape((label_small_mid.shape[0], label_small_mid.shape[1], 1))
                label_small = np.zeros([576, 1024, 1], dtype=np.float32)
                label_small[216:, :] = label_small_mid

                label_in = np.zeros([label_small.shape[0], label_small.shape[1], 1], dtype=np.float32)

                label_in[np.where((label_small < 0.5))] = 0.0
                label_in[np.where((label_small > 0.5))] = 1.0


                label = Y[index]
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                label_out = np.zeros([label.shape[0], label.shape[1], 2], dtype=np.float32)
                label_out[np.where((label < 0.5).all(axis=2))] = (1, 0)
                label_out[np.where((label > 0.5).all(axis=2))] = (0, 1)


                img2 = cv2.cvtColor(image * 255.0, cv2.COLOR_BGR2GRAY)
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
                cv2.imwrite("/hdd/temp/bork_data/edges{}.png".format(i), image2 * 255)
                cv2.imwrite("/hdd/temp/bork_data/labelin{}.png".format(i), label_in * 255)
                cv2.imwrite("/hdd/temp/bork_data/labelout{}.png".format(i), label_out[:, :, 0] * 255)
                image2 = 2 * (image2 - 0.5)
                image2 = np.concatenate((label_in, image2), axis=2)

                images[i] = image2
                labels[i] = label_out

                # Limit number of images to batch_size
                i += 1
                if i == batch_size:
                    break


            yield images, labels

    return gen()
