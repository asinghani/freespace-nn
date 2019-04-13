import tensorflow as tf
from tensorflow import keras as K

from PIL import Image
import io

import numpy as np
import cv2

from .vis import view_seg_map, img_stats
from .data_loader import reverse_preprocess

def poly_lr(initial, max_epoch, exp=0.9):
    def fun(epoch, lr):
        return initial * ((1.0 - float(epoch) / float(max_epoch)) ** exp)

    return fun

def make_tf_image(image):
    height, width, channel = image.shape

    image = Image.fromarray(image.astype("uint8"))
    output = io.BytesIO()
    image.save(output, format="PNG")
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

class SegCallback(K.callbacks.Callback):
    def __init__(self, data, config, writer):
        self.validation_data = None
        self.model = None
        self.data = data
        self.config = config
        self.writer = writer

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # Test network
        model = self.model
        generator = self.data
        config = self.config

        images, labels = next(generator)
        rgb, features, label = images[0][0], images[1][0], labels[0]

        p = model.predict([np.array([rgb]), np.array([features])])[0]
        p = p.reshape((config.image_size[0], config.image_size[1], 2)).argmax(axis=2)

        image1 = (rgb[:, :, 0:3] / 2.0) + 0.5
        features = features

        seg, overlay = view_seg_map(image1, p, color=(0, 1, 0), include_overlay=True)
        seg = seg * 255
        seg = make_tf_image(seg)

        gt = view_seg_map(image1, label.argmax(axis=2), color=(0, 1, 0)) * 255
        gt = make_tf_image(gt)

        features = make_tf_image(((features / 2.0) + 0.5) * 255)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Segmentation", image=seg)])
        features_summary = tf.Summary(value=[tf.Summary.Value(tag="Features", image=features)])
        groundtruth_summary = tf.Summary(value=[tf.Summary.Value(tag="Ground Truth Segmentation", image=gt)])

        self.writer.add_summary(summary, epoch)
        self.writer.add_summary(features_summary, epoch)
        self.writer.add_summary(groundtruth_summary, epoch)
        self.writer.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class SimpleTensorboardCallback(K.callbacks.Callback):
    def __init__(self, config, writer):
        self.validation_data = None
        self.model = None
        self.config = config
        self.epoch = 0
        self.hist = []
        self.writer = writer

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.hist = []
        return

    def on_epoch_end(self, epoch, logs={}):
        train_acc = tf.Summary(value=[tf.Summary.Value(tag="train_acc_epoch", simple_value=logs["acc"])])
        train_loss = tf.Summary(value=[tf.Summary.Value(tag="train_loss_epoch", simple_value=logs["loss"])])
        train_miou = tf.Summary(value=[tf.Summary.Value(tag="train_miou_epoch", simple_value=logs["mean_iou"])])
        val_acc = tf.Summary(value=[tf.Summary.Value(tag="val_acc_epoch", simple_value=logs["val_acc"])])
        val_loss = tf.Summary(value=[tf.Summary.Value(tag="val_loss_epoch", simple_value=logs["val_loss"])])
        val_miou = tf.Summary(value=[tf.Summary.Value(tag="val_miou_epoch", simple_value=logs["val_mean_iou"])])

        self.writer.add_summary(train_acc, epoch)
        self.writer.add_summary(train_loss, epoch)
        self.writer.add_summary(train_miou, epoch)
        self.writer.add_summary(val_acc, epoch)
        self.writer.add_summary(val_loss, epoch)
        self.writer.add_summary(val_miou, epoch)

        for batch, summary in self.hist:
            self.writer.add_summary(summary, self.epoch * self.config.samples_per_epoch + batch)

        self.writer.flush()
        self.epoch = self.epoch + 1
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.hist.append((batch, tf.Summary(value=[tf.Summary.Value(tag="train_acc_batch", simple_value=logs["acc"])])))
        self.hist.append((batch, tf.Summary(value=[tf.Summary.Value(tag="train_loss_batch", simple_value=logs["loss"])])))
        return
