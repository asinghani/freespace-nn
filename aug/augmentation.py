import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
Augumentations:
    - Brightness
    - ContrastNormalization

    - Dropout
    - Salt & Pepper
    - AverageBlur
    - Superpixels
    - AdditiveGaussianNoise

    - Fliplr
    - Affine (reflect)
        - Translate
        - Shear (small)
"""

gen = ImageDataGenerator(fill_mode="reflect", dtype=np.float32)

def augment(image, label):
    rotationAngle = np.random.normal(0.0, 4.0)
    shearAngle = np.random.normal(0.0, 8.0)

    shiftX = np.random.normal(0.0, 0.08)
    shiftY = np.random.normal(0.0, 0.08)

    zoom = np.random.normal(1.0, 0.04)
    horizFlip = bool(random.getrandbits(1))

    brightness = np.random.normal(0.95, 0.22)

    out = gen.apply_transform(image * 255.0, {"theta": rotationAngle, "shear": shearAngle, "tx": shiftX, "ty": shiftY, "zx": zoom, "zy": zoom, "flip_horizontal": horizFlip, "brightness": brightness})
    outLabel = gen.apply_transform(label * 255.0, {"theta": rotationAngle, "shear": shearAngle, "tx": shiftX, "ty": shiftY, "zx": zoom, "zy": zoom, "flip_horizontal": horizFlip})

    return out / 255.0, outLabel / 255.0

