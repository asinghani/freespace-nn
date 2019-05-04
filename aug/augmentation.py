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
    rotationAngle = np.random.normal(0.0, 3.0)
    shearAngle = np.random.normal(0.0, 3.0)

    shiftX = np.random.normal(0.0, 0.03)
    shiftY = np.random.normal(0.0, 0.03)

    zoom = np.random.normal(1.0, 0.04)
    horizFlip = bool(random.getrandbits(1))

    brightness = np.random.normal(1.00, 0.12)

    out = gen.apply_transform(image * 255.0, {"theta": rotationAngle, "shear": shearAngle, "tx": shiftX, "ty": shiftY, "zx": zoom, "zy": zoom, "flip_horizontal": horizFlip, "brightness": brightness})
    outLabel = gen.apply_transform(label * 255.0, {"theta": rotationAngle, "shear": shearAngle, "tx": shiftX, "ty": shiftY, "zx": zoom, "zy": zoom, "flip_horizontal": horizFlip})

    out = np.array(cv2.cvtColor(np.array(np.clip(out, 0, 255), dtype=np.uint8), cv2.COLOR_RGB2HSV), np.int16)
    out[:, :, 0] += np.int16(np.random.normal(0.0, 5.0))
    out[:, :, 1] += np.int16(np.random.normal(0.0, 8.0))
    out[:, :, 2] += np.int16(np.random.normal(0.0, 10.0))
    out[:, :, 0] = np.clip(out[:, :, 0], 0, 179)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB).astype(np.float32)

    return out / 255.0, outLabel / 255.0

