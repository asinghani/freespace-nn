import cv2
import numpy as np
import random

def get_patch(size, image, label, scale_min, scale_max):
    scale = random.uniform(scale_min, scale_max)

    image = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))))
    label = cv2.resize(label, (int(round(label.shape[1] * scale)), int(round(label.shape[0] * scale))))

    label[label > 0.5] = 1.0
    label[label <= 0.5] = 0.0

    start_row = random.randint(0, image.shape[0] - size[0] - 1)
    start_col = random.randint(0, image.shape[1] - size[1] - 1)

    end_row = start_row + size[0]
    end_col = start_col + size[1]

    return image[start_row:end_row, start_col:end_col], label[start_row:end_row, start_col:end_col]
