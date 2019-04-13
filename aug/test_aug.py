import cv2
import numpy as np
import sys
from . import augmentation as aug
import imgaug as ia

from ..util.vis import view_seg_map

image = cv2.imread(sys.argv[1])
label = cv2.imread(sys.argv[2], 0)
label = label.reshape((label.shape[0], label.shape[1], 1))

image = np.array([np.concatenate((image, image), axis=2)])
label = np.array([label])

geo = aug.aug_geo_light
noise = aug.combine_aug(aug.aug_noise_light, aug.aug_color_light)

for i in range(150):
    d = geo.to_deterministic()
    out = d.augment_images(image)[0]
    out = noise.augment_images(np.array([out]))[0]

    out2 = d.augment_images(label)[0] / 255.0
    out2[out2 > 0.5] = 1.0
    out2[out2 <= 0.5] = 0.0
    cv2.imwrite("/hdd/test/image{}.png".format(i), view_seg_map(out[:, :, ::2], out2, alpha=0.8))
    print(i+1)

