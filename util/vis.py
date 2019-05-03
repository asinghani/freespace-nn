import cv2
import numpy as np

"""
Visualize segmentation output
"""
def view_seg_map(img, seg, color=(0, 255, 0), alpha=0.4, include_overlay=False, include_mask=False):
    if len(seg.shape) > 2 and seg.shape[2] > 1:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

    if len(seg.shape) < 3:
        seg = np.expand_dims(seg, axis = 2)

    assert img.shape[0:2] == seg.shape[0:2]

    overlay = img.copy()
    overlay[np.where((seg > 0).all(axis = 2))] = color

    overlay2 = img.copy()
    overlay2[np.where((seg > 0).all(axis = 2))] = color
    overlay2[np.where((overlay2 != color).all(axis = 2))] = (0, 0, 0)

    output = img.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    if include_overlay:
        return output, overlay
    elif include_mask:
        return output, overlay2
    else:
        return output

def img_stats(img):
    flat = img.flatten()
    print("Mean = {}, Stddev = {}, Range = {}, Min = {}, Max = {}".format(np.mean(flat), np.std(flat), np.max(flat) - np.min(flat), np.min(flat), np.max(flat)))
