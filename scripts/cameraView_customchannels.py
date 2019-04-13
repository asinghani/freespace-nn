import cv2
import sys, argparse
from scipy.signal import convolve2d
import numpy as np

parser = argparse.ArgumentParser(description="View camera feed with OpenCV")
parser.add_argument("-c", "--camera", help="Camera number", default=0)
parser.add_argument("-r", "--resolution", help="Image resolution (i.e. \"1920x1080\")", default="none")
parser.add_argument("-s", "--scale", help="Scaling factor (i.e. 0.5)", default=0)
args = vars(parser.parse_args())

if "camera" in args:
    i = int(args["camera"])
else:
    i = 0

cap = cv2.VideoCapture(i)


kernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=np.float32) / 4.0

if "resolution" in args and args["resolution"] != "none":
    res = args["resolution"].lower().split("x")
    cap.set(3, int(res[0]))
    cap.set(4, int(res[1]))

if "scale" in args:
    scale = float(args["scale"])
else:
    scale = 0

while True:
    img = cap.read()[1]
    if scale != 0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    img = cv2.Laplacian(img, cv2.CV_16S)
    img = cv2.convertScaleAbs(img)

    print(img.flatten().mean(), img.flatten().max())

    cv2.imshow("Image", img)
    if cv2.waitKey(10) == 113: # Q
        break
