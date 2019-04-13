import cv2
import sys, argparse

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

    cv2.imshow("Image", img)
    if cv2.waitKey(10) == 113: # Q
        break
