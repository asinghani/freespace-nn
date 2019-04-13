import cv2
import sys, argparse

parser = argparse.ArgumentParser(description="Take pictures (for creating training dataset)")
parser.add_argument("-c", "--camera", help="Camera number", default=0)
parser.add_argument("-r", "--resolution", help="Image resolution (i.e. \"1920x1080\")", default="none")
parser.add_argument("-s", "--scale", help="Scaling factor (i.e. 0.5)", default=0)
parser.add_argument("-n", "--start", help="Image # to start at (filename is in the format \"image00000.png\")", default=0)
args = vars(parser.parse_args())

if "start" in args:
    n = int(args["start"])
else:
    n = 0

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

flash = 0

while True:
    img = cap.read()[1]
    if scale != 0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    if flash > 0:
        flash = flash - 1
        cv2.rectangle(img, (10, 10), (100, 100), (256, 85, 0), -1)
        cv2.imshow("Image", img)
    else:
        cv2.imshow("Image", img)
        key = cv2.waitKey(10)
        if key == 32 or key == 97: # Space (on MacOS) or A (on Linux)
            print("Image {} saved".format(n))
            cv2.imwrite("image{}.png".format(str(n).zfill(5)), img)
            n = n + 1
            flash = 4

        if key == 113: # Q
            break
