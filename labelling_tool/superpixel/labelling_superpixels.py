from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, LassoSelector
from matplotlib.path import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
import threading
import time
import sys, os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

if len(sys.argv) < 3:
    print("Usage: <output dir> <images>")

out_dir = sys.argv[1]
images = sys.argv[2:]

index = 0

image = None
segments = None
freespace = None

segments_next = None
img_frame = None

def make_next_seg(x):
    global segments_next
    next_image = cv2.imread(x)[:, :, ::-1]
    segments_next = slic(next_image / 255.0, n_segments = 400, sigma = 0)

def reload():
    global image, segments, freespace, index, segments_next, img_frame
    print("Loading {}/{}...".format(index + 1, len(images)))
    if img_frame:
        img_frame.set_data(np.array([[0, 0], [0, 0]], dtype=np.float32))
        plt.draw()

    _image = cv2.imread(images[index])[:, :, ::-1]
    segments = slic(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB) / 255.0, n_segments = 600, sigma = 0)
    image = _image
    freespace = set()

    print("Loaded")

reload()

def mark_freespace(x, y):
    return segments[x, y]

def render_freespace(image, segments, regions):
    image = image.copy()
    for region in list(regions):
        image[(segments == region)] = (0, 180, 0)

    return mark_boundaries(image, segments)

def onclick(event):
    global freespace, img_frame, ax, lasso
    if event.xdata is not None and event.ydata is not None and event.inaxes == ax and lasso is None:
        if event.button == 1:
            freespace.add(segments[int(event.ydata), int(event.xdata)])
        if event.button == 3:
            seg = segments[int(event.ydata), int(event.xdata)]
            if seg in freespace:
                freespace.remove(seg)

lasso = None

def onlasso(points):
    global lasso, image, freespace
    temp = Image.new("L", (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(temp).polygon(points, outline=0, fill=1)
    mask = np.array(temp)
    s = np.unique(segments[(mask == 1)])
    freespace.update(s)
    lasso.disconnect_events()
    lasso = None

force_refresh = False

def onnextbutton(event):
    # save
    global image, freespace
    mask = np.zeros_like(image)
    for region in list(freespace):
        mask[(segments == region)] = (255, 255, 255)

    cv2.imwrite(os.path.join(out_dir, images[index].split("/")[-1]), mask)

    global index, force_refresh
    if index < len(images) - 1:
        index = index + 1
        reload()
        force_refresh = True
    else:
        print("Finished")
        sys.exit(0)

def onprevbutton(event):
    global index, force_refresh
    if index > 0:
        index = index - 1
        reload()
        force_refresh = True

def onlassobutton(event):
    global lasso, ax
    lasso = LassoSelector(ax, onselect=onlasso)

def onresetbutton(event):
    global freespace
    freespace = set()

def onunborkbutton(event):
    global lasso, force_refresh
    lasso = None
    force_refresh = True

def refresh_loop():
    global lasso, force_refresh
    time.sleep(2)
    while True:
        if lasso is None or force_refresh:
            img_frame.set_data(render_freespace(image, segments, freespace))
            plt.draw()
            if force_refresh:
                onlassobutton(None)
            force_refresh = False
        time.sleep(0.5)

refreshThread = threading.Thread(target=refresh_loop, args=())
refreshThread.daemon = True

fig = plt.figure("Superpixels")
cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("motion_notify_event", onclick)
#ax = fig.add_subplot(1, 1, 1)
ax = plt.axes([0.05, 0.15, 0.9, 0.8])

button_next = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), "Next")
button_next.on_clicked(onnextbutton)

button_prev = Button(plt.axes([0.71, 0.05, 0.1, 0.075]), "Prev")
button_prev.on_clicked(onprevbutton)

button_lasso = Button(plt.axes([0.61, 0.05, 0.1, 0.075]), "Lasso")
button_lasso.on_clicked(onlassobutton)

button_reset = Button(plt.axes([0.51, 0.05, 0.1, 0.075]), "Reset")
button_reset.on_clicked(onresetbutton)

button_unbork = Button(plt.axes([0.41, 0.05, 0.1, 0.075]), "Unbork")
button_unbork.on_clicked(onunborkbutton)

plt.axes([0.31, 0.05, 0.1, 0.075])

img_frame = ax.imshow(render_freespace(image, segments, freespace))
plt.axis("off")
ax.axis("off")
plt.tight_layout()
refreshThread.start()
plt.show()
