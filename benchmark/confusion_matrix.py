import cv2
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(true, pred):
    # true, pred = 2D array of 0 and 1
    if pred.shape != true.shape:
        raise Exception("True and predicted labels must have the same shape")

    total = float(np.prod(np.array(true.shape)))

    true_positive = np.logical_and(true == 1, pred == 1).sum() / total
    false_positive = np.logical_and(true == 0, pred == 1).sum() / total

    false_negative = np.logical_and(true == 1, pred == 0).sum() / total
    true_negative = np.logical_and(true == 0, pred == 0).sum() / total

    return true_positive, false_positive, false_negative, true_negative

def draw_confusion_matrix(true, pred):
    pass

if __name__ == "__main__":
    true = np.zeros((224, 224))
    pred = np.zeros((224, 224))

    true[70:200, 50:100] = 1
    pred[150:200, 50:120] = 1

    # true positive = 0.04982461735
    # false positive = 0.01992984694
    # false negative = 0.07971938775
    # true negative = 0.850526148

    cm = confusion_matrix(true, pred)
    cm = np.array([[cm[0], cm[1]], [cm[2], cm[3]]])
    classes = ["Freespace", "Obstacle"]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.title("title", y=1.08)

    # Rotate the tick labels and set their alignment.

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.show()
