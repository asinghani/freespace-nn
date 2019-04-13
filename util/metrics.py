import numpy as np
import tensorflow as tf

def iou(true, pred):
    ######### BROKEN
    return -1
    assert true.shape == pred.shape
    return np.logical_and(true.flatten(), pred.flatten()).mean() / np.logical_or(true.flatten(), pred.flatten()).mean()

if __name__ == "__main__":
    arr1 = np.zeros((150, 150))
    arr2 = np.zeros((150, 150))

    arr1[0:50, :] = 1
    arr1[0:50, 50:80] = 0
    arr2[0:52, :] = 1

    print(iou(arr1, arr2))

    a = tf.constant(arr1)
    b = tf.constant(arr2)

    miou, update_op = tf.metrics.mean_iou(a, b, 2)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(update_op)
    print(sess.run(miou))
