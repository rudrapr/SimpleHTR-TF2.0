from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2


def pre_process(img, img_size, data_augmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"
    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([img_size[1], img_size[0]])

    # increase dataset size by applying random stretches to the images
    if data_augmentation:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        w_stretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(img, (w_stretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = img_size
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, new_size)
    target = np.ones([ht, wt]) * 255
    target[0:new_size[1], 0:new_size[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return np.expand_dims(img, axis=2)
