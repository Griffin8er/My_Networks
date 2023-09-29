from Image_process import *
import numpy as np


def choose(i, j):
    if j % 2 == 0:
        img = horz(f"Cat_images/cat{i+1 * j}.jpg")
        return img, np.array([1, 0])
    else:
        img = horz(f"Dog_images/dog{i+1 * j}.jpg")
        return img, np.array([0, 1])


def assign(img):
    px_val = []
    for i in range(197):
        for j in range(197):
            px_val.append(img.getpixel((i, j)))
    return px_val