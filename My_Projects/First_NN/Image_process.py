from PIL import Image, ImageOps
import numpy as np


def horz(file):
    img = Image.open(file)
    img = ImageOps.grayscale(img)
    horizontal = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])

    new_img_h = []
    for a in range(img.size[1]-3):
        for b in range(img.size[0]-3):
            val = []
            for i in range(3):
                for j in range(3):
                    val.append(img.getpixel((b+i, a+j)))
            val = np.array(val)
            new_img_h.append(np.dot(horizontal, val))

    new_img_horz = Image.new("L", (img.size[0] - 3, img.size[1] - 3))
    new_img_horz.putdata(new_img_h)

    return new_img_horz


def vert(file):
    img = Image.open(file)
    img = ImageOps.grayscale(img)
    vertical = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1])

    new_img_v = []
    for a in range(img.size[1]-3):
        for b in range(img.size[0]-3):
            val = []
            for i in range(3):
                for j in range(3):
                    val.append(img.getpixel((b+i, a+j)))
            val = np.array(val)
            new_img_v.append(np.dot(vertical, val))

    new_img_vert = Image.new("L", (img.size[0]-3, img.size[1]-3))
    new_img_vert.putdata(new_img_v)

    return new_img_vert


def show(type, file):
    return type(file).show()