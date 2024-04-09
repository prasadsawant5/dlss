import numpy as np
import tensorflow as tf
from PIL import Image
from config import *

def get_y_channel(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, LOW_RES_SIZE, method="area")

    # convert the color space from RGB to YUV and only keep the Y
    # channel (which is our target variable)
    ycbcr = tf.image.rgb_to_yuv(image)
    (y, cb, cr) = tf.split(ycbcr, 3, axis=-1)

    return (y, cb, cr)

def clip_numpy(image):
    # cast image to integer, clip its pixel range to [0, 255]
    image = tf.cast(image * 255.0, tf.uint8)
    image = tf.clip_by_value(image, 0, 255).numpy()
    # return the image
    return image


def postprocess_image(y, cb, cr):
    # do a bit of initial preprocessing, reshape it to match original
    # size, and then convert it to a PIL Image
    y = clip_numpy(y).squeeze()
    y = y.reshape(y.shape[0], y.shape[1])
    y = Image.fromarray(y, mode="L")

    # resize the other channels of the image to match the original
    # dimension
    outputCB = cb.resize(y.size, Image.BICUBIC)
    outputCR = cr.resize(y.size, Image.BICUBIC)
    # merge the resized channels altogether and return it as a numpy
    # array
    final = Image.merge("YCbCr", (y, outputCB, outputCR)).convert("RGB")
    return np.array(final)

def convert_to_yCbCr(image: Image):
    ycbcr = image.convert("YCbCr")
    (y, cb, cr) = ycbcr.split()

    y = np.array(y)
    y = y.astype("float32") / 255.0

    return (y, cb, cr)

# def convert_to_yuv(image: Image):
#     ycbcr = image.convert("YCbCr")
#     (y, cb, cr) = ycbcr.split()

#     denominator = 240. - 16.

#     y = np.array(y)
#     y = y.astype("float32") / 255.0

#     cb = np.array(cb)
#     cb = (cb.astype("float32") - 16.) / denominator

#     cr = np.array(cr)
#     cr = (cr.astype("float32") - 16.) / denominator

#     return np.array([y, cb, cr])
