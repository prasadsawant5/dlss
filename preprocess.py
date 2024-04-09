import tensorflow as tf
import tensorflow_io as tfio
from config import *


def process_input(imagePath: str, is_all_channels: bool = False) -> tuple:
    # load the original image from disk, decode it as a JPEG image,
    # scale its pixel values to [0, 1] range, and resize the image
    origImage = tf.io.read_file(imagePath)
    origImage = tf.image.decode_jpeg(origImage, 3)
    origImage = tf.image.convert_image_dtype(origImage, tf.float32)
    origImage = tf.image.resize(origImage, HI_RES_SIZE, method="area")

    # convert the color space from RGB to YUV and only keep the Y
    # channel (which is our target variable)
    origImageYUV = tf.image.rgb_to_yuv(origImage)

    if is_all_channels:
        downImage = tf.image.resize(origImageYUV, LOW_RES_SIZE, method="area")
        downImage = tf.clip_by_value(downImage, 0.0, 1.0)
        target = tf.clip_by_value(origImageYUV, 0.0, 1.0)
    else:
        (target, _, _) = tf.split(origImageYUV, 3, axis=-1)
        # resize the target to a lower resolution
        downImage = tf.image.resize(target, LOW_RES_SIZE, method="area")
        # clip the values of the input and target to [0, 1] range
        target = tf.clip_by_value(target, 0.0, 1.0)
        downImage = tf.clip_by_value(downImage, 0.0, 1.0)
        # return a tuple of the downsampled image and original image

    return (downImage, target)


def process_rgb_input(imagePath: str) -> tuple:
    origImage = tf.io.read_file(imagePath)
    origImage = tf.image.decode_jpeg(origImage, 3)
    origImage = tfio.experimental.color.rgb_to_bgr(origImage)
    origImage = tf.image.convert_image_dtype(origImage, tf.float32)
    origImage = tf.image.resize(origImage, HI_RES_SIZE, method="area")

    downImage = tf.image.resize(origImage, LOW_RES_SIZE, method="area")
    # clip the values of the input and target to [0, 1] range
    target = tf.clip_by_value(origImage, 0.0, 1.0)
    downImage = tf.clip_by_value(downImage, 0.0, 1.0)
    # return a tuple of the downsampled image and original image
    return (downImage, target)

def preprocess_raw_image(imagePath: str) -> tuple:
    # load the original image from disk, decode it as a JPEG image,
    # scale its pixel values to [0, 1] range, and resize the image
    origImage = tf.io.read_file(imagePath)
    origImage = tf.image.decode_jpeg(origImage, 3)
    origImage = tf.image.convert_image_dtype(origImage, tf.float32)
    origImageYUV = tf.image.rgb_to_yuv(origImage)
    
    target = tf.image.resize(origImageYUV, HI_RES_SIZE, method="area")
    downImage = tf.image.resize(origImageYUV, LOW_RES_SIZE, method="area")
    
    target = tf.clip_by_value(target, 0.0, 1.0)
    downImage = tf.clip_by_value(downImage, 0.0, 1.0)

    return (downImage, target)

def preprocess_rgb_image(imagePath: str) -> tuple:
    # load the original image from disk, decode it as a JPEG image,
    # scale its pixel values to [0, 1] range, and resize the image
    origImage = tf.io.read_file(imagePath)
    origImage = tf.image.decode_jpeg(origImage, 3)
    origImage = tf.image.convert_image_dtype(origImage, tf.float32)
    
    target = tf.image.resize(origImage, HI_RES_SIZE, method="area")
    downImage = tf.image.resize(origImage, LOW_RES_SIZE, method="area")
    
    target = tf.clip_by_value(target, 0.0, 1.0)
    downImage = tf.clip_by_value(downImage, 0.0, 1.0)

    return (downImage, target)
