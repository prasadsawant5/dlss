import tensorflow as tf
import cv2
import numpy as np
from imutils import paths
from config import *
from preprocess import *

if __name__ == '__main__':
    AUTO = tf.data.AUTOTUNE

    is_exit = False

    trainPaths = list(paths.list_images(DIV_HI_RES))
    trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)

    trainDS = trainDS.map(process_rgb_input, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

    for data in trainDS:
        low_res_images, hi_res_images = tf.cast(data[0] * 255., tf.uint8).numpy(), tf.cast(data[1] * 255., tf.uint8).numpy()
        for i in range(0, low_res_images.shape[0]):
            low_img = low_res_images[i]
            hi_img = hi_res_images[i]

            cv2.imshow('Low Res', low_img)
            cv2.moveWindow('Low Res', 10, 0)
            cv2.imshow('Hi Res', hi_img)
            cv2.moveWindow("Hi Res", 800, 0)

            key = cv2.waitKey(0)
            if key == 27:
                is_exit = True
                break

        if is_exit:
            cv2.destroyAllWindows()
            break
