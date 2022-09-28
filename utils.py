import os
import tensorflow as tf
from config import *
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from tensorflow.keras.preprocessing import image_dataset_from_directory


def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataset() -> MapDataset:
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


    if not os.path.exists(DIV_HI_RES):
        print('Dir does not exists')

    low_res_images = image_dataset_from_directory(
        DIV_HI_RES,
        label_mode=None,
        image_size=LOW_RES_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    hi_res_images = image_dataset_from_directory(
        DIV_HI_RES,
        label_mode=None,
        image_size=HI_RES_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((low_res_images, hi_res_images))
    dataset = dataset.map(lambda low_res, hi_res: (normalization(low_res), normalization(hi_res)))
    dataset = configure_for_performance(dataset)

    return dataset


def inference_psnr(orig, pred):
    # cast the target images to integer
    orig = orig * 255.0
    orig = tf.cast(orig, tf.uint8)
    orig = tf.clip_by_value(orig, 0, 255)
    # cast the predicted images to integer
    pred = pred * 255.0
    pred = tf.cast(pred, tf.uint8)
    pred = tf.clip_by_value(pred, 0, 255)
    # return the psnr
    return tf.image.psnr(orig, pred, max_val=255)
