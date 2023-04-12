import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from model.layers.rdb import Rdb
from tensorflow.keras.models import Model
from config import *


class MyModel:
    def build_model(self, padding: str = 'same', act: str = 'relu', k_init: str = 'Orthogonal', channels: int = 1, is_rgb: bool = False) -> Model:
        if channels == 3:
            inputs = Input(shape=tuple(list(LOW_RES_SIZE) + [channels]), name='inputs')

            if is_rgb:
                x = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init, name='conv0')(inputs)
            else:
                (Y, U, V) = tf.split(inputs, 3, axis=-1, name='split_channels')
                x = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init, name='conv0')(Y)

            x = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv1')(x)

            x = Rdb().get_layer(x)
            x = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init)(x)
            x = Rdb().get_layer(x)

            if is_rgb:
                x = Conv2D(12, kernel_size=3, padding=padding, kernel_initializer=k_init)(x)
            else:
                x = Conv2D(1 * (UPSCALING_FACTOR ** 2), kernel_size=3, padding=padding, kernel_initializer=k_init)(x)
            outputs = tf.nn.depth_to_space(x, UPSCALING_FACTOR, name='depth_to_space')

            if not is_rgb:
                U = tf.image.resize(U, size=HI_RES_SIZE, method="area", name='resize_u')
                V = tf.image.resize(V, size=HI_RES_SIZE, method="area", name='resize_v')
                outputs = tf.concat([outputs, U, V], axis=-1)

        else:
            inputs = Input(shape=tuple(list(LOW_RES_SIZE) + [channels]), name='inputs')

            x = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init)(inputs)
            x = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init)(x)

            x = Rdb().get_layer(x)
            x = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init)(x)
            x = Rdb().get_layer(x)

            x = Conv2D(channels * (UPSCALING_FACTOR ** 2), kernel_size=3, padding=padding, kernel_initializer=k_init)(x)
            outputs = tf.nn.depth_to_space(x, UPSCALING_FACTOR)

        return Model(inputs, outputs)
