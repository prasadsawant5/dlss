import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
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
                x = tf.image.rgb_to_yuv(inputs)
                (Y, U, V) = tf.split(x, 3, axis=-1, name='split_channels')
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
                U = UpSampling2D((2, 2), interpolation="nearest", name='upsample_u')(U)
                V = UpSampling2D((2, 2), interpolation="nearest", name='upsample_v')(V)
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
    
    def build_yuv_model(self, padding: str = 'same', act: str = 'relu', k_init: str = 'Orthogonal') -> Model:
        inputs = Input(shape=tuple(list(LOW_RES_SIZE) + [3]), name='inputs')
        (Y, U, V) = tf.split(inputs, 3, axis=-1, name='split_channels')
        
        x_y = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init, name='conv0_y')(Y)
        x_u = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init, name='conv0_u')(U)
        x_v = Conv2D(64, 5, padding=padding, activation=act, kernel_initializer=k_init, name='conv0_v')(V)

        x_y = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv1_y')(x_y)
        x_u = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv1_u')(x_u)
        x_v = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv1_v')(x_v)

        x_y = Rdb().get_layer(x_y)
        x_y = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv2_y')(x_y)
        x_y = Rdb().get_layer(x_y)

        x_u = Rdb().get_layer(x_u)
        x_u = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv2_u')(x_u)
        x_u = Rdb().get_layer(x_u)

        x_v = Rdb().get_layer(x_v)
        x_v = Conv2D(64, 3, padding=padding, activation=act, kernel_initializer=k_init, name='conv2_v')(x_v)
        x_v = Rdb().get_layer(x_v)

        x_y = Conv2D((UPSCALING_FACTOR ** 2), kernel_size=3, padding=padding, kernel_initializer=k_init, name='conv3_y')(x_y)
        x_u = Conv2D((UPSCALING_FACTOR ** 2), kernel_size=3, padding=padding, kernel_initializer=k_init, name='conv3_u')(x_u)
        x_v = Conv2D((UPSCALING_FACTOR ** 2), kernel_size=3, padding=padding, kernel_initializer=k_init, name='conv3_v')(x_v)

        outputs_y = tf.nn.depth_to_space(x_y, UPSCALING_FACTOR, name="outputs_y")
        outputs_u = tf.nn.depth_to_space(x_y, UPSCALING_FACTOR, name="outputs_u")
        outputs_v = tf.nn.depth_to_space(x_y, UPSCALING_FACTOR, name="outputs_v")

        outputs = tf.concat([outputs_y, outputs_u, outputs_v], axis=-1, name="outputs")

        return Model(inputs, outputs)
