import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.models.load_model('./saved_models/1')

    model.summary()