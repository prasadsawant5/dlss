import os
import tensorflow as tf
from imutils import paths
from config import *
from tensorflow.keras.optimizers import Adam
from preprocess import *
from model.my_model import MyModel
from utils import get_dataset


def psnr(orig, pred):
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    AUTO = tf.data.AUTOTUNE

    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    idx = os.listdir(logs_dir)
    if len(idx) == 0:
       idx = '1'
    else:
        idx = str(int(idx[-1]) + 1)

    tb_logs_dir = os.path.join(logs_dir, idx)
    if not os.path.exists(tb_logs_dir):
        os.mkdir(tb_logs_dir)

    trainPaths = list(paths.list_images(DIV_HI_RES))
    trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)

    trainDS = trainDS.map(process_rgb_input, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

    optimizer = Adam(LR)

    model = MyModel().build_model(channels=3, is_rgb=True)
    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_dir, profile_batch=5)
    model.compile(optimizer=optimizer, loss='mse', metrics=psnr)
    model.fit(trainDS, validation_data=None, epochs=EPOCHS, callbacks=[tensorboard_callback])

    save_path = os.path.join('saved_models_rgb', idx)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model.save(save_path)
    print('Model saved to path: {}'.format(save_path))

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(save_path, 'model_rgb.tflite'), 'wb') as f:
        f.write(tflite_model)
