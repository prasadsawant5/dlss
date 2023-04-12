import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import inference_psnr
from config import *
import cv2
import numpy as np

SAVED_MODEL = './saved_models_rgb/1'
VIDEO = 'test.mp4'

if __name__ == '__main__':
    model = load_model(SAVED_MODEL, custom_objects={"psnr": inference_psnr})
    hi_res_normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)

    cap = cv2.VideoCapture(VIDEO)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            image = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            img = tf.image.resize(frame, LOW_RES_SIZE, method="area")
            img = hi_res_normalization(img)
            img = tf.expand_dims(img, axis=0)

            prediction = model.predict(img, batch_size=1)
            prediction = np.array(prediction[0] * 255, dtype=np.uint8)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            cv2.imshow('Original', frame)
            cv2.moveWindow('Original', 10, 0)
            cv2.imshow('Input', image)
            cv2.moveWindow("Input", 800, 0)
            cv2.imshow('Inference', prediction)
            cv2.moveWindow('Inference', 10, 600)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()