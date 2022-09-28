import tensorflow as tf
import tensorflow_io as tfio
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from utils import inference_psnr
from tensorflow.keras.models import load_model
from config import *
import cv2
import numpy as np
from test_utils import *

SAVED_MODEL = './saved_models/3'
VIDEO = 'test.mp4'


if __name__ == '__main__':
    model = load_model(SAVED_MODEL, custom_objects={"psnr": inference_psnr})

    show_native = False

    cap = cv2.VideoCapture(VIDEO)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            image = cv2.resize(frame, OPENCV_LOW_RES)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            (y, cb, cr) = convert_to_yCbCr(pil_img)

            upscaledY = model.predict(y[None, ...])[0]
            finalOutput = postprocess_image(upscaledY, cb, cr)
            inference = np.array(finalOutput)
            inference = inference[:, :, ::-1].copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if show_native:
                native = cv2.resize(image, OPENCV_HI_RES, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Native Resizing', inference)
                cv2.moveWindow('Native Resizing', 10, 0)

                cv2.imshow('Inference', inference)
                cv2.moveWindow('Inference', 1000, 0)
            else:
                cv2.imshow('Input', image)
                cv2.moveWindow("Input", 10, 0)
                cv2.imshow('Inference', inference)
                cv2.moveWindow('Inference', 600, 0)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
