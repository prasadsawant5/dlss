import os
from test_utils import *
from tensorflow.python.compiler.tensorrt import trt_convert as trt

MODEL = './saved_models/1'
TENSORRT_MODEL = './tensorrt_model'


def my_input_fn():
    pass

if __name__ == '__main__':
    if not os.path.exists(MODEL):
        print('Model path does not exist')
        exit(0)

    if not os.path.exists(TENSORRT_MODEL):
        os.mkdir(TENSORRT_MODEL)

    conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP32)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=MODEL,
        conversion_params=conversion_params
    )
    converter.convert()

    converter.save(TENSORRT_MODEL, 'tensorrt_model')

