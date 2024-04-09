import os

HI_RES_SIZE = (1080, 1920)
LOW_RES_SIZE = (360, 640)

OPENCV_LOW_RES = (640, 360)
OPENCV_HI_RES = (1920, 1080)

# specify root path to the BSDS500 dataset
ROOT_PATH = os.path.join("BSR", "BSDS500", "data", "images")
# specify paths to the different splits of the dataset
TRAIN_SET = os.path.join(ROOT_PATH, "train")
VAL_SET = os.path.join(ROOT_PATH, "val")
TEST_SET = os.path.join(ROOT_PATH, "test")

DIV_LOW_RES = './data/low_res_2x/'
DIV_HI_RES = './data/hi_res/'

DIV_TRAIN_SET = './data/hi_res/'

BATCH_SIZE = 1

UPSCALING_FACTOR = 3

LR = 1e-3

EPOCHS = 10
