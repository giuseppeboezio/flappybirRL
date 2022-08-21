# name of the environments
FLAPPY_BASE_NAME = "FlappyBird-v0"
FLAPPY_IMAGE_NAME = "FlappyBird-rgb-v0"

# input shape base model
BASE_SHAPE = (1, 8, 3)
# shape of the preprocessed image
IMAGE_SHAPE = (84, 84)

# maximum value of a pixel
MAX_PIXEL_VALUE = 255
# weights of the luminance function
WEIGHT_R = 0.3
WEIGHT_G = 0.59
WEIGHT_B = 0.11

# length of the timeseries of the base model
SERIES_LENGTH = 8
# number of channels for the stack of the cnn model
NUM_CHANNELS = 4

# name of the default pretrained model
BASE = "trained_base"
CNN = "trained_cnn"
ENTROPY = "trained_entropy"

# name of the directory where models, data and plots are stored
DIR_MODELS = "training/saved_models"
DIR_DATA = "training/data"
DIR_PLOT = "training/plot"
