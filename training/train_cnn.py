import cv2
import numpy as np
from utils import IMAGE_SHAPE, MAX_PIXEL_VALUE


def grayscale(image):
    """
    Convert a 3d image to a 2d one keeping the maximum value along each channel
    :param image: 3d image
    :return 2d grayscale image
    """
    return np.max(image, axis=2)


def rescale(image):
    """
    Rescale the image to a given size
    :param image: image to rescale
    :return rescaled image
    """
    return cv2.resize(image, IMAGE_SHAPE, interpolation = cv2.INTER_AREA)


def normalize(image):
    """
    Normalize each pixel value between 0 and 1
    :param image: image to normalize
    :return normalized image
    """
    return image / MAX_PIXEL_VALUE


# preprocessing of the screenshot
def preprocess_image(functions_list, image):
    """
    Preprocess an image applying in sequence functions provided as parameters
    :param functions_list: sequence of functions to apply to the image
    :param image: image to be processed
    :return preprocessed image
    """
    output = image
    for function in functions_list:
        output = function(output)

    return output



