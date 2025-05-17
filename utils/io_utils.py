from skimage.io import imread
from PIL import Image
import numpy as np
import pydicom
import cv2

# use at you own volition
def load_image(path, grayscale=False):
    return imread(path, as_gray=grayscale)

def load_image_grayscale_255(path):
    img = imread(path, as_gray=True) * 255
    return img

def load_image_grayscale_float(path):
    img = imread(path, as_gray=True)
    return img

def load_image_RGB_255(path):
    img = imread(path, as_gray=False)
    return img

def load_dicom(path):
    return pydicom.dcmread(path).pixel_array.astype(np.int16)

def load_rgb_and_resize(path, size):
    return np.array(Image.open(path).convert('RGB').resize(size))

def load_hsv_image(path):
    return np.array(Image.open(path).convert('HSV'))
