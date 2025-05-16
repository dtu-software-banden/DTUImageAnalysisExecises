from skimage.io import imread
from PIL import Image
import numpy as np
import pydicom

def load_image(path, grayscale=False):
    return imread(path, as_gray=grayscale)

def load_dicom(path):
    return pydicom.dcmread(path).pixel_array.astype(np.int16)

def load_rgb_and_resize(path, size):
    return np.array(Image.open(path).convert('RGB').resize(size))

def load_hsv_image(path):
    return np.array(Image.open(path).convert('HSV'))
