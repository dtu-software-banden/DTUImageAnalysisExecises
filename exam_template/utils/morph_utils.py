import cv2
from skimage.morphology import closing, disk, label
from skimage.measure import regionprops
import numpy as np

def segment_blobs(image, threshold=200, disk_radius=3, min_area=500):
    binary = image > threshold
    closed = closing(binary, disk(disk_radius))
    labeled = label(closed)

    mask = np.zeros_like(image, dtype=bool)
    for region in regionprops(labeled):
        if region.area > min_area:
            mask[labeled == region.label] = True
    return mask


def clean_and_filter(binary_image, radius=3, min_area=500):
    cleaned = closing(binary_image, disk(radius))
    labeled = label(cleaned)

    mask = np.zeros_like(binary_image, dtype=bool)
    for region in regionprops(labeled):
        if region.area > min_area:
            mask[labeled == region.label] = True
    return mask


def dialate_circle(binary_image, radius=8):
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary_image,kernel)