from utils.io_utils import load_image
import cv2
import numpy as np
from skimage.filters import threshold_otsu

from utils.plot_utils import plot_image


image = load_image("exams/2023-spring-Thor/section9/lights.png",grayscale=True)


M = cv2.getRotationMatrix2D((40,40),11,1.0)

# Get the size of the image
(h, w) = image.shape[:2]

# Apply the rotation
rotated = cv2.warpAffine(image, M, (w, h))

otsu_thresh = threshold_otsu(rotated)
print("otsu: ",otsu_thresh)

otsu_image = rotated > otsu_thresh

print("average:",otsu_image.mean())