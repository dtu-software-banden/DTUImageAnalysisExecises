import cv2
import numpy as np
from skimage.util import img_as_float,img_as_ubyte
from utils.io_utils import load_image
from utils.plot_utils import plot_image
from utils.morph_utils import erode_circle

image = load_image("exams/2023-spring-Thor/section7/Letters.png",grayscale=True)


blurred = img_as_float(cv2.medianBlur(img_as_ubyte(image),7))
print("value at coord",blurred[100,100])
plot_image(blurred)