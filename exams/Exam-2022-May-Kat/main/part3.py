import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 
from skimage.morphology import closing, disk, label, erosion, dilation

def Question1():
    print("Running Question 1")
    print("A guess for now but , rho = 5 theta = 0??? come bacK!!")


def Question2():
    print("Running Question 2")
    print(1)

import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import disk, erosion, dilation
from skimage.color import rgb2hsv
from skimage.io import imread



img_hsv = load_hsv_from_rgb("./data/car.png")

saturation_img = img_hsv[:, :, 1] > 0.7

# Morphological erosion with disk radius 6, padding with 1 (MATLAB default for erosion)
padded = np.pad(saturation_img, pad_width=6, mode='constant', constant_values=1)
eroded = erosion(padded, disk(6))
eroded = eroded[6:-6, 6:-6]  # Remove padding

# Morphological dilation with disk radius 4, padding with 0 (MATLAB default for dilation)
padded = np.pad(eroded, pad_width=4, mode='constant', constant_values=0)
dilated = dilation(padded, disk(4))
dilated = dilated[4:-4, 4:-4]  # Remove padding

def Question3():
    print("Running Question 3")
    count = np.sum(dilated)
    print(count)

img_hsv = load_hsv_from_rgb("./data/road.png")

value_img = img_hsv[:, :, 2] > 0.9

# 8-connectivity, 8 connectivity
labeled = label(value_img, connectivity=2)
mask = np.zeros_like(value_img, dtype=bool)


def Question4():
    print("Running Question 4")
    min_area = 1040
    for region in regionprops(labeled):
        if region.area > min_area :
            mask[labeled == region.label] = True

    plot_image(mask)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()