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

from skimage.io import imread
import numpy as np

# Load zebra image and masks
zebra_img = load_image_grayscale_255("data/zebra/Zebra.png")
white_mask = imread("data/zebra/Zebra_whiteStripes.png") > 0
black_mask = imread("data/zebra/Zebra_blackStripes.png") > 0
roi_mask = imread("data/zebra/Zebra_MASK.png") > 0

# Get training data
white_values = zebra_img[white_mask]
black_values = zebra_img[black_mask]

# Compute means and stds for both classes
mu_white, sigma_white = gaussian_parameters(white_values)
mu_black, sigma_black = gaussian_parameters(black_values)

# Apply classifier inside ROI
roi_pixels = zebra_img[roi_mask]

# Gaussian likelihoods for class 1 (white) and class 2 (black)
lik_white = -((roi_pixels - mu_white) ** 2) / (2 * sigma_white ** 2)
lik_black = -((roi_pixels - mu_black) ** 2) / (2 * sigma_black ** 2)

# Classify as white if likelihood under white class is greater
classified_as_white = lik_white > lik_black




def Question1():
    print("Running Question 1")
    num_white_pixels = np.sum(classified_as_white)
    print(num_white_pixels)


def Question2():
    print("Running Question 2")

    # Check true intensity range for black stripe training pixels
    print(black_values)
    black_min = np.min(black_values)
    black_max = np.max(black_values)
    (black_min, black_max)

def Question3():
    print("Running Question 3")
    print(mu_white, sigma_white)


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()