import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 

from skimage.io import imread
from skimage.color import rgb2gray

# Step 1: Convert both images to grayscale
background_rgb = imread("./data/ChangeDetection/background.png")
new_frame_rgb = imread("./data/ChangeDetection/new_frame.png")

background_gray = rgb2gray(background_rgb)
new_frame_gray = rgb2gray(new_frame_rgb)

# Step 2: Update the background using exponential moving average
alpha = 0.90
updated_background = alpha * background_gray + (1 - alpha) * new_frame_gray

print(updated_background.shape, updated_background.min(), updated_background.max())

# Step 3: Compute the absolute difference image
difference_image = np.abs(new_frame_gray - updated_background)

print(difference_image.shape, difference_image.min(), difference_image.max())

# Step 4: Count changed pixels (value > 0.1 in the difference image)
changed_pixels = np.sum(difference_image > 0.1)


def Question1():
    print("Running Question 1")
    print(f"Changed pixels {changed_pixels}")


def Question2():
    print("Running Question 2")
    # Extract the region [150:200, 150:200] and compute its average value
    region_avg = np.mean(updated_background[150:200, 150:200])
    print(f"Region average{region_avg}")

if __name__ == "__main__":
    Question1()
    Question2()