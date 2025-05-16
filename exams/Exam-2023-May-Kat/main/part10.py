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
from skimage.filters import threshold_otsu
from skimage.transform import rotate

# Step 1: Load and rotate the image 11 degrees around (40, 40)
lights_rgb = imread("./data/lights.png")
rotated = rotate(lights_rgb, angle=11, center=(40, 40), preserve_range=True).astype(np.uint8)

# Step 2: Convert to grayscale
gray = rgb2gray(rotated)

# Step 3: Compute Otsu's threshold
threshold = threshold_otsu(gray)

# Step 4: Compute binary foreground mask and percentage
binary_mask = gray > threshold
foreground_percentage = np.mean(binary_mask) * 100  # in percent


def Question1():
    print("Running Question 1")
    print(foreground_percentage)


def Question2():
    print("Running Question 2")
    print(threshold)


if __name__ == "__main__":
    Question1()
    Question2()