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
from skimage.transform import SimilarityTransform, warp
import numpy as np

# Load images
shoe1 = imread("./data/LMRegistration/shoe_1.png")
shoe2 = imread("./data/LMRegistration/shoe_2.png")

# Landmarks for registration
src = np.array([[40, 320], [425, 120], [740, 330]])      # landmarks on shoe_1.png (source)
dst = np.array([[80, 320], [380, 155], [670, 300]])      # landmarks on shoe_2.png (destination)

# Step 1: Compute similarity transform
tform = SimilarityTransform()
tform.estimate(src, dst)

# Step 2: Apply transform to warp shoe_1 image
registered_shoe1 = warp(shoe1, inverse_map=tform.inverse, output_shape=shoe2.shape)

# Step 3: Extract the scale from the transform
found_scale = tform.scale

print(f"Found Scale:",found_scale)
# Before registration: compare original source landmarks to destination
F_before = np.sum(np.square(src - dst))

# After registration: transform source landmarks and compare
transformed_src = tform(src)
F_after = np.sum(np.square(transformed_src - dst))
print(f"F before: {F_before:.4f} F after: {F_after:.4f}")

# Ensure the registered image is in the same scale and type as the original (0-255, uint8) for fair color comparison
registered_shoe1_uint8 = (registered_shoe1 * 255).astype(np.uint8)

# Extract the blue channels from both images
blue_shoe1 = registered_shoe1_uint8[:, :, 2]
blue_shoe2 = shoe2[:, :, 2]

# Compute the mean blue value in each image
mean_blue_shoe1 = np.mean(blue_shoe1)
mean_blue_shoe2 = np.mean(blue_shoe2)

print(f"Blue shoe1 mean: {mean_blue_shoe1:.4f} Blue shoe2 mean: {mean_blue_shoe2:.4f}")


def Question1():
    print("Running Question 1")
    from skimage.util import img_as_ubyte

    # Convert both images to byte format
    shoe1_byte = img_as_ubyte(registered_shoe1)
    shoe2_byte = img_as_ubyte(shoe2)

    # Extract the blue component at position (200, 200)
    blue_shoe1_pixel = shoe1_byte[200, 200, 2]
    blue_shoe2_pixel = shoe2_byte[200, 200, 2]

    # Compute absolute difference
    abs_difference = abs(int(blue_shoe1_pixel) - int(blue_shoe2_pixel))
    print(abs_difference)

def Question2():
    print("Running Question 2")
    print(f"Found Scale:",found_scale)


def Question3():
    print("Running Question 3")
   
    print(f"Change in F { F_after-F_before:.4f}")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()