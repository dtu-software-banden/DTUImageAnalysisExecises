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

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import disk, dilation

# Load the image
image_path = "./data/nike.png"
rgb_image = imread(image_path)

# Step 1: Convert RGB to HSV
hsv_image = rgb2hsv(rgb_image)

# Step 2: Extract H component
h_component = hsv_image[:, :, 0]

# Step 3: Threshold H values between 0.3 and 0.7 to create binary mask
binary_mask = ((h_component > 0.3) & (h_component < 0.7)).astype(int)

# Step 4: Morphological dilation with a disk of radius 8
dilated_mask = dilation(binary_mask, disk(8))

# Show original and processed images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb_image)
axes[0].set_title("Original RGB")
axes[1].imshow(binary_mask, cmap='gray')
axes[1].set_title("Binary H ∈ (0.3, 0.7)")
axes[2].imshow(dilated_mask, cmap='gray')
axes[2].set_title("Dilated Mask")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

def Question1():
    print("Running Question 1")
    # Count number of foreground pixels (value = 1) in the dilated binary mask
    foreground_pixel_count = np.sum(dilated_mask == 1)
    print(f"Number of foreground pixels: {foreground_pixel_count:.4f}")


if __name__ == "__main__":
    Question1()