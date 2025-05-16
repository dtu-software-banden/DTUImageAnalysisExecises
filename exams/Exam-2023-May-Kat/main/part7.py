import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.morph_utils import *
from utils.plot_utils import *
from skimage.morphology import erosion, disk

# Load the image
letters_image = imread("./data/Letters.png")

# Extract RGB channels
R = letters_image[:, :, 0]
G = letters_image[:, :, 1]
B = letters_image[:, :, 2]

# Create binary image for red detection
binary_red = ((R > 100) & (G < 100) & (B < 100)).astype(np.uint8)

# Apply morphological erosion with disk(radius=3)
eroded = erosion(binary_red, disk(3))

# Count foreground pixels in the eroded image
foreground_pixel_count = np.sum(eroded == 1)


print(foreground_pixel_count)


from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import square

# Step 1: Convert to grayscale
letters_gray = rgb2gray(letters_image)

# Step 2: Apply median filter with square footprint of size 8
filtered = median(letters_gray, footprint=square(8))

# Step 3: Extract the value at pixel (100, 100)
pixel_value = filtered[100, 100]
print(pixel_value)


def Question3():
    from skimage.measure import label, regionprops
    from skimage.morphology import disk

    # Step 1: Compute binary image where R > 100, G < 100, B < 100
    R = letters_image[:, :, 0]
    G = letters_image[:, :, 1]
    B = letters_image[:, :, 2]
    binary_image = ((R > 100) & (G < 100) & (B < 100)).astype(np.uint8)

    eroded_image = erosion(binary_image, disk(3))
    mask = filter(eroded_image, min_area=1000, max_area=4000, min_perim = 300, max_perim = np.inf)

    overlay = letters_image.copy()
    overlay[~mask] = 0

    # Display original and overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(letters_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title("Filtered Letters")
    axes[1].axis('off')

    axes[2].imshow(binary_image)
    axes[2].set_title("Binary")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

Question3()