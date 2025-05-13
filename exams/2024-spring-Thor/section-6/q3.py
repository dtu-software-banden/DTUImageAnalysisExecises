import numpy as np
import cv2
from scipy.ndimage import median_filter
from PIL import Image

# Load the image
image = Image.open("pots.jpg").convert("RGB")  # Make sure it's RGB
image_np = np.array(image)

# Extract the red channel
red_channel = image_np[:, :, 0]

# Apply median filter with a 10x10 footprint
filtered_red = median_filter(red_channel, size=10)

# Threshold the image: pixels > 200 are foreground (value 1), others are background (value 0)
foreground_mask = filtered_red > 200

# Count foreground pixels
foreground_pixel_count = np.sum(foreground_mask)

print(f"Number of foreground pixels: {foreground_pixel_count}")
