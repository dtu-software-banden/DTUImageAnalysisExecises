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

img1 = load_hsv_image("./data/ChangeDetection/frame_1.jpg")
img2 = load_hsv_image("./data/ChangeDetection/frame_2.jpg")

print(img1.shape)

# Extract and scale the S channel to [0, 255]
s1 = (img1[:, :, 1] * 255).astype(np.uint8)
s2 = (img2[:, :, 1] * 255).astype(np.uint8)

#print(s1,s2)

abs_diff = np.abs(s1.astype(np.int16) - s2.astype(np.int16)).astype(np.uint8)

mean_diff = np.mean(abs_diff)
std_diff = np.std(abs_diff)

threshold = mean_diff + 2 * std_diff

binary = (abs_diff > threshold).astype(np.uint8)

changed_pixels = np.sum(binary)

from skimage.measure import label, regionprops

# Label the connected components in the binary image
label_image = label(binary)

# Extract region properties
regions = regionprops(label_image)

# Number of blobs
num_blobs = len(regions)

# Centroids of each blob
centroids = [region.centroid for region in regions]

def Question1():
    print("Running Question 1")
    # Compute the size (area) of each blob
    blob_areas = [region.area for region in regions]

    # Find the size of the largest blob
    largest_blob_area = max(blob_areas) if blob_areas else 0
    print(largest_blob_area)


def Question2():
    print("Running Question 2")
    print(threshold)


def Question3():
    print("Running Question 3")
    print(changed_pixels)


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()