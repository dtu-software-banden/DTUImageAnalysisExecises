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


import os
import numpy as np
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

img_dir = "data/screws/"


image_filenames = sorted([
    f"screws_{i:03d}.jpg" for i in range(20)
])

# Load and stack all images into a 4D array (N, H, W, C)
images = [imread(os.path.join(img_dir, fname)) for fname in image_filenames]
image_stack = np.stack(images, axis=0)

# Compute average image
average_screw = np.mean(image_stack, axis=0).astype(np.uint8)

# Display average pizza
plt.imshow(average_screw)
plt.title("Average Screw (RGB)")
plt.axis('off')
plt.show()

# Flatten images for PCA (shape: 10 x (H*W*3))
flat_images = image_stack.reshape(len(image_stack), -1)

# Run PCA on flattened images
pca, projections = compute_pca(flat_images,7)

def Question1():
    print("Running Question 1")
    # Compute PCA

    print(pca.explained_variance_ratio_)
    explained = 0
    num_components = 0
    for i in pca.explained_variance_ratio_:
        num_components += 1
        explained = explained + i
        if explained >= 0.44:
            break
        
    print(f"Number of PCA components needed for ≥44% variance: {num_components}")


def Question2():
    print("Running Question 2")
    # Track best pair and minimum distance
    min_dist = np.inf
    best_pair = (None, None)

    for i in range(len(projections)):
        for j in range(len(projections)): 
            if j == i:
                continue 
            dist = np.linalg.norm(projections[i] - projections[j])
            if dist < min_dist:
                min_dist = dist
                best_pair = (i, j)

    print(f"The two most similar photos in PCA space are: screws_{best_pair[0]:03d}.jpg and screws_{best_pair[1]:03d}.jpg")



def Question3():
    print("Running Question 3")
    # First principal component values
    pc1_values = projections[:, 0]

    min_index = np.argmin(pc1_values)
    max_index = np.argmax(pc1_values)

    print(f"Lowest PC1: screws_{min_index:03d}.jpg")
    print(f"Highest PC1: screws_{max_index:03d}.jpg")

def Question4():
    print("Running Question 4")
    # Compute Euclidean distance
    coords, distance = most_similair_indexs(projections)

    print(f"Distance between screws_007.jpg and screws_008.jpg in PCA space: {distance[7][8]}")

import matplotlib.pyplot as plt

def Question5():
    print("Running Question 5")

    # Get PC1 and PC2 values
    pc1 = projections[:, 0]
    pc2 = projections[:, 1]

    # Plot all points
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c='lightgray', label='All screws')
    plt.scatter(pc1[7], pc2[7], c='red', label='screws_007.jpg', marker='X', s=100)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Screw Images in PCA Space (PC1 vs PC2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()
    Question5()
    #Question6()