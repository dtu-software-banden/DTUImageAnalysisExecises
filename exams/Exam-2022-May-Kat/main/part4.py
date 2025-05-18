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
from skimage.io import imread

# Directory containing spoon images
spoon_dir = "./data/Spoonz"
spoon_filenames = sorted([f"spoon{i}.png" for i in range(1, 7)])

# Load and stack all spoon images into a 4D array (N, H, W, C)
spoon_images = [imread(os.path.join(spoon_dir, fname)) for fname in spoon_filenames]
spoon_stack = np.stack(spoon_images, axis=0)

# Compute average spoon image
average_spoon = np.mean(spoon_stack, axis=0)
flat_images = spoon_stack.reshape(len(spoon_stack), -1)

pca, projection = compute_pca(flat_images, 2)

def Question1():
    print("Running Question 1")
    print(pca.explained_variance_ratio_[:2].sum())
    print(projection[0])




def Question2():
    print("Running Question 2")
    # Load and stack all spoon images into a 4D array (N, H, W, C)
    spoon_images = [imread(os.path.join(spoon_dir, fname)) > 100 for fname in spoon_filenames]
    spoon_stack = np.stack(spoon_images, axis=0)

    # Compute average spoon image
    average_spoon = np.mean(spoon_stack, axis=0).astype(np.uint8)
    flat_images = spoon_stack.reshape(len(spoon_stack), -1)

    pca, projection = compute_pca(flat_images, 2)

    print(pca.explained_variance_ratio_)


def Question3():
    print("Running Question 3")
    plot_image(average_spoon)
    print(average_spoon[499][99])

from utils.affine_trans_utils import * 
def Question4():
    print("Running Question 4")
    # Step 1: Rotation matrix (20 degrees CCW)
    theta = np.radians(20)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # Step 2: Scaling matrix (uniform scale of 2)
    S = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ])

    # Step 3: Translation matrix (+3.1, -3.3)
    T = np.array([
        [1, 0, 3.1],
        [0, 1, -3.3],
        [0, 0, 1]
    ])

    # Combine transformations: T * S * R
    transformation_matrix = T @ S @ R

    # Point (10, 10) in homogeneous coordinates
    point = np.array([10, 10, 1])

    # Apply transformation
    transformed_point = transformation_matrix @ point
    transformed_point[:2]


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()