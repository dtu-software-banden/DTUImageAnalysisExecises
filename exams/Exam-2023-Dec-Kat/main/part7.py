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

import numpy as np
import matplotlib.pyplot as plt

# Define landmarks
standing = np.array([
    [1, 0],
    [2, 4],
    [3, 6],
    [4, 4],
    [5, 0]
])

running = np.array([
    [3, 1],
    [3.5, 3],
    [4.5, 6],
    [5.5, 5],
    [7, 1]
])

# Step 1: Compute SSD before alignment
ssd_initial = np.sum((standing - running) ** 2)

# Step 2: Compute centroids
centroid_standing = np.mean(standing, axis=0)
centroid_running = np.mean(running, axis=0)

# Step 3: Compute optimal translation (Δx, Δy)
translation = centroid_standing - centroid_running
running_translated = running + translation

# Step 4: Compute SSD after translation
ssd_translated = np.sum((standing - running_translated) ** 2)

# Step 5: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(*standing.T, color='blue', label='Standing (Reference)', marker='o')
plt.scatter(*running.T, color='red', label='Running (Template)', marker='x')
plt.scatter(*running_translated.T, color='green', label='Translated Running', marker='^')

# Draw lines between matched points before translation
for p1, p2 in zip(standing, running):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.3)

# Draw lines between matched points after translation
for p1, p2 in zip(standing, running_translated):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.5)

plt.legend()
plt.axis('equal')
plt.title(f"Landmark Alignment\nInitial SSD: {ssd_initial:.2f} → After Translation SSD: {ssd_translated:.2f}")
plt.grid(True)
plt.show()

def Question1():
    print("Running Question 1")
    # Print the optimal translation vector (Δx, Δy)
    print(f"Optimal translation vector (Δx, Δy): ({translation[0]:.2f}, {translation[1]:.2f})")

def Question2():
    print("Running Question 2")
    # Print the SSD before any transformation
    print(f"Sum of Squared Distances (before transformation): {ssd_initial:.2f}")

def Question3():
    print("Running Question 3")

    from skimage.transform import estimate_transform

    # Estimate similarity transform (includes rotation, translation, scaling)
    tform = estimate_transform('similarity', src=running, dst=standing)

    # Extract absolute value of the rotation (in degrees)
    rotation_degrees = abs(np.rad2deg(tform.rotation))
    print(rotation_degrees)



def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()