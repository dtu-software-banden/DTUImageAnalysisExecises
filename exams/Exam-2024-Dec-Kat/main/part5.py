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


from skimage.io import imread
import numpy as np

def extract_landmarks(label_img):
    landmarks = []
    for label in range(1, 6):
        y, x = np.where(label_img == label)
        if len(x) == 0 or len(y) == 0:
            print(f"Warning: Label {label} not found.")
            landmarks.append((np.nan, np.nan))
        else:
            landmarks.append((x[0], y[0]))
    return landmarks


fixed = imread("data/cells/LabelsFixedImg.png")
moving = imread("data/cells/LabelsMovingImg.png")

fixed_landmarks = extract_landmarks(fixed)
moving_landmarks = extract_landmarks(moving)

print("Fixed image landmarks (x, y):")
print(fixed_landmarks)

print("Moving image landmarks (x, y):")
print(moving_landmarks)

from skimage.morphology import opening, disk
from skimage.measure import label


def Question1():
    print("Running Question 1")

    # Load grayscale images
    img_x = imread("data/cells/x_NisslStain_9-260.81.png", as_gray = True)
    img_y = imread("data/cells/y_NisslStain_9-260.81.png", as_gray = True)

    threshold_value = 30 / 255  # ≈ 0.1176

    bin_x = img_x > threshold_value
    bin_y = img_y > threshold_value

    # Morphological opening
    opened_x = opening(bin_x, disk(3))
    opened_y = opening(bin_y, disk(3))

    # Label connected components
    labeled_x = label(opened_x)
    labeled_y = label(opened_y)

    # Plot the results again
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(labeled_x, cmap='nipy_spectral')
    axes[0].set_title("Labeled BLOBs: x_NisslStain")
    axes[0].axis("off")

    axes[1].imshow(labeled_y, cmap='nipy_spectral')
    axes[1].set_title("Labeled BLOBs: y_NisslStain")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



def Question2():
    print("Running Question 2")
    from scipy.spatial.distance import euclidean
    # Compute centroids (average positions)
    fixed_mean = np.mean(fixed_landmarks, axis=0)
    moving_mean = np.mean(moving_landmarks, axis=0)

    # Compute Euclidean distance between the centroids
    avg_distance = euclidean(fixed_mean, moving_mean)

    print(avg_distance)

def Question3():
    print("Running Question 3")
    print("Between 2 and 5")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    #Question1()
    Question2()
    Question3()
    Question4()