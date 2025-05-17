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
from skimage.io import imread
import os

roi_data = {
    "A": np.loadtxt("data/dice/A_Cubes.txt"),
    "B": np.loadtxt("data/dice/B_Cubes.txt"),
    "C": np.loadtxt("data/dice/C_Cubes.txt"),
    "D": np.loadtxt("data/dice/D_Cubes.txt"),
    "E": np.loadtxt("data/dice/E_Cubes.txt")
}

# Define the desired combinations
combinations_list = [
    ("A", "B", "C"),
    ("A", "D", "E"),
    ("C", "D", "E"),
    ("B", "C", "D"),
    ("B", "D", "E")
]

classifiers = {}

for combo in combinations_list:
    # Compute the mean for each class
    means = np.array([roi_data[label].mean() for label in combo]).reshape(-1, 1)
    classifiers[combo] = means

def Question1():
    print("Running Question 1")
    from skimage.io import imread
    from sklearn.metrics import pairwise_distances_argmin
    import matplotlib.pyplot as plt

    # Reload image
    image = imread("data/dice/CubesG.png")

    # Create segmentation masks for each classifier
    segmentations = {}

    for combo, means in classifiers.items():
        # Apply minimum distance classifier
        flat_image = image.flatten().reshape(-1, 1)
        labels = pairwise_distances_argmin(flat_image, means) + 1  # Add 1 to keep class labels from 1 to 3
        labels = labels.reshape(image.shape)

        # Set background (intensity 0) back to 0
        labels[image == 0] = 0

        segmentations[combo] = labels

    # Plot all segmentations for visual comparison
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (combo, seg) in zip(axes, segmentations.items()):
        ax.imshow(seg, cmap='viridis')
        ax.set_title(f"Classifier {combo}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    print("B,C,D")



def get_gaussian_discriminant_coeffs(mu1, var1, mu2, var2):
    """
    Compute coefficients A, B, C for the quadratic equation that defines the optimal threshold
    between two Gaussian distributions with means mu1, mu2 and variances var1, var2.
    """
    A = 1 / var2 - 1 / var1
    B = -2 * mu2 / var2 + 2 * mu1 / var1
    C = (mu2**2 / var2) - (mu1**2 / var1) - np.log(var2 / var1)
    return A, B, C

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np

def Question2():
    print("Running Question 2")

    # Reload ROI D and E
    roi_d = np.loadtxt("data/dice/D_Cubes.txt")
    roi_e = np.loadtxt("data/dice/E_Cubes.txt")

    # Compute class statistics
    mu_d, var_d = np.mean(roi_d), np.var(roi_d)
    mu_e, var_e = np.mean(roi_e), np.var(roi_e)

    A, B, C = get_gaussian_discriminant_coeffs(mu_d, var_d, mu_e, var_e)
    thresholds = np.roots([A, B, C])
    print(thresholds)



    # Stack the samples from both ROIs
    X = np.concatenate([roi_d, roi_e]).reshape(-1, 1)

    # Create labels: 0 for D, 1 for E
    y = np.array([0] * len(roi_d) + [1] * len(roi_e))

    # Train QDA
    qda = QDA()
    qda.fit(X, y)
    # Evaluate across a range of values to find decision boundary
    test_values = np.linspace(50, 200, 1000).reshape(-1, 1)
    predictions = qda.predict(test_values)

    # Find threshold where prediction switches from 0 to 1
    switch_index = np.where(np.diff(predictions) != 0)[0]
    qda_threshold = test_values[switch_index[0]][0] if switch_index.size > 0 else None

    print(qda_threshold)

def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()