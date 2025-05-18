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

# Load the data while skipping the header line
data = np.loadtxt("./data/pistachio_data.txt", skiprows=1)

# Step 1: Subtract the mean
mean_vec = np.mean(data, axis=0)
data_centered = data - mean_vec

# Step 2: Compute standard deviation
std_vec = np.std(data_centered, axis=0)

# Step 3: Standardize the data
data_standardized = data_centered / std_vec

# Step 4: Perform PCA using eigendecomposition of the covariance matrix
cov_matrix = np.cov(data_standardized.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Sort by eigenvalue magnitude
sorted_indices = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[sorted_indices]
eig_vecs = eig_vecs[:, sorted_indices]

# Project data onto first 2 principal components
projected_data = data_standardized @ eig_vecs[:, :2]

# Plot the result
plt.figure(figsize=(7, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1], s=30, alpha=0.7)
plt.title("PCA Projection of Pistachio Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()


def Question1():
    print("Running Question 1")
    first_sample = data_standardized[0]
    projected_first = first_sample @ eig_vecs  # all components

    # Compute the sum of squared projected values (L2 norm squared)
    sum_squared = np.sum(projected_first ** 2)
    print(sum_squared)


def Question2():
    print("Running Question 2")
    # Recalculate standard deviation of original (non-standardized) data
    std_values = np.std(data, axis=0)

    # Index of feature with smallest std
    min_std_index = np.argmin(std_values)
    min_std_value = std_values[min_std_index]

    # Feature names (from the task description)
    features = [
        "AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ",
        "SOLIDITY", "CONVEX_AREA", "EXTENT", "ASPECT_RATIO", "ROUNDNESS", "COMPACTNESS"
    ]

    # Get corresponding feature name
    min_std_feature = features[min_std_index]
    print(min_std_feature, min_std_value)

def Question3():
    print("Running Question 3")
    pca,projections = compute_pca(data_standardized)

    print("Top 1 ratio:",pca.explained_variance_ratio_[:1].sum())
    print("Top 2 ratio:",pca.explained_variance_ratio_[:2].sum())
    print("Top 3 ratio:",pca.explained_variance_ratio_[:3].sum())
    print("Top 4 ratio:",pca.explained_variance_ratio_[:4].sum())


def Question4():
    print("Running Question 4")
    max_abs_cov_value = np.max(np.abs(cov_matrix))
    print(max_abs_cov_value)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()