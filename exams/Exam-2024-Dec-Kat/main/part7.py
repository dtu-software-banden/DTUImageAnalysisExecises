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

from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Standardize data
scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
print(scaled)


def Question1():
    print("Running Question 1")
    # Compute PCA using eigendecomposition
    cov_matrix = np.cov(scaled.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # Sort eigenvectors by eigenvalues (descending)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # Project data onto first principal component
    pc1_projection = scaled @ eig_vecs[:, 0]

    # Classify: negative projection → positive (no cancer = 1), else negative (cancer = 0)
    predicted_positive = pc1_projection < 0

    positive_count = np.sum(predicted_positive)
    print(positive_count)

def Question2():
    print("Running Question 2")


def Question3():
    print("Running Question 3")
    cov_matrix = np.cov(scaled.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    pc1_projection = scaled @ eig_vecs[:, 0]
    # Compute average values of PC1 projections for each class
    pc1_positive_mean = np.mean(pc1_projection[target == 1])
    pc1_negative_mean = np.mean(pc1_projection[target == 0])

    (pc1_negative_mean, pc1_positive_mean)


def Question4():
    print("Running Question 4")
    print(x.shape)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()