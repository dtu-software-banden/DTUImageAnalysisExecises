import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 


import numpy as np
import pandas as pd

# Manually define correct column names from the file header
column_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

# Reload the data with correct header handling
data = pd.read_csv("./data/glass_data.txt", delim_whitespace=True, comment="%", names=column_names)
X = data.values  # Convert to NumPy array
X_scaled = normalize(X)

def Question1():
    print("Running Question 1")
    pca, projections = compute_pca(X_scaled)
    # Total variance explained by the first 3 components
    explained_first_three = np.sum(pca.explained_variance_ratio_[:3])
    print(f"Explained variance by first 3 components: {explained_first_three:.4f}")


def Question2():
    print("Running Question 2")
    # Compute covariance matrix (columns are variables)
    cov_matrix = np.cov(X_scaled, rowvar=False)

    # Extract the value at position (0, 0)
    value_00 = cov_matrix[0, 0]
    print(f"Covariance matrix value at (0, 0): {value_00:.4f}")


def Question3():
    print("Running Question 3")
    first_na_value = X_scaled[0, 1] 
    print(f"First scaled Sodium (Na) value: {first_na_value:.4f}")


def Question4():
    print("Running Question 4")
    pca, projections = compute_pca(X_scaled)
    # Compute the maximum absolute value among all projected values
    max_abs_projection = np.max(np.abs(projections))
    print(f"Max absolute projection: {max_abs_projection:.4f}")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()