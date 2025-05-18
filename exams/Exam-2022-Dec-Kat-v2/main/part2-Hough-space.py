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


data = np.array([
    [7, 13],
    [9, 10],
    [6, 10],
    [6, 8],
    [3, 6]
])

# Angle in degrees
theta_deg = 151
theta_rad = np.deg2rad(theta_deg)

# Direction vector for Hough transform
direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

# Compute all rho values as dot product
rhos = data @ direction  # equivalent to: np.dot(points, direction)

# Find which points are close to rho = 0.29
target_rho = 0.29
diff = np.abs(rhos - target_rho)

# Show the 2 closest matches
closest_indices = np.argsort(diff)[:2]
matching_points = data[closest_indices]

print("Matching points:")
print(matching_points)
