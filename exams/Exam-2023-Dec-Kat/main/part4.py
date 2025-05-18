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

# Define Option 4 Hough points
option4 = np.array([
    [-45, -1.4],
    [45, 5.7],
    [0, 3],
    [-45, 0.7],
    [45, 3.5]
])

# Function to convert Hough (theta, rho) to Cartesian line
def hough_to_line(theta_deg, rho, x_range=(-4, 4)):
    theta_rad = np.deg2rad(theta_deg)
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    x_vals = np.linspace(x_range[0], x_range[1], 500)

    if np.abs(b) > 1e-6:
        y_vals = (rho - x_vals * a) / b
        return x_vals, y_vals
    else:
        # Vertical line
        x_fixed = np.full_like(x_vals, rho / a)
        return x_fixed, x_vals

# Plot the lines
plt.figure(figsize=(6, 6))
for theta, rho in option4:
    x_vals, y_vals = hough_to_line(theta, rho)
    plt.plot(x_vals, y_vals, label=f"θ={theta}°, ρ={rho}")

plt.title("Option 4: Hough Lines in Cartesian Coordinates")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.xlim(0, 8)
plt.ylim(0,10)
plt.show()


def Question1():
    print("Running Question 1")


def Question2():
    print("Running Question 2")


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()