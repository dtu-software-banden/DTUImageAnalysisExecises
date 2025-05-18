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
from utils.optimization_utils import gradient_descent

import matplotlib.pyplot as plt
import numpy as np
from utils.optimization_utils import gradient_descent

# Cost and gradient definitions
def cost(x1, x2):
    return 7 * x1**2 + x1 * x2 + 3 * x2**2

def grad(x1, x2):
    dc_dx1 = 14 * x1 + x2
    dc_dx2 = x1 + 6 * x2
    return np.array([dc_dx1, dc_dx2])

# Run and collect path
def run_descent(x_init, step_size=0.1, max_iter=200, tol=0.2, itter= False):
    x = x_init.copy()
    path = [x.copy()]
    for i in range(max_iter):
        g = grad(*x)
        x = x - step_size * g
        path.append(x.copy())
        if itter: print(i + 1)
        if cost(*x) < tol:
            break
    return np.array(path)

def Question1():
    print("Running Question 1")

    # Grid for contour
    x1_vals = np.linspace(-6, 6, 200)
    x2_vals = np.linspace(-6, 6, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = cost(X1, X2)

    # Candidate starting points (based on green dot locations in your answer options)
    starts = {
        "A": np.array([-4.5, -5.5]),
        "B": np.array([-5, 4]),
        "C": np.array([4.5, 4.5]),
        "D": np.array([-4.5, -4.0]),
        "E": np.array([4.5, 4.5]),
    }

    for label, x_init in starts.items():
        path = run_descent(x_init)
        print(f"Candidate {label}: {len(path)} iterations")

        plt.contour(X1, X2, Z, levels=50, cmap='gray')
        plt.plot(path[:, 0], path[:, 1], marker='o', color='black', linewidth=1)
        plt.scatter(path[0, 0], path[0, 1], color='limegreen', label='Start', zorder=5)
        plt.scatter(path[-1, 0], path[-1, 1], color='red', label='End', zorder=5)
        plt.title(f"Candidate {label} — Iterations: {len(path)}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        #plt.show()
    print("Candidate B")

def Question2():
    print("Running Question 2")

    x = np.array([-5, 4])
    run_descent(x, tol= 2.0, itter= True)

def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()