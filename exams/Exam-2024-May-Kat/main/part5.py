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


def gradient_descent(func, grad_func, x_init, step_size, max_iter=100, tol=0.2):
    x = x_init
    for i in range(max_iter):
        if func(*x) < tol:
            return x, i
        grad = grad_func(*x)
        x -= step_size * grad
    return x, max_iter

# Cost function from the task
def cost_function(x1, x2):
    return x1**2 - x1 * x2 + 3 * x2**2 + x1**3

# Its gradient
def grad_cost(x1, x2):
    dC_dx1 = 2 * x1 - x2 + 3 * x1**2
    dC_dx2 = -x1 + 6 * x2
    return np.array([dC_dx1, dC_dx2])


def run_fixed_iterations(func, grad_func, x_init, step_size, num_iter):
    x = x_init.copy()
    for _ in range(num_iter):
        grad = grad_func(*x)
        x -= step_size * grad
    return x

def Question1():
    print("Running Question 1")
    import numpy as np


    # Initial setup
    x_init = np.array([4.0, 3.0])
    step_size = 0.07

    res = run_fixed_iterations(cost_function, grad_cost, x_init, step_size, 5)
    print(f"Optimal x: {res}")



def Question2():
    print("Running Question 2")


    # Initial setup
    x_init = np.array([4.0, 3.0])
    step_size = 0.07

    # Call your gradient descent
    result, iterations = gradient_descent(cost_function, grad_cost, x_init, step_size)

    # Print result
    print(f"Optimal x: {result}")
    print(f"Reached in {iterations} iterations")


import numpy as np

# Class means
mu1 = np.array([24, 3])   # Class 1
mu2 = np.array([45, 7])   # Class 2

# Shared covariance matrix (2x2)
Sigma = np.array([
    [2, 0],
    [0, 2]
])

# Prior probabilities
prior1 = 0.5
prior2 = 0.5

# Inverse of shared covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Compute weight vector w for LDA
w = Sigma_inv @ (mu2 - mu1)

# Compute threshold x0
x0 = 0.5 * (mu1 + mu2) @ w

# Observation to classify
x = np.array([30, 10])

# Compute g(x) = w^T x
gx = w @ x

# Determine class
assigned_class = 2 if gx > x0 else 1




def Question3():
    print("Running Question 3")
    gx_corrected = gx - x0
    gx_corrected, (2 if gx_corrected > 0 else 1)
    print(gx_corrected, assigned_class)


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()