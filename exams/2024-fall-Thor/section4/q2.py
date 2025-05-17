# Define the cost function and its gradient
import numpy as np

from utils.optimization_utils import gradient_descent


def cost_function(x1,x2):
    return 7 * x1**2 + x1 * x2 + 3 * x2**2

def gradient(x1,x2):
    dc_dx1 = 14 * x1 + x2
    dc_dx2 = x1 + 6 * x2
    return np.array([dc_dx1, dc_dx2])

coords,steps = gradient_descent(cost_function,gradient,[-5.0,4.0],0.1,tol=2.0)
print(cost_function(coords[0],coords[1]))