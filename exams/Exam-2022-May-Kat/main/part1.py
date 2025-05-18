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
from utils.analysis_utils import * 

import numpy as np

matrix = np.array([
    [177, 195, 181,  30, 192],
    [ 81, 203, 192, 127,  65],
    [242,  48,  70, 245, 129],
    [  9, 125, 173,  87, 178],
    [112, 114, 167, 149, 227]
])

acc = accumulator_image(matrix)

print(acc)

def Question1():
    print("Running Question 1")
    print(compute_optimal_path(acc))
    print(acc[2][1])

def Question2():
    print("Running Question 2")
    path = compute_optimal_path(acc)
    path_plus_one = [(i + 1, j + 1) for i, j in path]
    print(path_plus_one)


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()