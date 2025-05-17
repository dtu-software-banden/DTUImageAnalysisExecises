import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 


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