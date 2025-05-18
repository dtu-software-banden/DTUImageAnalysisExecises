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
from utils.dicom_utils import * 


cows = np.array([26, 46, 33, 23, 35, 28, 21, 30, 38, 43])
sheep = np.array([67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100])

thresh = threshold_min_dist_classification(cows,sheep)
s_mu, s_std = gaussian_parameters(sheep)
c_mu, c_std = gaussian_parameters(cows)


def Question1():
    print("Running Question 1")
    print(thresh)


def Question2():
    print("Running Question 2")
    val = 38
    print(parametric_classifier_predict(val,cows), parametric_classifier_predict(val,sheep))


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()