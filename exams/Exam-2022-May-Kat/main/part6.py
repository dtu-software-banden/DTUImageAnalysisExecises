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
from utils.lda_utils import *


def Question1():
    print("Running Question 1")
    print(decision_boundary(52,2,150,30))

def Question2():
    print("Running Question 2")
    print((25 + 52) / 2 )


# Passive volcanoes (class 1)
class1 = np.array([
    [1.2, 1.1],
    [2.9, 0.4],
    [1.7, -2.7],
    [1.8, -0.3],
    [3.2, 1.3],
    [3.1, -0.9]
])

# Erupting volcanoes (class 2)
class2 = np.array([
    [0.5, 1.7],
    [1.4, -2.1],
    [2.7, -0.8],
    [2.0, 0.5]
])

def Question3():
    print("Running Question 3")
    mu1, mu2, sigma = lda_train(class1,class2)
    print(lda_predict_batch(class2, mu1, mu2, sigma))

def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()