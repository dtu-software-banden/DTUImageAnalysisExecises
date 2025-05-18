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

from skimage.filters import prewitt
image = load_image_grayscale_float("./data/rocket.png")
prew = prewitt(image) > 0.06


def Question1():
    print("Running Question 1")
    white_pixel_count = np.sum(prew)
    print(white_pixel_count)


mean1 = np.array([24, 3])  # Class 1
mean2 = np.array([30, 7])  # Class 2

covariance = np.array([
    [2, 0],
    [0, 2]
])

def Question2():
    print("Running Question 2")
    X = np.array([23, 5])
    print(lda_predict_batch(X,mean1,mean2,covariance))


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()