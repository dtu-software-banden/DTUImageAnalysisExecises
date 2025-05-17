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

image_float = load_image_grayscale_float("./data/pixelwise.png")

scaled = linear_transformation(image_float,0.1,0.6)

from skimage.filters import threshold_otsu
threshold_float = threshold_otsu(scaled)
binary = scaled > threshold_float

def Question1():
    print("Running Question 1")
    plot_image(binary)


def Question2():
    print("Running Question 2")
    print(threshold_float)


if __name__ == "__main__":
    Question1()
    Question2()
