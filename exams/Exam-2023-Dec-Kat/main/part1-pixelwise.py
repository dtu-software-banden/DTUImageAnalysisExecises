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

def Question1():
    print("Running Question 1")
    trans_speed = 35 * 1000000
    img_size = 2400 * 1200 * 3
    analysis = 0.130

    trans_pr_s = trans_speed / img_size
    img_pr_s = 1 / analysis

    print(min(trans_pr_s,img_pr_s))


image_float = load_image_grayscale_float("./data/ardeche_river.jpg")

scaled = linear_transformation(image_float,0.2,0.8)

average = np.mean(scaled)

from skimage.filters import prewitt_h
edges_horizontal = prewitt_h(scaled)

max_abs_value = np.max(np.abs(edges_horizontal))

binary_image = (scaled > average).astype(np.uint8)

foreground_count = np.sum(binary_image)

def Question2():
    print("Running Question 2")
    print(foreground_count)


def Question3():
    print("Running Question 3")
    print(max_abs_value)


def Question4():
    print("Running Question 4")
    print(average)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()