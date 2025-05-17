import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 
from utils.lda_utils import *

img_gray = load_image("./data/pixelwise.png", grayscale= True)
# normalize
gray_float = img_gray.astype(np.float32) / 255.0

# Step 2: Linear scaling to [0.1, 0.6]
min_val = gray_float.min()
max_val = gray_float.max()
scaled = 0.1 + ((gray_float - min_val) * (0.6 - 0.1) / (max_val - min_val))


from skimage.filters import threshold_otsu
threshold_float = threshold_otsu(scaled)
binary = scaled > threshold_float


def Question1():
    print("Running Question 1")
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Foreground (White) / Background (Black)")
    plt.axis('off')
    plt.show()


def Question2():
    print("Running Question 2")
    print(threshold_float)


    


from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import prewitt
import numpy as np
img = imread("./data/rocket.png")
gray = rgb2gray(img)




def Question3():
    print("Running Question 3")
    # Thresholding
    edges = prewitt(gray)
    binary = edges > 0.06

    # Count white pixels
    white_pixel_count = np.sum(binary)
    print(white_pixel_count)



def Question4():
    print("Running Question 4")
    # Mean vectors
    mu1 = np.array([24, 3])
    mu2 = np.array([30, 7])

    # Shared covariance matrix
    Sigma = np.array([[2, 0],
                    [0, 2]])
    
    x = np.array([23, 5])
    print(lda_predict(x , mu1,mu2,Sigma))


# Define training data
cows = np.array([26, 46, 33, 23, 35, 28, 21, 30, 38, 43])
sheep = np.array([67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100])

# Minimum distance classifier: use class means
mean_cow = np.mean(cows)
mean_sheep = np.mean(sheep)

std_cow = np.std(cows)
std_sheep = np.std(sheep)


def Question5():
    print("Running Question 5")
    # Threshold = midpoint between means
    threshold_min_dist = (mean_cow + mean_sheep) / 2
    print(threshold_min_dist)

def Question6():
    print("Running Question 6")
    from scipy.stats import norm

    # Evaluate both Gaussians at value = 38
    value = 38
    cow_pdf = norm.pdf(value, loc=mean_cow, scale=std_cow)
    sheep_pdf = norm.pdf(value, loc=mean_sheep, scale=std_sheep)

    print(cow_pdf, sheep_pdf)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()
    Question5()
    Question6()