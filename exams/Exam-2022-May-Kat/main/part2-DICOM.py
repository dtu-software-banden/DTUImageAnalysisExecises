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


# === Step 1: Load DICOM and convert to Hounsfield Units ===
dicom_img = load_dicom("./data/Aorta/1-442.dcm")


# === Step 2: Load ROI masks ===
aorta_roi = imread("./data/Aorta/AortaROI.png") 
back_roi = imread("./data/Aorta/BackROI.png") 
liver_roi = imread("./data/Aorta/LiverROI.png") 



thresh = qda_1d_threshold(dicom_img[aorta_roi],dicom_img[liver_roi])


def Question1():
    print("Running Question 1")
    print(thresh)


def Question2():
    print("Running Question 2")
    thresh_img = dicom_img > 90
    labeled = label(thresh_img, connectivity=2)
    mask = np.zeros_like(thresh_img, dtype=bool)

    for region in regionprops(labeled):
        if region.area > 200 :
            circ = (4 * 3.14 * region.area) / (region.perimeter**2)
            if circ > 0.9:
                mask[labeled == region.label] = True

    plot_image(mask)

    pixel_size = 0.75 ** 2
    count = np.count_nonzero(mask)

    print(count * pixel_size)



def Question3():
    print("Running Question 3")
    img = dicom_img[aorta_roi]
    mean = np.mean(img)
    std = np.std(img)

    print(mean, std)


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()