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


dicom_img = load_dicom("./data/kidney/1-162.dcm")

roi_list = load_roi_list([
    "./data/kidney/AortaROI.png",
    "./data/kidney/BackROI.png",
    "./data/kidney/KidneyROI.png",
    "./data/kidney/LiverROI.png",
])

aorta_area = dicom_img[roi_list[0]]
bak_area = dicom_img[roi_list[1]]
kidney_area = dicom_img[roi_list[2]]
liver_area = dicom_img[roi_list[3]]

t1 = threshold_min_dist_classification(liver_area, kidney_area)
t2 = threshold_min_dist_classification(kidney_area, aorta_area)

segmented = (dicom_img > t1) & (dicom_img < t2)

score = dice_score(segmented, roi_list[2])

def Question1():
    print("Running Question 1")
    print(t1,t2)


def Question2():
    print("Running Question 2")
    print(score)

def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()