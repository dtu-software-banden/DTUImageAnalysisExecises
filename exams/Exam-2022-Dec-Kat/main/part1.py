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



# Step 1: Load DICOM and ROI annotations
aorta_roi_path = "./data/kidney_analysis/AortaROI.png"
back_roi_path = "./data/kidney_analysis/backROI.png"
kidney_roi_path = "./data/kidney_analysis/KidneyROI.png"
liver_roi_path = "./data/kidney_analysis/LiverROI.png"

# Read the DICOM file
dicom_image = load_dicom("./data/kidney_analysis/1-162.dcm")

# Read the expert annotations
liver_mask = imread(liver_roi_path) > 0
kidney_mask = imread(kidney_roi_path) > 0
aorta_mask = imread(aorta_roi_path) > 0
back_mask = imread(back_roi_path) > 0

# Step 2: Extract Hounsfield unit values
liver_values = dicom_image[liver_mask]
kidney_values = dicom_image[kidney_mask]
aorta_values = dicom_image[aorta_mask]
back_values = dicom_image[back_mask]

# Compute means of each region
mean_liver = np.mean(liver_values)
mean_kidney = np.mean(kidney_values)
mean_aorta = np.mean(aorta_values)

# Compute midpoints (minimum distance thresholding)
t1 = (mean_liver + mean_kidney) / 2  # Separates liver and kidney
t2 = (mean_kidney + mean_aorta) / 2  # Separates kidney and aorta

def Question1():
    print("Running Question 1")
    print(f"Threshold t1 (liver/kidney): {t1:.2f}")
    print(f"Threshold t2 (kidney/aorta): {t2:.2f}")

def Question2():
    print("Running Question 2")
    segmented = (dicom_image > t1) & (dicom_image < t2)

    # Use binary masks
    kidney_mask = imread(kidney_roi_path) > 0
    dice = dice_score(segmented, kidney_mask)

    print(f"DICE Score: {dice:.4f}")


def Question3():
    print("Running Haar Question")
    res = 168+217+159+223-178-60-155-252+97+136+32+108
    print(res)


def Question4():
    print("Running Integral Image")
    res1 = 32+12+200+54
    res2 = res1 +110+81+220+120+107
    print(res1 ,res2)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()