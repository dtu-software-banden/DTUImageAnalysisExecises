import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error
import sys
import os

# Get the path to the project root by going two levels up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import your utility functions
from utils.io_utils import *  # or import specific functions
from utils.morph_utils import * 

# Step 1: Load DICOM and ROI annotations
liver_roi_path = "./data/Abdominal/LiverROI.png"
kidney_l_path = "./data/Abdominal/KidneyRoi_l.png"
kidney_r_path = "./data/Abdominal/KidneyRoi_r.png"

# Read the DICOM file
dicom_image = load_dicom("./data/Abdominal/1-166.dcm")

# Read the expert annotations
liver_mask = imread(liver_roi_path) > 0
kidney_l_mask = imread(kidney_l_path) > 0
kidney_r_mask = imread(kidney_r_path) > 0

# Step 2: Extract Hounsfield unit values
liver_values = dicom_image[liver_mask]
kidney_l_values = dicom_image[kidney_l_mask]
kidney_r_values = dicom_image[kidney_r_mask]

# Check the basic info
print(f"Liver pixels: {len(liver_values)}, Left Kidney: {len(kidney_l_values)}, Right Kidney: {len(kidney_r_values)}")


def Question1():
    print("Running question 1")
    # Compute the average Hounsfield Units
    avg_l = np.mean(kidney_l_values)
    avg_r = np.mean(kidney_r_values)
    print(f"Average HU - Left Kidney: {avg_l:.2f}")
    print(f"Average HU - Right Kidney: {avg_r:.2f}")


def Question2():
    print("Running question 2")
    # Compute mean and standard deviation
    liver_mean = np.mean(liver_values)
    liver_std = np.std(liver_values)

    # Compute thresholds
    t_1 = liver_mean - liver_std
    t_2 = liver_mean + liver_std

    print(f"Liver HU Mean: {liver_mean:.2f}")
    print(f"Liver HU Std Dev: {liver_std:.2f}")
    print(f"Hounsfield Unit Limits for Segmentation: [{t_1:.2f}, {t_2:.2f}]")


# Compute mean and standard deviation
liver_mean = np.mean(liver_values)
liver_std = np.std(liver_values)

# Compute thresholds
t_1 = liver_mean - liver_std
t_2 = liver_mean + liver_std


def Question3():
    print("Running question 3")
    from skimage.morphology import disk, dilation, erosion
    from skimage.measure import label, regionprops

    # Binary segmentation using thresholds
    liver_segment = ((dicom_image >= t_1) & (dicom_image <= t_2)).astype(np.uint8)

    # Morphological ops
    liver_segment = dilation(liver_segment, disk(3))
    liver_segment = erosion(liver_segment, disk(10))
    liver_segment = dilation(liver_segment, disk(10))

    filtered_mask = filter(liver_segment, min_area = 1500, max_area = 7000, min_perim = 300)
    # Load the ground truth liver mask
    gt_liver_mask = imread("./data/Abdominal/LiverROI.png") > 0

    # Compute DICE score
    dice = dice_score(filtered_mask, gt_liver_mask)
    print(f"DICE Score: {dice:.4f}")

def Question4():
    print("Running question 4")


if __name__ == "__main__":
    #Question1()
    #Question2()
    Question3()
    #Question4()