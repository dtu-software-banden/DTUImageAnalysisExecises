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


import numpy as np
import pydicom
from skimage.io import imread
from skimage.morphology import closing, opening, disk
from skimage.measure import label, regionprops
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# === Step 1: Load DICOM and convert to Hounsfield Units ===
dicom_img = pydicom.dcmread("./data/HeartCT/1-001.dcm")
pixels = dicom_img.pixel_array.astype(np.int16)
hu_image = pixels * dicom_img.RescaleSlope + dicom_img.RescaleIntercept


# === Step 2: Load ROI masks ===
blood_roi = imread("./data/HeartCT/bloodROI.png") > 0
myocardium_roi = imread("./data/HeartCT/MyocardiumROI.png") > 0
ground_truth = imread("./data/HeartCT/BloodGT.png") > 0

# === Step 3: Get HU values within the blood ROI ===
blood_values = hu_image[blood_roi]
mu = np.mean(blood_values)
sigma = np.std(blood_values)

# === Step 4: Compute class range and segment blood ===
lower_bound = mu - 3 * sigma
upper_bound = mu + 3 * sigma
binary_mask = ((hu_image > lower_bound) & (hu_image < upper_bound)).astype(np.uint8)

# === Step 5: Morphological closing and opening ===
closed = closing(binary_mask, disk(3))
opened = opening(closed, disk(5))

# === Step 6: BLOB analysis + filtering ===
labeled = label(opened)
filtered_mask = np.zeros_like(opened, dtype=bool)

for region in regionprops(labeled):
    if 2000 < region.area < 5000:
        filtered_mask[labeled == region.label] = True

# === Step 7: Compute DICE score ===
intersection = np.logical_and(filtered_mask, ground_truth).sum()
dice = 2 * intersection / (filtered_mask.sum() + ground_truth.sum())
print(f"DICE score: {dice:.4f}")


def Question1():
    print("Running Question 1")
    print(f"Class range (in Hounsfield units): {lower_bound:.2f} to {upper_bound:.2f}")


def Question2():
    print("Running Question 2")
    plot_blob_mask(opened)
    print("5")


def Question3():
    print("Running Question 3")
    print(dice_score(filtered_mask,ground_truth))


def Question4():
    print("Running Question 4")
    print(threshold_min_dist_classification(hu_image[myocardium_roi],hu_image[blood_roi]))


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()