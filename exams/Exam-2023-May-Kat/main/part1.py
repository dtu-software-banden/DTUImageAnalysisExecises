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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from utils.io_utils import *


print("hello")
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
