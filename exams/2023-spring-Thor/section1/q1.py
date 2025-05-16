import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops

# Step 1: Load the DICOM file and the ROI images
dicom = pydicom.dcmread('1-166.dcm')
ct_image = dicom.pixel_array.astype(np.int16)

# Rescale if necessary (to get Hounsfield units)
if 'RescaleSlope' in dicom and 'RescaleIntercept' in dicom:
    ct_image = ct_image * dicom.RescaleSlope + dicom.RescaleIntercept

# Load the expert annotations (assumed to be binary masks)
kidney_left_mask = cv2.imread('KidneyRoi_l.png', cv2.IMREAD_GRAYSCALE)
kidney_right_mask = cv2.imread('KidneyRoi_r.png', cv2.IMREAD_GRAYSCALE)

# Ensure binary masks (0 and 1)
kidney_left_mask = (kidney_left_mask > 0).astype(np.uint8)
kidney_right_mask = (kidney_right_mask > 0).astype(np.uint8)

# Step 2: Extract pixel values using masks
left_kidney_pixels = ct_image[kidney_left_mask == 1]
right_kidney_pixels = ct_image[kidney_right_mask == 1]

# Step 3: Compute average Hounsfield unit values
left_kidney_avg = np.mean(left_kidney_pixels)
right_kidney_avg = np.mean(right_kidney_pixels)

print(f"Average HU in left kidney: {left_kidney_avg:.2f}")
print(f"Average HU in right kidney: {right_kidney_avg:.2f}")

# Load liver mask
liver_mask = cv2.imread('LiverROI.png', cv2.IMREAD_GRAYSCALE)
liver_mask = (liver_mask > 0).astype(np.uint8)

# Extract liver pixel values
liver_pixels = ct_image[liver_mask == 1]

# Compute mean and std
liver_mean = np.mean(liver_pixels)
liver_std = np.std(liver_pixels)

# Compute thresholds
t1 = liver_mean - liver_std
t2 = liver_mean + liver_std

print(f"Mean HU in liver: {liver_mean:.2f}")
print(f"Std. dev. of HU in liver: {liver_std:.2f}")
print(f"Liver segmentation thresholds: t1 = {t1:.2f}, t2 = {t2:.2f}")

# Step 7: Binary thresholding
binary_liver = ((ct_image >= t1) & (ct_image <= t2)).astype(np.uint8)

# Step 8–10: Morphological operations
binary_liver = dilation(binary_liver, disk(3))
binary_liver = erosion(binary_liver, disk(10))
binary_liver = dilation(binary_liver, disk(10))

# Step 11: Extract blobs
labeled_blobs = label(binary_liver)
regions = regionprops(labeled_blobs)

# Step 12–13: Filter blobs by area and perimeter
final_mask = np.zeros_like(binary_liver)
for region in regions:
    if 1500 < region.area < 7000 and region.perimeter > 300:
        final_mask[labeled_blobs == region.label] = 1

# Step 14: Compute DICE score
ground_truth = (cv2.imread('LiverROI.png', cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

intersection = np.sum((final_mask == 1) & (ground_truth == 1))
dice_score = 2 * intersection / (np.sum(final_mask) + np.sum(ground_truth))

print(f"DICE score: {dice_score:.4f}")


