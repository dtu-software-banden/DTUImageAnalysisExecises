import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from skimage.io import imread

# Load DICOM image and extract pixel values
dicom_file = pydicom.dcmread("1-353.dcm")
image_hu = dicom_file.pixel_array.astype(np.int16)

# Convert to Hounsfield Units (simplified, assumes rescale slope/intercept are standard)
if 'RescaleIntercept' in dicom_file and 'RescaleSlope' in dicom_file:
    intercept = dicom_file.RescaleIntercept
    slope = dicom_file.RescaleSlope
    image_hu = image_hu * slope + intercept

# Thresholding - bone is bright in CT, so threshold >200 HU
binary = image_hu > 200

# Morphological closing with a disk (radius=3)
binary_cleaned = closing(binary, disk(3))

# Label connected regions (BLOB analysis)
label_image = label(binary_cleaned)

# Filter BLOBs by area > 500 pixels
final_mask = np.zeros_like(binary_cleaned, dtype=bool)
for region in regionprops(label_image):
    if region.area > 500:
        final_mask[label_image == region.label] = True

# Load expert mask (assumed to be binary: vertebra=1, background=0)
# If it's a grayscale image, threshold it to get binary mask
expert_mask = imread("vertebra_gt.png", as_gray=True) > 0.5

# Ensure mask and DICOM image are the same shape
assert expert_mask.shape == image_hu.shape, "Mask and DICOM image dimensions do not match."

# Extract HU values where mask is True
masked_hu_values = image_hu[expert_mask]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(masked_hu_values, bins=100, color='gray', edgecolor='black')
plt.title("Histogram of Hounsfield Units in Expert Mask")
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
