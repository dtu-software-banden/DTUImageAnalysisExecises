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

# Optional: Load and show ground truth
gt_mask = imread("vertebra_gt.png", as_gray=True) > 0.5

# Plot results
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(image_hu, cmap='gray')
axes[0].set_title("Original DICOM (HU)")
axes[1].imshow(binary, cmap='gray')
axes[1].set_title("Thresholded > 200 HU")
axes[2].imshow(binary_cleaned, cmap='gray')
axes[2].set_title("After Morphological Closing")
axes[3].imshow(final_mask, cmap='gray')
axes[3].set_title("Final Mask (Area > 500)")
plt.tight_layout()
plt.show()
