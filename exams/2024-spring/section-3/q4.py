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

# Perform BLOB analysis before filtering
label_image = label(binary_cleaned)
regions = regionprops(label_image)

# Get all areas
areas = [region.area for region in regions]

# Compute min and max
if areas:
    min_area = min(areas)
    max_area = max(areas)
    print(f"Minimum BLOB area: {min_area} pixels")
    print(f"Maximum BLOB area: {max_area} pixels")
else:
    print("No BLOBs found.")
