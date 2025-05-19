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

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from skimage.io import imread

# Load DICOM slice
dicom_image = sitk.ReadImage("data/vertebraCT/1-353.dcm")
image_array = sitk.GetArrayFromImage(dicom_image)[0]

# Threshold at 200 HU
binary_image = image_array > 200

# Morphological closing with disk radius=3
cleaned = closing(binary_image, disk(3))

# BLOB analysis
labeled = label(cleaned)
regions = regionprops(labeled)

# Keep only blobs with area > 500 pixels
mask = np.zeros_like(image_array, dtype=bool)
for region in regions:
    if region.area > 500:
        mask[labeled == region.label] = True

# Display original and mask
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_array, cmap="gray")
axs[0].set_title("Original DICOM Slice")
axs[1].imshow(mask, cmap="gray")
axs[1].set_title("Segmented Vertebra Mask")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()


def Question1():
    print("Running Question 1")

    import SimpleITK as sitk
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread

    # Load DICOM slice and extract array
    dicom_image = sitk.ReadImage("data/vertebraCT/1-353.dcm")
    image_array = sitk.GetArrayFromImage(dicom_image)[0]

    # Load expert mask (PNG assumed to be binary mask)
    mask = imread("data/vertebraCT/vertebra_gt.png") > 0

    # Extract HU values inside mask
    masked_values = image_array[mask]

    # Plot histogram with 100 bins
    plt.hist(masked_values, bins=100, color='skyblue', edgecolor='black')
    plt.title("Histogram of Hounsfield Units in Vertebra Mask")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def Question2():
    print("Running Question 2")

    import SimpleITK as sitk
    import numpy as np
    from skimage.morphology import closing, disk
    from skimage.measure import label, regionprops

    # Load DICOM slice
    dicom_image = sitk.ReadImage("data/vertebraCT/1-353.dcm")
    image_array = sitk.GetArrayFromImage(dicom_image)[0]

    # Thresholding
    binary = image_array > 200

    # Morphological closing
    cleaned = closing(binary, disk(3))

    # BLOB labeling
    labeled = label(cleaned)
    regions = regionprops(labeled)

    # Create final mask of large blobs
    mask = np.zeros_like(image_array, dtype=bool)
    for region in regions:
        if region.area > 500:
            mask[labeled == region.label] = True

    # Extract HU values within mask
    values = image_array[mask]
    mean_val = np.mean(values)
    std_val = np.std(values)

    print(f"Mean HU value: {mean_val:.2f}")
    print(f"Standard deviation: {std_val:.2f}")


def Question3():
    print("Running Question 3")

    import SimpleITK as sitk
    import numpy as np
    from skimage.morphology import closing, disk
    from skimage.measure import label, regionprops

    # Load DICOM slice
    dicom_image = sitk.ReadImage("data/vertebraCT/1-353.dcm")
    image_array = sitk.GetArrayFromImage(dicom_image)[0]

    # Threshold and close
    binary = image_array > 200
    cleaned = closing(binary, disk(3))

    # BLOB labeling
    labeled = label(cleaned)
    regions = regionprops(labeled)

    # Compute all areas
    areas = [region.area for region in regions]

    if areas:
        print(f"Minimum area: {min(areas)} pixels")
        print(f"Maximum area: {max(areas)} pixels")
    else:
        print("No BLOBs detected.")



def Question4():
    print("Running Question 4")
    expert_mask = imread("data/vertebraCT/vertebra_gt.png") > 0
    print(dice_score(mask, expert_mask))


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()