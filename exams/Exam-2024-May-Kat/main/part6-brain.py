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
from utils.affine_trans_utils import * 

import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import ball, closing, erosion
import matplotlib.pyplot as plt

# === Load the image ===
image = sitk.ReadImage("data/brain/T1_brain_template.nii.gz")
image_np = sitk.GetArrayFromImage(image)  # (z, y, x) shape

# === Apply transformation: rotation (yaw=10°, pitch=-30°) ===
# Yaw: around z-axis; Pitch: around x-axis
yaw_deg=10
pitch_deg=-30

pitch_rad = np.deg2rad(pitch_deg)  # X
yaw_rad = np.deg2rad(yaw_deg)      # Z'

transform = sitk.Euler3DTransform()
transform.SetRotation(pitch_rad, 0, yaw_rad)  # pitch, roll, yaw
transform.SetCenter(np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)

# Apply transform to generate the moving image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(image)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(transform)
moving = resampler.Execute(image)

# === Otsu threshold + morphology to generate mask ===
thresh = threshold_otsu(image_np)
binary = image_np > thresh

closed = closing(binary, ball(5))
final_mask = erosion(closed, ball(3))

# Convert to SimpleITK image (for aligned comparison)
mask_sitk = sitk.GetImageFromArray(final_mask.astype(np.uint8))
mask_sitk.CopyInformation(image)

# === Apply mask to both images ===
image_masked = sitk.Mask(image, mask_sitk)
moving_masked = sitk.Mask(moving, mask_sitk)

# === Compute normalized correlation coefficient ===
def normalized_correlation(img1, img2):
    a = sitk.GetArrayFromImage(img1).astype(np.float32)
    b = sitk.GetArrayFromImage(img2).astype(np.float32)
    a, b = a[final_mask], b[final_mask]  # apply mask directly
    a_mean, b_mean = np.mean(a), np.mean(b)
    return np.sum((a - a_mean) * (b - b_mean)) / (np.sqrt(np.sum((a - a_mean)**2)) * np.sqrt(np.sum((b - b_mean)**2)))

ncc = normalized_correlation(image_masked, moving_masked)
print(f"Normalized Correlation Coefficient: {ncc:.4f}")


def Question1():
    print("Running Question 1")
    imshow_orthogonal_view(final_mask)
    

def Question2():
    print("Running Question 2")
    print(f"Normalized Correlation Coefficient: {ncc:.4f}")


def Question3():
    print("Running Question 3")
    # Load the image
    image = sitk.ReadImage("data/brain/T1_brain_template.nii.gz")
    image_np = sitk.GetArrayFromImage(image)

    # Convert degrees to radians
    pitch = np.deg2rad(-30)  # rotation around x
    yaw = np.deg2rad(10)     # rotation around z

    # No translation, no roll, no scaling, no shear
    dx = dy = dz = 0
    roll = 0
    sx = sy = sz = 1
    sxy = sxz = syz = 0

    # Build full affine transformation matrix
    A = affine_transformation(dx, dy, dz, pitch, roll, yaw, sx, sy, sz, sxy, sxz, syz)

    # Center the rotation: we must translate to center, apply A, then translate back
    z, y, x = image_np.shape
    center = np.array([x, y, z]) / 2

    # Translate to center, apply A, translate back
    T_to_center = np.eye(4)
    T_to_center[:3, 3] = -center
    T_back = np.eye(4)
    T_back[:3, 3] = center
    A_centered = T_back @ A @ T_to_center

    transformed_np = apply_affine_to_image_np(image_np, A_centered)
    imshow_orthogonal_view(transformed_np)
   


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()