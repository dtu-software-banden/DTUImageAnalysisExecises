import SimpleITK as sitk
import numpy as np

# Step 1: Read images
img_fixed = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii")
img_moving = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii")

# Step 2: Apply -20 degree roll (rotation around Z-axis) to moving image
# Default center = image center in physical space
theta_deg = -20
theta_rad = np.deg2rad(theta_deg)

# Create Euler3D transform (roll is rotation around Z-axis)
transform = sitk.Euler3DTransform()
center = img_moving.TransformContinuousIndexToPhysicalPoint(
    [sz/2.0 for sz in img_moving.GetSize()]
)
transform.SetCenter(center)
transform.SetRotation(0.0, 0.0, theta_rad)  # (Rx, Ry, Rz)

# Resample the transformed image
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(img_fixed)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)

img_moving_rotated = resampler.Execute(img_moving)

# Step 3: Create brain mask from img_fixed where intensities > 50
brain_mask = sitk.Cast(img_fixed > 50, sitk.sitkUInt8)

# Step 4: Convert to NumPy arrays
arr_fixed = sitk.GetArrayFromImage(img_fixed)
arr_moving_rot = sitk.GetArrayFromImage(img_moving_rotated)
arr_mask = sitk.GetArrayFromImage(brain_mask)

# Step 5: Compute MSE within the mask
masked_diff_sq = ((arr_fixed - arr_moving_rot) ** 2)[arr_mask == 1]
mse = np.mean(masked_diff_sq)

print("Mean Squared Error within brain mask:", mse)
