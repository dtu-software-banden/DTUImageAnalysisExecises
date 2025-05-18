import SimpleITK as sitk
import numpy as np

# Step 1: Load the images
fixed_image = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii.gz")
moving_image = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii.gz")

# Step 2: Apply -20 degree roll (rotation around Z-axis) to moving image
theta = np.deg2rad(-20)  # Convert to radians

transform = sitk.Euler3DTransform()
transform.SetRotation(0.0, theta,0.0)  # Roll around Z
# Default center is (0, 0, 0) – as specified

# Resample moving image to fixed image space
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)

moved_image = resampler.Execute(moving_image)

# Step 3: Create brain mask from fixed image (voxels > 50)
brain_mask = sitk.GetArrayFromImage(fixed_image) > 50

# Step 4: Convert images to NumPy arrays
fixed_array = sitk.GetArrayFromImage(fixed_image)
moved_array = sitk.GetArrayFromImage(moved_image)

# Step 5: Compute MSE within the mask
mse = np.mean((fixed_array[brain_mask] - moved_array[brain_mask]) ** 2)

print("Mean Squared Error within brain mask:", mse)
