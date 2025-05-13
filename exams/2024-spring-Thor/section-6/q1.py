import SimpleITK as sitk
import numpy as np
from skimage.morphology import ball, closing, erosion
from skimage.filters import threshold_otsu

# Step 1: Load the 3D MRI image
template_img = sitk.ReadImage("T1_brain_template.nii.gz")
template_array = sitk.GetArrayFromImage(template_img)

# Step 2: Apply rigid transformation (yaw: around z-axis, pitch: around x-axis)
def apply_rigid_transform(image, yaw_deg=10, pitch_deg=-30):
    transform = sitk.Euler3DTransform()
    # Convert degrees to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Set rotation (pitch -> X axis, yaw -> Z axis)
    transform.SetRotation(pitch, 0, yaw)
    # Center of rotation
    transform.SetCenter(np.array(image.GetSize()) / 2.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    return resampler.Execute(image)

moving_img = apply_rigid_transform(template_img)

# Step 3: Generate mask
template_array = sitk.GetArrayFromImage(template_img)
threshold = threshold_otsu(template_array)
binary_mask = template_array > threshold

# Morphological closing with ball(radius=5) and erosion with ball(radius=3)
closed_mask = closing(binary_mask, ball(5))
eroded_mask = erosion(closed_mask, ball(3))

# Step 4: Apply mask to both template and moving images
moving_array = sitk.GetArrayFromImage(moving_img)
template_masked = template_array * eroded_mask
moving_masked = moving_array * eroded_mask

# Flatten masked voxel values
template_masked_vals = template_array[eroded_mask]
moving_masked_vals = moving_array[eroded_mask]

# Compute NCC
def normalized_correlation_coefficient(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    numerator = np.sum((a - a_mean) * (b - b_mean))
    denominator = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    return numerator / denominator

ncc_value = normalized_correlation_coefficient(template_masked_vals, moving_masked_vals)
print(f"Normalized Correlation Coefficient (NCC): {ncc_value:.4f}")

import matplotlib.pyplot as plt

def show_orthogonal_slices(volume, title="Orthogonal View"):
    """
    Display orthogonal slices from a 3D volume (z, y, x), correcting orientation.
    """
    z, y, x = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial (x-y plane at middle z)
    axes[0].imshow(np.flipud(volume[z // 2, :, :]), cmap="gray")
    axes[0].set_title("Axial (z={})".format(z // 2))

    # Coronal (x-z plane at middle y)
    axes[1].imshow(np.flipud(volume[:, y // 2, :]), cmap="gray")
    axes[1].set_title("Coronal (y={})".format(y // 2))

    # Sagittal (y-z plane at middle x)
    axes[2].imshow(np.flipud(volume[:, :, x // 2]), cmap="gray")
    axes[2].set_title("Sagittal (x={})".format(x // 2))

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Visualize the final binary mask
show_orthogonal_slices(eroded_mask.astype(np.uint8), title="Binary Mask After Morphology")
