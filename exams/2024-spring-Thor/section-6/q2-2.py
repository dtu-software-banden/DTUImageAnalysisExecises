import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load the image
template_img = sitk.ReadImage("T1_brain_template.nii.gz")
template_array = sitk.GetArrayFromImage(template_img)

# Function to apply rigid transform (yaw + pitch)
def apply_rigid_transform(image, yaw_deg=10, pitch_deg=-30):
    transform = sitk.Euler3DTransform()

    pitch_rad = np.deg2rad(pitch_deg)  # X
    yaw_rad = np.deg2rad(yaw_deg)      # Z

    transform.SetRotation(pitch_rad, 0, yaw_rad)

    size = image.GetSize()
    center_phys = image.TransformIndexToPhysicalPoint([sz // 2 for sz in size])
    transform.SetCenter(center_phys)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

# Apply transformation
moving_img = apply_rigid_transform(template_img)
moving_array = sitk.GetArrayFromImage(moving_img)

# Function to show comparison
def show_comparison(template, volume, title="Template vs. Moving"):
    z, y, x = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial: horizontal slice (top-down)
    axial = volume[z // 2, :, :]
    axes[0].imshow(axial, cmap="gray", origin="lower")
    axes[0].set_title("Axial (Top-down)")

    # Coronal: front view
    coronal = volume[:, y // 2, :]
    axes[1].imshow(np.rot90(coronal), cmap="gray", origin="lower")
    axes[1].set_title("Coronal (Front)")

    # Sagittal: side view
    sagittal = volume[:, :, x // 2]
    axes[2].imshow(np.rot90(sagittal), cmap="gray", origin="lower")
    axes[2].set_title("Sagittal (Side)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Run the comparison
show_comparison(template_array, moving_array)
