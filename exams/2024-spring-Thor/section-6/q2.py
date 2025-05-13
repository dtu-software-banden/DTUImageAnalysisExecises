import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
template_img = sitk.ReadImage("T1_brain_template.nii.gz")

# Step 2: Apply rigid transformation (pitch around X, then yaw around Z)
def apply_rigid_transform(image, yaw_deg=10, pitch_deg=-30):
    transform = sitk.Euler3DTransform()

    # Convert degrees to radians
    pitch_rad = np.deg2rad(pitch_deg)  # rotation around X
    yaw_rad = np.deg2rad(yaw_deg)      # rotation around Z

    # Set rotation: (alpha, beta, gamma) => (X, Y, Z)
    transform.SetRotation(pitch_rad, 0, yaw_rad)

    # Center the rotation around the image center
    size = image.GetSize()
    spacing = image.GetSpacing()
    center_phys = image.TransformIndexToPhysicalPoint([sz // 2 for sz in size])
    transform.SetCenter(center_phys)

    # Apply the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

# Step 3: Apply the transform
moving_img = apply_rigid_transform(template_img)

# Convert to NumPy array for visualization
moving_array = sitk.GetArrayFromImage(moving_img)

# Step 4: Visualize orthogonal slices
def show_orthogonal_slices(volume, title="Transformed Image"):
    z, y, x = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.flipud(volume[z // 2, :, :]), cmap='gray')
    axes[0].set_title(f'Axial (z={z // 2})')

    axes[1].imshow(np.flipud(volume[:, y // 2, :]), cmap='gray')
    axes[1].set_title(f'Coronal (y={y // 2})')

    axes[2].imshow(np.flipud(volume[:, :, x // 2]), cmap='gray')
    axes[2].set_title(f'Sagittal (x={x // 2})')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Display result
show_orthogonal_slices(moving_array, title="Moving Image after Rigid Transform")
