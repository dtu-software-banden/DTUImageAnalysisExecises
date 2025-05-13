import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and flatten color images
image_list = [f"flower{str(i).zfill(2)}.jpg" for i in range(1, 16)]
image_data = []

for img_name in image_list:
    with Image.open(img_name) as img:
        img = img.convert('RGB')  # Keep color
        img = img.resize((64, 64))  # Resize for uniformity
        img_array = np.array(img).astype(np.float32)  # (64, 64, 3)
        img_flat = img_array.flatten()  # (64*64*3,)
        image_data.append(img_flat)

X = np.stack(image_data)  # Shape: (15, 64*64*3)

# Perform PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Fraction of total variance explained by the first PC
explained_pc1 = pca.explained_variance_ratio_[0]
print(f"PC1 explains {explained_pc1 * 100:.2f}% of the total variance.")


# Compute the average image
average_image = np.mean(X, axis=0)

# Amount of variation along first principal component
variation_amount = 3 * np.sqrt(pca.explained_variance_[0])
direction_vector = pca.components_[0, :]

# Function to reshape flat RGB array to image
def to_color_image(img_array, shape=(64, 64, 3)):
    img = img_array.reshape(shape)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8))

# Project images onto the first principal component (1D projection)
pc1_values = X_pca[:, 0]  # Shape: (15,)

# Find indices of the max and min values along PC1
idx_min = np.argmin(pc1_values)
idx_max = np.argmax(pc1_values)

# Get the corresponding image names
img_min_name = image_list[idx_min]
img_max_name = image_list[idx_max]

# Print result
print(f"The two most different flowers along PC1 are:")
print(f"  - {img_min_name} (lowest PC1 value: {pc1_values[idx_min]:.2f})")
print(f"  - {img_max_name} (highest PC1 value: {pc1_values[idx_max]:.2f})")

# Optional: Show both images
img_min = to_color_image(X[idx_min])
img_max = to_color_image(X[idx_max])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_min)
axes[0].set_title(f"{img_min_name}\nLowest on PC1")
axes[1].imshow(img_max)
axes[1].set_title(f"{img_max_name}\nHighest on PC1")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
