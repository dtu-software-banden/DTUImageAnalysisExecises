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
        img = img.resize((200, 200))  # Resize for uniformity
        img_array = np.array(img).astype(np.float32)  # (64, 64, 3)
        img_flat = img_array.flatten()  # (64*64*3,)
        image_data.append(img_flat)

X = np.stack(image_data)  # Shape: (15, 64*64*3)

# Perform PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Load and process the ideal flower image
with Image.open("idealflower.jpg") as img:
    img = img.convert('RGB')
    img = img.resize((200, 200))
    img_array = np.array(img).astype(np.float32).flatten()  # (64*64*3,)

# Project the ideal flower onto the PCA space
ideal_pca = pca.transform([img_array])  # Shape: (1, n_components)
ideal_pc2_value = ideal_pca[0, 1]  # 2nd principal component (PC2)

# Compare with the PC2 values of the 15 flowers
pc2_values = X_pca[:, 1]  # 2nd column from the PCA projection

# Find the closest match in PC2
closest_idx = np.argmin(np.abs(pc2_values - ideal_pc2_value))
closest_match_name = image_list[closest_idx]

# Report result
print(f"The closest flower to idealflower.jpg along PC2 is: {closest_match_name}")
print(f"  Ideal PC2 value: {ideal_pc2_value:.3f}")
print(f"  Closest match PC2 value: {pc2_values[closest_idx]:.3f}")

# Function to reshape flat RGB array to image
def to_color_image(img_array, shape=(200, 200, 3)):
    img = img_array.reshape(shape)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8))

# Optional: show both images
img_ideal = to_color_image(img_array)
img_match = to_color_image(X[closest_idx])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_ideal)
axes[0].set_title("Ideal Flower")
axes[1].imshow(img_match)
axes[1].set_title(f"Closest Match: {closest_match_name}")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
