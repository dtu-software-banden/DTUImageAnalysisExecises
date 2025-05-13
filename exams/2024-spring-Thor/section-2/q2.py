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

# Compute the average image
average_image = np.mean(X, axis=0)

# Amount of variation along first principal component
variation_amount = 3 * np.sqrt(pca.explained_variance_[0])
direction_vector = pca.components_[0, :]

# Synthesize images
synth_image_plus = average_image + variation_amount * direction_vector
synth_image_minus = average_image - variation_amount * direction_vector

# Function to reshape flat RGB array to image
def to_color_image(img_array, shape=(64, 64, 3)):
    img = img_array.reshape(shape)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8))

# Convert arrays to color images
img_avg = to_color_image(average_image)
img_plus = to_color_image(synth_image_plus)
img_minus = to_color_image(synth_image_minus)

# Display the three images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_minus)
axes[0].set_title("−3σ PC1")
axes[1].imshow(img_avg)
axes[1].set_title("Average Image")
axes[2].imshow(img_plus)
axes[2].set_title("+3σ PC1")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
