import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and flatten the images
image_list = [f"flower{str(i).zfill(2)}.jpg" for i in range(1, 16)]
image_data = []

for img_name in image_list:
    with Image.open(img_name) as img:
        img = img.convert('L')  # grayscale
        img = img.resize((64, 64))  # consistent size
        img_array = np.array(img).flatten()
        image_data.append(img_array)

X = np.stack(image_data)  # Shape: (15, 4096)

# Perform PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Compute the average image
average_image = np.mean(X, axis=0)

# Amount of variation to apply (3 std dev along PC1)
variation_amount = 3 * np.sqrt(pca.explained_variance_[0])
direction_vector = pca.components_[0, :]

# Synthesize images
synth_image_plus = average_image + variation_amount * direction_vector
synth_image_minus = average_image - variation_amount * direction_vector

# Reshape back to image size
def to_image(img_array, shape=(64, 64)):
    img = img_array.reshape(shape)
    img = np.clip(img, 0, 255)  # Ensure pixel values are valid
    return Image.fromarray(img.astype(np.uint8))

# Convert arrays to images
img_avg = to_image(average_image)
img_plus = to_image(synth_image_plus)
img_minus = to_image(synth_image_minus)

# Display the three images side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_minus, cmap='gray')
axes[0].set_title("−3σ PC1")
axes[1].imshow(img_avg, cmap='gray')
axes[1].set_title("Average Image")
axes[2].imshow(img_plus, cmap='gray')
axes[2].set_title("+3σ PC1")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
