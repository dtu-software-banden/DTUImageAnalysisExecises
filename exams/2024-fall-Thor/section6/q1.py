import numpy as np
import matplotlib.pyplot as plt

from utils.io_utils import load_image

image = load_image("exams/2024-fall-Thor/section6/CubesG.png")

# Example mean RGB vectors for 3 classes + background (you should replace these with your own)
# Format: [R, G, B]

with open("exams/2024-fall-Thor/section6/A_Cubes.txt", 'r') as file:
    int_list = [int(line.strip()) for line in file if line.strip()]
    Amean = np.mean(int_list)

with open("exams/2024-fall-Thor/section6/B_Cubes.txt", 'r') as file:
    int_list = [int(line.strip()) for line in file if line.strip()]
    Bmean = np.mean(int_list)

with open("exams/2024-fall-Thor/section6/C_Cubes.txt", 'r') as file:
    int_list = [int(line.strip()) for line in file if line.strip()]
    Cmean = np.mean(int_list)

with open("exams/2024-fall-Thor/section6/D_Cubes.txt", 'r') as file:
    int_list = [int(line.strip()) for line in file if line.strip()]
    Dmean = np.mean(int_list)

with open("exams/2024-fall-Thor/section6/E_Cubes.txt", 'r') as file:
    int_list = [int(line.strip()) for line in file if line.strip()]
    Emean = np.mean(int_list)

print(Amean,Bmean,Cmean,Dmean,Emean)

import numpy as np
import matplotlib.pyplot as plt

# Assume grayscale image of dtype uint8 and shape (H, W)
# image = ...

# Define mean grayscale intensities for background and 3 classes
# Replace with actual values for your problem
class_means = {
    0: 0,   # Background
    1: Bmean,  # Class 1
    2: Cmean,  # Class 2
    3: Dmean,  # Class 3
}

# Flatten image for vectorized processing
H, W = image.shape
pixels = image.flatten().astype(np.float32)  # shape (H*W,)

# Compute distances to each class mean
distances = np.zeros((pixels.shape[0], len(class_means)))
for class_id, mean_val in class_means.items():
    distances[:, class_id] = np.abs(pixels - mean_val)  # Euclidean distance in 1D

# Classify each pixel based on the minimum distance
classified_flat = np.argmin(distances, axis=1)
classified = classified_flat.reshape((H, W))

# Optional: color visualization (assign RGB colors to each class)
colors = np.array([
    [0, 0, 0],         # Background - black
    [0, 0, 200],       # Class 1 - red
    [0, 200, 0],       # Class 2 - green
    [200, 200, 0],       # Class 3 - blue
], dtype=np.uint8)

# Convert class labels to RGB image for display
classified_rgb = colors[classified]

# Display the result
plt.imshow(classified_rgb)
plt.title("Minimum Distance Classification (Grayscale)")
plt.axis("off")
plt.show()
