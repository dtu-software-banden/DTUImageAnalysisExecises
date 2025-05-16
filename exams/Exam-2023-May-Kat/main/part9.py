import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 

import os
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

# Directory containing training images
training_dir = "data/PizzaPCA/training"
image_filenames = sorted([
    "BewareOfOnions.png", "BigSausage.png", "CucumberParty.png", "FindTheOlives.png",
    "GreenHam.png", "Leafy.png", "PaleOne.png", "SnowAndGrass.png", "TheBush.png", "WhiteSnail.png"
])

# Load and stack all images into a 4D array (N, H, W, C)
images = [imread(os.path.join(training_dir, fname)) for fname in image_filenames]
image_stack = np.stack(images, axis=0)

# Compute average image
average_pizza = np.mean(image_stack, axis=0).astype(np.uint8)

# Display average pizza
plt.imshow(average_pizza)
plt.title("Average Pizza (RGB)")
plt.axis('off')
plt.show()

# Flatten images for PCA (shape: 10 x (H*W*3))
flat_images = image_stack.reshape(len(image_stack), -1)

# Run PCA on flattened images
pca, projections = compute_pca(flat_images)

def Question1():
    print("Running Question 1")
    # Compute sum of squared differences from the average for each pizza
    ssd_values = [np.sum((img.astype(np.float32) - average_pizza) ** 2) for img in image_stack]

    # Find the index of the pizza with the largest SSD
    most_unique_index = int(np.argmax(ssd_values))
    most_unique_pizza = image_filenames[most_unique_index]

    print(most_unique_pizza)



def Question2():
    print("Running Question 2")
    first_component_variance = pca.explained_variance_ratio_[0]
    print(first_component_variance)


def Question3():
    print("Running Question 3")
    # projections should be shape (10, 5) if 5 PCs were used
    first_pc_coords = projections[:, 0]

    # Index of most positive and most negative on PC1
    max_idx = np.argmax(first_pc_coords)
    min_idx = np.argmin(first_pc_coords)

    # Get corresponding filenames
    signature_positive = image_filenames[max_idx]
    signature_negative = image_filenames[min_idx]

    print("Signature pizzas:")
    print("  Positive extreme:", signature_positive)
    print("  Negative extreme:", signature_negative)


def Question4():
    print("Running Question 4")
    from skimage.io import imread
    import numpy as np

    # Load and flatten super pizza
    super_pizza = imread("data/PizzaPCA/super_pizza.png").reshape(1, -1)

    # Project to PCA space
    super_proj = pca.transform(super_pizza)

    # Compute Euclidean distances
    distances = np.linalg.norm(projections - super_proj, axis=1)

    # Find closest pizza
    closest_idx = np.argmin(distances)
    closest_pizza = image_filenames[closest_idx]

    print("Most similar pizza to super_pizza:", closest_pizza)




if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()