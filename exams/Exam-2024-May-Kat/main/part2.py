import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 


# Directory containing flower images
training_dir = "data/flowers"
image_filenames = sorted([
    f"flower{str(i).zfill(2)}.jpg" for i in range(1, 16)
])


ideal_floweer = load_image("data/flowers/idealflower.jpg")


# Load and stack all images into a 4D array (N, H, W, C)
images = [imread(os.path.join(training_dir, fname)) for fname in image_filenames]
image_stack = np.stack(images, axis=0)

# Compute average image
average_flower  = np.mean(image_stack, axis=0).astype(np.uint8)

# Display average pizza
plt.imshow(average_flower)
plt.title("Average flower (RGB)")
plt.axis('off')
plt.show()

# Flatten images for PCA (shape: 10 x (H*W*3))
flat_images = image_stack.reshape(len(image_stack), -1)

# Run PCA on flattened images
pca, projections = compute_pca(flat_images)



def Question1():
    print("Running Question 1")
    synth_image_plus = flat_images + 3 * np.sqrt(pca.explained_variance_[0]) * pca.components_[0, :]
    synth_image_minus = flat_images - 3 * np.sqrt(pca.explained_variance_[0]) * pca.components_[0, :]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(synth_image_minus)
    axs[0].set_title("Synthetic Image - PC1")
    axs[1].imshow(flat_images)
    axs[1].set_title("Average Image")
    axs[2].imshow(synth_image_plus)
    axs[2].set_title("Synthetic Image + PC1")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    #plt.show()



def Question2():
    print("Running Question 2")
    # Project all flower vectors onto the first principal component
    flower_pc1_values = projections[:, 0]

    # Find indices of the two flowers with minimum and maximum projection values
    min_index = np.argmin(flower_pc1_values)
    max_index = np.argmax(flower_pc1_values)

    print(min_index + 1, max_index + 1)  # +1 to match file numbering (flower01 to flower15)



def Question3():
    print("Running Question 3")
    explained_variance_pc1 = pca.explained_variance_ratio_[0] * 100  # convert to percent
    print(explained_variance_pc1)


def Question4():
    print("Running Question 4")
    # Load ideal flower image
    ideal_image_path = os.path.join(training_dir, "idealflower.jpg")
    ideal_image = imread(ideal_image_path).astype(np.float32)

    # Reshape and project to PCA space
    ideal_vector = ideal_image.reshape(1, -1)
    ideal_proj = pca.transform(ideal_vector)

    # Compare only the second principal component
    flower_proj = pca.transform(flat_images)
    ideal_pc2 = ideal_proj[0, 1]
    flower_pc2_values = flower_proj[:, 1]

    # Find index of flower closest to ideal in PC2
    closest_index = np.argmin(np.abs(flower_pc2_values - ideal_pc2))
    print(closest_index + 1)  # convert to 1-based indexing (flower01.jpg to flower15.jpg)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()