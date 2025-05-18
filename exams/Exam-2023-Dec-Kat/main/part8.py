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

from skimage.io import imread

# Path to fish images
fish_dir = "./data/Fish"
fish_filenames = sorted([
    "discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
    "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"
])

# Load and stack all fish images into a 4D array (N, H, W, C)
fish_images = [imread(os.path.join(fish_dir, fname)) for fname in fish_filenames]
fish_stack = np.stack(fish_images, axis=0)

# Compute average fish image
average_fish = np.mean(fish_stack, axis=0).astype(np.uint8)

# Display average fish
plt.imshow(average_fish)
plt.title("Average Fish (RGB)")
plt.axis('off')
plt.show()

# Reshape fish images for PCA: (N, H, W, C) -> (N, H*W*C)
N, H, W, C = fish_stack.shape
fish_flat = fish_stack.reshape(N, -1)

# Fit PCA with 6 components
pca = PCA(n_components=6)
projections = pca.fit_transform(fish_flat)


def Question1():
    print("Running Question 1")
    # Compute total variance explained by the first 2 components
    explained_variance = pca.explained_variance_ratio_
    explained_first_two = np.sum(explained_variance[:2])
    print(explained_first_two)


def Question2():
    print("Running Question 2")
    neon = load_image("./data/Fish/neon.jpg")
    guppy = load_image("./data/Fish/guppy.jpg")

    ssd = np.sum((neon - guppy) ** 2)
    print(ssd)

def Question3():
    print("Running Question 3")
    # Get the index of the neon fish in the PCA projection
    neon_proj = projections[3]

    # Compute Euclidean distance from neon fish to all others in PCA space
    distances = np.linalg.norm(projections - neon_proj, axis=1)

    # Find index of the furthest fish
    furthest_index = np.argmax(distances)
    furthest_fish = fish_filenames[furthest_index]
    print(furthest_fish)


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()