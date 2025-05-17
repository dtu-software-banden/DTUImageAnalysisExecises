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


image = load_dicom("data/kidneys/1-189.dcm")

# === Step 2: Thresholding to create binary mask ===
lower, upper = 100, 250

closed_mask = segment_blobs_2_threshold(image, lower, upper, 3, 0)


# === Optional: Show result ===
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original DICOM")
axes[2].imshow(closed_mask, cmap='gray')
axes[2].set_title("Final Mask after Filtering + Closing")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

def Question1():
    print("Running Question 1")


def Question2():
    print("Running Question 2")


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()