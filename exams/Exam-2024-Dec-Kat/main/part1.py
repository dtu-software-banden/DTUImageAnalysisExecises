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

closed_mask = everything_blobs(image, lower, upper,disk_radius=3,min_area= 2000,max_area= 5000, min_perim=400, max_perim=600)


# === Optional: Show result ===
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original DICOM")
axes[1].imshow(closed_mask, cmap='gray')
axes[1].set_title("Final Mask after Filtering + Closing")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

def Question1():
    print("Running Question 1")
    print("2000")


def Question2():
    print("Running Question 2")
    expert_binary = load_image("./data/kidneys_gt.png")
    score = dice_score(closed_mask, expert_binary)
    print(score)


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()