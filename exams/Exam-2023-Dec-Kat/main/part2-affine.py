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

import SimpleITK as sitk

# Load both NIfTI images
img1 = sitk.ReadImage('./data/ImgT1_v1.nii.gz')
img2 = sitk.ReadImage('./data/ImgT1_v2.nii.gz')

# Define the affine matrix A and translation vector
matrix_A = [0.98, -0.16, 0.17,
            0.26,  0.97, 0.00,
           -0.17,  0.04, 0.98]  # Row-major order for SimpleITK

translation = [0.0, -15.0, 0.0]  # Translation vector

# Create affine transform and set parameters
affine = sitk.AffineTransform(3)

affine.SetTranslation(translation)
affine.SetMatrix(matrix_A)

# Apply the transform using a resampler
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(img1)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(affine)

# Apply transformation
transformed_img = resampler.Execute(img1)

# Optional: visualize or save
# imshow_orthogonal_view(transformed_img, origin=None)
sitk.WriteImage(transformed_img, './data/ImgT1_v1_transformed.nii.gz')


def Question1():
    print("Running Question 1")
    imshow_orthogonal_view(sitk.GetArrayFromImage(transformed_img), origin=None)

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