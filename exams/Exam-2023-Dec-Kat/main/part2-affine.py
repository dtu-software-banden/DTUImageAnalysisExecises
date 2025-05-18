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

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_closing, binary_opening
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import median
from scipy.stats import norm
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
import SimpleITK as sitk


def Question1():
    print("Running Question 1")
     # Load both MRI volumes
    img1 = sitk.ReadImage("data/ImgT1_v1.nii.gz")
    img2 = sitk.ReadImage("data/ImgT1_v2.nii.gz")

    # Define the affine matrix A
    A = np.array([
        [ 0.98, -0.16,  0.17,   0],
        [ 0.26,  0.97,  0.00, -15],
        [-0.17,  0.04,  0.98,   0],
        [ 0.00,  0.00,  0.00,   1]
    ])

    # Extract 3x3 matrix and translation
    rotation_matrix = A[:3, :3].flatten().tolist()
    translation = A[:3, 3].tolist()

    # Create and configure the affine transform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rotation_matrix)
    transform.SetTranslation(translation)

    # Resample img2 to align with img1
    resampled = sitk.Resample(img2, img1, transform, sitk.sitkLinear, 0.0, img2.GetPixelID())

    # Show orthogonal view of result
    imshow_orthogonal_view(sitk.GetArrayFromImage(resampled), title="Transformed ImgT1_v2 aligned to ImgT1_v1")


import numpy as np
import SimpleITK as sitk
def Question2():
    print("Running Question 2")
    fixedImage = sitk.ReadImage("data/ImgT1_v1.nii.gz")
    movingImage = sitk.ReadImage("data/ImgT1_v2.nii.gz")

    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid scheule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid
    initTransform = sitk.CenteredTransformInitializer(fixedImage, movingImage, sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track to the optimization process
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixedImage, movingImage)

    # Apply the estimated transformation to the moving image
    ImgT1_B = sitk.Resample(movingImage, tform_reg)
    imshow_orthogonal_view(sitk.GetArrayFromImage(ImgT1_B), title='Moving image')
    overlay_slices(fixedImage, ImgT1_B, title='Overlay')

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))

def Question3():
    print("Running Question 3")

    # Load fixed and moving volumes
    fixed = sitk.ReadImage("data/ImgT1_v1.nii.gz")
    moving = sitk.ReadImage("data/ImgT1_v2.nii.gz")

    # === Step 1: Create rigid transform with -20° roll (rotation around Y-axis) ===
    transform = sitk.Euler3DTransform()
    transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([s/2 for s in fixed.GetSize()]))
    transform.SetRotation(0.0, np.deg2rad(-20.0), 0.0)  # (Rx, Ry=-20deg, Rz)

    # === Step 2: Create a brain mask from fixed image where intensity > 50 ===
    fixed_array = sitk.GetArrayFromImage(fixed)
    brain_mask_array = fixed_array > 50
    brain_mask = sitk.GetImageFromArray(brain_mask_array.astype(np.uint8))
    brain_mask.CopyInformation(fixed)

    # === Step 3: Resample moving image to fixed space ===
    resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    # === Step 4: Compute Mean Squared Error inside the mask ===
    resampled_array = sitk.GetArrayFromImage(resampled)
    mse = np.mean((resampled_array[brain_mask_array] - fixed_array[brain_mask_array]) ** 2)

    print("Mean Squared Error inside brain mask:", mse)



def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()