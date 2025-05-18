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


def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)



# Step 1: Load the images
fixedImage = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii.gz")
movingImage = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii.gz")

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
imshow_orthogonal_view(ImgT1_B, title='Moving image')
# overlay_slices(fixedImage, ImgT1_B, title='Overlay')

params = tform_reg.GetParameters()
angles = params[:3]
trans = params[3:6]
print('Estimated translation: ')
print(np.round(trans, 3))
print('Estimated rotation (deg): ')
print(np.round(np.rad2deg(angles), 3))