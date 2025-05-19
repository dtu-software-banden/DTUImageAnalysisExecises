import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
from skimage.util import img_as_ubyte

from utils.those_stupid_ass_3d_utils import apply_transform, imshow_orthogonal_view

rotation = [
    0.98, -0.16,  0.17,
    0.26,  0.97,  0.0,
   -0.17,  0.04,  0.98
]

# Define the translation vector (from the 4th column of the matrix)
translation = [0.0, -15.0, 0.0]

# Create and configure the affine transform
affine_transform = sitk.AffineTransform(3)  # 3D transform
affine_transform.SetMatrix(rotation)
affine_transform.SetTranslation(translation)

img = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii.gz")

rotated_img = apply_transform(img,affine_transform)

imshow_orthogonal_view(rotated_img)