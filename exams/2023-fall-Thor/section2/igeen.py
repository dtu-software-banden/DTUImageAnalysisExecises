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

from utils.those_stupid_ass_3d_utils import apply_transform, find_affine, imshow_orthogonal_view, overlay_slices, rotation_matrix

# rotation = [
#     0.98, -0.16,  0.17,
#     0.26,  0.97,  0.0,
#    -0.17,  0.04,  0.98
# ]

# # Define the translation vector (from the 4th column of the matrix)
# translation = [0.0, -15.0, 0.0]

# # Create and configure the affine transform
# affine_transform = sitk.AffineTransform(3)  # 3D transform
# affine_transform.SetMatrix(rotation)
# affine_transform.SetTranslation(translation)

imgv1 = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v1.nii.gz")
imgv2 = sitk.ReadImage("exams/2023-fall-Thor/section2/ImgT1_v2.nii.gz")

# rotated_img = apply_transform(imgv2,affine_transform)

# imshow_orthogonal_view(rotated_img,origin=None)


# trans,rot,rotatedv2 = find_affine(imgv1,imgv2,step_size=0.1,plot_progess=False)

# overlay_slices(imgv1, rotatedv2, title = 'ImgT1 (red) vs. ImgT1_A (green)')

# print(trans,rot)
roll_radians = np.deg2rad(-20)
_,rotation = rotation_matrix(0,roll_radians,0)
print(rotation)

affine_transform = sitk.AffineTransform(3)  # 3D transform
affine_transform.SetMatrix(rotation)

rotated_img = apply_transform(imgv2,affine_transform,rotate_center=False)

overlay_slices(imgv1,rotated_img)

imgv2_arr = sitk.GetArrayFromImage(rotated_img)

imgv1_arr = sitk.GetArrayFromImage(imgv1)
mask = imgv1_arr > 50

diff = np.mean((imgv2_arr[mask] - imgv1_arr[mask])**2)
print("Diff:",diff)
