import numpy as np
from skimage import transform as tf
from skimage.transform import warp
from skimage import color, io, measure, img_as_ubyte, img_as_float



from utils.affine_trans_utils import landmark_transform
from utils.io_utils import load_image
from utils.plot_utils import plot_image

img = load_image("exams/2022-fall-Thor/section8/CPHSun.png")

plot_image(img)
from skimage.transform import AffineTransform, warp
import numpy as np

# Define rotation parameters
angle_deg = -16
angle_rad = np.deg2rad(angle_deg)
center = (20, 20)

# Construct the affine transform:
# Step 1: Translate image so rotation center is at origin
# Step 2: Rotate
# Step 3: Translate back

# Step 1: shift image so center is at origin
tform_shift_to_origin = AffineTransform(translation=(-center[0], -center[1]))

# Step 2: rotate around origin
tform_rotate = AffineTransform(rotation=angle_rad)

# Step 3: shift back
tform_shift_back = AffineTransform(translation=center)

# Combine transformations (right to left)
tform = tform_shift_to_origin + tform_rotate + tform_shift_back

# Apply transformation
rotated_img = warp(img, tform.inverse, preserve_range=True)

# If you want to keep data type consistent:
rotated_img = rotated_img.astype(img.dtype)

plot_image(rotated_img)

print(rotated_img[200,200])