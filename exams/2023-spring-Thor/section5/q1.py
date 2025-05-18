import numpy as np
from utils.io_utils import load_image
from skimage.util import img_as_ubyte
from utils.affine_trans_utils import landmark_transform,compute_alignment_error

shoe1 = load_image("exams/2023-spring-Thor/section5/shoe_1.png")
shoe2 = load_image("exams/2023-spring-Thor/section5/shoe_2.png")

print(shoe1.shape)

lm1 = np.array([[40, 320], [425, 120], [740, 330]])
lm2 = np.array([[80, 320], [380, 155], [670, 300]])

transformed_dst,tform = landmark_transform(shoe1,shoe2,lm1,lm2)
print(transformed_dst.shape)

print("Scale",tform.scale)

# plot_image(shoe1)
# plot_image(shoe2)
# plot_image(transformed_dst)

F_before,F_after = compute_alignment_error(lm1,lm2,tform)

print("before after",F_before,F_after)

shoe1_trans = img_as_ubyte(transformed_dst)

print(shoe1.dtype)

print(shoe2[200,200,2], shoe1_trans[200,200,2])
print(np.abs(shoe2[200,200,2]- shoe1_trans[200,200,2]))