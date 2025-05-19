import numpy as np
from skimage import transform as tf
from skimage.transform import warp
from skimage import color, io, measure, img_as_ubyte, img_as_float



from utils.affine_trans_utils import landmark_transform
from utils.io_utils import load_image
from utils.plot_utils import plot_image


src_lm = np.array([(220, 55), (105, 675), (315, 675)])
dst_lm = np.array([(100, 165), (200, 605), (379, 525)])

print("Pre error:",np.sum((src_lm-dst_lm)**2))

rocket_img = img_as_ubyte(load_image("exams/2022-fall-Thor/section7/rocket.png",grayscale=True))

tform = tf.estimate_transform('euclidean', src_lm, dst_lm)

trans_dst = tform(src_lm)
print(dst_lm)
print(trans_dst)
print("Post error:",np.sum((trans_dst-dst_lm)**2))

trns_img = img_as_ubyte(warp(rocket_img, inverse_map=tform.inverse, output_shape=rocket_img.shape))

plot_image(rocket_img)
plot_image(trns_img)

from skimage.filters import gaussian

gauss_img = img_as_ubyte(gaussian(rocket_img,3))

print("Pixel 100 100:",gauss_img[100,100])
plot_image(gauss_img)