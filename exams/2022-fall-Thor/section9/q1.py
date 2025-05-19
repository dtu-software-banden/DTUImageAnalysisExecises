import numpy as np
from utils.io_utils import load_image
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu

from utils.morph_utils import erode_circle
from utils.plot_utils import plot_image


img = load_image("exams/2022-fall-Thor/section9/pixelwise.png")
plot_image(img)
hsv_img = color.rgb2hsv(img)
plot_image(hsv_img)
h_channel = hsv_img[:,:,1]
plot_image(h_channel)
thresh = threshold_otsu(h_channel)

fg = h_channel > thresh

plot_image(fg)

eroded = erode_circle(img_as_ubyte( fg),4)

plot_image(eroded)
print("Pixel count:",np.sum(eroded > 0))