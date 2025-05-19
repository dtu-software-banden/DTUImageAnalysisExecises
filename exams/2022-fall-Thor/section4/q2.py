import numpy as np
from utils.io_utils import load_image
from skimage.filters import prewitt,prewitt_h
from skimage import color, io, measure, img_as_ubyte, img_as_float

from utils.plot_utils import plot_image


image = load_image("exams/2022-fall-Thor/section4/rocket.png",grayscale=True)


edges = prewitt(image)
print(edges.shape,edges.dtype,np.max(edges),np.min(edges))

fg_mask = edges > 0.06

print(np.sum(np.where(fg_mask,1,0)))

plot_image(img_as_ubyte(fg_mask))