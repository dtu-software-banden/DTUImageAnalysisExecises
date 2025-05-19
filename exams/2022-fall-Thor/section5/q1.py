import numpy as np
from utils.io_utils import load_image
from skimage.filters import threshold_otsu
from skimage import color, io, measure, img_as_ubyte, img_as_float

from utils.morph_utils import everything_blobs,filter
from utils.plot_utils import plot_image
from skimage.morphology import closing, disk, label
from skimage.measure import regionprops


image = load_image("exams/2022-fall-Thor/section5/figures.png",grayscale=True)

thresh = threshold_otsu(image)

fg = image < thresh


blobs = filter(fg,min_area=13000)

plot_image(img_as_ubyte(blobs))

labels = label(blobs)
print("Blob count:",np.max(labels))



maxblob = filter(fg,min_area=27000).astype(np.uint)
print(regionprops(maxblob)[0].perimeter)