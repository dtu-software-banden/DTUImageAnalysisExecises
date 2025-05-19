from utils.analysis_utils import linear_transformation
from utils.io_utils import load_image
from skimage.filters import threshold_otsu

from utils.plot_utils import plot_image


image = load_image("exams/2022-fall-Thor/section4/pixelwise.png",grayscale=True)

timage = linear_transformation(image,0.1,0.6)

thresh = threshold_otsu(timage)
print("Otsu:",thresh)

foreground_mask = timage > thresh

plot_image(foreground_mask)