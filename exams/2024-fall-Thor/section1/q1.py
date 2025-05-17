import numpy as np
from utils.io_utils import load_dicom, load_image
from utils.plot_utils import plot_image
from utils.morph_utils import dice_score, filter,morph_closing
from skimage.util import img_as_ubyte



dicom_data = load_dicom("exams/2024-fall-Thor/section1/1-189.dcm")
threshholded = np.where((dicom_data > 100) & (dicom_data < 250),1,0)

# plot_image(threshholded)

blobs = filter(threshholded,min_perim=400,max_perim=600,max_area=5000,min_area=2000)

closed = morph_closing(img_as_ubyte(blobs),radius=3)

# plot_image(blobs)

expert_image = img_as_ubyte(load_image("exams/2024-fall-Thor/section1/kidneys_gt.png"))

# plot_image(closed)
# plot_image(expert_image)


print("DICE: ",dice_score(expert_image,closed))
area = closed.sum()