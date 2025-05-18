import numpy as np
from utils.classifier_utils import threshold_min_dist_classification
from utils.io_utils import load_dicom, load_image
from utils.morph_utils import dice_score, morph_closing, morph_opening,filter
from utils.plot_utils import plot_image
from skimage.util import img_as_float,img_as_ubyte
from skimage.morphology import closing, disk, label




dicom = load_dicom("exams/2023-fall-Thor/section4/1-001.dcm")

myoRoi = load_image("exams/2023-fall-Thor/section4/MyocardiumROI.png")
bloodRoi = load_image("exams/2023-fall-Thor/section4/BloodROI.png")

bloodGT = load_image("exams/2023-fall-Thor/section4/BloodGT.png")

blood_houns_avg = np.mean(dicom[bloodRoi])
blood_houns_std = np.std(dicom[bloodRoi])

l_range = blood_houns_avg - 3 * blood_houns_std
u_range = blood_houns_avg + 3 * blood_houns_std

print("Range:",[l_range,u_range])

foreground = img_as_ubyte((dicom > l_range) & (dicom < u_range))

morphed_foreground = morph_opening(morph_closing(foreground,radius=3),radius=5)

print("Blob count:",np.max(label(morphed_foreground)))

blobs = filter(morphed_foreground,max_area=5000,min_area=2000)

print("Dice score:",dice_score(bloodGT,blobs))

# plot_image(dicom)
# plot_image(foreground)
# plot_image(morphed_foreground)
# plot_image(bloodRoi)
# plot_image(blobs)

thresh = threshold_min_dist_classification(dicom[bloodRoi],dicom[myoRoi])
print("Class thresh:",thresh)