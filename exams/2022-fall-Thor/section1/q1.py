import numpy as np
from utils.classifier_utils import threshold_min_dist_classification
from utils.io_utils import load_dicom, load_image
from utils.morph_utils import dice_score
from utils.plot_utils import plot_image


dicom = load_dicom("exams/2022-fall-Thor/section1/1-162.dcm")
plot_image(dicom)


aorta_mask = load_image("exams/2022-fall-Thor/section1/AortaROI.png")
liver_mask = load_image("exams/2022-fall-Thor/section1/LiverROI.png")
kidney_mask = load_image("exams/2022-fall-Thor/section1/KidneyROI.png")

aorta_values = dicom[aorta_mask]
liver_values = dicom[liver_mask]
kidney_values = dicom[kidney_mask]

t1 = threshold_min_dist_classification(liver_values,kidney_values)
t2 = threshold_min_dist_classification(aorta_values,kidney_values)

print("t1, t2 =",t1,t2)

foreground = (dicom > t1) & (dicom < t2)

plot_image(foreground)

print("Dice:",dice_score(foreground,kidney_mask))