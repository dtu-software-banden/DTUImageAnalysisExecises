import numpy as np
from utils.io_utils import load_image
from skimage.color import rgb2hsv
from skimage.morphology import closing, disk, label

from utils.plot_utils import plot_image



img1 = load_image("exams/2023-fall-Thor/section3/frame_1.jpg")
img2 = load_image("exams/2023-fall-Thor/section3/frame_2.jpg")

img1hsv = rgb2hsv(img1)
img2hsv = rgb2hsv(img2)

s1 = img1hsv[:,:,1] * 255
s2 = img2hsv[:,:,1] * 255

diff_img = np.abs(s1 - s2)

diff_avg = np.mean(diff_img)
diff_std = np.std(diff_img)

thresh = diff_avg + 2*diff_std
print("Threshhold:",thresh)
change_img = diff_img > thresh

change_cnt = np.sum(change_img > 0)
print("Changed pixels:",change_cnt)

labels = label(change_img)
label_count = np.unique(labels)

maxLabel = 0
for i in label_count[1:]:
    mask = labels == i
    value = np.sum(mask)
    # print("Label",i,":",value)
    if value > maxLabel:
        maxLabel = value
    
print("Max label:",maxLabel)
plot_image(change_img)