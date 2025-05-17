import math
import numpy as np
from utils.io_utils import load_image
from skimage.util import img_as_float,img_as_ubyte

from utils.morph_utils import morph_opening
from utils.plot_utils import plot_image



x_image = img_as_ubyte(load_image("exams/2024-fall-Thor/section5/x_NisslStain_9-260.81.png",grayscale=True))
y_image = img_as_ubyte(load_image("exams/2024-fall-Thor/section5/y_NisslStain_9-260.81.png",grayscale=True))

x_thresh = morph_opening(img_as_ubyte(x_image > 30),radius=3)
y_thresh = morph_opening(img_as_ubyte(y_image > 30),radius=3)


# plot_image(x_thresh)
# plot_image(y_thresh)


fixed_landmarks_img = load_image("exams/2024-fall-Thor/section5/LabelsFixedImg.png")
moving_landmarks_img = load_image("exams/2024-fall-Thor/section5/LabelsMovingImg.png")

fixed_landmarks = []
fixed_mean_x = 0
fixed_mean_y = 0
for i in range(1,6):
    coord = np.argwhere(fixed_landmarks_img == i)
    fixed_mean_x += coord[0,0] / 5.0
    fixed_mean_y += coord[0,1] / 5.0
    fixed_landmarks.append((coord[0][0],coord[0][1]))

moving_landmarks = []
moving_mean_x = 0
moving_mean_y = 0
for i in range(1,6):
    coord = np.argwhere(moving_landmarks_img == i)
    moving_mean_x += coord[0,0] / 5.0
    moving_mean_y += coord[0,1] / 5.0
    moving_landmarks.append((coord[0][0],coord[0][1]))

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

print(euclidean_distance((fixed_mean_x,fixed_mean_y),(moving_mean_x,moving_mean_y)))
