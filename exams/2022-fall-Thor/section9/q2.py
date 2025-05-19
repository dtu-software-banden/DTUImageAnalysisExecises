import numpy as np
from utils.io_utils import load_image


img1 = load_image("exams/2022-fall-Thor/section9/change1.png",grayscale=True)
img2 = load_image("exams/2022-fall-Thor/section9/change2.png",grayscale=True)

percent = np.sum(np.abs(img1-img2) > 0.3) / (360 * 457)

print("%=",percent)