import numpy as np
from utils.io_utils import load_image
from utils.plot_utils import plot_image
from utils.morph_utils import erode_circle

image = load_image("exams/2023-spring-Thor/section7/Letters.png")

red_mask = np.apply_along_axis(lambda pixel: 1.0 if pixel[0] > 100 and pixel[1] < 100 and pixel[2] < 100 else 0.0 ,2,image)

plot_image(red_mask)

eroded_mask = erode_circle(red_mask,3)

plot_image(eroded_mask)

print("white count:",eroded_mask.sum())


