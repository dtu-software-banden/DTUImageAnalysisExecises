import numpy as np
from utils.io_utils import load_image
from utils.plot_utils import plot_image
from utils.morph_utils import erode_circle
from utils.morph_utils import filter

image = load_image("exams/2023-spring-Thor/section7/Letters.png")

red_mask = np.apply_along_axis(lambda pixel: 1.0 if pixel[0] > 100 and pixel[1] < 100 and pixel[2] < 100 else 0.0 ,2,image)

eroded_mask = erode_circle(red_mask,3)

blob = filter(eroded_mask,min_area=1000,max_area=4000,min_perim=300)

plot_image(blob)