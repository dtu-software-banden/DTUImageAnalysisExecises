import exam_template.utils.io_utils as io
import exam_template.utils.plot_utils as pu
import numpy as np
import cv2


img = io.load_hsv_image("exams/2023-spring-Thor/section4/nike.png")

h_img = img[:,:,0]
h_img_norm = h_img / 256.0

bw_img = np.vectorize(lambda x: 1.0 if 0.3< x and x < 0.7 else 0.0 )(h_img_norm)

# Create a circular structuring element with radius 8
radius = 8
kernel_size = 2 * radius + 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

dialted = cv2.dilate(bw_img,kernel)

pu.plot_image(bw_img)
pu.plot_image(dialted)

print("FG pixels:",dialted.sum())

