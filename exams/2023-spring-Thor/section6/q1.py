import numpy as np
from utils.io_utils import load_image

bg_img = load_image("exams/2023-spring-Thor/section6/background.png",grayscale=True)
new_frame = load_image("exams/2023-spring-Thor/section6/new_frame.png",grayscale=True)

alpha = 0.9

new_background = alpha * bg_img + (1-alpha) * new_frame

diff_img = np.abs(new_background - new_frame)

print("count:",np.vectorize(lambda x: 1 if x > 0.1 else 0)(diff_img).sum())

sub_image = new_background[150:200,150:200]
print("sub mean:",sub_image.mean())