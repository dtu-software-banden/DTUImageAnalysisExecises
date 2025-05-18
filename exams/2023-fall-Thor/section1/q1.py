import numpy as np
from utils.io_utils import load_image
from utils.morph_utils import histogram_stretch
from utils.plot_utils import plot_histogram, plot_image
from skimage.filters import prewitt,prewitt_h


img = load_image("exams/2023-fall-Thor/section1/ardeche_river.jpg",grayscale=True)

stretched = histogram_stretch(img,min=0.2,max=0.8)

# plot_histogram(img)
# plot_histogram(stretched)

avg_value = np.mean(stretched)
print("avg stretched",avg_value)

edges = prewitt_h(stretched)

plot_image(edges)

abs_edges = np.abs(edges)

max_edge_value = np.max(abs_edges)
print("Max edge",max_edge_value)

mask = img > avg_value

foreground_count = np.sum(mask)
print("Count foreground",foreground_count)