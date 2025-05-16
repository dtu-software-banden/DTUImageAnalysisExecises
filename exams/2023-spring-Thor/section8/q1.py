import numpy as np
from utils.io_utils import load_image
from utils.pca_utils import compute_pca, most_similair_index, project_onto_pc1

import glob

from utils.plot_utils import plot_image


paths = glob.glob("exams/2023-spring-Thor/section8/training/*")

images = [load_image(path) for path in paths]
flat_images = [image.flatten() for image in images]

pca,projections = compute_pca(flat_images)

mean_pca = projections.mean(axis=0)

ssd_pca = np.sum((projections - mean_pca) ** 2, axis=1)  # shape: (10,)

max_ssd_index = np.argmax(ssd_pca)
most_different_image = images[max_ssd_index]  # original shape: (height, width, 3)

print("MDI:",paths[max_ssd_index])

print("PC ratio:",pca.explained_variance_ratio_)
# plot_image(most_different_image)

pc1_values = projections[:,0]

# Find indices of min and max along the first principal axis
min_index = np.argmin(pc1_values)
max_index = np.argmax(pc1_values)

# These are the most divergent items along the 1st principal component
item_1 = images[min_index]
item_2 = images[max_index]

# plot_image(item_1)
# plot_image(item_2)

most_similar_pizza = most_similair_index(pca,projections, load_image("exams/2023-spring-Thor/section8/super_pizza.png").reshape(1,-1))
print(paths[most_similar_pizza])
plot_image(images[most_similar_pizza])
