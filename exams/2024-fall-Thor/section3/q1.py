import numpy as np
from utils.io_utils import load_image
from utils.pca_utils import compute_pca, most_similair_indexs
from utils.plot_utils import plot_image


screw_paths = ["screws_" + str(x).zfill(3) + ".jpg" for x in range(0,20)]

screw_images = [load_image("exams/2024-fall-Thor/section3/" + path) for path in screw_paths]

flat_images = [image.flatten() for image in screw_images]

pca,projections = compute_pca(flat_images,7)

print("RATIOS:",pca.explained_variance_ratio_)

coords,distances = most_similair_indexs(projections)
i,j = coords
print("similair:",screw_paths[i],screw_paths[j])

first_pc = projections[:,0]
print("Smallest & largest",screw_paths[np.argmin(first_pc)],screw_paths[np.argmax(first_pc)])

# plot_image(screw_images[np.argmin(first_pc)])
# plot_image(screw_images[np.argmax(first_pc)])

print("Dist between 7 & 8",distances[7,8])