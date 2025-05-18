import glob

import numpy as np

from utils.io_utils import load_image
from utils.pca_utils import compute_pca, least_similair_index


paths = glob.glob("exams/2023-fall-Thor/section7/*.jpg")
paths.sort()

images = [load_image(path) for path in paths]

flat_images = [image.flatten() for image in images]

pca,projections = compute_pca(flat_images,n_components=6)

print("Pca explained:",pca.explained_variance_ratio_[:2].sum())

gubby = flat_images[1]
neon = flat_images[3]

sqDiff = np.sum((neon - gubby) ** 2)
print("Diff:",sqDiff)

neon_proj = projections[3,:]

minval = 1000000000000
minindex = 0
for i in range(10):
    proj = projections[i,:]
    val = np.sum((neon_proj - proj)**2)
    if i != 3 and val < minval:
        minval = val
        minindex = i
    
print(paths[minindex])

# print("Least similair",coords)