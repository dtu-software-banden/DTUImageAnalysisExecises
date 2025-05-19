import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 


x_org = np.loadtxt("data/winePCA/wine-data.txt", comments="%")
x = x_org[:, :13]
producer = x_org[:, 13]


x_mean = np.mean(x, axis=0)
x_range = np.ptp(x, axis=0)  # ptp = max - min
x_norm = (x - x_mean) / x_range # normalised

data = x_norm 

pca, projections = compute_pca(data)

# PCA
cov_matrix = np.cov(x_norm, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
x_pca = x_norm @ eig_vecs

def Question1():
    print("Running Question 1")
    # Compute average projected value on PC1 for producers 1 and 2
    pc1_proj_producer1 = np.mean(x_pca[producer == 1, 0])
    pc1_proj_producer2 = np.mean(x_pca[producer == 2, 0])
    difference = np.abs(pc1_proj_producer1 - pc1_proj_producer2)
    print(difference)

    pc1 = projections[:, 0]
    mean_pc1_prod1 = np.mean(pc1[producer == 1])
    mean_pc1_prod2 = np.mean(pc1[producer == 2])

    # Compute the absolute difference
    difference = np.abs(mean_pc1_prod1 - mean_pc1_prod2)

    print("Difference between average PC1 values:", difference)

def Question2():
    print("Running Question 2")
    pc1 = projections[:, 0]
    max_ = max(pc1)
    min_ = min(pc1)
    print(min_, max_ , max_ - min_)
    


def Question3():
    print("Running Question 3")
    print(x_org[0][0])
    print(data[0][0])

def Question4():
    print("Running Question 4")
    avg_val = np.mean(cov_matrix)
    print(avg_val)

        # Perform PCA on the normalized data
    pca = PCA(n_components=13)
    pca.fit(x_norm)

    # Calculate the percentage of total variance explained by the first 5 components
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_by_top5 = np.sum(explained_variance_ratio[:5]) * 100  # convert to percent
    print(explained_by_top5)


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()