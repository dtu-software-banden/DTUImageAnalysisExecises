from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import euclidean_distances

def normalize(data):
    mean = np.mean(data, axis=0)
    range_ = np.ptp(data, axis=0)
    return (data - mean) / range_

def normalize_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def compute_pca(data, n_components=5):
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(data)
    return pca, projections

def compute_dyn_pca(data):
    pca = PCA()
    projections = pca.fit_transform(data)
    return pca, projections


def synthesize_from_pc(average_img, component, lambda_, scale=3):
    return average_img + scale * np.sqrt(lambda_) * component


def project_onto_pc1(x_normalized, eigenvectors, eigenvalues):
    idx = np.argsort(-np.abs(eigenvalues))
    first_pc = eigenvectors[:, idx[0]]
    return x_normalized @ first_pc


def project_onto_pc2(x_normalized, eigenvectors, eigenvalues):
    idx = np.argsort(-np.abs(eigenvalues))
    first_pc = eigenvectors[:, idx[1]]
    return x_normalized @ first_pc

def max_projected_abs_value(projections):
    return np.abs(projections).max()

def most_similair_index(pca,projections,new_item):
    new_projection = pca.transform(new_item)
    distances = euclidean_distances(projections, new_projection)
    return np.argmin(distances)

def least_similair_index(pca,projections,new_item):
    new_projection = pca.transform(new_item)
    distances = euclidean_distances(projections, new_projection)
    return np.argmax(distances)

def most_similair_indexs(projections):
    distances = euclidean_distances(projections)
    # Ignore diagonal (self-distances)
    np.fill_diagonal(distances, np.inf)

    # Find the indices of the smallest distance
    return np.unravel_index(np.argmin(distances), distances.shape),distances

def count_needed_to_explain(pca, percent: float): #percent er float fra 0 til 1
    ratios = pca.explained_variance_ratio_
    sum = 0
    for i in range(len(ratios)):
        sum += ratios[i]
        if sum >= percent:
            return i + 1
    
    return len(ratios)

