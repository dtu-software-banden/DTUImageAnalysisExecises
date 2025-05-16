from sklearn.decomposition import PCA
import numpy as np

def normalize(data):
    mean = np.mean(data, axis=0)
    range_ = np.ptp(data, axis=0)
    return (data - mean) / range_

def compute_pca(data, n_components=5):
    pca = PCA(n_components=n_components)
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