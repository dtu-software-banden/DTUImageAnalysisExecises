import numpy as np


def lda_train(X1, X2):
    """Trains LDA parameters given class 1 and class 2 samples"""
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    sigma1 = np.cov(X1, rowvar=False)
    sigma2 = np.cov(X2, rowvar=False)
    sigma = (sigma1 + sigma2) / 2
    return mu1, mu2, sigma

def lda_predict_batch(X, mu1, mu2, sigma):
    """Predicts using LDA for a batch of samples"""
    sigma_inv = np.linalg.inv(sigma)
    w = sigma_inv @ (mu2 - mu1)
    w0 = -0.5 * (mu2 @ sigma_inv @ mu2) + 0.5 * (mu1 @ sigma_inv @ mu1)
    y = X @ w + w0
    return y, np.where(y > 0, 2, 1) 
