import numpy as np

def lda_predict(x, mu1, mu2, sigma):
    sigma_inv = np.linalg.inv(sigma)
    w = sigma_inv @ (mu2 - mu1)
    w0 = -0.5 * (mu2 @ sigma_inv @ mu2) + 0.5 * (mu1 @ sigma_inv @ mu1)
    y = w @ x + w0
    return y, 2 if y > 0 else 1
