from scipy.stats import norm
import numpy as np

def gaussian_parameters(pixels):
    return np.mean(pixels), np.std(pixels)

def classify_pixels(pixels, mean1, std1, mean2, std2):
    likelihood1 = norm.pdf(pixels, mean1, std1)
    likelihood2 = norm.pdf(pixels, mean2, std2)
    return likelihood1 > likelihood2

def decision_boundary(mean1, std1, mean2, std2):
    from scipy.optimize import brentq
    def diff(x): return norm.pdf(x, mean1, std1) - norm.pdf(x, mean2, std2)
    return brentq(diff, min(mean1, mean2), max(mean1, mean2))
