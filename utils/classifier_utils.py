from scipy.stats import norm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np


def threshold_min_dist_classification(values1, values2):
    return (np.mean(values1) + np.mean(values2)) / 2 


def parametric_classifier_predict(value, img):
    mean, std = gaussian_parameters(img)
    # Evaluate both Gaussians at value = 38
    predict = norm.pdf(value, loc=mean, scale=std)
    return predict


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


def qda_1d_threshold(class1, class2, value_range=(0, 255), resolution=1000):
    """
    parametric classifier
    Trains a 1D QDA classifier on two sets of values and returns the threshold where
    class prediction switches from class1 to class2.
    """
    X = np.concatenate([class1, class2]).reshape(-1, 1)
    y = np.array([0] * len(class1) + [1] * len(class2))

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    test_values = np.linspace(*value_range, resolution).reshape(-1, 1)
    predictions = qda.predict(test_values)

    switch_indices = np.where(np.diff(predictions) != 0)[0]
    if switch_indices.size > 0:
        return test_values[switch_indices[0]][0]
    return None