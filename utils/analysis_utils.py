import numpy as np
from skimage.filters import threshold_otsu

def hounsfield_of_dicom(dicom):
    # Rescale if necessary (to get Hounsfield units)
    ct_image = dicom
    if 'RescaleSlope' in dicom and 'RescaleIntercept' in dicom:
        ct_image = ct_image * dicom.RescaleSlope + dicom.RescaleIntercept

    return np.mean(ct_image)


def accumulator_image(img: np.ndarray):
    # img is an np array

    # Initialize accumulator with first row same as original
    acc = img.copy()

    # Dynamic programming loop
    for row in range(1, img.shape[0]):
        for col in range(img.shape[1]):
            # Get upper-left, upper, upper-right values (with bounds check)
            upper_vals = []
            if col > 0:
                upper_vals.append(acc[row - 1, col - 1])
            upper_vals.append(acc[row - 1, col])
            if col < img.shape[1] - 1:
                upper_vals.append(acc[row - 1, col + 1])
            # Add current cell value to minimum of upper neighbors
            acc[row, col] += min(upper_vals)

    return acc

def integral_image(img: np.ndarray):
    return img.cumsum(axis=0).cumsum(axis=1)



def linear_transformation(img,min_value,max_value):
    min_val = img.min()
    max_val = img.max()
    scaled = min_value + ((img - min_val) * (max_value - min_value) / (max_val - min_val))
    return scaled




def compute_optimal_path(acc):
    rows, cols = acc.shape
    path = []

    # Start at the minimum value in the bottom row
    j = np.argmin(acc[-1])
    path.append((rows - 1, j))

    for i in reversed(range(rows - 1)):
        candidates = []
        for dj in [-1, 0, 1]:
            nj = j + dj
            if 0 <= nj < cols:
                candidates.append((acc[i, nj], nj))

        min_val, j = min(candidates)
        path.append((i, j))

    # Reverse the path to go from top to bottom if needed
    path.reverse()
    return path