import numpy as np
from skimage.io import imread
import pydicom

def load_roi_list(paths):
    ls = []
    for path in paths:
        ls.append(imread(path) > 0)
    return ls



def hounsfield_of_dicom(dicom):
    # Rescale if necessary (to get Hounsfield units)
    ct_image = dicom
    if 'RescaleSlope' in dicom and 'RescaleIntercept' in dicom:
        ct_image = ct_image * dicom.RescaleSlope + dicom.RescaleIntercept

    return np.mean(ct_image)