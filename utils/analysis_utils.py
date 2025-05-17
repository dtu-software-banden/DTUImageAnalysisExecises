import numpy as np


def hounsfield_of_dicom(dicom):
    # Rescale if necessary (to get Hounsfield units)
    ct_image = dicom
    if 'RescaleSlope' in dicom and 'RescaleIntercept' in dicom:
        ct_image = ct_image * dicom.RescaleSlope + dicom.RescaleIntercept

    return np.mean(ct_image)