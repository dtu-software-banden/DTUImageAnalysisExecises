import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.util import img_as_ubyte


def plot_blob_mask(mask):
    plt.imshow(mask, cmap='nipy_spectral')
    plt.title("Labeled BLOBs")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_image(img, title='', cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_histogram(data, bins=100, title='Histogram', xlabel='Value'):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def imshow_orthogonal_view(data, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    #data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def rotation_matrix(pitch, roll, yaw, deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R


import matplotlib.pyplot as plt

def plot_pca_components(X_pca, labels=None, highlight_index=None):
    """
    Plots the first two principal components from PCA-transformed data.

    Parameters:
        X_pca (array-like): PCA-transformed data with shape (n_samples, n_components).
        labels (array-like, optional): Optional labels for coloring the points.
        highlight_index (int, optional): Index of a point to highlight specially.
    """
    plt.figure(figsize=(6, 5))

    # Plot all points
    if labels is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
    else:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', color='grey')

    # Highlight a specific point if requested
    if highlight_index is not None:
        plt.scatter(X_pca[highlight_index, 0], X_pca[highlight_index, 1],
                    color='red', edgecolor='black', s=120, marker='X', label='Highlighted')
        plt.legend()

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('First Two Principal Components')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
