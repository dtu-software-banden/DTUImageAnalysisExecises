import numpy as np
from skimage import transform as tf
from skimage.transform import warp
import numpy as np
from scipy.ndimage import map_coordinates


def translation_matrix(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def rotation_matrix_x_pitch(alpha):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_y_roll(beta):
    return np.array([
        [np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_z_yaw(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def shear_matrix(sxy, sxz, syz):
    return np.array([
        [1, sxy, sxz, 0],
        [0, 1, syz, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def affine_transformation(dx, dy, dz, alpha, beta, gamma, sx, sy, sz, sxy, sxz, syz):
    T = translation_matrix(dx, dy, dz)
    Rx = rotation_matrix_x_pitch(alpha)
    Ry = rotation_matrix_y_roll(beta)
    Rz = rotation_matrix_z_yaw(gamma)
    S = scaling_matrix(sx, sy, sz)
    Z = shear_matrix(sxy, sxz, syz)
    return T @ (Rx @ Ry @ Rz) @ Z @ S


def landmark_transform(src_float_image,dst_float_image,src_landmarks: np.array,dst_landmarks: np.array):
    tform = tf.estimate_transform('similarity', src_landmarks, dst_landmarks)
    # Assuming `src_image` is the image you want to transform
    # You may also want to specify output shape = destination image shape
    

    return warp(src_float_image, inverse_map=tform.inverse, output_shape=dst_float_image.shape),tform

def compute_alignment_error(src_landmarks, dst_landmarks, tform):
    # Convert to NumPy arrays if needed
    src_landmarks = np.asarray(src_landmarks)
    dst_landmarks = np.asarray(dst_landmarks)
    
    # Before registration: SSD between original points
    F_before = np.sum((src_landmarks - dst_landmarks) ** 2)

    # After registration: transform src landmarks to align with dst
    transformed_src = tform(src_landmarks)
    F_after = np.sum((transformed_src - dst_landmarks) ** 2)

    return F_before, F_after


def apply_affine_to_image_np(image_np, A):
    # Get image shape
    z, y, x = image_np.shape

    # Create a grid of coordinates
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones_like(xx.ravel())], axis=0)  # shape (4, N)

    # Apply inverse of A (because we're pulling values from input to output)
    A_inv = np.linalg.inv(A)
    new_coords = A_inv @ coords
    x_new, y_new, z_new = new_coords[:3]

    # Interpolate using map_coordinates
    warped = map_coordinates(image_np, [z_new, y_new, x_new], order=1, mode='constant', cval=0.0)
    return warped.reshape(image_np.shape)