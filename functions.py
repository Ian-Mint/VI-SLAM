from typing import List, Union

import numba
import numpy as np
import scipy.linalg
import scipy.sparse as sparse

__all__ = ['expm', 'homogeneous', 'homo_mul', 'inv_pose', 'hat', 'pi', 'd_pi_dx', 'vee', 'img_to_camera_frame',
           'adj_hat', 'lstsq_broadcast', 'coord_to_cell', 'get_coords', 'expand_dim', 'vector_to_diag', 'pose_to_axis',
           'pose_to_angle', 'o_dot', 'kalman_gain', 'block_to_bsr', 'kalman_gain']

expm = scipy.linalg.expm
logm = scipy.linalg.logm


def o_dot(x):
    out = np.zeros((4, 6, *x.shape[1:]))
    out[:3, :3] = np.eye(3)[..., None]
    _hat_map(-x, out[:3, 3:])
    return out


def pose_to_angle(pose):
    return np.arccos((np.trace(pose[:3, :3], axis1=0, axis2=1) - 1) / 2)


def pose_to_axis(pose):
    out = np.zeros(3)
    out[0] = pose[2, 1] - pose[1, 2]
    out[1] = pose[0, 2] - pose[2, 0]
    out[2] = pose[1, 0] - pose[0, 1]

    angle = pose_to_angle(pose)
    return out / (2 * np.sin(angle))


def homo_mul(mat: np.ndarray, vec: np.ndarray):
    if vec.ndim > 1:
        out = mat[:, :3] @ vec + mat[:, 3][..., None]
    else:
        out = mat[:, :3] @ vec + mat[:, 3]
    return out


def homogeneous(x: np.ndarray) -> np.ndarray:
    """
    Convert to homogeneous coordinates along the first dimension
    
    Args:
        x: numpy array 

    Returns:
        x_
    """
    out = np.ones(shape=(x.shape[0] + 1, *x.shape[1:]), dtype=x.dtype)
    out[:-1] = x[...]
    return out


def inv_pose(x: np.ndarray):
    out = np.zeros_like(x)
    r = x[:3, :3]
    p = x[:3, 3]

    out[3, 3] = 1.
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def _hat_map(x, out):
    """
    The 3x3 hat map

    Args:
        x: 3-vector
        out: 3x3 array

    Returns:
        3x3 x^
    """
    out[0, 1] = -x[2]
    out[0, 2] = x[1]
    out[1, 0] = x[2]
    out[1, 2] = -x[0]
    out[2, 0] = -x[1]
    out[2, 1] = x[0]


def _vee_map(x, out):
    """
    The 3x3 vee map
    Args:
        out: 3-vector
        x: 3x3 array

    Returns:
        Inverse of the 3x3 hat map
    """
    out[0] = x[2, 1]
    out[1] = x[0, 2]
    out[2] = x[1, 0]


def hat(x):
    x_hat = np.zeros((4, 4))

    x_hat[:3, -1] = x[:3]
    _hat_map(x[3:], x_hat)
    return x_hat


def vee(x_hat):
    x = np.zeros(6)

    x[:3] = x_hat[:3, 3]
    _vee_map(x_hat, x[3:])
    return x


def adj_hat(x):
    x_hat = np.zeros((6, 6))

    _hat_map(x[3:], x_hat)
    _hat_map(x[3:], x_hat[3:, 3:])
    _hat_map(x[:3], x_hat[:3, 3:])
    return x_hat


def coord_to_cell(point: np.ndarray, minima: np.ndarray, resolution: float) -> np.ndarray:
    """
    Discretize according to the map layout.
    Convert from xy (m) coordinates to ij (px) coordinates.

    Args:
        minima:
        resolution:
        point: a point or set of points in meters

    Returns:
        point or set of points in coordinate indices
    """
    return np.ceil((point - minima) / resolution).astype(np.int16) - 1


def get_coords(disparity) -> np.ndarray:
    """
    Return xyz coordinates in the camera frame associated with each pixel

    Args:
        disparity: the disparity image

    Returns:
        h*w x 3 array
    """
    b = 475.143600050775 / 1000
    fsu = 7.7537235550066748e+02
    cu = 6.1947309112548828e+02
    cv = 2.5718049049377441e+02

    height, width = disparity.shape

    u = np.arange(width)
    v = np.arange(height)

    z = fsu * b / disparity
    y = np.expand_dims((v - cv) / fsu, axis=1) * z
    x = np.expand_dims((u - cu) / fsu, axis=0) * z
    return np.stack([x, y, z], axis=2)


def img_to_camera_frame(observation: np.ndarray, fsu, fsv, cu, cv, b):
    """
    convert stereo pixel observation coordinates to the camera frame

    Args:
        observation: (xL, yL, xR, yR)
        fsu:
        fsv:
        cu:
        cv:
        b:

    Returns:
        (x, y, z) in camera frame of reference
    """
    if observation.ndim == 2:
        out = np.zeros((3, observation.shape[1]))
    elif observation.ndim == 1:
        out = np.zeros(3)
    else:
        raise RuntimeError("Observations can only be 1D or 2D")
    out[2] = fsu * b / (observation[0] - observation[2])
    out[1] = out[2] * (observation[1] - cv) / fsv
    out[0] = out[2] * (observation[0] - cu) / fsu
    return out


def pi(x: np.ndarray):
    return x / x[2]


def d_pi_dx(x: np.ndarray):
    if x.ndim > 1:
        out = np.zeros((4, 4, *x.shape[1:]))
    else:
        out = np.zeros((4, 4))
    z_inv = 1 / x[2]

    out[0, 0] = 1.
    out[1, 1] = 1.
    out[3, 3] = 1.
    out[0, 2] = -x[0] * z_inv
    out[1, 2] = -x[1] * z_inv
    out[3, 2] = -x[3] * z_inv
    return z_inv * out


@numba.njit()
def lstsq_broadcast(a, b):
    x = np.zeros_like(b)
    for i in range(len(a)):
        a_mat = a[i]
        b_mat = b[i]
        x_mat, _, _, _ = np.linalg.lstsq(a_mat, b_mat)
        x[i] = x_mat
    return x


def expand_dim(arrays: List[np.ndarray], axis: int) -> List[np.ndarray]:
    """
    Expand all dimensions in arrays along the specified axis

    Args:
        arrays: a list of numpy arrays
        axis: the axis along which to expand

    Returns:
        List of new views of the original arrays with axis expanded
    """
    results = []
    for arr in arrays:
        results.append(np.expand_dims(arr, axis=-1))
    return results


def vector_to_diag(x):
    return np.eye(4) * x.T[..., None]


def block_to_bsr(x: np.ndarray, replicate):
    """
    Replaces vector_to_diag when using bsr sparse matrices

    Args:
        replicate: number of times to replicate x along the diagonal of the output
        x: numpy array vector

    Returns:
        bsr sparse diagonal array
    """
    tiled = np.stack([x] * replicate, axis=0)
    out = sparse.bsr_matrix((tiled, np.arange(replicate), np.arange(replicate + 1)),
                            blocksize=(4, 4),
                            shape=(4 * replicate, 4 * replicate))
    return out


hint = Union[sparse.bsr_matrix, np.ndarray]


def kalman_gain(cv: hint, h: hint, noise: hint) -> np.ndarray:
    b = cv @ h.T
    a = h @ b + noise
    if isinstance(a, sparse.bsr_matrix):
        kt, *_ = scipy.linalg.lstsq(a.T.toarray(), b.T.toarray(), overwrite_a=True, overwrite_b=True)
    else:
        kt, *_ = scipy.linalg.lstsq(a.T, b.T, overwrite_a=True, overwrite_b=True)
    return kt.T
