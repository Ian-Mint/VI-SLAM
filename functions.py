import numba
import numpy as np
import scipy.linalg

expm = scipy.linalg.expm


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


@numba.njit()
def inv_pose(x: np.ndarray):
    out = np.zeros_like(x)
    r = x[:3, :3]
    p = x[:3, 3]

    out[3, 3] = 1.
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


@numba.njit()
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


@numba.njit()
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


@numba.njit()
def hat(x):
    x_hat = np.zeros((4, 4))

    x_hat[:3, -1] = x[:3]
    _hat_map(x[3:], x_hat)
    return x_hat


@numba.njit()
def vee(x_hat):
    x = np.zeros(6)

    x[:3] = x_hat[:3, 3]
    _vee_map(x_hat, x[3:])
    return x


@numba.njit()
def adj_hat(x):
    x_hat = np.zeros((6, 6))

    _hat_map(x[3:], x_hat)
    _hat_map(x[3:], x_hat[3:, 3:])
    _hat_map(x[:3], x_hat[:3, 3:])
    return x_hat


@numba.njit()
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


@numba.njit()
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
