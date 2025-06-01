import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_inverse(q):
    """Returns the inverse of a quaternion."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_multiply(q1, q2):
    """Multiplies two quaternions q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    :param roll: Roll angle in degrees
    :param pitch: Pitch angle in degrees
    :param yaw: Yaw angle in degrees
    :return: Quaternion as a tuple (qx, qy, qz, qw)
    """

    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return (qx, qy, qz, qw)


def pose_to_matrix(
    pose: tuple[float, float, float, float, float, float, float]
) -> np.ndarray:
    """
    Convert a pose (translation + quaternion) to a 4x4 transformation matrix.

    Parameters:
        pose: Tuple (x, y, z, qx, qy, qz, qw)

    Returns:
        4x4 transformation matrix as np.ndarray
    """
    x, y, z, qx, qy, qz, qw = pose
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose(
        T: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4x4 transformation matrix to pose (translation + quaternion).

    Parameters:
        T: 4x4 transformation matrix

    Returns:
        Tuple (x, y, z, qx, qy, qz, qw)
    """
    assert T.shape == (4, 4), "Input must be a 4x4 transformation matrix."
    translation = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()  # returns (qx, qy, qz, qw)
    return (*translation, *quat)


def pose_difference(
    T1: np.ndarray, T2: np.ndarray
) -> tuple[float, float]:
    """
    Compute the translational and rotational difference between two 4x4 poses.

    Parameters:
        T1, T2: Two 4x4 transformation matrices

    Returns:
        Tuple (translation_distance, rotation_angle_degrees)
    """
    assert T1.shape == (4, 4) and T2.shape == (4, 4), "Both inputs must be 4x4 matrices."

    # Translation difference
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    translation_diff = np.linalg.norm(t1 - t2)

    # Rotation difference
    R1 = R.from_matrix(T1[:3, :3])
    R2 = R.from_matrix(T2[:3, :3])
    relative_rot = R1.inv() * R2
    rotation_diff_deg = relative_rot.magnitude() * 180.0 / np.pi

    return translation_diff, rotation_diff_deg