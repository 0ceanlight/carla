import math
import numpy as np
import gtsam
from gtsam import Pose3, Rot3, Point3
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
    try:
        x, y, z, qx, qy, qz, qw = pose
    except ValueError:
        print(pose)
        raise ValueError("Pose must be a tuple of (x, y, z, qx, qy, qz, qw)")

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


def calc_offset_margin(transform_arr_1, transform_arr_2, weight=1.0):
    """
    Calculate the absolute offset margin per-pose between two lists of transforms.

    Args:
        transform_arr_1 (list): First list of N transforms (as 4x4 matrices).
        transform_arr_2 (list): Second list of N transforms (as 4x4 matrices).
        weight (float): Weighting factor for the error margin.
    
    Returns:
        list: N float values representing the offset margin for each pose pair.
    """
    if len(transform_arr_1) != len(transform_arr_2):
        raise ValueError("Both transform arrays must have the same length.")
    
    diffs = []

    for i, j in zip(transform_arr_1, transform_arr_2):
        trans, _ = pose_difference(i, j)
        # TODO: consider rotation difference?
        diffs.append(trans * weight)

    return diffs


def pose_to_gtsam_pose3(
    pose: tuple[float, float, float, float, float, float, float]
) -> Pose3:
    """
    Convert a pose (translation + quaternion) to a GTSAM Pose3 object.

    Parameters:
        pose: Tuple (x, y, z, qx, qy, qz, qw)

    Returns:
        GTSAM Pose3 object
    """
    x, y, z, qx, qy, qz, qw = pose
    return Pose3(Rot3.Quaternion(qx, qy, qz, qw), Point3(x, y, z))


def gtsam_pose3_to_pose(
    pose: Pose3
) -> tuple[float, float, float, float, float, float, float]:
    """
    Convert a GTSAM Pose3 object to a pose (translation + quaternion).

    Parameters:
        pose: GTSAM Pose3 object

    Returns:
        Tuple (x, y, z, qx, qy, qz, qw)
    """
    translation = (pose.x(), pose.y(), pose.z())
    # import IPython
    # IPython.embed()
    q = pose.rotation().toQuaternion()  # returns (qx, qy, qz, qw)
    rotation = (q.x(), q.y(), q.z(), q.w())
    return (*translation, *rotation)

# def gtsam_pose3_to_pose(
#     pose: Pose3
# ) -> tuple[float, float, float, float, float, float, float]:
#     """
#     Convert a GTSAM Pose3 object to a pose (translation + quaternion).

#     Parameters:
#         pose: GTSAM Pose3 object

#     Returns:
#         Tuple (x, y, z, qx, qy, qz, qw)
#     """
#     return (
#         pose.x(),
#         pose.y(),
#         pose.z(),
#         pose.rotation().x(),
#         pose.rotation().y(),
#         pose.rotation().z(),
#         pose.rotation().w()
#     )


# def pose_to_gtsam_pose3(
#     pose: tuple[float, float, float, float, float, float, float]
# ) -> Pose3:
#     """
#     Convert a pose (translation + quaternion) to a GTSAM Pose3 object.

#     Parameters:
#         pose: Tuple (x, y, z, qx, qy, qz, qw)

#     Returns:
#         GTSAM Pose3 object
#     """
#     x, y, z, qx, qy, qz, qw = pose
#     rotation = Rot3.Quaternion(qw, qx, qy, qz)
#     translation = Point3(x, y, z)
#     return Pose3(rotation, translation)