import math
import numpy as np
from gtsam import Pose3, Rot3, Point3
from scipy.signal import savgol_filter
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


def matrix_euclidean_distance(T2: np.ndarray, T1: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two 4x4 transformation matrices.

    Args:
        T2 (np.ndarray): Second transformation matrix (4x4).
        T1 (np.ndarray): First transformation matrix (4x4).

    Returns:
        float: Euclidean distance between the translations of T1 and T2.
    """
    assert T1.shape == (4, 4) and T2.shape == (4, 4), "Both inputs must be 4x4 matrices."
    translation_diff = T2[:3, 3] - T1[:3, 3]
    return np.linalg.norm(translation_diff)


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


def align_matrix_list_to_matrix(matrix_list, matrix):
    """
    Align a trajectory (list of matrices) to start at a given single transform (matrix).

    Args:
        matrix_list (list): Trajectory to align as list of 4x4 np.ndarrays.
        matrix (np.ndarray): Matrix to align the list to.
    """
    T_list0 = matrix_list[0]
    T_start0 = matrix
    T_align = T_start0 @ np.linalg.inv(T_list0)

    aligned_matrices = [T_align @ T for T in matrix_list]
    return aligned_matrices


def align_matrix_list_to_matrix_rotation_only(matrix_list, matrix):
    """
    Align the rotations of entries in a trajectory (list of matrices) so they start
    with the same rotation as the given reference matrix. This does not affect 
    the translation of each matrix.

    Args:
        matrix_list (list): Trajectory to align, as list of 4x4 np.ndarrays.
        matrix (np.ndarray): Reference 4x4 transformation matrix to align rotations to.
    """
    assert isinstance(matrix_list, list) and all(m.shape == (4, 4) for m in matrix_list), \
        "matrix_list must be a list of 4x4 matrices."
    assert matrix.shape == (4, 4), "matrix must be a 4x4 matrix."

    # Get rotation part of reference and first trajectory matrix
    R_target = matrix[:3, :3]
    R_first = matrix_list[0][:3, :3]

    # Compute rotation that brings R_first to R_target
    R_align = R_target @ R_first.T  # Equivalent to R_target * inverse(R_first)

    aligned_list = []
    for mat in matrix_list:
        aligned = np.eye(4)
        aligned[:3, :3] = R_align @ mat[:3, :3]
        aligned[:3, 3] = mat[:3, 3]  # Preserve translation
        aligned_list.append(aligned)

    return aligned_list


def align_pose_list_to_pose(pose_list, pose):
    """ Align trajectory (list of poses) to start at a given single transform (pose).
    
    Args:
        poses (list): List of poses as (x, y, z, qx, qy, qz, qw).
        pose (tuple): Pose as (x, y, z, qx, qy, qz, qw) to align to.
    Returns:
        list: Aligned poses in the same format.
    """
    from utils.math_utils import pose_to_matrix, matrix_to_pose
    matrix = pose_to_matrix(pose)
    matrix_list = [pose_to_matrix(p) for p in pose_list]
    aligned_matrices = align_matrix_list_to_matrix(matrix_list, matrix)
    return [matrix_to_pose(m) for m in aligned_matrices]


def align_slam_to_gps_timestamps(slam_poses, gps_poses):
    """
    Align SLAM poses to GPS timestamps.

    Args:
        slam_poses (list): List of SLAM poses in TUM format.
        gps_poses (list): List of GPS poses in TUM format.

    Returns:
        list: Aligned SLAM poses with GPS timestamps.
    """
    from utils.tum_file_comparator import find_closest_entry

    aligned_slam = []
    for gps_pose in gps_poses:
        closest_slam = find_closest_entry(gps_pose[0], slam_poses)
        if closest_slam:
            aligned_slam.append((gps_pose[0], *closest_slam[1:]))

    return aligned_slam


def get_smooth_max_envelope(y, window=15, margin=0.05):
    """
    Compute a smooth upper envelope of the input signal using a sliding max and Savitzky–Golay filter.

    Args:
        y (list[float]): Input signal.
        window (int): Window size for local maximum and smoothing.
        margin (float): Margin to determine acceptance below the envelope.

    Returns:
        tuple[np.ndarray, list[int]]: Smoothed envelope and list of accepted indices where value >= envelope - margin.
    """
    y = np.array(y)
    local_max = np.array([
        np.max(y[max(0, i - window // 2):min(len(y), i + window // 2 + 1)])
        for i in range(len(y))
    ])
    smooth_env = savgol_filter(local_max, window_length=15, polyorder=2)
    accepted = [i for i in range(len(y)) if y[i] >= smooth_env[i] - margin]
    return smooth_env, accepted


def get_smooth_min_envelope(y, window=15, margin=0.05):
    """
    Compute a smooth lower envelope of the input signal using a sliding min and Savitzky–Golay filter.

    Args:
        y (list[float]): Input signal.
        window (int): Window size for local minimum and smoothing.
        margin (float): Margin to determine acceptance above the envelope.

    Returns:
        tuple[np.ndarray, list[int]]: Smoothed envelope and list of accepted indices where value <= envelope + margin.
    """
    y = np.array(y)
    local_min = np.array([
        np.min(y[max(0, i - window // 2):min(len(y), i + window // 2 + 1)])
        for i in range(len(y))
    ])
    smooth_env = savgol_filter(local_min, window_length=15, polyorder=2)
    accepted = [i for i in range(len(y)) if y[i] <= smooth_env[i] + margin]
    return smooth_env, accepted


def compute_umeyama_transform(src_points, dst_points):
    """
    Computes the rigid transformation T that maps src_points to dst_points
    using the Umeyama method (least-squares rigid alignment).

    Args:
        src_points (np.ndarray): Nx3 array of source points (e.g. SLAM)
        dst_points (np.ndarray): Nx3 array of destination points (e.g. Registration)

    Returns:
        np.ndarray: 4x4 rigid transformation matrix
    """
    assert src_points.shape == dst_points.shape

    mu_src = np.mean(src_points, axis=0)
    mu_dst = np.mean(dst_points, axis=0)

    src_centered = src_points - mu_src
    dst_centered = dst_points - mu_dst

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_dst - R @ mu_src

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
