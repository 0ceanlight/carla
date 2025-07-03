import math
import numpy as np
from typing import List, Tuple, Optional
from .math_utils import quaternion_inverse, quaternion_multiply
from .tum_file_parser import tum_load_as_tuples


def quaternion_angle_difference(q1, q2):
    """
    Computes the absolute angular difference in degrees between two unit quaternions.

    Args:
        q1 (tuple or list): The first quaternion (x, y, z, w).
        q2 (tuple or list): The second quaternion (x, y, z, w).

    Returns:
        float: The absolute angular difference in degrees between q1 and q2.
    """
    # Normalize the quaternions to ensure they are unit quaternions
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Compute the dot product
    dot = np.dot(q1, q2)

    # Clamp the dot product to [-1, 1] to avoid numerical errors with arccos
    dot = np.clip(dot, -1.0, 1.0)

    # Compute the angular difference
    angle_rad = 2 * np.arccos(abs(dot))  # abs(dot) handles the double-cover of quaternions

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# Takes numpy arrays as input and outpu
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    Args:
        q (np.ndarray): Quaternion represented as (x, y, z, w).

    Returns:
        np.ndarray: Normalized quaternion.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return q / norm


def align_sequences(
    seq1: List[Tuple[float, float, float, float, float, float, float,
                     float]], seq2: List[Tuple[float, float, float, float,
                                               float, float, float, float]]
) -> Tuple[List[Tuple[float, float, float, float, float, float, float, float]],
           List[Tuple[float, float, float, float, float, float, float,
                      float]]]:
    """
    Align two sequences so that their starting positions, orientations, and timestamps are equal.

    Args:
        seq1: First sequence of poses.
        seq2: Second sequence of poses.

    Returns:
        Tuple containing the aligned sequences.
    """
    if not seq1 or not seq2:
        raise ValueError("Input sequences must not be empty.")

    # Extract starting poses
    t1_0, x1_0, y1_0, z1_0, qx1_0, qy1_0, qz1_0, qw1_0 = seq1[0]
    t2_0, x2_0, y2_0, z2_0, qx2_0, qy2_0, qz2_0, qw2_0 = seq2[0]

    # Compute translation and time offsets
    dx = x2_0 - x1_0
    dy = y2_0 - y1_0
    dz = z2_0 - z1_0
    dt = t2_0 - t1_0

    # Compute rotation offset
    q1_0 = (qx1_0, qy1_0, qz1_0, qw1_0)
    q2_0 = (qx2_0, qy2_0, qz2_0, qw2_0)
    q1_0_inv = quaternion_inverse(q1_0)
    q_rot = quaternion_multiply(q1_0_inv, q2_0)
    q_rot = normalize_quaternion(q_rot)

    # Align sequences
    aligned_seq1 = []
    for t, x, y, z, qx, qy, qz, qw in seq1:
        aligned_seq1.append((t, x, y, z, qx, qy, qz, qw))

    aligned_seq2 = []
    for t, x, y, z, qx, qy, qz, qw in seq2:
        # Adjust position
        x_adj = x - dx
        y_adj = y - dy
        z_adj = z - dz

        # Adjust orientation
        q = (qx, qy, qz, qw)
        q_adj = quaternion_multiply(quaternion_inverse(q_rot), q)
        q_adj = normalize_quaternion(q_adj)

        # Adjust timestamp
        t_adj = t - dt

        aligned_seq2.append((t_adj, x_adj, y_adj, z_adj, *q_adj))

    return aligned_seq1, aligned_seq2


def find_closest_entry(
    timestamp: float, sequence: List[Tuple[float, float, float, float, float,
                                           float, float, float]]
) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    """
    Find the entry in the sequence with the closest timestamp to the given timestamp.

    Args:
        timestamp (float): Target timestamp.
        sequence: Sequence of poses.

    Returns:
        The closest pose entry or None if the sequence is empty.
    """
    if not sequence:
        return None
    return min(sequence, key=lambda entry: abs(entry[0] - timestamp))


def compute_differences(
    seq1: List[Tuple[float, float, float, float, float, float, float, float]], 
    seq2: List[Tuple[float, float, float, float, float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """
    Compute per-entry differences between two aligned sequences.

    Args:
        seq1: First aligned sequence.
        seq2: Second aligned sequence.

    Returns:
        List of tuples containing:
            - Timestamp from seq1
            - Time difference between matched entries
            - Translation difference (Euclidean norm)
            - Rotation difference (angle in degrees)
    """
    differences = []
    for entry1 in seq1:
        t1, x1, y1, z1, qx1, qy1, qz1, qw1 = entry1
        closest_entry2 = find_closest_entry(t1, seq2)
        if closest_entry2 is None:
            continue
        t2, x2, y2, z2, qx2, qy2, qz2, qw2 = closest_entry2

        # Time difference
        time_diff = t2 - t1

        # Translation difference
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        trans_diff = math.sqrt(dx**2 + dy**2 + dz**2)

        # Rotation difference
        q1 = (qx1, qy1, qz1, qw1)
        q2 = (qx2, qy2, qz2, qw2)

        # This can be replaced with...
        # q_rel = quaternion_multiply(quaternion_inverse(q1), q2)
        # q_rel = normalize_quaternion(q_rel)
        # rot_diff_deg = quaternion_to_angle_degrees(q_rel)
        # ...this (once it works!!)
        rot_diff_deg = quaternion_angle_difference(q1, q2)

        differences.append((t1, time_diff, trans_diff, rot_diff_deg))
    return differences


def compute_average_difference(
    differences: List[Tuple[float, float, float,
                            float]]) -> Tuple[float, float]:
    """
    Compute average translation and rotation differences.

    Args:
        differences: List of per-entry differences.

    Returns:
        Tuple containing:
            - Average translation difference
            - Average rotation difference in degrees
    """
    if not differences:
        return (0.0, 0.0)
    total_trans = sum(diff[2] for diff in differences)
    total_rot = sum(diff[3] for diff in differences)
    count = len(differences)
    return (total_trans / count, total_rot / count)


def get_all_differences(file1: str, file2: str, align=True, start: Optional[int] = None, end: Optional[int] = None) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Display per-entry differences between two TUM trajectory files.

    Args:
        file1 (str): Path to the first TUM file.
        file2 (str): Path to the second TUM file.
        align (bool): Whether to align the sequences before computing differences.
        start (Optional[int]): Start index for cropping the differences.
        end (Optional[int]): End index for cropping the differences.

    Returns:

        Tuple containing lists of:
            - Timestamps
            - Time differences
            - Translation differences (meters)
            - Rotation differences (degrees)
    """
    seq1 = tum_load_as_tuples(file1)
    seq2 = tum_load_as_tuples(file2)
    differences = None
    if align:
        aligned_seq1, aligned_seq2 = align_sequences(seq1, seq2)
        differences = compute_differences(aligned_seq1, aligned_seq2)
    else:
        # If not aligning, compute differences directly
        differences = compute_differences(seq1, seq2)
    # Crop the differences array if start and end are specified
    # This allows for partial analysis of the data
    if end is not None:
        # end first bcz otherwise it can be offset by the start :)
        differences = differences[:end]
    if start is not None:
        differences = differences[start:]

    ret = ([], [], [], [])
    for t, time_diff, trans_diff, rot_diff in differences:
        ret[0].append(t)
        ret[1].append(time_diff)
        ret[2].append(trans_diff)
        ret[3].append(rot_diff)
    return ret


def get_average_difference(file1: str, file2: str, align=True, start: Optional[int] = None, end: Optional[int] = None) -> Tuple[float, float]:
    """
    Display average translation and rotation differences between two TUM trajectory files.

    Args:
        file1 (str): Path to the first TUM file.
        file2 (str): Path to the second TUM file.
        align (bool): Whether to align the sequences before computing differences.
        start (Optional[int]): Start index for cropping the differences.
        end (Optional[int]): End index for cropping the differences.

    Returns:
        Tuple[float, float]: Average translation difference (meters) and average rotation difference (degrees).
    """
    seq1 = tum_load_as_tuples(file1)
    seq2 = tum_load_as_tuples(file2)
    differences = None
    if align:
        aligned_seq1, aligned_seq2 = align_sequences(seq1, seq2)
        differences = compute_differences(aligned_seq1, aligned_seq2)
    else:
        # If not aligning, compute differences directly
        differences = compute_differences(seq1, seq2)
    # Crop the differences array if start and end are specified
    # This allows for partial analysis of the data
    if end is not None:
        # end first bcz otherwise it can be offset by the start :)
        differences = differences[:end]
    if start is not None:
        differences = differences[start:]
    avg_trans_diff, avg_rot_diff = compute_average_difference(differences)
    print(f"Average Translation Difference (meters): {avg_trans_diff:.6f}m")
    print(f"Average Rotation Difference (degrees): {avg_rot_diff:.6f}°")


def show_all_differences(file1: str, file2: str, align=True, start: Optional[int] = None, end: Optional[int] = None) -> None:
    """
    Display per-entry differences between two TUM trajectory files.

    Args:
        file1 (str): Path to the first TUM file.
        file2 (str): Path to the second TUM file.
        align (bool): Whether to align the sequences before computing differences.
        start (Optional[int]): Start index for cropping the differences.
        end (Optional[int]): End index for cropping the differences.
    """
    seq1 = tum_load_as_tuples(file1)
    seq2 = tum_load_as_tuples(file2)
    differences = None
    if align:
        aligned_seq1, aligned_seq2 = align_sequences(seq1, seq2)
        differences = compute_differences(aligned_seq1, aligned_seq2)
    else:
        # If not aligning, compute differences directly
        differences = compute_differences(seq1, seq2)
    # Crop the differences array if start and end are specified
    # This allows for partial analysis of the data
    if end is not None:
        # end first bcz otherwise it can be offset by the start :)
        differences = differences[:end]
    if start is not None:
        differences = differences[start:]
    print("Timestamp\tTimeDiff\tTransDiff\tRotDiff(deg)")
    for t, time_diff, trans_diff, rot_diff in differences:
        print(f"{t:.6f}\t{time_diff:.6f}\t{trans_diff:.6f}\t{rot_diff:.6f}")


def show_average_difference(file1: str, file2: str, align=True, start: Optional[int] = None, end: Optional[int] = None) -> None:
    """
    Display average translation and rotation differences between two TUM trajectory files.

    Args:
        file1 (str): Path to the first TUM file.
        file2 (str): Path to the second TUM file.
        align (bool): Whether to align the sequences before computing differences.
        start (Optional[int]): Start index for cropping the differences.
        end (Optional[int]): End index for cropping the differences.
    """
    seq1 = tum_load_as_tuples(file1)
    seq2 = tum_load_as_tuples(file2)
    differences = None
    if align:
        aligned_seq1, aligned_seq2 = align_sequences(seq1, seq2)
        differences = compute_differences(aligned_seq1, aligned_seq2)
    else:
        # If not aligning, compute differences directly
        differences = compute_differences(seq1, seq2)
    # Crop the differences array if start and end are specified
    # This allows for partial analysis of the data
    if end is not None:
        # end first bcz otherwise it can be offset by the start :)
        differences = differences[:end]
    if start is not None:
        differences = differences[start:]
    avg_trans_diff, avg_rot_diff = compute_average_difference(differences)
    print(f"Average Translation Difference (meters): {avg_trans_diff:.6f}m")
    print(f"Average Rotation Difference (degrees): {avg_rot_diff:.6f}°")
