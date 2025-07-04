# utils script
# utils/pose_graph_optimization.py

import gtsam
import numpy as np
from gtsam import (Values, Pose3, NonlinearFactorGraph, noiseModel, BetweenFactorPose3, PriorFactorPose3)
from scipy.spatial.transform import Rotation as R
from utils.math_utils import get_smooth_max_envelope, get_smooth_min_envelope, matrix_euclidean_distance


def pose_to_gtsam(pose):
    """
    Convert 4x4 numpy transformation matrix to GTSAM Pose3.

    Args:
        pose (np.ndarray): 4x4 transformation matrix.

    Returns:
        gtsam.Pose3: Equivalent Pose3 object.
    """
    rot = R.from_matrix(pose[:3, :3])
    trans = pose[:3, 3]
    return Pose3(gtsam.Rot3(rot.as_matrix()), gtsam.Point3(*trans))


def gtsam_to_matrix(pose):
    """
    Convert GTSAM Pose3 to 4x4 numpy transformation matrix.

    Args:
        pose (gtsam.Pose3): Pose3 object.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


def optimize_pose_graph(slam_transforms, reg_transforms, accepted_indices):
    """
    Fuse SLAM trajectory with registration anchors using GTSAM pose graph optimization.

    Args:
        slam_transforms (list[np.ndarray]): World-aligned SLAM poses.
        reg_transforms (list[np.ndarray]): Registration poses.
        accepted_indices (list[int]): Indices where registration is trusted.

    Returns:
        list[np.ndarray]: Optimized trajectory as 4x4 matrices.
    """
    graph = NonlinearFactorGraph()
    initial = Values()

    # Noise models
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    reg_noise = noiseModel.Diagonal.Sigmas(np.array([0.01]*6))
    slam_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))

    # Add prior at first pose
    initial.insert(0, pose_to_gtsam(slam_transforms[0]))
    graph.add(PriorFactorPose3(0, pose_to_gtsam(slam_transforms[0]), prior_noise))

    # Add SLAM chain
    for i in range(1, len(slam_transforms)):
        T_prev = np.linalg.inv(slam_transforms[i-1]) @ slam_transforms[i]
        graph.add(BetweenFactorPose3(i-1, i, pose_to_gtsam(T_prev), slam_noise))
        initial.insert(i, pose_to_gtsam(slam_transforms[i]))

    # Add registration constraints
    for idx in accepted_indices:
        graph.add(PriorFactorPose3(idx, pose_to_gtsam(reg_transforms[idx]), reg_noise))

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    optimized = [gtsam_to_matrix(result.atPose3(i)) for i in range(len(slam_transforms))]
    return optimized


# Unused as of right now... We believe the idea of final smoothing is valid, but
# in practice it increases overall error in the fused trajectory.
def robust_smooth_matrix_list(pose_list, window_size=5):
    """
    Smooth a list of SE(3) transformation matrices using robust median filtering
    on translation and rotation averaging.

    Args:
        pose_list (list[np.ndarray]): List of 4x4 transformation matrices.
        window_size (int): Number of poses to consider in each local window (must be odd).

    Returns:
        list[np.ndarray]: Smoothed list of poses.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    half_w = window_size // 2
    n = len(pose_list)
    smoothed = []

    for i in range(n):
        # Define window bounds
        i_start = max(0, i - half_w)
        i_end = min(n, i + half_w + 1)
        window = pose_list[i_start:i_end]

        # Extract translations and rotations
        translations = np.array([T[:3, 3] for T in window])
        rotations = R.from_matrix([T[:3, :3] for T in window])

        # Median translation
        median_translation = np.median(translations, axis=0)

        # Rotation averaging via mean quaternion (better than SLERP between 2)
        mean_rotation = rotations.mean().as_matrix()

        # Reconstruct smoothed transform
        T_smoothed = np.eye(4)
        T_smoothed[:3, :3] = mean_rotation
        T_smoothed[:3, 3] = median_translation
        smoothed.append(T_smoothed)

    return smoothed


def combine_fitness_rmse_acceptance(
    fitness,
    rmse,
    fitness_margin=0.04,
    rmse_margin=0.03,
    weight_fitness=1.0,
    weight_rmse=1.0
):
    """
    Combine fitness and RMSE acceptance criteria using smooth envelopes.

    Args:
        fitness (list[float]): List of registration fitness scores.
        rmse (list[float]): List of inlier RMSE values.
        fitness_margin (float): Margin below the fitness envelope for acceptance.
        rmse_margin (float): Margin above the RMSE envelope for acceptance.
        weight_fitness (float): Weight to scale fitness margin (looser or stricter).
        weight_rmse (float): Weight to scale RMSE margin (looser or stricter).

    Returns:
        tuple[list[int], np.ndarray, np.ndarray]:
            - List of accepted frame indices,
            - Smoothed fitness envelope,
            - Smoothed RMSE envelope.
    """
    fitness_env, fitness_ok = get_smooth_max_envelope(fitness, margin=fitness_margin * weight_fitness)
    rmse_env, rmse_ok = get_smooth_min_envelope(rmse, margin=rmse_margin * weight_rmse)
    accepted = sorted(set(fitness_ok).intersection(set(rmse_ok)))
    return accepted, fitness_env, rmse_env


def remove_outliers_accepted_reg_indices(accepted_indices, reg_transforms, max_speed_mps=22, fps=20, max_offset_m=0.5):
    """
    Given a list of accepted indices and a registration trajectory as a list of 
    registration transforms, remove all accepted indices that correspond to a
    transform with a jump greater than max_offset_m from the previous accepted 
    transform.

    Example:
        - max speed := 200km/h ~ 55m/s
        - frame rate/sensor capture frequency := 20Hz
        -> maximum possible offset in consecutive frames = 55m/s / 20Hz = 2.75m

        Thus, 2.75m is the maximum offset allowed for consecutive frames.

        If we have, for example, one accepted frame, 2 non-accepted, and then another accepted, that's a gap of 3 frames, and thus the max offset should be 3 * 2.75m = 8.25m.
    
    Args:
        accepted_indices (list[int]): List of indices to check.
        reg_transforms (list[np.ndarray]): List of registration transforms as 4x4 matrices.
        max_speed_mps (float): Maximum speed of the tracked object in meters per second.
        fps (int): Frames per second of the registration sensor.
        max_offset_m (float): Maximum allowed offset in meters. A consecutive accepted frame with a distance discrepancy higher than this from the last frame is ignored no matter what.
    Returns:
        list[int]: Filtered list of accepted indices.
    """

    max_consecutive_offset = (max_speed_mps / fps)

    filtered_indices = []
    prev_idx = accepted_indices[0]
    prev_transform = reg_transforms[prev_idx]

    for idx in accepted_indices:
        current_transform = reg_transforms[idx]
        offset = matrix_euclidean_distance(prev_transform, current_transform)

        frames_in_between = idx - prev_idx

        allowed_offset = frames_in_between * max_consecutive_offset

        if offset <= max_offset_m and offset <= allowed_offset:
            # Note: first accepted frame is always added
            filtered_indices.append(idx)
            prev_transform = current_transform

        prev_transform = current_transform
        prev_idx = idx
    return filtered_indices
