import open3d as o3d
import numpy as np
import gtsam
from gtsam import Pose3, Rot3, Point3
from utils.math_utils import pose_to_matrix, matrix_to_pose


def fuse_slam_gps_poses(slam_poses, gps_poses):
    """
    Fuse SLAM and GPS trajectories using GTSAM.
    
    Args:
        slam_poses (List[Pose3]): Relative SLAM poses, starting at origin.
        gps_poses (List[Pose3]): GPS poses in the world frame.
        
    Returns:
        List[Pose3]: Fused trajectory in world frame.
    """
    assert len(slam_poses) == len(gps_poses), "SLAM and GPS must be same length"
    
    N = len(slam_poses)
    
    # Initialize factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Noise models
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0]*3 + [0.5]*3))  # GPS prior: loose
    slam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*3 + [0.05]*3))  # SLAM relative motion: tighter
    
    # Add GPS priors (soft constraints)
    for i in range(N):
        graph.add(gtsam.PriorFactorPose3(i, gps_poses[i], gps_noise))
    
    # Add relative SLAM constraints as BetweenFactors
    for i in range(N - 1):
        T_i = slam_poses[i]
        T_j = slam_poses[i + 1]
        T_ij = T_i.between(T_j)
        graph.add(gtsam.BetweenFactorPose3(i, i + 1, T_ij, slam_noise))
    
    # Initial guess: align SLAM to world using first GPS pose
    T_align = gps_poses[0].compose(slam_poses[0].inverse())
    
    for i in range(N):
        aligned_pose = T_align.compose(slam_poses[i])
        initial.insert(i, aligned_pose)
    
    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    
    # Extract fused poses
    fused_poses = [result.atPose3(i) for i in range(N)]
    return fused_poses


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
