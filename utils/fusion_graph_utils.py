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


