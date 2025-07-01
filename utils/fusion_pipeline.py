import numpy as np
from collections import defaultdict
from utils.tum_file_parser import tum_load_as_tuples, tum_save_tuples
from utils.tum_file_comparator import find_closest_entry
from utils.math_utils import pose_to_matrix, pose_difference, pose_to_gtsam_pose3, gtsam_pose3_to_pose
from .fusion_graph_utils import fuse_slam_gps_poses, align_pose_list_to_pose, align_slam_to_gps_timestamps


def load_scalar_series(path):
    """
    Load a scalar time series file in the format <timestamp> <value>.

    Args:
        path (str): Path to the file.

    Returns:
        dict[float, float]: Mapping from timestamp to value.
    """
    values = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            t, v = map(float, line.strip().split())
            values[t] = v
    return values



def run_pose_graph_fusion(
    gps_file,
    slam_file,
    reg_file,
    fitness_file,
    rmse_file,
    output_file,
):
    """
    Run the full pipeline to fuse GPS, SLAM, and registration into an optimized pose graph.

    Args:
        gps_file (str): Path to GPS poses in TUM format.
        slam_file (str): Path to SLAM poses in TUM format.
        reg_file (str): Path to registration poses in TUM format.
        fitness_file (str): Path to fitness scores.
        rmse_file (str): Path to RMSE scores.
        groundtruth_file (str): Path to ground truth poses (used for alignment).
        output_file (str): Output path for the optimized trajectory in TUM format.
    """
    gps_poses = tum_load_as_tuples(gps_file)
    slam_poses = tum_load_as_tuples(slam_file)
    reg_poses = tum_load_as_tuples(reg_file)
    fitness_map = load_scalar_series(fitness_file)
    rmse_map = load_scalar_series(rmse_file)

    # Use GPS timestamps as reference
    reference_timestamps = [pose[0] for pose in gps_poses]
    aligned_data = defaultdict(dict)

    # Align SLAM poses to gps
    aligned_slam_poses = align_pose_list_to_pose(slam_poses, gps_poses)

    for t_ref in reference_timestamps:
        gps_pose = find_closest_entry(t_ref, gps_poses)
        slam_pose = find_closest_entry(t_ref, aligned_slam_poses)
        reg_pose = find_closest_entry(t_ref, reg_poses)

        aligned_data[t_ref]['gps'] = gps_pose
        aligned_data[t_ref]['slam'] = slam_pose
        aligned_data[t_ref]['reg'] = reg_pose

        aligned_data[t_ref]['fitness'] = fitness_map.get(reg_pose[0], 0.0) if reg_pose else 0.0
        aligned_data[t_ref]['rmse'] = rmse_map.get(reg_pose[0], 1e3) if reg_pose else 1e3

        aligned_data[t_ref]['use_reg'] = (
            reg_pose and slam_pose and
            is_registration_trustworthy(
                aligned_data[t_ref]['fitness'],
                aligned_data[t_ref]['rmse'],
                slam_pose[1:], reg_pose[1:]
            )
        )

    # Construct and optimize the pose graph
    pose_graph, trajectory = build_pose_graph(aligned_data)
    optimized_trajectory = optimize_pose_graph(pose_graph, trajectory)

    # Save result
    tum_save_tuples(output_file, optimized_trajectory)


def fuse_slam_gps_files(output_file, gps_file, slam_file):
    """
    Fuse SLAM and GPS poses from TUM files into a single output trajectory TUM file.
    """

    gps_tum_poses = tum_load_as_tuples(gps_file)
    slam_tum_poses = tum_load_as_tuples(slam_file)

    # Align SLAM poses to GPS timestamps
    slam_tum_poses_aligned = align_slam_to_gps_timestamps(slam_tum_poses, gps_tum_poses)

    # Convert poses to lists of gtsam Pose3 objects
    gps_gtsam_poses = [pose_to_gtsam_pose3(pose[1:]) for pose in gps_tum_poses]
    slam_gtsam_poses = [pose_to_gtsam_pose3(pose[1:]) for pose in slam_tum_poses_aligned]

    # Fuse the poses using the gtsam pose optimization
    fused_gtsam_poses = fuse_slam_gps_poses(slam_gtsam_poses, gps_gtsam_poses)

    # Convert back to TUM format
    fused_tum_poses = [
        (gps_tum_poses[i][0], *gtsam_pose3_to_pose(pose))
        for i, pose in enumerate(fused_gtsam_poses)
    ]

    # Save the fused trajectory to a TUM file
    tum_save_tuples(output_file, fused_tum_poses)
