import open3d as o3d
import numpy as np
from utils.sensor_data_merger import SensorDataMerger
from utils.merge_plys import load_point_cloud, transform_point_cloud

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.5  # More lenient threshold
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 10000))
    return result

def refine_registration(source, target, initial_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# Using clouds centered at origin: ---------------------------------------------
# # Input file paths
# file_1 = "output_v1.log/sensor_captures_v3_cropped/ego_lidar/lidar_frames/10969.ply"
# file_2 = "output_v1.log/sensor_captures_v3_cropped/infrastruct_lidar/lidar_frames/10969.ply"

# # Load point clouds
# source = o3d.io.read_point_cloud(file_1)
# target = o3d.io.read_point_cloud(file_2)

# SANITY CHECK: Use already transformed point clouds ---------------------------

base_directory = "output_v1.log/sensor_captures_v3"
sensors = ["ego_lidar", "infrastruct_lidar"]
max_discrepancy = 0.2  # seconds

# Initialize the data manager
merger = SensorDataMerger(base_dir=base_directory, sensors=sensors, max_timestamp_discrepancy=max_discrepancy)

merger.get_relative_match_for_ego_index(57)

source_data = merger.get_relative_match_for_ego_index(57)[0]
target_data = merger.get_relative_match_for_ego_index(57)[1]

source_file, _, source_pose = source_data
target_file, _, target_pose = target_data

# Load point clouds from the matched files
source_pcd = load_point_cloud(source_file)
target_pcd = load_point_cloud(target_file)

# Transform the source point cloud using the provided pose
source = transform_point_cloud(source_pcd, source_pose)
target = transform_point_cloud(target_pcd, target_pose)


# Using clouds with artificial pose drift --------------------------------------
# Here we load ground truth poses, then shift them to simulate pose drift
# Worst-case GPS perpendicular drifts are most common with a mean of 4.4 and a 
# standard deviation of 3.6 (Source: 
# https://www.sciencedirect.com/science/article/pii/S2214367X15300120)
# NOTE: Not applying rotation shift, only translation shift for simplicity

# base_directory = "output_v1.log/sensor_captures_v3"
# sensors = ["ego_lidar", "infrastruct_lidar"]
# max_discrepancy = 0.2  # seconds

# # Initialize the data manager
# merger = SensorDataMerger(base_dir=base_directory, sensors=sensors, max_timestamp_discrepancy=max_discrepancy)

# merger.get_relative_match_for_ego_index(57)

# ego_data = merger.get_relative_match_for_ego_index(57)[0]
# infra_data = merger.get_relative_match_for_ego_index(57)[1]

# ego_file, _, ego_pose = ego_data
# infra_file, _, infra_pose = infra_data

# # Load point clouds from the matched files
# source_pcd = load_point_cloud(ego_file)
# target_pcd = load_point_cloud(infra_file)

# # Apply a simulated pose drift to the ego pose (infrastruct location is known)
# random_shift = np.random.normal(loc=0.0, scale=4.4, size=3)  # Mean 0, stddev 4.4
# ego_pose_shifted = tuple(ego_pose[:3] + random_shift) + ego_pose[3:]  # Apply shift to translation part of pose

# print("Original Ego Pose:", ego_pose)
# print("Shifted Ego Pose:", ego_pose_shifted)
# print("Shift :", random_shift)

# # Transform the source point cloud using the provided pose
# source = transform_point_cloud(source_pcd, ego_pose_shifted)
# target = transform_point_cloud(target_pcd, infra_pose)

# ------------------------------------------------------------------------------

# Preprocess and extract features
voxel_size = 1.0  # More aggressive downsampling for stability
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Global registration
ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print("RANSAC result:\n", ransac_result.transformation)

# Estimate normals for full-resolution clouds (required for PointToPlane ICP)
source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

# Refine with ICP
refined_result = refine_registration(source, target, ransac_result.transformation, voxel_size)
print("ICP refined result:\n", refined_result.transformation)

# Apply final transformation
source.transform(refined_result.transformation)

# Visualize result
source.paint_uniform_color([1, 0, 0])  # red
target.paint_uniform_color([0, 1, 0])  # green
o3d.visualization.draw_geometries([source, target])