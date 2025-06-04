import logging
import numpy as np
import open3d as o3d
from utils.lidar_viewer import PointCloudViewer
from utils.registration import *
from utils.sensor_data_merger import SensorDataMerger
from utils.math_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Using clouds centered at origin: ---------------------------------------------
# # Input file paths
# file_1 = "output_v1.log/sensor_captures_v3_cropped/ego_lidar/lidar_frames/10969.ply"
# file_2 = "output_v1.log/sensor_captures_v3_cropped/infrastruct_lidar/lidar_frames/10969.ply"

# # Load point clouds
# source = o3d.io.read_point_cloud(file_1)
# target = o3d.io.read_point_cloud(file_2)

# SANITY CHECK: Use already transformed point clouds ---------------------------

# base_directory = "output_v1.log/sensor_captures_v3"
# sensors = ["ego_lidar", "infrastruct_lidar"]
# max_discrepancy = 0.2  # seconds

# # Initialize the data manager
# merger = SensorDataMerger(base_dir=base_directory, sensors=sensors, max_timestamp_discrepancy=max_discrepancy)

# merger.get_relative_match_for_ego_index(57)

# source_data = merger.get_relative_match_for_ego_index(57)[0]
# target_data = merger.get_relative_match_for_ego_index(57)[1]

# source_file, _, source_pose = source_data
# target_file, _, target_pose = target_data

# register_and_view_multiple_point_clouds([(source_file, source_pose), (target_file, target_pose)])

# Using clouds with artificial pose drift --------------------------------------
# Here we load ground truth poses, then shift them to simulate pose drift
# Worst-case GPS perpendicular drifts are most common with a mean of 4.4 and a 
# standard deviation of 3.6 (Source: 
# https://www.sciencedirect.com/science/article/pii/S2214367X15300120)
# NOTE: Not applying rotation shift, only translation shift for simplicity

# base_directory = "output_v1.log/sensor_captures_v3_cropped"
base_directory = "output_v1.log/sensor_captures_v3"
sensors = ["ego_lidar", "infrastruct_lidar"]
max_discrepancy = 0.2  # seconds

# Initialize the data manager
merger = SensorDataMerger(base_dir=base_directory, sensors=sensors, max_timestamp_discrepancy=max_discrepancy)

index = 57 # Example index for testing

merger.get_relative_match_for_ego_index(index)

ego_data = merger.get_relative_match_for_ego_index(index)[0]
infra_data = merger.get_relative_match_for_ego_index(index)[1]

ego_file, _, ego_pose = ego_data
infra_file, _, infra_pose = infra_data

ego_color = (255, 0, 0)  # Red color for ego point cloud
infra_color = (0, 255, 0)  # Green color for infrastructure point cloud

T_ground_truth = [
    pose_to_matrix(ego_pose),           # Ego pose
    pose_to_matrix(infra_pose)          # Infrastructure pose remains unchanged
]

# SHIFT THE POSES --------------------------------------------------------------
# Apply a simulated pose drift to the ego pose (infrastruct location is known)
mean, stddev = 0.0, 4.4  # Mean 0, stddev 4.4 for translation shift
random_shift = np.random.normal(loc=mean, scale=stddev, size=3)
ego_pose_shifted = tuple(ego_pose[:3] + random_shift) + ego_pose[3:]  # Apply shift to translation part of pose

# print("Original Ego Pose:", ego_pose)
# print("Shifted Ego Pose:", ego_pose_shifted)
# print("Shift :", random_shift)

# register_and_view_multiple_point_clouds([(ego_file, ego_pose_shifted, ego_color), (infra_file, infra_pose, infra_color)])

merged_pcd, T_registered, fitness, inlier_rmse = register_multiple_point_clouds([(ego_file, ego_pose_shifted, ego_color), (infra_file, infra_pose, infra_color)])

print("====== Registration Summary ======")
print(f"Fitness: {fitness:.4f}, Inlier RMSE: {inlier_rmse:.4f}")

# TODO: this RUINS it
# shift T_ground_truth such that first pose is at origin
T0_inv = np.linalg.inv(T_ground_truth[0])
T_ground_truth[1] = T0_inv @ T_ground_truth[1]
T_ground_truth[0] = np.eye(4)  # Set first pose to identity (origin)

# shift T_registered such that first pose is at origin
T0_inv = np.linalg.inv(T_registered[0])
T_registered[1] = T0_inv @ T_registered[1]
T_registered[0] = np.eye(4)  # Set first pose to identity (origin)

# apply the transformation to the point clouds and visualize with open3d
ego_pcd = o3d.io.read_point_cloud(ego_file)
infra_pcd = o3d.io.read_point_cloud(infra_file)

ego_pcd.transform(T_registered[0])
infra_pcd.transform(T_registered[1])

# apply colors
ego_pcd.paint_uniform_color(np.array(ego_color) / 255.0)
infra_pcd.paint_uniform_color(np.array(infra_color) / 255.0)

# Get information on how well the registration worked
trans_err, rot_err = pose_difference(T_ground_truth[1], T_registered[1])
print("====== Registration Results ======")
print(f"Registration error margin: {trans_err:.2f}m translation, {rot_err:.2f} degrees rotation\n")

# Merge the point clouds for visualization
merged_pcd = ego_pcd + infra_pcd
viewer = PointCloudViewer.from_pointcloud(merged_pcd).run()


# register_and_view_multiple_point_clouds([(ego_file, None, ego_color), (infra_file, None, infra_color)])


# ------------------------------------------------------------------------------
