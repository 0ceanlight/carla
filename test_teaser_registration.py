import os
import logging
import numpy as np
from utils.lidar_viewer import PointCloudViewer
from utils.teaser_registration import register_and_save_multiple_point_clouds
from utils.sensor_data_merger import SensorDataMerger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Using clouds centered at origin: ---------------------------------------------
# # Input file paths
# file_1 = "output_v1.log/sensor_captures_v3_cropped/ego_lidar/frames/10969.ply"
# file_2 = "output_v1.log/sensor_captures_v3_cropped/infrastruct_lidar/frames/10969.ply"

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
base_directory = "output_v1.log/sensor_captures_v3_cropped"
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

# Apply a simulated pose drift to the ego pose (infrastruct location is known)
random_shift = np.random.normal(loc=0.0, scale=4.4, size=3)  # Mean 0, stddev 4.4
ego_pose_shifted = tuple(ego_pose[:3] + random_shift) + ego_pose[3:]  # Apply shift to translation part of pose

print("Original Ego Pose:", ego_pose)
# print("Shifted Ego Pose:", ego_pose_shifted)
print("Shifted Ego Pose:", ego_pose)
print("Shift :", random_shift)

output_dir = "teaser_output.log"

o1 = os.path.join(output_dir, "shifted.ply")
o2 = os.path.join(output_dir, "gt.ply")
o3 = os.path.join(output_dir, "zero.ply")

register_and_save_multiple_point_clouds(o1, [(ego_file, ego_pose_shifted, ego_color), (infra_file, infra_pose, infra_color)])
register_and_save_multiple_point_clouds(o2, [(ego_file, ego_pose, ego_color), (infra_file, infra_pose, infra_color)])
register_and_save_multiple_point_clouds(o3, [(ego_file, None, ego_color), (infra_file, None, infra_color)])


# ------------------------------------------------------------------------------
