import open3d as o3d
import numpy as np

# Input file paths
file_1 = "output_v1.log/sensor_captures_v3_cropped/ego_lidar/lidar_frames/10969.ply"
file_2 = "output_v1.log/sensor_captures_v3_cropped/infrastruct_lidar/lidar_frames/10969.ply"

# Load point clouds
source = o3d.io.read_point_cloud(file_1)  # The point cloud that will be transformed
target = o3d.io.read_point_cloud(file_2)  # The reference point cloud

# Preprocessing: downsample for faster ICP (optional but recommended)
voxel_size = 0.2
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# Estimate normals (required for ICP if using point-to-plane or colored ICP)
source_down.estimate_normals()
target_down.estimate_normals()

# Initial transformation (identity)
initial_trans = np.eye(4)

# Run ICP
threshold = 1.0  # Maximum correspondence points-pair distance
icp_result = o3d.pipelines.registration.registration_icp(
    source_down, target_down, threshold, initial_trans,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

print("ICP fitness:", icp_result.fitness)
print("ICP inlier RMSE:", icp_result.inlier_rmse)
print("Transformation matrix:\n", icp_result.transformation)

# Apply transformation to the full-resolution source
source.transform(icp_result.transformation)

# Visualize result
source.paint_uniform_color([1, 0, 0])  # red
target.paint_uniform_color([0, 1, 0])  # green
o3d.visualization.draw_geometries([source, target],
                                  zoom=0.455,
                                  front=[0.6452, -0.3036, -0.7011],
                                  lookat=[1.9892, 2.0208, 1.8945],
                                  up=[-0.2779, -0.9482, 0.1556])