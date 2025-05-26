import open3d as o3d
import numpy as np

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

# Input file paths
file_1 = "output_v1.log/sensor_captures_v3_cropped/ego_lidar/lidar_frames/10969.ply"
file_2 = "output_v1.log/sensor_captures_v3_cropped/infrastruct_lidar/lidar_frames/10969.ply"

# Load point clouds
source = o3d.io.read_point_cloud(file_1)
target = o3d.io.read_point_cloud(file_2)

# Preprocess and extract features
voxel_size = 1.0  # More aggressive downsampling for stability
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Global registration
ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print("RANSAC result:\n", ransac_result.transformation)

# Refine with ICP
refined_result = refine_registration(source, target, ransac_result.transformation, voxel_size)
print("ICP refined result:\n", refined_result.transformation)

# Apply final transformation
source.transform(refined_result.transformation)

# Visualize result
source.paint_uniform_color([1, 0, 0])  # red
target.paint_uniform_color([0, 1, 0])  # green
o3d.visualization.draw_geometries([source, target])