import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return pcd_down, fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def main():
    # Load point clouds
    source_file = "run.log/sensor_captures_v3/ego_lidar/lidar_frames/10969.ply"
    target_file = "run.log/sensor_captures_v3/infrastruct_lidar/lidar_frames/10969.ply"

    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    voxel_size = 0.2  # Adjust depending on scale and density

    # Preprocess
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("RANSAC transformation:\n", result_ransac.transformation)

    # Before calling refine_registration
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Refine with ICP
    result_icp = refine_registration(source, target, result_ransac, voxel_size)
    print("ICP refined transformation:\n", result_icp.transformation)

    # Visualize
    source.transform(result_icp.transformation)
    o3d.visualization.draw_geometries([source.paint_uniform_color([1, 0, 0]), 
                                       target.paint_uniform_color([0, 1, 0])])

if __name__ == "__main__":
    main()
