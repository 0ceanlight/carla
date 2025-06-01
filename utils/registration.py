import open3d as o3d
import numpy as np
from typing import List, Optional, Tuple
import logging
from .merge_plys import load_point_cloud, save_point_cloud, transform_point_cloud
from .lidar_viewer import PointCloudViewer


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

    logging.info(f"RANSAC Registration:")
    logging.info(f"  Fitness: {result.fitness:.4f}")
    logging.info(f"  Inlier RMSE: {result.inlier_rmse:.4f}")
    logging.debug(f"  Transformation:\n{result.transformation}")

    return result

def refine_registration(source, target, initial_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    logging.info(f"ICP Refinement:")
    logging.info(f"  Fitness: {result.fitness:.4f}")
    logging.info(f"  Inlier RMSE: {result.inlier_rmse:.4f}")
    logging.debug(f"  Transformation:\n{result.transformation}")
    
    return result


def register_multiple_point_clouds(
    files_with_transforms,
    voxel_size: float = 1.0
) -> o3d.geometry.PointCloud:
    """
    Register multiple point clouds into one.
    
    Parameters:
        files_with_transforms: List of tuples, each (file_path, transform, color)
            where transform is (x, y, z, qx, qy, qz, qw) or None
            where color is (r, g, b) Values in [0, 255] or None
        voxel_size: for downsampling and registration precision

    Returns:
        Merged point cloud (Open3D PointCloud object)
    """
    assert len(files_with_transforms) >= 2, "Need at least two point clouds for registration."

    base_file, base_transform, base_color = files_with_transforms[0]

    # Load the first point cloud
    merged_cloud = load_point_cloud(base_file)
    if base_transform is not None:
        merged_cloud = transform_point_cloud(merged_cloud, base_transform)

    if base_color is not None:
        # Assign color if provided
        merged_cloud.paint_uniform_color(np.array(base_color) / 255.0)

    for next_file, next_transform, next_color in files_with_transforms[1:]:
        # Load the next point cloud
        next_cloud = load_point_cloud(next_file)
        if next_transform is not None:
            next_cloud = transform_point_cloud(next_cloud, next_transform)

        if next_color is not None:
            # Assign color if provided
            next_cloud.paint_uniform_color(np.array(next_color) / 255.0)

        # Preprocess both point clouds
        merged_down, merged_fpfh = preprocess_point_cloud(merged_cloud, voxel_size)
        next_down, next_fpfh = preprocess_point_cloud(next_cloud, voxel_size)

        # Global registration
        ransac_result = execute_global_registration(merged_down, next_down, merged_fpfh, next_fpfh, voxel_size)

        # Estimate normals for full-resolution clouds (required for PointToPlane ICP)
        merged_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        next_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # Refine registration using ICP
        refined_result = refine_registration(
            merged_cloud, next_cloud, ransac_result.transformation, voxel_size)

        # Apply the refined transformation to the next cloud
        merged_cloud.transform(refined_result.transformation)

        # Merge the point clouds
        merged_cloud += next_cloud

    # Downsample the final merged cloud
    # merged_cloud = merged_cloud.voxel_down_sample(voxel_size)
    # merged_cloud.remove_non_finite_points()  # Clean up any non-finite points
    # merged_cloud.remove_duplicated_points()  # Remove duplicate points
    # merged_cloud.remove_radius_outlier(nb_points=16, radius=voxel_size * 2)  # Remove outliers
    # merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # Remove statistical outliers
    return merged_cloud


def register_and_save_multiple_point_clouds(
    files_with_transforms: List[Tuple[str, Optional[Tuple[float, float, float, float, float, float, float]]]],
    output_file: str,
    voxel_size: float = 1.0
):
    """
    Register multiple point clouds and save the merged result to a file.
    
    Parameters:
        files_with_transforms: List of tuples, each (file_path, transform, color)
            where transform is (x, y, z, qx, qy, qz, qw) or None
            where color is (r, g, b) Values in [0, 255] or None
        output_file: Path to save the merged point cloud
        voxel_size: for downsampling and registration precision
    """
    merged_cloud = register_multiple_point_clouds(files_with_transforms, voxel_size)
    save_point_cloud(output_file, merged_cloud)


def register_and_view_multiple_point_clouds(
    files_with_transforms: List[Tuple[str, Optional[Tuple[float, float, float, float, float, float, float]]]],
    voxel_size: float = 1.0
):
    """
    Register multiple point clouds and visualize the merged result.
    
    Parameters:
        files_with_transforms: List of tuples, each (file_path, transform, color)
            where transform is (x, y, z, qx, qy, qz, qw) or None
            where color is (r, g, b) Values in [0, 255] or None
        voxel_size: for downsampling and registration precision
    """
    merged_cloud = register_multiple_point_clouds(files_with_transforms, voxel_size)
    o3d.visualization.draw_geometries([merged_cloud])


if __name__ == "__main__":
    # Example usage:
    files_with_poses = [
        ("path/to/first.ply", None),
        ("path/to/second.ply", (1.0, 2.0, 3.0, 0, 0, 0, 1)),  # with transform
        ("path/to/third.ply", None)
    ]

    final_cloud = register_multiple_point_clouds(files_with_poses, voxel_size=1.0)
    PointCloudViewer.from_pointcloud(final_cloud).run()