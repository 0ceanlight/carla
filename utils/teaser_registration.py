import numpy as np
import open3d as o3d
import teaserpp_python
from .lidar_viewer import PointCloudViewer
from .merge_plys import load_point_cloud, save_point_cloud, transform_point_cloud

NOISE_BOUND = 0.05  # You may expose this as a parameter if needed


def register_pairwise(source, target):
    """
    Register two point clouds using TEASER++.

    Parameters:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.

    Returns:
        o3d.geometry.PointCloud: Transformed source point cloud.
    """
    src = np.transpose(np.asarray(source.points))
    dst = np.transpose(np.asarray(target.points))

    # Setup TEASER++ parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()

    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = solution.rotation
    T[:3, 3] = solution.translation

    # Apply transformation
    registered = source.transform(T)
    return registered


def register_multiple_point_clouds(files_with_transforms) -> o3d.geometry.PointCloud:
    """
    Register multiple point clouds into one.

    Parameters:
        files_with_transforms: List of tuples, each (file_path, transform, color)
            - transform is (x, y, z, qx, qy, qz, qw) or None
            - color is (r, g, b) in [0, 255] or None

    Returns:
        o3d.geometry.PointCloud: Merged and registered point cloud.
    """
    merged_pcd = None

    for file_path, transform, color in files_with_transforms:
        pcd = load_point_cloud(file_path)

        if transform:
            pcd = transform_point_cloud(pcd, transform)

        if color:
            rgb = np.array(color) / 255.0
            colors = np.tile(rgb, (np.asarray(pcd.points).shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if merged_pcd is None:
            merged_pcd = pcd
        else:
            # Register new point cloud to existing merged cloud
            pcd = register_pairwise(pcd, merged_pcd)
            merged_pcd += pcd

    return merged_pcd


def register_and_save_multiple_point_clouds(output_file: str, files_with_transforms):
    """
    Register multiple point clouds and save the merged result to a file.

    Parameters:
        output_file: Path to save the merged point cloud (.ply).
        files_with_transforms: List of tuples, each (file_path, transform, color)
            - transform is (x, y, z, qx, qy, qz, qw) or None
            - color is (r, g, b) in [0, 255] or None
    """
    merged_pcd = register_multiple_point_clouds(files_with_transforms)
    save_point_cloud(output_file, merged_pcd)


def register_and_view_multiple_point_clouds(files_with_transforms):
    """
    Register multiple point clouds and visualize the merged result.

    Parameters:
        files_with_transforms: List of tuples, each (file_path, transform, color)
            - transform is (x, y, z, qx, qy, qz, qw) or None
            - color is (r, g, b) in [0, 255] or None
    """
    merged_pcd = register_multiple_point_clouds(files_with_transforms)
    PointCloudViewer.from_pointcloud(merged_pcd).run()
