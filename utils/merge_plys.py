import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging


def load_point_cloud(ply_path):
    """
    Loads a point cloud from a .ply file.
    
    Parameters:
        ply_path (str): Path to the .ply file.
    
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    logging.debug(
        f"Loaded point cloud from {ply_path} with {len(pcd.points)} points.")
    return pcd


def save_point_cloud(out_file, pcd):
    """
    Saves a point cloud to a .ply file.
    
    Parameters:
        out_file (str): Output filename for the point cloud.
        pcd (o3d.geometry.PointCloud): The point cloud to save.
    """

    dirname = os.path.dirname(out_file)

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    o3d.io.write_point_cloud(out_file, pcd)
    logging.debug(
        f"Saved point cloud to {out_file} with {len(pcd.points)} points.")


def transform_point_cloud(pcd, pose):
    """
    Transforms a point cloud using the given pose (translation and rotation).

    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to transform.
        pose (tuple): A tuple containing translation (x, y, z) and quaternion (qx, qy, qz, qw).
    Returns:
        o3d.geometry.PointCloud: The transformed point cloud.
    """
    x, y, z, qx, qy, qz, qw = pose

    # Convert quaternion to rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Apply rotation and translation
    points = np.asarray(pcd.points)
    rotated_points = points @ rotation.T
    transformed_points = rotated_points + np.array([x, y, z])

    # Update point cloud with transformed points
    pcd.points = o3d.utility.Vector3dVector(transformed_points)

    return pcd


def combine_point_clouds_with_poses(cloud_params, out_file):
    """
    Combines multiple .ply files into one point cloud using the given offset in 
    absolute world coordinates and quaternions. Also OPTIONALLY assigns colors 
    to each different point cloud for better visualization IF the 'color' key 
    is present in the cloud_params dict. Make sure that either ALL or NO entries
    have the 'color' key.
    
    Parameters:
        cloud_params (list of dict): Each dict contains:
            {
                'file': str,
                'pose': (x, y, z, qx, qy, qz, qw),
                'color': (r, g, b) Values in [0, 255] # OPTIONAL value
            }
        out_file (str): Output filename for the merged point cloud.
    """
    combined_point_cloud = o3d.geometry.PointCloud()

    for cloud in cloud_params:
        in_file = cloud['file']
        pose = cloud['pose']

        pcd = load_point_cloud(in_file)

        # Apply the transformation
        pcd = transform_point_cloud(pcd, pose)

        # If color is provided, apply it
        if 'color' in cloud:
            color = cloud['color']
            pcd.paint_uniform_color(np.array(color) / 255.0)

        combined_point_cloud += pcd

    o3d.io.write_point_cloud(out_file, combined_point_cloud)
    save_point_cloud(out_file, combined_point_cloud)
