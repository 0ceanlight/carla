import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def combine_point_clouds_with_offset_and_colors(cloud_params, out_file):
    """
    Combines multiple .ply files into one point cloud using absolute world coordinates and quaternions.
    
    Parameters:
        cloud_params (list of dict): Each dict contains:
            {
                'file': str,
                'pose': (x, y, z, qx, qy, qz, qw),
                'color': (r, g, b) Values in [0, 255]
            }
        out_file (str): Output filename for the merged point cloud.
    """
    combined_point_cloud = o3d.geometry.PointCloud()

    for cloud in cloud_params:
        in_file = cloud['file']
        x, y, z, qx, qy, qz, qw = cloud['pose']
        color_r, color_g, color_b = cloud['color']

        pcd = o3d.io.read_point_cloud(in_file)
        points = np.asarray(pcd.points)

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Apply translation first, then rotation because the points are in absolute world coordinates
        translated_points = points + np.array([x, y, z])
        rotated_points = translated_points @ rotation.T

        # Set the color for all points
        # colors = np.full_like(transformed_points, (color_r / 255.0, color_g / 255.0, color_b / 255.0))
        colors = np.full_like(rotated_points, (color_r / 255.0, color_g / 255.0, color_b / 255.0))

        # pcd.points = o3d.utility.Vector3dVector(transformed_points)
        pcd.points = o3d.utility.Vector3dVector(rotated_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        combined_point_cloud += pcd

    o3d.io.write_point_cloud(out_file, combined_point_cloud)
    print(f"Combined point cloud saved to {out_file}")
