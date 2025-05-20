import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def combine_point_clouds_with_offset_and_colors(cloud_params, out_file):
    """
    Combines multiple .ply files into one point cloud after applying specified
    quaternion rotations and XYZ offsets, while setting colors.

    Parameters:
        cloud_params (list of dict): Each dict contains:
            {
                'file': str,
                'offset': (x, y, z),
                'quaternion': (qx, qy, qz, qw),
                'color': (r, g, b)
            }
        out_file (str): Output filename for the merged point cloud.
    """
    combined_point_cloud = o3d.geometry.PointCloud()

    for cloud in cloud_params:
        in_file = cloud['file']
        offset = np.array(cloud['offset'])
        qx, qy, qz, qw = cloud['quaternion']
        color_r, color_g, color_b = cloud['color']

        pcd = o3d.io.read_point_cloud(in_file)
        points = np.asarray(pcd.points)

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Apply rotation and then translation
        rotated_points = points @ rotation.T
        transformed_points = rotated_points + offset

        # Set color for all points
        colors = np.full_like(transformed_points, (color_r / 255.0, color_g / 255.0, color_b / 255.0))

        pcd.points = o3d.utility.Vector3dVector(transformed_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        combined_point_cloud += pcd

    o3d.io.write_point_cloud(out_file, combined_point_cloud)
    print(f"Combined point cloud saved to {out_file}")


# Example usage with quaternion rotation added (no rotation = [0, 0, 0, 1])
dx = -61.2 + 52.3109
dy = 36.8 - 1.5852
dz = 7.6 - 2.5857

clouds = [
    {
        'file': 'run.log/sensor_captures_v1/ego_lidar/lidar_frames/1.ply',
        'offset': (0.0, 0.0, 0.0),
        'quaternion': (0, 0, 0, 1),
        'color': (255, 0, 0)
    },
    {
        'file': 'run.log/sensor_captures_v1/infrastruct_lidar/lidar_frames/2.ply',
        'offset': (dx, -dy, dz),
        'quaternion': (0, 0, 0, 1),
        'color': (0, 255, 0)
    }
]

combine_point_clouds_with_offset_and_colors(clouds, out_file='out_combined_with_colors.ply')

