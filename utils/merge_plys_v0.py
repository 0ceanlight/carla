import open3d as o3d
import numpy as np

def combine_point_clouds_with_offset_and_colors(*args, out_file):
    """
    Combines multiple .ply files into one point cloud after applying specified offsets to each,
    while preserving or setting colors for each point cloud.

    Parameters:
        *args: Each argument is a pair: (input_file, offset_x, offset_y, offset_z, color_r, color_g, color_b)
        out_file (str): The output file where the combined point cloud will be saved.
    """
    # Initialize an empty list to hold the point clouds
    combined_point_cloud = o3d.geometry.PointCloud()

    for i in range(0, len(args), 7):
        # Read the input point cloud file
        in_file = args[i]
        offset_x = args[i+1]
        offset_y = args[i+2]
        offset_z = args[i+3]
        color_r = args[i+4]
        color_g = args[i+5]
        color_b = args[i+6]
        
        # Read the .ply file
        pcd = o3d.io.read_point_cloud(in_file)
        
        # Convert to numpy array for easy manipulation
        points = np.asarray(pcd.points)

        # Apply the offset to each point in the point cloud
        points += np.array([offset_x, offset_y, offset_z])

        # Set the color for all points (RGB values between 0 and 1)
        colors = np.full_like(points, (color_r / 255.0, color_g / 255.0, color_b / 255.0), dtype=np.float64)

        # Update the point cloud with the new points and colors
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Combine this point cloud with the existing ones
        combined_point_cloud += pcd

    # Save the combined point cloud to the output file
    o3d.io.write_point_cloud(out_file, combined_point_cloud)
    print(f"Combined point cloud saved to {out_file}")


# EGO LiDAR at -52.3109 1.5852 2.5857
# INFRASTRUCTURE LiDAR at -61.2, 36.8, 7.6
dx = -61.2 + 52.3109
dy = 36.8 - 1.5852
dz = 7.6 - 2.5857

# TODO: rotations????

# Combine point clouds with offsets and colors
combine_point_clouds_with_offset_and_colors(
    'run.log/sensor_captures_v1/ego_lidar/lidar_frames/1.ply', 0.0, 0.0, 0.0, 255, 0, 0,  # Offset (1,0,0) and color red
    'run.log/sensor_captures_v1/infrastruct_lidar/lidar_frames/2.ply', dx, -dy, dz, 0, 255, 0,  # Offset (0,1,0) and color green
    # 'in_file_3.ply', 0.0, 0.0, 1.0, 0, 0, 255,  # Offset (0,0,1) and color blue
    out_file='out_combined_with_colors.ply'
)


# IDEAS:
# zipper-combine merging (ASSUMES time synchronization, + time-labeled clouds)
# 1. start with the first point cloud in ego sequence (ego will be base-cloud, trying to match with closest others)
# 2. for each point in the base-cloud, find the closest ply in each of the other clouds
# 3. For each of these closest .plys, only mearge if within a certain distance threshold (X meters), and time threshold (X ms)
# 4. Merge the closest points from each cloud into the base-cloud, using given offsets (this data is paired with each ply sequence)
