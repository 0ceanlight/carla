import open3d as o3d
import numpy as np
import os

def remove_points_below_z_threshold(input_file, output_file, z_threshold=-1.62):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Create mask for points above or equal to the threshold
    mask = points[:, 2] >= z_threshold
    filtered_points = points[mask]

    # If colors exist, filter them too
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_colors = colors[mask]
    else:
        filtered_colors = None

    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Save the result
    o3d.io.write_point_cloud(output_file, filtered_pcd)
    print(f"Saved filtered point cloud with {len(filtered_points)} points to {output_file}")

input_file = "test_lidar_merge.log/out_combined_with_colors_v8.ply"
output_dir = "test_lidar_merge.log/cropped_ply/"

# Create path if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# z_threshold = -4  # Set your desired low Z threshold here, e.g., -1 to keep all points above this value

# # Example usage:
# for i in range(1, 100):
#     z_threshold += .01 * 7  # Set your desired Z threshold
#     output_file = os.path.join(output_dir, f"{i} thresh {z_threshold}.ply")
#     remove_points_below_z_threshold(input_file, output_file, z_threshold)

# Final threshold
final_thresh = -1.62
output_file = os.path.join(output_dir, f"0_final_thresh.ply")
remove_points_below_z_threshold(input_file, output_file, final_thresh)

input_file2 = "test_lidar_merge.log/out_combined_with_colors_v6.ply"

output_file = os.path.join(output_dir, f"1_final_thresh.ply")
remove_points_below_z_threshold(input_file2, output_file, final_thresh)


input_file3 = "test_lidar_merge.log/out_combined_with_colors_v1.ply"
output_file = os.path.join(output_dir, f"2_final_thresh.ply")
remove_points_below_z_threshold(input_file3, output_file, final_thresh)