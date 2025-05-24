import os
from utils.sensor_data_merger import SensorDataManager
from utils.merge_plys import combine_point_clouds_with_offset
import open3d as o3d

if __name__ == "__main__":
    # Example setup
    base_directory = "output_v1.log/sensor_captures_v3"
    sensors = ["ego_lidar", "infrastruct_lidar"]
    max_discrepancy = 1.0  # seconds

    # Initialize the data manager
    manager = SensorDataManager(base_dir=base_directory, sensors=sensors, max_timestamp_discrepancy=max_discrepancy)

    # Print summary of matching status
    print("=== Matching Summary ===")
    manager.summary()

# TESTING ALL MATCHES WITH SENSOR_DATA_MERGER MERGING FUNCS --------------------

frame = 57

# TODO: dis a problem child
manager.save_merged_ply_at_index(frame, output_file='test_merger.log/out_combined_with_colors_v4.ply', relative_match=False, colored=True)

manager.save_merged_ply_at_index(frame, output_file='test_merger.log/out_combined_with_colors_v5.ply', relative_match=True, colored=True)

# manager.save_all_merged_plys("test_merger.log/out_frames_abs_v0/", relative_match=False)
# manager.save_all_merged_plys("test_merger.log/out_frames_rel_v0/", relative_match=True)

# TESTING SINGLE MATCHES W/ MANUAL MERGING -------------------------------------

#     # Print the 57th matched ego frame with its corresponding matches
#     print("\n=== 57th Frame Match ===")
#     match = manager.get_match_for_ego_index(56)  # 57th frame is index 56
#     if match:
#         print(f"Ego Frame 57: {match}")
#     else:
#         print("No match found for ego frame 57.")

#     # Print the 57th matched ego frame with its corresponding RELATIVE coord matches
#     print("\n=== 57th Frame RELATIVE Match ===")
#     match = manager.get_relative_match_for_ego_index(56)  # 57th frame is index 56
#     if match:
#         print(f"Ego Frame 57: {match}")

#         # unpack match for further use
#         ego_filename, ego_pose = match[0]
#         infrastructure_filename, infrastructure_pose = match[1]
#         i_x, i_y, i_z, i_qx, i_qy, i_qz, i_qw = infrastructure_pose

#         # cyan
#         ego_color = (0, 255, 255)
#         # magenta
#         infrastruct_color = (255, 0, 255)

#         clouds = [
#             {
#                 'file': ego_filename,
#                 'pose': ego_pose,
#                 'color': ego_color
#             },
#             {
#                 'file': infrastructure_filename,
#                 # 'pose': infrastructure_pose, # (v0)
#                 # 'pose': (i_x + 1, i_y + 1, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift, using index 64 (v1)
#                 # 'pose': (i_x + 7, i_y - 7, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift (v2)
#                 # 'pose': (i_x + 10, i_y - 6.5, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift (v3)
#                 # 'pose': (i_x + 10.25, i_y - 6.25, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift (v4)
#                 # 'pose': (i_x + 10.5, i_y - 6.15, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift (v5)
#                 # 'pose': (i_x + 10.5, i_y - 6.15, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift, using a new index 56 (v6)
#                 # 'pose': infrastructure_pose, # (v7) back to no shift for index 56
#                 # 'pose': (i_x + 18, i_y + 0.5, i_z, i_qx, i_qy, i_qz, i_qw),  # manual shift (v9)
#                 'pose': infrastructure_pose, # (v10) back to no shift for index 56, now flipping y in the merging part (before rotation) (THAT DID NOT WORK)
#                 'color': infrastruct_color
#             }
#         ]

# # === 65th Frame Match ===
# # Ego Frame 65: [('run.log/sensor_captures_v3/ego_lidar/lidar_frames/10975.ply', (139.952, -52.4286, -33.9597, 2.0592, -0.0032, -0.0022, -0.7027, 0.7114)), ('run.log/sensor_captures_v3/infrastruct_lidar/lidar_frames/10969.ply', (139.752, -61.2, -36.8, 7.6, -0.0, 0.0, -0.0, 1.0))]

# # === 65th Frame RELATIVE Match ===
# # Ego Frame 65: [('run.log/sensor_captures_v3/ego_lidar/lidar_frames/10975.ply', (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ('run.log/sensor_captures_v3/infrastruct_lidar/lidar_frames/10969.ply', (-8.7714, -2.840299999999999, 5.540799999999999, 0.0032, 0.0022, 0.7027, 0.7114))]
# # Combined point cloud saved to out_combined_with_colors_v5.ply

# # === 57th Frame Match ===
# # Ego Frame 57: [('run.log/sensor_captures_v3/ego_lidar/lidar_frames/10957.ply', (139.152, -52.492, -27.3943, 2.0446, -0.0009, -0.002, -0.7077, 0.7065)), ('run.log/sensor_captures_v3/infrastruct_lidar/lidar_frames/10957.ply', (139.152, -61.2, -36.8, 7.6, -0.0, 0.0, -0.0, 1.0))]

# # === 57th Frame RELATIVE Match ===
# # Ego Frame 57: [('run.log/sensor_captures_v3/ego_lidar/lidar_frames/10957.ply', (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ('run.log/sensor_captures_v3/infrastruct_lidar/lidar_frames/10957.ply', (-8.708000000000006, -9.405699999999996, 5.5554, 0.0009, 0.002, 0.7077, 0.7065))]
# # Combined point cloud saved to out_combined_with_colors_v9.ply

#         combine_point_clouds_with_offset_and_colors(clouds, out_file='test_lidar_merge.log/out_combined_with_colors_v11.ply')

#     else:
#         print("No match found for ego frame 57.")

#     # Print the first 5 matched ego frames with their corresponding matches
#     print("\n=== First 5 Frame Matches ===")
#     for i in range(5):
#         match = manager.get_match_for_ego_index(i)
#         if match:
#             print(f"\nEgo Frame {i}:")
#             for sensor_idx, entry in enumerate(match):
#                 if entry:
#                     filename, pose = entry
#                     print(f"  Sensor {sensors[sensor_idx]}: {filename}, pose: {pose}")
#                 else:
#                     print(f"  Sensor {sensors[sensor_idx]}: No match found")

#     # Report unmatched frame indices
#     unmatched_indices = manager.get_unmatched_indices()
#     if unmatched_indices:
#         print(f"\nUnmatched ego frame indices: {unmatched_indices}")
#     else:
#         print("\nAll ego frames have matches from all sensors.")
