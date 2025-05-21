from utils.tum_sequence_comparator import show_all_differences, show_average_difference

if __name__ == "__main__":
    # Paths to your TUM trajectory files
    file1 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"
    file2 = "output_v1.log/sensor_captures_v3/test_pin_2025-05-21_06-36-45/slam_poses_tum.txt"
    # Sanity check:
    # file2 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"

    # Display per-entry differences
    show_all_differences(file1, file2)

    # Display average differences
    show_average_difference(file1, file2)
