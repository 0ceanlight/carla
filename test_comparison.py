from utils.tum_file_comparator import show_all_differences, show_average_difference

if __name__ == "__main__":
    # Paths to your TUM trajectory files
    print("WITHOUT MERGE ==============================")
    # gt_0 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"
    # cmp_0 = "output_v1.log/sensor_captures_v3/test_pin_2025-05-21_06-36-45/slam_poses_tum.txt"
    gt_0 = "build/sim_output/sim_4/ego_lidar/ground_truth_poses_tum.txt"
    cmp_0 = "build/slam_output_ego_only/sim_4/test_pin_2025-06-26_20-44-31/slam_poses_tum.txt"
    # Sanity check:
    # cmp_0 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"

    # Display per-entry differences
    show_all_differences(gt_0, cmp_0)

    # Display average differences
    show_average_difference(gt_0, cmp_0)


    # print("WITH MERGE =================================")
    # gt_1 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"
    # cmp_1 = "output_v1.log/sensor_captures_v3/test_pin_2025-05-24_12-42-24/slam_poses_tum.txt"
    # # Sanity check:
    # # cmp_1 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"

    # # Display per-entry differences
    # show_all_differences(gt_1, cmp_1)

    # # Display average differences
    # show_average_difference(gt_1, cmp_1)
