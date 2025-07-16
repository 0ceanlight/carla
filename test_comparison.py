from utils.tum_file_comparator import show_all_differences, show_average_difference
import logging
import os
import subprocess
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "dataset_structure.jsonc")
BASE_INPUT_DIR = os.path.join(SCRIPT_DIR, "build", "sim_output")
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "build", "slam_output_ego_only")
DOCKER_MOUNT_DIR = os.path.join(SCRIPT_DIR, "build")

def run_slam_on_sim():
    input_path = f"/storage/gt_merged_sim_output/sim_0/2_infra_1_ego/frames"
    output_path_container = f"/storage/slam_output_ego_only/test_augmented"
    output_path_host = os.path.join(BASE_OUTPUT_DIR, "sim_0")

    os.makedirs(output_path_host, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm", "--ipc", "host", "--privileged", "--network", "host",
        "-p", "8080:8081", "--gpus", "all",
        "-v", f"{DOCKER_MOUNT_DIR}:/storage/",
        "pinslam:localbuild",
        "/usr/bin/python3", "pin_slam.py", "-sm",
        "-i", input_path,
        "-o", output_path_container
    ]

    logging.info(f"Running SLAM")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Finished SLAM ")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ SLAM failed with error: {e}")
    except Exception as ex:
        logging.exception(f"Unexpected error during SLAM: {ex}")

if __name__ == "__main__":
    # Paths to your TUM trajectory files
    print("WITHOUT MERGE ==============================")
    # gt_0 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"
    # cmp_0 = "output_v1.log/sensor_captures_v3/test_pin_2025-05-21_06-36-45/slam_poses_tum.txt"

    # gt_0 = "build/sim_output/sim_4/ego_lidar/ground_truth_poses_tum.txt"
    # cmp_0 = "build/slam_output_ego_only/sim_4/test_pin_2025-06-26_20-44-31/slam_poses_tum.txt"

    gt_0 = "build/sim_output/sim_0/ego_lidar/ground_truth_poses_tum.txt"
    # cmp_0 = "build/gt_merged_sim_output/sim_0/2_infra_1_ego/ground_truth_poses_tum.txt" # sanity check
    # cmp_0 = "build/slam_output_ego_only/sim_0/test_pin_2025-06-27_13-51-06/slam_poses_tum.txt" # initial slam
    cmp_0 = "build/slam_output_ego_only/test_augmented/test_pin_2025-07-06_16-57-29/slam_poses_tum.txt" # augmented slam


    # run_slam_on_sim()



    # Sanity check:
    # cmp_0 = "output_v1.log/sensor_captures_v3/ego_lidar/ground_truth_poses_tum.txt"

    # Display per-entry differences
    # show_all_differences(gt_0, cmp_0)

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
