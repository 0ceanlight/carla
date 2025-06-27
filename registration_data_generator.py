import logging
import numpy as np
from utils.sensor_data_merger import SensorDataMerger
from utils.registration import register_multiple_point_clouds, save_point_cloud
from utils.tum_file_parser import *
from utils.tum_file_comparator import show_average_difference
from utils.math_utils import *
from utils.lidar_viewer import PointCloudViewer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input data directory
sensor_data_dir = "build/sim_output/sim_4"
# Output data/results directory
out_dir = "test_registration_GPS_drifted_data_sim_4.log"

ego_drifted_file = os.path.join(out_dir, "ego_drifted_tum.txt")
merged_frames_dir = os.path.join(out_dir, "merged_frames")
registration_est_file = os.path.join(out_dir, "registration_est_tum.txt")
registration_fitness_file = os.path.join(out_dir, "registration_fit.txt")
inlier_rmse_file = os.path.join(out_dir, "registration_inlier_rmse.txt")


def simulate_gps_drift(poses, bias_std=0.15, noise_std=1.2, jump_prob=0.03, jump_std=3.5, seed=None):
    """
    Apply realistic GPS drift to ground truth poses.

    Args:
        poses: np.ndarray of shape (N, 7) where each row is (x, y, z, qx, qy, qz, qw)
        bias_std: Standard deviation for the random walk bias
        noise_std: Standard deviation for the Gaussian noise
        jump_prob: Probability of a jump occurring at each step
        jump_std: Standard deviation for the jump noise

    Returns:
        np.ndarray of shape (N, 7) with drifted poses
    """
    if seed is not None:
        np.random.seed(seed)
    drifted_poses = poses.copy()
    n = poses.shape[0]
    bias = np.zeros(3)
    for i in range(n):
        # Random walk for bias
        bias += np.random.normal(0, bias_std, 3)
        # Occasional jump
        if np.random.rand() < jump_prob:
            bias += np.random.normal(0, jump_std, 3)
        # Gaussian noise
        noise = np.random.normal(0, noise_std, 3)
        drifted_poses[i, :3] += bias + noise
    return drifted_poses


def run_registration():
    """Run the registration process on the sensor data.
    
    This saves the drifted ego poses, merged point clouds, and registration
    results to files.
    """
    merger = SensorDataMerger(
        base_dir=sensor_data_dir,
        sensors=["ego_lidar", "merged_infrastruct_lidar"],
        max_timestamp_discrepancy=0.2
    )

    matches = merger.get_all_matches()

    # Get all ego data, which is the first element in each sub-array
    ego_data = [match[0] for match in matches]
    # Get all infrastructure data, which is the first element in each sub-array
    # infra_data = [match[1] for match in matches]

    # Each element of these lists is a tuple (file_path, timestamp, pose)

    # extract ego timestamps
    ego_timestamps = np.array([data[1] for data in ego_data])

    # extract ego poses and apply gps drift
    ego_poses = np.array([data[2] for data in ego_data])

    drifted_ego_poses = simulate_gps_drift(ego_poses)

    # # Print drifted and original poses side-by-side for comparison
    # for original, drifted in zip(ego_poses, drifted_ego_poses):
    #     print(f"Original: {original[:3]} Drifted: {drifted[:3]}")

    # Save drifted ego poses to TUM file
    drifted_ego_tum_data = [(timestamp,) + tuple(pose) for timestamp, pose in zip(ego_timestamps, drifted_ego_poses)]
    save_tum_file(ego_drifted_file, drifted_ego_tum_data)

    show_average_difference(ego_drifted_file, os.path.join(sensor_data_dir, "ego_lidar/ground_truth_poses_tum.txt"))

    fitness_values = []
    inlier_rmse_values = []

    post_registration_ego_poses = []

    logging.info(f"Registering {len(matches)} pairs of point clouds...")

    # Apply registration and save results
    for i, (ego_frame, infra_frame) in enumerate(matches):
        # Uncomment the following to process only a specific range of pairs
        # if i < 52 or i > 60:
        #     continue
        logging.info(f"Processing pair {i+1}/{len(matches)}...")
        if ego_frame is None or infra_frame is None:
            continue

        ego_file, ego_timestamp, ego_pose = ego_frame
        infra_file, _, infra_pose = infra_frame

        # 0 - 255
        magenta = (255, 0, 255)
        cyan = (0, 255, 255)

        # drifted 
        merged_pcd, transforms, fitness, inlier_rmse = register_multiple_point_clouds(
            [(ego_file, drifted_ego_poses[i], magenta), (infra_file, infra_pose, cyan)]
        )
        # ground truth (sanity check)
        # merged_pcd, transforms, fitness, inlier_rmse = register_multiple_point_clouds(
        #     [(ego_file, ego_pose, magenta), (infra_file, infra_pose, cyan)]
        # )

        # saved merged point cloud
        merged_pcd_file = os.path.join(merged_frames_dir, f"{i:04d}.ply")

        save_point_cloud(merged_pcd_file, merged_pcd)

        ego_reg_transform = transforms[0]
        infra_reg_transform = transforms[1]


        fitness_values.append(fitness)
        inlier_rmse_values.append(inlier_rmse)

        # Convert original infra pose to 4x4 transformation matrix
        infra_transform = pose_to_matrix(infra_pose)

        # Get relative transform between the two
        # TODO: right order? this describes "movement from infra to ego"
        rel_reg_transform = np.linalg.inv(infra_reg_transform) @ ego_reg_transform

        # Transform original pose by relative transform
        # TODO: right order?
        est_ego_transform = infra_transform @ rel_reg_transform
        est_ego_pose = matrix_to_pose(est_ego_transform)

        # Append the transformed pose with timestamp
        post_registration_ego_poses.append((ego_timestamp,) + tuple(est_ego_pose))

    # Save fitness and inlier RMSE to files
    np.savetxt(registration_fitness_file, fitness_values, fmt='%.6f')
    np.savetxt(inlier_rmse_file, inlier_rmse_values, fmt='%.6f')

    # Save the registration results to TUM file
    save_tum_file(registration_est_file, post_registration_ego_poses)


def eval_registration_results():
    # Show average difference between the drifted poses and the original ground truth
    print("==== Drifted poses and ground truth ====")
    show_average_difference(
        ego_drifted_file,
        os.path.join(sensor_data_dir, "ego_lidar/ground_truth_poses_tum.txt")
    )

    # Show average difference between the registration results and the original ground truth
    print("==== Registration results and ground truth ====")
    show_average_difference(
        registration_est_file,
        os.path.join(sensor_data_dir, "ego_lidar/ground_truth_poses_tum.txt")
    )

def view_results():
    PointCloudViewer(path=merged_frames_dir).run()

run_registration()
# eval_registration_results()
# view_results()

# GOAL: Find max RMSE/min fitness a certain threshold (e.g. 0.1m, 2deg) is not exceeded

