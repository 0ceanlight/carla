import os
import shutil
import logging
import numpy as np
from tqdm import tqdm

import config.dataset_structure_parser as dataset_parser
from utils.sensor_data_merger import SensorDataMerger
from utils.registration import register_multiple_point_clouds, save_point_cloud
from utils.tum_file_parser import load_tum_file, save_tum_file
from utils.math_utils import pose_to_matrix, matrix_to_pose

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Absolute path bases
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "build")
SIM_INPUT_DIR = os.path.join(BUILD_DIR, "sim_output")
MERGED_INPUT_DIR = os.path.join(BUILD_DIR, "gt_merged_sim_output")
REG_OUTPUT_DIR = os.path.join(BUILD_DIR, "registered_sim_output")


def register_sim_permutation(sim_name: str, permutation: str, ego_sensor: str):
    """
    Registers ego LiDAR point clouds to a merged point cloud (infrastructure or agent sensors).

    Args:
        sim_name (str): Name of the simulation, e.g., 'sim_0'
        permutation (str): Sensor permutation name, e.g., '2_agent'
        ego_sensor (str): Name of the ego sensor, e.g., 'ego_lidar'
    """
    logging.debug(f"🚗 Loading data for {sim_name}/{permutation}...")

    ego_dir = os.path.join(SIM_INPUT_DIR, sim_name, ego_sensor)
    merged_dir = os.path.join(MERGED_INPUT_DIR, sim_name, permutation)
    output_dir_final = os.path.join(REG_OUTPUT_DIR, sim_name, permutation)
    output_dir_tmp = output_dir_final + ".part"

    if os.path.exists(output_dir_final):
        logging.info(f"✅ Skipping {sim_name}/{permutation} — already completed.")
        return
    if os.path.exists(output_dir_tmp):
        logging.warning(f"⚠️ Skipping {sim_name}/{permutation} — .part directory already exists (incomplete run?)")
        return

    os.makedirs(os.path.join(output_dir_tmp, "frames"), exist_ok=True)

    # Load poses
    ego_gps_data = load_tum_file(os.path.join(ego_dir, "gps_poses_tum.txt"))
    ego_gt_data = load_tum_file(os.path.join(ego_dir, "ground_truth_poses_tum.txt"))
    merged_gt_data = load_tum_file(os.path.join(merged_dir, "ground_truth_poses_tum.txt"))

    # Use relative paths for sensor data merger
    ego_rel_path = os.path.relpath(ego_dir, start=BUILD_DIR)
    merged_rel_path = os.path.relpath(merged_dir, start=BUILD_DIR)

    # Use merger to get timestamp-aligned pairs
    merger = SensorDataMerger(
        base_dir=BUILD_DIR,
        sensors=[ego_rel_path, merged_rel_path],
        max_timestamp_discrepancy=0.2
    )
    matches = merger.get_all_matches()

    # Index poses by timestamp
    gps_pose_dict = {ts: pose for ts, *pose in ego_gps_data}
    merged_pose_dict = {ts: pose for ts, *pose in merged_gt_data}

    reg_poses = []
    fitness_values = []
    rmse_values = []


    tqdm_desc = "🔁 Registering {sim_name}/{permutation} frames"
    for i, (ego_frame, merged_frame) in tqdm(enumerate(matches), total=len(matches), desc=tqdm_desc, ncols=100):
        if ego_frame is None or merged_frame is None:
            logging.warning(f"⚠️ Skipping frame {i} due to missing ego frame {ego_frame} or merged frame {merged_frame}.")
            continue

        ego_file, ego_ts, _ = ego_frame
        merged_file, merged_ts, _ = merged_frame

        gps_pose = gps_pose_dict.get(ego_ts)
        merged_pose = merged_pose_dict.get(merged_ts)

        if gps_pose is None or merged_pose is None:
            logging.warning(f"⚠️ Skipping frame {i} due to missing gps pose {gps_pose} or merged pose {merged_pose}.")
            continue

        # optional TODO: remove colors
        # ego gets orange
        orange = (255, 165, 0)
        # merged gets blue
        blue = (0, 165, 255)

        # logging.debug(f"🔁 Registering frame {i} of {sim_name}/{permutation} at ego_ts={ego_ts:.6f} / merged_ts={merged_ts:.6f}")

        try:
            merged_pcd, transforms, fitness, inlier_rmse = register_multiple_point_clouds(
                [(ego_file, gps_pose, orange), (merged_file, merged_pose, blue)]
            )
        except Exception as e:
            logging.error(f"❌ Registration failed for frame {i}: {e}")
            continue

        save_point_cloud(os.path.join(output_dir_tmp, "frames", f"{i:04d}.ply"), merged_pcd)

        ego_transform = transforms[0]
        merged_transform = transforms[1]
        merged_gt_transform = pose_to_matrix(merged_pose)
        relative_transform = np.linalg.inv(merged_transform) @ ego_transform
        est_ego_pose = matrix_to_pose(merged_gt_transform @ relative_transform)

        reg_poses.append((ego_ts,) + tuple(est_ego_pose))
        fitness_values.append((ego_ts, fitness))
        rmse_values.append((ego_ts, inlier_rmse))

    # Save all results
    save_tum_file(os.path.join(output_dir_tmp, "reg_est_poses_tum.txt"), reg_poses)
    save_tum_file(os.path.join(output_dir_tmp, "ground_truth_poses_tum.txt"), ego_gt_data)
    np.savetxt(os.path.join(output_dir_tmp, "reg_fitness.txt"), fitness_values, fmt='%.6f')
    np.savetxt(os.path.join(output_dir_tmp, "reg_inlier_rmse.txt"), rmse_values, fmt='%.6f')

    # Finalize by renaming directory
    os.rename(output_dir_tmp, output_dir_final)
    logging.info(f"✅ Finished registration for {sim_name}/{permutation}")


def main():
    """
    Main script entrypoint. Loads the simulation config and performs registration
    for each simulation and sensor permutation defined in the config.
    """
    logging.info("📦 Loading simulation config...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)

    for sim_name, sim_data in sim_configs.items():
        ego_sensor = sim_data["ego"]
        for permutation in sim_data.get("sensor_permutations", {}).keys():
            logging.debug(f"Processing {sim_name}/{permutation} with ego sensor {ego_sensor}")
            try:
                register_sim_permutation(sim_name, permutation, ego_sensor)
            except Exception as e:
                logging.error(f"❌ Error in {sim_name}/{permutation}: {e}")

    logging.info("🎉 All registration jobs processed.")


if __name__ == "__main__":
    main()