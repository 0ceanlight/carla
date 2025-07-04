import os
import logging

from utils.tum_file_parser import tum_load_as_matrices, tum_save_matrices
from utils.pose_graph_optimization import optimize_pose_graph, combine_fitness_rmse_acceptance, remove_outliers_accepted_reg_indices
from utils.math_utils import align_matrix_list_to_matrix
import config.dataset_structure_parser as dataset_parser
from config.graph_colors import graph_colors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "build")
FIGURE_DIR = os.path.join(BUILD_DIR, "figures", "pose_error_plots")
os.makedirs(FIGURE_DIR, exist_ok=True)


def process_simulation(sim_name: str, permutation_name: str):
    logging.info(f"🗺️ Generating optimized fused SLAM/Registration trajectory for {sim_name}/{permutation_name}")

    # LOAD DATA ----------------------------------------------------------------
    # Paths
    sensor_data_ego_dir = os.path.join(BUILD_DIR, "sim_output", sim_name, "ego_lidar")
    reg_dir = os.path.join(BUILD_DIR, "registered_sim_output", sim_name, permutation_name)
    slam_dir = os.path.join(BUILD_DIR, "slam_output_ego_only", sim_name)
    slam_test_pin = [d for d in os.listdir(slam_dir)][0]
    # Output
    fused_output_dir = os.path.join(BUILD_DIR, "fused_output", sim_name, permutation_name)
    os.makedirs(fused_output_dir, exist_ok=True)

    ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
    ego_drifted_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")
    registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
    registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
    inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
    slam_file = os.path.join(slam_dir, slam_test_pin, "slam_poses_tum.txt")
    # Output
    fused_output_file = os.path.join(fused_output_dir, "fused_poses_tum.txt")

    # Load pose data
    gt_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_ground_truth_file)]
    gps_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_drifted_file)]
    reg_transforms = [matrix for _, matrix in tum_load_as_matrices(registration_est_file)]
    slam_transforms = [matrix for _, matrix in tum_load_as_matrices(slam_file)]

    # DATA ALIGNMENT / PREP ----------------------------------------------------
    # Align SLAM
    slam_transforms_aligned = align_matrix_list_to_matrix(slam_transforms, gt_transforms[0])

    # Load fitness and RMSE
    with open(registration_fitness_file, 'r') as f:
        fitness = [float(line.split(' ')[1].strip()) for line in f]
    with open(inlier_rmse_file, 'r') as f:
        inlier_rmse = [float(line.split(' ')[1].strip()) for line in f]

    # FUSE POSES ---------------------------------------------------------------
    accepted_indices, _, _ = combine_fitness_rmse_acceptance(
        fitness, inlier_rmse,
        fitness_margin=0.04,
        rmse_margin=0.03,
        weight_fitness=1.0,
        weight_rmse=1.2
    )

    accepted_indices_filtered = remove_outliers(accepted_indices, reg_transforms, max_offset_m=0.5)

    fused_transforms = optimize_pose_graph(slam_transforms_aligned, reg_transforms, accepted_indices_filtered)

    # SAVE RESULTS -------------------------------------------------------------
    # Save results
    # pull timestamps from ground truth poses
    gt_timestamps = [i for i, _ in tum_load_as_matrices(ego_ground_truth_file)]

    fused_output_data = [(ts, tr) for ts, tr in zip(gt_timestamps, fused_transforms)]
    tum_save_matrices(fused_output_file, fused_output_data)


def main():
    logging.info("📁 Loading simulation config...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)

    for sim_name, sim_data in sim_configs.items():
        permutations = sim_data.get("sensor_permutations", {})
        for perm_name in permutations:
            try:
                process_simulation(sim_name, perm_name)
            except Exception as e:
                logging.exception(f"❌ Failed to process {sim_name}/{perm_name}: {e}")

    logging.info("📈 All plots generated.")


if __name__ == "__main__":
    main()
