import os
import logging
import matplotlib.pyplot as plt

from utils.tum_file_parser import tum_load_as_matrices
from utils.math_utils import calc_offset_margin
from utils.fusion_graph_utils import align_matrix_list_to_matrix
from utils.data_viz import get_split_pose_plot
import config.dataset_structure_parser as dataset_parser

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
FIGURE_DIR = os.path.join(BUILD_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def process_simulation(sim_name: str, permutation_name: str):
    logging.info(f"📊 Generating plots for {sim_name}/{permutation_name}")

    # Paths
    sensor_data_ego_dir = os.path.join(BUILD_DIR, "sim_output", sim_name, "ego_lidar")
    reg_dir = os.path.join(BUILD_DIR, "registered_sim_output", sim_name, permutation_name)
    slam_dir = os.path.join(BUILD_DIR, "slam_output_ego_only", sim_name)
    slam_test_pin = [d for d in os.listdir(slam_dir)][0]

    ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
    ego_drifted_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")
    registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
    registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
    inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
    slam_file = os.path.join(slam_dir, slam_test_pin, "slam_poses_tum.txt")

    # Load pose data
    gt_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_ground_truth_file)]
    gps_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_drifted_file)]
    reg_transforms = [matrix for _, matrix in tum_load_as_matrices(registration_est_file)]
    slam_transforms = [matrix for _, matrix in tum_load_as_matrices(slam_file)]

    # Align SLAM
    slam_transforms_aligned = align_matrix_list_to_matrix(slam_transforms, gt_transforms[0])

    # Calculate errors
    gps_err_margins = calc_offset_margin(gt_transforms, gps_transforms)
    reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms)
    slam_err_margins = calc_offset_margin(gt_transforms, slam_transforms_aligned)

    # Load fitness and RMSE
    with open(registration_fitness_file, 'r') as f:
        fitness = [float(line.split(' ')[1].strip()) for line in f]
    with open(inlier_rmse_file, 'r') as f:
        inlier_rmse = [float(line.split(' ')[1].strip()) for line in f]

    # Generate plot
    fig = get_split_pose_plot(
        top_pose_sets=[gps_err_margins, slam_err_margins, reg_err_margins],
        # red, cyan-ish, blue-ish
        # top_colors=['red', '#0412b0', '#00a7b3'],
        top_colors=['red', 'blue', 'green'],
        top_labels=['GPS Error (m)', 'SLAM Error (m)', 'Registration Error (m)'],

        bottom_pose_sets=[fitness, inlier_rmse],
        bottom_colors=['orange', 'purple'],
        bottom_labels=['Registration Fitness', 'Registration Inlier RMSE'],

        # min_x=0, max_x=len(gps_err_margins),
        min_x=0, max_x=150,
        top_min_y=-0.5, top_max_y=7,
        bottom_min_y=0.0, bottom_max_y=0.6,
        title=permutation_name,
    )

    # Save
    out_path = os.path.join(FIGURE_DIR, f"{sim_name}_{permutation_name}.svg")
    fig.savefig(out_path, format='svg')
    logging.info(f"✅ Saved plot to {out_path}")


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
