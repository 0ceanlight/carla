import os
import logging

from utils.tum_file_parser import tum_load_as_matrices
from utils.math_utils import calc_offset_margin
from utils.data_viz import get_split_pose_plot
import config.dataset_structure_parser as dataset_parser
from config.graph_colors import graph_colors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

FORMAT = "pdf"  # or "svg"
FONT_SCALE = 1.05

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "build")
FIGURE_DIR = os.path.join(BUILD_DIR, "figures", "registration_pose_error_plots")
os.makedirs(FIGURE_DIR, exist_ok=True)


def process_simulation(sim_name: str, permutation_name: str):
    logging.info(f"📊 Generating registration-only plot for {sim_name}/{permutation_name}")

    # Paths
    sensor_data_ego_dir = os.path.join(BUILD_DIR, "sim_output", sim_name, "ego_lidar")
    reg_dir = os.path.join(BUILD_DIR, "registered_sim_output", sim_name, permutation_name)

    ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
    registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
    registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
    inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")

    # Load poses
    gt_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_ground_truth_file)]
    reg_transforms = [matrix for _, matrix in tum_load_as_matrices(registration_est_file)]

    # Compute registration error
    reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms)

    # Load fitness and RMSE
    with open(registration_fitness_file, 'r') as f:
        fitness = [float(line.split(' ')[1].strip()) for line in f]
    with open(inlier_rmse_file, 'r') as f:
        inlier_rmse = [float(line.split(' ')[1].strip()) for line in f]

    # Generate plot
    plt = get_split_pose_plot(
        top_pose_sets=[reg_err_margins],
        top_colors=[graph_colors.registration],
        top_labels=['Registration Error (m)'],

        bottom_pose_sets=[fitness, inlier_rmse],
        bottom_colors=[graph_colors.fitness, graph_colors.inlier_rmse],
        bottom_labels=['Registration Fitness', 'Registration Inlier RMSE'],

        min_x=0,
        max_x=min(len(reg_err_margins), 150),
        top_min_y=-0.5,
        top_max_y=80.0,
        bottom_min_y=0.0,
        bottom_max_y=0.7,
        title=permutation_name
    )

    # Save plot
    out_path = os.path.join(FIGURE_DIR, f"{sim_name}_{permutation_name}.{FORMAT}")
    plt.savefig(out_path, format=FORMAT)
    plt.close()
    logging.info(f"✅ Saved registration-only plot to {out_path}")


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

    logging.info("📈 All registration-only plots generated.")


if __name__ == "__main__":
    main()
