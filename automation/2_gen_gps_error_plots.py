import os
import logging

import config.dataset_structure_parser as dataset_parser
from config.graph_colors import graph_colors
from utils.tum_file_parser import tum_load_as_matrices
from utils.math_utils import calc_offset_margin
from utils.data_viz import get_pose_plot
from utils.tum_file_comparator import show_average_difference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Save format
FORMAT = "pdf"
# How much to scale up font size
FONT_SCALE = 2.0

# Absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "build")
FIGURES_DIR = os.path.join(BUILD_DIR, "figures", "gps_error_plots")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load simulation config
sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)

for sim_name in sim_configs:
    logging.info(f"📈 Processing GPS error plot for {sim_name}...")

    # Paths
    sensor_data_ego_dir = os.path.join(BUILD_DIR, "sim_output", sim_name, "ego_lidar")
    gt_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
    gps_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")

    if not os.path.exists(gt_file) or not os.path.exists(gps_file):
        logging.warning(f"❌ Missing TUM files for {sim_name}. Skipping.")
        continue

    # Optional: print average difference between ground truth and GPS poses
    show_average_difference(gt_file, gps_file)

    gt_transforms = [matrix for ts, matrix in tum_load_as_matrices(gt_file)]
    gps_transforms = [matrix for ts, matrix in tum_load_as_matrices(gps_file)]

    gps_err_margins = calc_offset_margin(gt_transforms, gps_transforms)

    fig = get_pose_plot(
        pose_sets=[gps_err_margins],
        colors=[graph_colors.gps],
        labels=["GPS Error (m)"],
        min_x=0, max_x=len(gps_err_margins) - 1,
        min_y=0.0, max_y=20.0,
        title=f"GPS Error - {sim_name}",
        ylabel="Distance from Ground Truth (m)"
    )

    output_path = os.path.join(FIGURES_DIR, f"gps_error_{sim_name}.{FORMAT}")
    fig.savefig(output_path, format=FORMAT)
    fig.close()

    logging.info(f"✅ Saved GPS error plot to {output_path}")