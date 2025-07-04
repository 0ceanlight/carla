import os
import logging

from utils.tum_file_parser import tum_load_as_matrices
from utils.tum_file_comparator import get_average_difference 
from utils.math_utils import align_matrix_list_to_matrix, matrix_euclidean_distance
from utils.pose_graph_optimization import combine_fitness_rmse_acceptance, remove_outliers
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


# This is so dumb... used to filter for where to start and stop considering 
# registration results, should operate on list of accepted registration indices
def find_consecutive_streaks(arr, N):
    """
    Find the first integer that starts a sequence of N+1 consecutive integers,
    and the last integer that ends a sequence of N+1 consecutive integers.

    Args:
        arr (list of int): List of integers (not necessarily sorted or unique).
        N (int): Number of additional consecutive frames required.

    Returns:
        tuple: (first_start, last_end)
            - first_start: First integer that has N more consecutive integers after it.
            - last_end: Last integer that has N more consecutive integers before it.
    """
    arr_set = set(arr)
    sorted_arr = sorted(arr)

    first_start = None
    last_end = None

    for num in sorted_arr:
        # Check for start of sequence: num, num+1, ..., num+N
        if all((num + i) in arr_set for i in range(N + 1)):
            if first_start is None:
                first_start = num

    for num in reversed(sorted_arr):
        # Check for end of sequence: num-N, ..., num-1, num
        if all((num - i) in arr_set for i in range(N + 1)):
            last_end = num
            break

    return first_start, last_end


def process_simulation(sim_name: str, permutation_name: str):
    # LOAD DATA ----------------------------------------------------------------
    # Paths
    sensor_data_ego_dir = os.path.join(BUILD_DIR, "sim_output", sim_name, "ego_lidar")
    reg_dir = os.path.join(BUILD_DIR, "registered_sim_output", sim_name, permutation_name)
    slam_dir = os.path.join(BUILD_DIR, "slam_output_ego_only", sim_name)
    slam_test_pin = [d for d in os.listdir(slam_dir)][0]
    fused_output_dir = os.path.join(BUILD_DIR, "fused_output", sim_name, permutation_name)
    os.makedirs(fused_output_dir, exist_ok=True)

    ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
    ego_drifted_gps_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")
    registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
    registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
    inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
    slam_file = os.path.join(slam_dir, slam_test_pin, "slam_poses_tum.txt")
    fused_output_file = os.path.join(fused_output_dir, "fused_poses_tum.txt")

    # # Load pose data
    gt_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_ground_truth_file)]
    # gps_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_drifted_gps_file)]
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

    
    # COMPUTE N PRINT ERRORS ---------------------------------------------------
    accepted_indices, _, _ = combine_fitness_rmse_acceptance(
        fitness, inlier_rmse,
        fitness_margin=0.04,
        rmse_margin=0.03,
        weight_fitness=1.0,
        weight_rmse=1.2
    )

    accepted_indices_filtered = remove_outliers(accepted_indices, reg_transforms, max_offset_m=0.5)

    # in the accepted_indices_filtered array, find the first and last indices with 2 consecutive numbers
    valid_reg_frame_start, valid_reg_frame_end = find_consecutive_streaks(accepted_indices_filtered, 2)

    if valid_reg_frame_start is None or valid_reg_frame_end is None:
        logging.warning(f"No valid registration frames found for {sim_name}/{permutation_name}. Skipping error calculation.")
        print(accepted_indices_filtered)
        return

    # SLAM ERROR
    # GPS ERROR
    # REGISTRATION ERROR
    # REGISTRATION ERROR FOR VALID REGISTRATION FRAMES
    # FUSED ERROR
    # FUSED ERROR FOR VALID REGISTRAION FRAMES

    # TODO: this is just too high... is it being aligned properly???
    # slam_translation_error, _ = get_average_difference(ego_ground_truth_file, slam_file)
    # average over all matrix euclidean distances between the aligned SLAM transforms and the ground truth
    nframes = len(gt_transforms)
    slam_translation_error = sum(
        matrix_euclidean_distance(slam_transforms_aligned[i], gt_transforms[i])
        for i in range(nframes)
    ) / nframes


    gps_translation_error, _ = get_average_difference(ego_ground_truth_file, ego_drifted_gps_file)
    registration_translation_error, _ = get_average_difference(ego_ground_truth_file, registration_est_file)
    registration_translation_error_cropped, _ = get_average_difference(ego_ground_truth_file, registration_est_file, start= valid_reg_frame_start, end=valid_reg_frame_end)
    fused_translation_error, _ = get_average_difference(ego_ground_truth_file, fused_output_file)
    fused_translation_error_cropped, _ = get_average_difference(ego_ground_truth_file, fused_output_file, start=valid_reg_frame_start, end=valid_reg_frame_end)
    
    # Log all errors, including sim name and permutation name in each message
    print(f"Simulation: {sim_name}, Permutation: {permutation_name}\t"
          f"SLAM Error: {slam_translation_error:.3f} m, "
          f"GPS Error: {gps_translation_error:.3f} m, "
          f"Registration Error: {registration_translation_error:.3f} m, "
          f"Registration Error (Valid Frames): {registration_translation_error_cropped:.3f} m, "
          f"Fused Error: {fused_translation_error:.3f} m, "
          f"Fused Error (Valid Frames): {fused_translation_error_cropped:.3f} m")


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

    logging.info("📈 All data generated.")


if __name__ == "__main__":
    main()
