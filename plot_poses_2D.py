import os
from utils.tum_file_parser import tum_load_as_matrices
from utils.math_utils import *
from utils.data_viz import get_split_pose_plot


base_dir = "build.old.log"
base_dir = "build"
sim = "sim_0"
permutation = "6_infra_2_agent"
# permutation = "2_agent"
# permutation = "1_infra"
# permutation = "1_agent"

# Input data directory
sensor_data_ego_dir = os.path.join(base_dir, "sim_output", sim, "ego_lidar")
# Output data/results directory
reg_dir = os.path.join(base_dir, "registered_sim_output", sim, permutation)
slam_dir = os.path.join(base_dir, "slam_output_ego_only", sim)
slam_test_pin = [d for d in os.listdir(slam_dir)][0]


ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
ego_drifted_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")
registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
slam_file = os.path.join(slam_dir, slam_test_pin, "slam_poses_tum.txt")

gt_transforms = [matrix for ts, matrix in tum_load_as_matrices(ego_ground_truth_file)]
gps_transforms = [matrix for ts, matrix in tum_load_as_matrices(ego_drifted_file)]
reg_transforms = [matrix for ts, matrix in tum_load_as_matrices(registration_est_file)]
slam_transforms = [matrix for ts, matrix in tum_load_as_matrices(slam_file)]

# Align SLAM to ground truth coordinate system since SLAM starts at origin
slam_transforms_aligned = align_matrix_list_to_matrix(slam_transforms, gt_transforms[0])

gps_err_margins = calc_offset_margin(gt_transforms, gps_transforms)
reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms)
slam_err_margins = calc_offset_margin(gt_transforms, slam_transforms_aligned)

# Load fitness and RMSE data (stored as 1 float per line in text files)
with open(registration_fitness_file, 'r') as f:
    fitness = [float(line.split(' ')[1].strip()) for line in f]
with open(inlier_rmse_file, 'r') as f:
    inlier_rmse = [float(line.split(' ')[1].strip()) for line in f]

plt = get_split_pose_plot(
    top_pose_sets=[gps_err_margins, reg_err_margins, slam_err_margins],
    top_colors=['red', 'blue', 'black'],
    top_labels=['GPS Error (m)', 'Registration Error (m)', 'SLAM Error (m)'],
    
    bottom_pose_sets=[fitness, inlier_rmse],
    bottom_colors=['orange', 'purple'],
    bottom_labels=['Registration Fitness', 'Registration Inlier RMSE'],

    min_x=0, max_x=200,
    top_min_y=0.0, top_max_y=6,
    bottom_min_y=0.0, bottom_max_y=0.6,

    title=permutation
)

# save as SVG
plt.show()
plt.savefig("pose_plot.svg", format='svg')
