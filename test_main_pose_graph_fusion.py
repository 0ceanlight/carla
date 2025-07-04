# main script
# main.py

import os
import matplotlib.pyplot as plt
from utils.tum_file_parser import tum_load_as_matrices, tum_save_matrices
from utils.tum_file_comparator import show_average_difference
from utils.math_utils import calc_offset_margin, align_matrix_list_to_matrix
from utils.pose_graph_optimization import optimize_pose_graph, combine_fitness_rmse_acceptance, remove_outliers_accepted_reg_indices
from config.graph_colors import graph_colors

# === CONFIG ===
base_dir = "build"
sim = "sim_0"
# permutation = "4_infra_2_agent"
permutation = "4_infra"
permutation = "2_infra"
# permutation = "1_agent"
# permutation = "1_infra"

sensor_data_ego_dir = os.path.join(base_dir, "sim_output", sim, "ego_lidar")
reg_dir = os.path.join(base_dir, "registered_sim_output", sim, permutation)
slam_dir = os.path.join(base_dir, "slam_output_ego_only", sim)
slam_test_pin = [d for d in os.listdir(slam_dir)][0]

ego_ground_truth_file = os.path.join(sensor_data_ego_dir, "ground_truth_poses_tum.txt")
ego_drifted_gps_file = os.path.join(sensor_data_ego_dir, "gps_poses_tum.txt")
registration_est_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
registration_fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
inlier_rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
slam_file = os.path.join(slam_dir, slam_test_pin, "slam_poses_tum.txt")

# === LOAD TRANSFORMS ===
gt_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_ground_truth_file)]
gps_transforms = [matrix for _, matrix in tum_load_as_matrices(ego_drifted_gps_file)]
reg_transforms = [matrix for _, matrix in tum_load_as_matrices(registration_est_file)]
slam_transforms = [matrix for _, matrix in tum_load_as_matrices(slam_file)]

# === LOAD METRICS ===
with open(registration_fitness_file, 'r') as f:
    fitness = [float(line.split(' ')[1].strip()) for line in f]
with open(inlier_rmse_file, 'r') as f:
    inlier_rmse = [float(line.split(' ')[1].strip()) for line in f]

# === FILTER REGISTRATION ===
accepted_indices, _, _ = combine_fitness_rmse_acceptance(
    fitness, inlier_rmse,
    fitness_margin=0.04,
    rmse_margin=0.03,
    weight_fitness=1.0,
    weight_rmse=1.2
)

accepted_indices = remove_outliers_accepted_reg_indices(accepted_indices, reg_transforms, max_offset_m=0.5)

print(f"Accepted indices for registration: {accepted_indices}")

# === FUSE SLAM + REGISTRATION ===

# ALIGN to ground truth / GPS / WORLD SYSTEM
slam_transforms_world_sys = align_matrix_list_to_matrix(slam_transforms, gt_transforms[0])
# slam_transforms_world_sys = align_matrix_list_to_matrix(slam_transforms, gps_transforms[0])
fused_transforms = optimize_pose_graph(slam_transforms_world_sys, reg_transforms, accepted_indices)

# === SAVE FINAL RESULTS ===
# pull timestamps from ground truth poses
gt_timestamps = [i for i, _ in tum_load_as_matrices(ego_ground_truth_file)]

fused_output_data = [(ts, tr) for ts, tr in zip(gt_timestamps, fused_transforms)]
fused_output_dir = os.path.join(base_dir, "fused_output", sim, permutation)
os.makedirs(fused_output_dir, exist_ok=True)
fused_output_file = os.path.join(fused_output_dir, "fused_poses_tum.txt")
tum_save_matrices(fused_output_file, fused_output_data)

# === CMP ===
show_average_difference(fused_output_file, ego_ground_truth_file)

# === COMPUTE ERRORS ===
gps_err_margins = calc_offset_margin(gt_transforms, gps_transforms)
reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms)
slam_err_margins = calc_offset_margin(gt_transforms, slam_transforms_world_sys)
fused_err_margins = calc_offset_margin(gt_transforms, fused_transforms)

# === PLOT ===
frames = list(range(len(gt_transforms)))
plt.figure(figsize=(10, 6))
# plt.plot(frames, gps_err_margins, label="GPS Error", linestyle="dotted", color="gray")
# plt.plot(frames, reg_err_margins, label="Registration Error", color=graph_colors.registration)
plt.plot(frames, slam_err_margins, label="SLAM Error (Aligned)", color=graph_colors.slam)
plt.plot(frames, fused_err_margins, label="SLAM/Reg (Optimized) Error", color=graph_colors.final_results)
plt.scatter(accepted_indices, [reg_err_margins[i] for i in accepted_indices],
            label="Accepted Reg Frames", color=graph_colors.dots, s=30, marker='o')
plt.xlabel("Frame")
plt.ylabel("Positional Error (m)")
plt.title("Pose Estimation Errors after Trajectory Optimization")
plt.grid(True)
plt.legend()
plt.tight_layout()

max_slam_val = max(slam_err_margins)
plt.ylim(bottom=0, top=max_slam_val)  # Set upper limit for better visibility
plt.xlim(left=0, right=len(gt_transforms) - 1)  # Set x-axis limits to match frames

plt.show()
