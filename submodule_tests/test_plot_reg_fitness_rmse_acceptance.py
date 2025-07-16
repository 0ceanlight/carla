import os
import numpy as np
import matplotlib.pyplot as plt
from utils.tum_file_parser import tum_load_as_matrices
from utils.math_utils import calc_offset_margin
from utils.pose_graph_optimization import combine_fitness_rmse_acceptance
from config.graph_colors import graph_colors
import matplotlib as mpl

mpl.rcParams['font.size'] *= 1.35

# === CONFIGURATION ===
# BASE_PATH = "build/registered_sim_output/sim_1/4_infra_2_agent"
# BASE_PATH = "build/registered_sim_output/sim_1/4_infra"
# BASE_PATH = "build/registered_sim_output/sim_1/2_infra"
# BASE_PATH = "build/registered_sim_output/sim_1/2_agent"
# note that 50 and 63 are outliers / graph wrongly here
BASE_PATH = "build/registered_sim_output/sim_1/1_infra"
# BASE_PATH = "build/registered_sim_output/sim_1/1_agent"
GT_FILE = os.path.join(BASE_PATH, "ground_truth_poses_tum.txt")
REG_FILE = os.path.join(BASE_PATH, "reg_est_poses_tum.txt")
FITNESS_FILE = os.path.join(BASE_PATH, "reg_fitness.txt")
RMSE_FILE = os.path.join(BASE_PATH, "reg_inlier_rmse.txt")

# === LOAD DATA ===
gt_transforms = [matrix for _, matrix in tum_load_as_matrices(GT_FILE)]
reg_transforms = [matrix for _, matrix in tum_load_as_matrices(REG_FILE)]

with open(FITNESS_FILE, 'r') as f:
    fitness = [float(line.split()[1]) for line in f]
with open(RMSE_FILE, 'r') as f:
    inlier_rmse = [float(line.split()[1]) for line in f]

# === CALCULATE REGISTRATION ERROR ===
reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms)


# === ACCEPTANCE LOGIC ===
accepted_indices, fitness_env, rmse_env = combine_fitness_rmse_acceptance(
    fitness, inlier_rmse,
    fitness_margin=0.04,
    rmse_margin=0.03,
    weight_fitness=1.0,
    weight_rmse=1.2
)

# === PLOTTING ===
frames = np.arange(len(fitness))
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# -- Plot 1: Fitness --
axs[0].plot(frames, fitness, label="Fitness", color=graph_colors.fitness)
axs[0].plot(frames, fitness_env, label="Smooth Max Envelope", linestyle="--", color="black")
axs[0].scatter(accepted_indices, [fitness[i] for i in accepted_indices], color=graph_colors.dots, label="Accepted", s=20)
axs[0].set_title("Registration Fitness")
axs[0].set_ylabel("Fitness")
axs[0].legend()

# -- Plot 2: RMSE --
axs[1].plot(frames, inlier_rmse, label="Inlier RMSE", color=graph_colors.inlier_rmse)
axs[1].plot(frames, rmse_env, label="Smooth Min Envelope", linestyle="--", color="black")
axs[1].scatter(accepted_indices, [inlier_rmse[i] for i in accepted_indices], color=graph_colors.dots, label="Accepted", s=20)
axs[1].set_title("Inlier RMSE")
axs[1].set_ylabel("RMSE")
axs[1].legend()

# -- Plot 3: Registration Error --
axs[2].plot(frames, reg_err_margins, label="Registration Position Error", color=graph_colors.registration)
axs[2].scatter(accepted_indices, [reg_err_margins[i] for i in accepted_indices], color=graph_colors.dots, label="Accepted", s=20)
axs[2].set_title("Registration Error vs Ground Truth")
axs[2].set_ylabel("Position Error (m)")
axs[2].set_xlabel("Frame")
axs[2].legend()

plt.tight_layout()
plt.show()
