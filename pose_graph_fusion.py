import os
from utils.fusion_pipeline import fuse_slam_gps_files

if __name__ == "__main__":
    DATA_DIR = "build"
    SIMULATION = "sim_0"
    INFRA_VERSION = "6_infra_2_agent" 

    base_sim_path = os.path.join(DATA_DIR, "sim_output", SIMULATION)
    base_reg_path = os.path.join(DATA_DIR, "registered_sim_output", SIMULATION)

    gps_file = os.path.join(base_sim_path, "ego_lidar", "gps_poses_tum.txt")
    slam_file = os.path.join(base_sim_path, "ego_lidar", "ground_truth_poses_tum.txt")
    reg_dir = os.path.join(base_reg_path, INFRA_VERSION)
    reg_file = os.path.join(reg_dir, "reg_est_poses_tum.txt")
    fitness_file = os.path.join(reg_dir, "reg_fitness.txt")
    rmse_file = os.path.join(reg_dir, "reg_inlier_rmse.txt")
    gt_file = os.path.join(base_sim_path, "ego_lidar", "ground_truth_poses_tum.txt")

    out_path = os.path.join(DATA_DIR, f"fused_gps_slam_{SIMULATION}_{INFRA_VERSION}.txt")

    fuse_slam_gps_files(
        gps_file=gps_file,
        slam_file=slam_file,
        output_file=out_path
    )