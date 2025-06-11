import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.tum_file_parser import load_tum_file

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_poses(pose_sets, colors, labels):
    """
    Plot multiple sets of 3D poses.

    Parameters:
    - pose_sets: list of lists of poses [(x, y, z, qx, qy, qz, qw)]
    - colors: list of color strings for each set
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for poses, color, label in zip(pose_sets, colors, labels):
        xs = [pose[0] for pose in poses]
        ys = [pose[1] for pose in poses]
        zs = [pose[2] for pose in poses]
        ax.plot(xs, ys, zs, color=color, label=label)
        ax.scatter(xs, ys, zs, color=color, s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Trajectories')
    plt.tight_layout()
    plt.show()

# Input data directory
sensor_data_dir = "output_v1.log/sensor_captures_v3"
# Output data/results directory
out_dir = "test_registration_GPS_drifted_data.log"

ego_ground_truth_file = os.path.join(sensor_data_dir, "ego_lidar/ground_truth_poses_tum.txt")
ego_drifted_file = os.path.join(out_dir, "ego_drifted_tum.txt")
merged_frames_dir = os.path.join(out_dir, "merged_frames")
registration_est_file = os.path.join(out_dir, "registration_est_tum.txt")
registration_fitness_file = os.path.join(out_dir, "registration_fit.txt")
inlier_rmse_file = os.path.join(out_dir, "registration_inlier_rmse.txt")

# Example usage
gt_set = load_tum_file(ego_ground_truth_file)
gps_set = load_tum_file(ego_drifted_file)
reg_set = load_tum_file(registration_est_file)

# Load fitness and RMSE data (stored as 1 float per line in text files)
with open(registration_fitness_file, 'r') as f:
    fitness = [float(line.strip()) for line in f]
with open(inlier_rmse_file, 'r') as f:
    inlier_rmse = [float(line.strip()) for line in f]

# Add these to ground truth set for visualization
fit_set = [(pose[0], pose[1], pose[2] + fit * 20, 0, 0, 0, 1) for pose, fit in zip(gt_set, fitness)]
rmse_set = [(pose[0], pose[1], pose[2] + rmse * 20, 0, 0, 0, 1) for pose, rmse in zip(gt_set, inlier_rmse)]

plot_poses(
    [gps_set, gt_set, reg_set, fit_set, rmse_set],
    ['red', 'green', 'blue', 'orange', 'purple'],
    ['GPS Drifted', 'Ground Truth', 'Registration Estimate', 
     'Registration Fitness', 'Registration Inlier RMSE']
)
