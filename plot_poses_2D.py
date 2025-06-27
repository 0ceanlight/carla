import os
import matplotlib.pyplot as plt
import mplcursors
from mpl_toolkits.mplot3d import Axes3D
from utils.tum_file_parser import load_tum_file
from utils.math_utils import *

# Prompt: Implement the following function header. Flesh out the function to 
# display a 2D graph of the input data, plotting the ith element as a point
# (x, y) = (i, data_value: float). Additionally, color the points of each list
# with the respective given colors, and add a key, labeled with the respective 
# labels. Ideally, the graph should support hovering to view the y values at the
# x value being hovered over (with numbers displayed in respective colors).
# Update the docstring so it's nicely descriptive and formatted correctly.
def plot_poses(pose_sets, colors, labels):
    """
    Plot multiple sequences of float values as 2D scatter plots.

    Each set of values in `pose_sets` is plotted such that each element is a point
    (x, y) = (i, value), where i is the index of the value in the list.

    Parameters:
    - pose_sets (list of list of float): A list where each inner list is a sequence
      of float values to be plotted.
    - colors (list of str): A list of color names or codes for each set.
    - labels (list of str): A list of labels for each dataset to be used in the legend.

    Features:
    - Each dataset is displayed in its respective color.
    - A legend (key) is shown using the provided labels.
    - Hovering over any point displays its (x, y) coordinates, with annotation colored
      to match the data point.
    """

    assert len(pose_sets) == len(colors) == len(labels), \
        "Input arrays must be of the same length."

    plt.figure(figsize=(10, 6))

    for pose_set, color, label in zip(pose_sets, colors, labels):
        x = list(range(len(pose_set)))
        y = pose_set
        scatter = plt.scatter(x, y, color=color, label=label)
        # plt.plot(x, y, color=color, linewidth=1.0, alpha=0.7)  # adjust line

        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", lambda sel, c=color: (
            sel.annotation.set_text(f"x={int(sel.target[0])}, y={sel.target[1]:.3f}"),
            sel.annotation.get_bbox_patch().set(fc=c, alpha=0.8)
        ))

    plt.title("Pose Sets | Note: Cropped to max 3.0")
    plt.xlabel("Frame Index")
    plt.ylabel("Distance from ground truth (m) / Fitness / RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calc_offset_margin(transform_arr_1, transform_arr_2, weight=1.0, max=None):
    """
    Calculate offset of each entry in array...

    Parameters:
        2 lists containing 4x4 np.ndarrays as entries
    Returns:
        float
    """
    
    ret = []

    for i, j in zip(transform_arr_1, transform_arr_2):
        trans, rot = pose_difference(i, j)
        # TODO: consider degree diff separately?
        # err = (rot + trans) * weight
        err = trans * weight

        if max is not None and err > max:
            err = max

        ret.append(err)

    return ret


# Input data directory
sensor_data_dir = "build/sim_output/sim_4"
# Output data/results directory
out_dir = "test_registration_GPS_drifted_data_sim_4.log"

ego_ground_truth_file = os.path.join(sensor_data_dir, "ego_lidar/ground_truth_poses_tum.txt")
ego_drifted_file = os.path.join(out_dir, "ego_drifted_tum.txt")
merged_frames_dir = os.path.join(out_dir, "merged_frames")
registration_est_file = os.path.join(out_dir, "registration_est_tum.txt")
registration_fitness_file = os.path.join(out_dir, "registration_fit.txt")
inlier_rmse_file = os.path.join(out_dir, "registration_inlier_rmse.txt")

# TODO: Add functionality to tum file parser to load transforms?
gt_poses = load_tum_file(ego_ground_truth_file)
gps_poses = load_tum_file(ego_drifted_file)
reg_poses = load_tum_file(registration_est_file)

# [1:] omits timestamp
gt_transforms = [pose_to_matrix(pose[1:]) for pose in gt_poses][:56]
gps_transforms = [pose_to_matrix(pose[1:]) for pose in gps_poses][:56]
reg_transforms = [pose_to_matrix(pose[1:]) for pose in reg_poses][:56]

max = 3.0 # Maximum error margin for better visualization

gps_err_margins = calc_offset_margin(gt_transforms, gps_transforms, max=max)
reg_err_margins = calc_offset_margin(gt_transforms, reg_transforms, max=max)

# Load fitness and RMSE data (stored as 1 float per line in text files)
with open(registration_fitness_file, 'r') as f:
    fitness = [float(line.strip()) for line in f][:56]
with open(inlier_rmse_file, 'r') as f:
    inlier_rmse = [float(line.strip()) for line in f][:56]

plot_poses(
    [gps_err_margins, reg_err_margins, fitness, inlier_rmse],
    ['red', 'blue', 'orange', 'purple'],
    ['GPS Error (m)', 'Registration Error (m)', 
     'Registration Fitness', 'Registration Inlier RMSE']
)
