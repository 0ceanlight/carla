import open3d as o3d
import numpy as np
from pycpd import RigidRegistration

# === Load the PLY point clouds ===
def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

source_file = "run.log/sensor_captures_v3/ego_lidar/frames/10969.ply"
target_file = "run.log/sensor_captures_v3/infrastruct_lidar/frames/10969.ply"

source_points = load_point_cloud(source_file)
target_points = load_point_cloud(target_file)

# Optional: Downsample for speed (keep this if your PLYs are large)
source_points = source_points[::5]
target_points = target_points[::5]

# === Run Rigid CPD Registration ===
reg = RigidRegistration(X=target_points, Y=source_points)

# Callback to monitor progress (optional)
def visualize(iteration, error, X, Y):
    print(f"Iteration {iteration}, Error: {error:.6f}")

TY, (s, R, t) = reg.register(callback=visualize)

# === Visualize the result ===
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(source_points)

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(target_points)

aligned_pcd = o3d.geometry.PointCloud()
aligned_pcd.points = o3d.utility.Vector3dVector(TY)

# Paint colors for visual distinction
source_pcd.paint_uniform_color([1, 0, 0])    # red = original source
target_pcd.paint_uniform_color([0, 1, 0])    # green = target
aligned_pcd.paint_uniform_color([0, 0, 1])   # blue = aligned result

# Show them together
o3d.visualization.draw_geometries([target_pcd, aligned_pcd])






# TODO: this 
# - algorithm is very slow
# - stops after 100 iterations
# - has LiDAR issue (merges rings)
