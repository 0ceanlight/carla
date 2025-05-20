import open3d as o3d
import os
import sys
import numpy as np
from natsort import natsorted

class PointCloudViewer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Sort files numerically based on the number in the filename (ignores extensions)
        self.files = natsorted(
            [f for f in os.listdir(folder_path) if f.endswith('.ply')]
        )
        if not self.files:
            raise ValueError("No .ply files found in the directory.")
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = None
        self.view_ctl = None
        self.camera_params = None

    def load_point_cloud(self, index):
        file_path = os.path.join(self.folder_path, self.files[index])
        print(f"Loading: {file_path}")
        return o3d.io.read_point_cloud(file_path)

    def save_camera(self):
        if self.view_ctl:
            self.camera_params = self.view_ctl.convert_to_pinhole_camera_parameters()

    def load_camera(self):
        if self.view_ctl and self.camera_params:
            self.view_ctl.convert_from_pinhole_camera_parameters(self.camera_params)

    def update_geometry(self, index):
        self.save_camera()
        new_pcd = self.load_point_cloud(index)
        self.vis.clear_geometries()
        self.vis.add_geometry(new_pcd)
        self.pcd = new_pcd
        self.index = index
        self.load_camera()

    def next_cloud(self, vis):
        if self.index < len(self.files) - 1:
            self.update_geometry(self.index + 1)
        return False

    def prev_cloud(self, vis):
        if self.index > 0:
            self.update_geometry(self.index - 1)
        return False

    def run(self):
        self.vis.create_window("LiDAR Viewer", 1280, 720)
        self.pcd = self.load_point_cloud(self.index)
        self.vis.add_geometry(self.pcd)
        self.view_ctl = self.vis.get_view_control()

        # Set the background color to black
        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([0, 0, 0])  # RGB black background

        # Register key callbacks
        self.vis.register_key_callback(ord("N"), self.next_cloud)
        self.vis.register_key_callback(ord("P"), self.prev_cloud)

        print("Controls:")
        print("  N - Next point cloud")
        print("  P - Previous point cloud")
        print("  Use mouse and arrows to navigate")

        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lidar_viewer.py /path/to/ply_folder")
        sys.exit(1)

    folder = sys.argv[1]
    viewer = PointCloudViewer(folder)
    viewer.run()

