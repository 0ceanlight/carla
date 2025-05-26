import open3d as o3d
import os
import sys
import numpy as np
from natsort import natsorted

class PointCloudViewer:
    def __init__(self, path):
        self.single_file_mode = path.endswith(".ply")
        if self.single_file_mode:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            self.files = [os.path.abspath(path)]
        else:
            self.folder_path = path
            self.files = natsorted(
                [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ply')]
            )
            if not self.files:
                raise ValueError("No .ply files found in the directory.")
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = None
        self.view_ctl = None
        self.camera_params = None
        self.quit_flag = False

    def load_point_cloud(self, index):
        file_path = self.files[index]
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

    def quit_viewer(self, vis):
        print("Exiting viewer.")
        self.quit_flag = True
        self.vis.close()
        return False

    def run(self):
        self.vis.create_window("LiDAR Viewer", 1280, 720)
        self.pcd = self.load_point_cloud(self.index)
        self.vis.add_geometry(self.pcd)
        self.view_ctl = self.vis.get_view_control()

        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([0, 0, 0])  # Black background

        # Always allow quitting
        quit_keys = [ord("Q"), 256]  # 'q' and ESC
        for k in quit_keys:
            self.vis.register_key_callback(k, self.quit_viewer)

        # If not single-file mode, register navigation keys
        if not self.single_file_mode:
            next_keys = [ord("N"), ord("L"), 262]  # Right arrow = 262
            prev_keys = [ord("P"), ord("J"), 263]  # Left arrow = 263
            for k in next_keys:
                self.vis.register_key_callback(k, self.next_cloud)
            for k in prev_keys:
                self.vis.register_key_callback(k, self.prev_cloud)

            print("Controls:")
            print("  N / L / → : Next point cloud")
            print("  P / J / ← : Previous point cloud")
        else:
            print("Single file mode: navigation disabled")

        print("  Q / Esc  : Quit")

        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lidar_viewer.py /path/to/ply_folder_or_file.ply")
        sys.exit(1)

    path = sys.argv[1]
    viewer = PointCloudViewer(path)
    viewer.run()
