import os
import sys
import numpy as np
import open3d as o3d
from natsort import natsorted

class DirectoryCropViewer:
    """
    View and interactively crop point clouds from a single file or an entire directory using Z-thresholding.
    Use keyboard to browse point clouds (if directory) and adjust Z-threshold dynamically.
    """

    def __init__(self, path, initial_threshold=-1.0, step=0.1):
        self.is_dir = os.path.isdir(path)
        self.files = [path]
        self.folder_path = os.path.dirname(path) if not self.is_dir else path

        if self.is_dir:
            self.files = natsorted([os.path.join(self.folder_path, f)
                                    for f in os.listdir(self.folder_path) if f.endswith('.ply')])
            if not self.files:
                raise ValueError("No .ply files found in the directory.")

        self.index = 0
        self.original_pcd = o3d.io.read_point_cloud(self.files[self.index])
        self.threshold = initial_threshold
        self.step = step
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.view_ctl = None
        self.camera_params = None
        self.pcd = None

    def crop_point_cloud(self, pcd, z_thresh):
        points = np.asarray(pcd.points)
        mask = points[:, 2] >= z_thresh
        return pcd.select_by_index(np.where(mask)[0])

    def update_view(self):
        self.vis.clear_geometries()
        cropped_pcd = self.crop_point_cloud(self.original_pcd, self.threshold)
        self.vis.add_geometry(cropped_pcd)
        self.pcd = cropped_pcd
        self.load_camera()

    def save_camera(self):
        if self.view_ctl:
            self.camera_params = self.view_ctl.convert_to_pinhole_camera_parameters()

    def load_camera(self):
        if self.view_ctl and self.camera_params:
            self.view_ctl.convert_from_pinhole_camera_parameters(self.camera_params)

    def increase_threshold(self, vis):
        self.save_camera()
        self.threshold += self.step
        print(f"Z threshold increased to: {self.threshold:.3f}")
        self.update_view()
        return False

    def decrease_threshold(self, vis):
        self.save_camera()
        self.threshold -= self.step
        print(f"Z threshold decreased to: {self.threshold:.3f}")
        self.update_view()
        return False

    def next_cloud(self, vis):
        if self.is_dir and self.index < len(self.files) - 1:
            self.save_camera()
            self.index += 1
            self.original_pcd = o3d.io.read_point_cloud(self.files[self.index])
            print(f"Loading: {self.files[self.index]}")
            self.update_view()
        return False

    def prev_cloud(self, vis):
        if self.is_dir and self.index > 0:
            self.save_camera()
            self.index -= 1
            self.original_pcd = o3d.io.read_point_cloud(self.files[self.index])
            print(f"Loading: {self.files[self.index]}")
            self.update_view()
        return False

    def quit_viewer(self, vis):
        print("Exiting viewer.")
        self.vis.close()
        return False

    def run(self):
        self.vis.create_window("Z-Crop Point Cloud Viewer", 1280, 720)
        self.pcd = self.crop_point_cloud(self.original_pcd, self.threshold)
        self.vis.add_geometry(self.pcd)
        self.view_ctl = self.vis.get_view_control()

        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([0, 0, 0])

        # Register common keys
        z_up     = [ord("I"), 265]             # ↑ Arrow
        z_down   = [ord("K"), 264]             # ↓ Arrow
        quit_keys = [ord("Q"), 256]            # Esc

        for k in z_up:
            self.vis.register_key_callback(k, self.increase_threshold)
        for k in z_down:
            self.vis.register_key_callback(k, self.decrease_threshold)
        for k in quit_keys:
            self.vis.register_key_callback(k, self.quit_viewer)

        # Navigation keys only if directory
        if self.is_dir:
            nav_next = [ord("N"), ord("L"), 262]  # → Arrow
            nav_prev = [ord("P"), ord("J"), 263]  # ← Arrow
            for k in nav_next:
                self.vis.register_key_callback(k, self.next_cloud)
            for k in nav_prev:
                self.vis.register_key_callback(k, self.prev_cloud)

        print("Controls:")
        if self.is_dir:
            print("  N / L / → : Next point cloud")
            print("  P / J / ← : Previous point cloud")
        print("  I / ↑     : Increase Z threshold")
        print("  K / ↓     : Decrease Z threshold")
        print("  Q / Esc   : Quit")
        print(f"Starting at Z threshold = {self.threshold:.3f}")

        self.vis.run()
        self.vis.destroy_window()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python interactive_directory_crop_viewer.py <path_to_ply_or_directory>")
        sys.exit(1)

    viewer = DirectoryCropViewer(sys.argv[1])
    viewer.run()
