import open3d as o3d
import os
import sys
import numpy as np
from natsort import natsorted

class PointCloudViewer:
    """
    An interactive point cloud viewer for visualizing .ply files using Open3D.
    
    Supports:
    - Viewing a single .ply file
    - Browsing through a directory of .ply files with navigation keys
    - Directly displaying an in-memory Open3D point cloud object
    """

    def __init__(self, path=None, direct_pcd=None):
        """
        Initializes the viewer.

        Args:
            path (str, optional): Path to a .ply file or directory of .ply files.
            direct_pcd (open3d.geometry.PointCloud, optional): An in-memory point cloud.
        """
        self.direct_pcd_mode = direct_pcd is not None
        self.single_file_mode = False
        self.quit_flag = False

        if self.direct_pcd_mode:
            # Directly visualize a single point cloud object
            self.pcd = direct_pcd
            self.files = []
        elif path:
            self.single_file_mode = path.endswith(".ply")
            if self.single_file_mode:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"File not found: {path}")
                self.files = [os.path.abspath(path)]
            else:
                # Load all .ply files from directory
                self.folder_path = path
                self.files = natsorted(
                    [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ply')]
                )
                if not self.files:
                    raise ValueError("No .ply files found in the directory.")
            self.index = 0
            self.pcd = None
        else:
            raise ValueError("Either 'path' or 'direct_pcd' must be provided.")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.view_ctl = None
        self.camera_params = None

    @classmethod
    def from_pointcloud(cls, pcd):
        """
        Alternate constructor to create a viewer from a point cloud object.
        
        Args:
            pcd (open3d.geometry.PointCloud): Point cloud to visualize.

        Returns:
            PointCloudViewer: A viewer instance for the given point cloud.
        """
        return cls(direct_pcd=pcd)

    def load_point_cloud(self, index):
        """
        Loads a point cloud from file at a specific index.

        Args:
            index (int): Index of the .ply file in the list.

        Returns:
            open3d.geometry.PointCloud: Loaded point cloud.
        """
        file_path = self.files[index]
        print(f"Loading: {file_path}")
        return o3d.io.read_point_cloud(file_path)

    def save_camera(self):
        """Saves the current camera parameters for reuse."""
        if self.view_ctl:
            self.camera_params = self.view_ctl.convert_to_pinhole_camera_parameters()

    def load_camera(self):
        """Restores the previously saved camera parameters."""
        if self.view_ctl and self.camera_params:
            self.view_ctl.convert_from_pinhole_camera_parameters(self.camera_params)

    def update_geometry(self, index):
        """
        Replaces the current point cloud with a new one from file.

        Args:
            index (int): Index of the new point cloud.
        """
        self.save_camera()
        new_pcd = self.load_point_cloud(index)
        self.vis.clear_geometries()
        self.vis.add_geometry(new_pcd)
        self.pcd = new_pcd
        self.index = index
        self.load_camera()

    def next_cloud(self, vis):
        """Callback: Shows the next point cloud in the list."""
        if self.index < len(self.files) - 1:
            self.update_geometry(self.index + 1)
        return False

    def prev_cloud(self, vis):
        """Callback: Shows the previous point cloud in the list."""
        if self.index > 0:
            self.update_geometry(self.index - 1)
        return False

    def quit_viewer(self, vis):
        """Callback: Quits the viewer window."""
        print("Exiting viewer.")
        self.quit_flag = True
        self.vis.close()
        return False

    def run(self):
        """Runs the interactive Open3D viewer."""
        self.vis.create_window("LiDAR Viewer", 1280, 720)

        # Initial geometry
        if self.direct_pcd_mode:
            self.vis.add_geometry(self.pcd)
        else:
            self.pcd = self.load_point_cloud(self.index)
            self.vis.add_geometry(self.pcd)

        self.view_ctl = self.vis.get_view_control()

        # Viewer style
        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([0, 0, 0])  # Black background

        # Register quit keys
        quit_keys = [ord("Q"), 256]  # ESC = 256
        for k in quit_keys:
            self.vis.register_key_callback(k, self.quit_viewer)

        # Register navigation keys if applicable
        if not self.single_file_mode and not self.direct_pcd_mode:
            next_keys = [ord("N"), ord("L"), 262]  # → = 262
            prev_keys = [ord("P"), ord("J"), 263]  # ← = 263
            for k in next_keys:
                self.vis.register_key_callback(k, self.next_cloud)
            for k in prev_keys:
                self.vis.register_key_callback(k, self.prev_cloud)

            print("Controls:")
            print("  N / L / → : Next point cloud")
            print("  P / J / ← : Previous point cloud")
        elif self.direct_pcd_mode:
            print("Direct point cloud mode: navigation disabled")
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
    viewer = PointCloudViewer(path=path)
    viewer.run()