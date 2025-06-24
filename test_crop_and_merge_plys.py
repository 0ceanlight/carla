import os
from utils.crop_plys import remove_points_below_z_threshold

def crop_all_plys(input_dir, output_dir, z_threshold=-1.62):
    """
    Crop all PLY files in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing the input PLY files.
        output_dir (str): Directory to save the cropped PLY files.
        z_threshold (float): Z threshold below which points will be removed.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.ply'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            remove_points_below_z_threshold(input_file, output_file, z_threshold)
            print(f"Cropped {filename} and saved to {output_file}")

if __name__ == "__main__":
    input_data_dir = "output_v1.log/sensor_captures_v3"
    output_data_dir = "output_v1.log/sensor_captures_v3_cropped"

    sensors = [("ego_lidar", -1.2), ("infrastruct_lidar", -7.3)]

    for sensor, z_threshold in sensors:
        input_dir = os.path.join(input_data_dir, sensor, "frames")
        output_dir = os.path.join(output_data_dir, sensor, "frames")
        crop_all_plys(input_dir, output_dir, z_threshold=z_threshold)