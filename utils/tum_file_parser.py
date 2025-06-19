import os
from .math_utils import euler_to_quaternion

def load_tum_file(file_path):
    """
    Load a TUM RGB-D trajectory file.

    Args:
        file_path (str): Path to the TUM file.

    Returns:
        list: A list of tuples, where each tuple contains:
              (timestamp, x, y, z, qx, qy, qz, qw)
    """

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Ignore comments and empty lines
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Invalid line format: {line}")
            timestamp, x, y, z, qx, qy, qz, qw = map(float, parts)
            data.append((timestamp, x, y, z, qx, qy, qz, qw))
    return data

def save_tum_file(file_path, data):
    """
    Save data to a TUM RGB-D trajectory file.

    Args:
        file_path (str): Path to the TUM file.
        data (list): A list of tuples, where each tuple contains:
                     (timestamp, x, y, z, qx, qy, qz, qw)
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        for entry in data:
            if len(entry) != 8:
                raise ValueError(f"Invalid entry format: {entry}")
            file.write(f"{entry[0]:.6f} {entry[1]:.6f} {entry[2]:.6f} {entry[3]:.6f} "
                       f"{entry[4]:.6f} {entry[5]:.6f} {entry[6]:.6f} {entry[7]:.6f}\n")

# function to append any number of poses (as optionally long list of tuples)
def append_tum_poses(file_path, *poses):
    """
    Append multiple poses to a TUM RGB-D trajectory file.

    Args:
        file_path (str): Path to the TUM file.
        *poses: A variable number of tuples, where each tuple contains:
                (timestamp, x, y, z, qx, qy, qz, qw)
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        for pose in poses:
            if len(pose) != 8:
                raise ValueError(f"Invalid pose format: {pose}")
            # Write in TUM format: timestamp tx ty tz qx qy qz qw
            file.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} "
                       f"{pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")

def append_right_handed_tum_pose(filename, transform, timestamp):
        # Get the location of the vehicle
        location = transform.location
        rotation = transform.rotation

        # Save the location to a file in TUM format
        # TUM format: timestamp tx ty tz qx qy qz qw
        # where qx, qy, qz, qw are the quaternion components 
        # quaternions will require conversion because carla uses pitch, roll, yaw
        
        # TODO: is -pitch, -yaw conversion correct?
        # ALTERNATIVE: swap qx and qz to account for carla's left handed 
        # coordinate system, leave all else untouched
        qx, qy, qz, qw = euler_to_quaternion(
            rotation.roll,
            -rotation.pitch,
            -rotation.yaw
        )

        x, y, z = location.x, location.y, location.z

        # Save negative y to convert to right-handed coordinate system
        append_tum_poses(filename, (timestamp, x, -y, z, qx, qy, qz, qw))

# Example usage:
# data = load_tum_file("trajectory.txt")
# save_tum_file("output_trajectory.txt", data)
# append_tum_file("output_trajectory.txt", (1234567890.123456, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))