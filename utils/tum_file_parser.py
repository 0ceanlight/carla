import os
import open3d as o3d
import numpy as np
from .math_utils import euler_to_quaternion, matrix_to_pose, pose_to_matrix

def tum_load_as_tuples(file_path):
    """
    Load a TUM RGB-D trajectory file.

    Args:
        file_path (str): Path to the TUM file.

    Returns:
        list: A list of tuples, where each tuple contains:
              (timestamp, x, y, z, qx, qy, qz, qw)
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
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

def tum_save_tuples(file_path, list_of_tuples):
    """
    Save data to a TUM RGB-D trajectory file. This overwrites any existing file.

    Args:
        file_path (str): Path to the TUM file.
        data (list): A list of tuples, where each tuple contains:
                     (timestamp, x, y, z, qx, qy, qz, qw)
    Raises:
        ValueError: If any entry in data does not have exactly 8 elements.
        IOError: If there is an issue writing to the file.
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        for entry in list_of_tuples:
            if len(entry) != 8:
                raise ValueError(f"Invalid entry format: {entry}")
            file.write(f"{entry[0]:.6f} {entry[1]:.6f} {entry[2]:.6f} {entry[3]:.6f} "
                       f"{entry[4]:.6f} {entry[5]:.6f} {entry[6]:.6f} {entry[7]:.6f}\n")

# function to append any number of poses (as optionally long list of tuples)
def tum_append_tuples(file_path, *poses):
    """
    Append multiple poses to a TUM RGB-D trajectory file. Appends poses to the
    end of the file if it exists, and creates the file if it doesn't exist.

    Args:
        file_path (str): Path to the TUM file.
        *poses: A variable number of tuples, where each tuple contains:
                (timestamp, x, y, z, qx, qy, qz, qw)
    Raises:
        ValueError: If any pose does not have exactly 8 elements.
        IOError: If there is an issue appending to the file.
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

def tum_append_right_handed_carla_transform(filename, transform, timestamp):
    """
    Append a single pose (given in right-handed coordinate system) to a TUM 
    RGB-D trajectory file (which uses left-handed coordinates).

    Args:
        filename (str): Path to the TUM file.
        transform: A CARLA transform object containing location and rotation.
        timestamp (float): Timestamp for the pose.
    Raises:
        IOError: If there is an issue appending to the file.
    """
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
    tum_append_tuples(filename, (timestamp, x, -y, z, qx, qy, qz, qw))

def tum_load_as_matrices(path):
    """
    Load a TUM RGB-D trajectory file and convert it to a list of tuples of 
    timestamps and 4x4 transformation matrices.

    Args:
        path (str): Path to the TUM file.
    Returns:
        list (List[Tuple[float, np.ndarray]]): A list of tuples of timestamps and 4x4 
        numpy arrays representing the poses.
    """
    tum_data = tum_load_as_tuples(path)
    matrix_entries = []
    for entry in tum_data:
        timestamp = entry[0]
        pose = entry[1:]
        matrix = pose_to_matrix(pose)
        matrix_entries.append((timestamp, matrix))

    return matrix_entries


def tum_save_matrices(path, matrix_pose_list):
    """
    Save a list of 4x4 transformation matrices as a TUM-formatted trajectory.

    Args:
        path (str): Output file path
        pose_list (List[Tuple[float, np.ndarray]]): List of tuples containing 
        timestamp and 4x4 numpy arrays.
    """
    tum_data = []
    for timestamp, matrix in matrix_pose_list:
        pose = matrix_to_pose(matrix)
        tum_data.append((timestamp, *pose))

    tum_save_tuples(path, tum_data)
