import os
import bisect
from typing import List, Tuple, Dict, Optional
from natsort import natsorted
import numpy as np
import logging
from scipy.spatial.transform import Rotation
from .tum_file_parser import load_tum_file
from .quaternion_utils import quaternion_inverse, quaternion_multiply
from .merge_plys import combine_point_clouds_with_poses

# TODO: for debugging only -----------------------------------------------------
import random
import colorsys


def random_bright_color():
    """Generates a random bright RGB color."""
    h = random.random()
    s = 1.0
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# end for debugging only -------------------------------------------------------


class SensorDataMerger:
    """
    A class to manage and synchronize LIDAR sensor data and their associated TUM 
    pose files.

    If you get a ValueError during initialization, it means that the number of 
    TUM entries does not match the number of .ply files for one of the sensors. 
    This is a critical error that needs to be resolved before proceeding. Ensure 
    that each sensor's TUM file and corresponding .ply files are correctly 
    aligned and complete. 
    
    If you encounter errors during merging or extracting matches, it may be due 
    to missing or misaligned data. Make sure that your sensors have 
    corresponding entries within the set timestamp discrepancy. This should be 
    avoidable if the sensors are synchronized correctly.

    Attributes:
        base_dir (str): Root directory containing sensor data.
        sensors (List[str]): List of sensor names. The first is assumed to be the reference (ego) sensor.
        max_timestamp_discrepancy (float): Maximum allowed timestamp difference in seconds for matching.
        sensor_data (Dict[str, List[Tuple]]): Parsed TUM entries for each sensor.
        sensor_filenames (Dict[str, List[str]]): Corresponding sorted .ply filenames for each sensor.
        matched_frames (List[List[Optional[Tuple[str, Tuple]]]]): Matched frames across sensors.
            The outer list contains entries for each ego frame.
            The inner list contains that frame's matches for each other sensor.
            Each match is a tuple of (filename, timestamp, pose) or None if no match was found.
    """

    def __init__(self,
                 base_dir: str,
                 sensors: List[str],
                 max_timestamp_discrepancy: float = 1.0):
        """
        Initialize the SensorDataMerger.

        Args:
            base_dir (str): Path to the base directory containing sensor folders.
            sensors (List[str]): List of sensor names (first one is ego sensor).
            max_timestamp_discrepancy (float): Max allowed time difference for matching frames.
        """
        self.base_dir = base_dir
        self.ego_sensor = sensors[0]
        self.other_sensors = sensors[1:]
        self.max_timestamp_discrepancy = max_timestamp_discrepancy

        self.sensor_data = {}  # sensor -> list of TUM tuples
        self.sensor_filenames = {}  # sensor -> list of .ply filenames

        self._load_all_data()
        self.matched_frames = self._match_frames()

    def _load_all_data(self):
        """
        Loads TUM files and .ply filenames for all sensors.
        Ensures correspondence between entries and frames.
        """
        for sensor in [self.ego_sensor] + self.other_sensors:
            sensor_path = os.path.join(self.base_dir, sensor)
            tum_path = os.path.join(sensor_path, 'ground_truth_poses_tum.txt')
            frames_path = os.path.join(sensor_path, 'lidar_frames')

            tum_data = load_tum_file(tum_path)
            filenames = natsorted(
                [f for f in os.listdir(frames_path) if f.endswith('.ply')],
                key=lambda x: int(x.split('.')[0]))
            full_paths = [os.path.join(frames_path, f) for f in filenames]

            if len(tum_data) != len(filenames):
                raise ValueError(
                    f"Mismatch between TUM entries and .ply files for sensor {sensor}. There are {len(tum_data)} TUM entries and {len(filenames)} .ply files."
                )

            self.sensor_data[sensor] = tum_data
            self.sensor_filenames[sensor] = full_paths

    def _match_frames(self) -> List[List[Optional[Tuple[str, float, Tuple]]]]:
        """
        Matches each ego frame to the closest frames from other sensors based on timestamp.

        Returns:
            List of lists, each containing tuples of (filename, timestamp, TUM entry) or None if no match found.
        """
        matches = []
        ego_timestamps = [
            entry[0] for entry in self.sensor_data[self.ego_sensor]
        ]

        for idx, ego_entry in enumerate(self.sensor_data[self.ego_sensor]):
            frame_matches = []
            ego_filename = self.sensor_filenames[self.ego_sensor][idx]
            frame_matches.append(
                (ego_filename, ego_entry[0], tuple(ego_entry[1:])))

            for sensor in self.other_sensors:
                timestamps = [entry[0] for entry in self.sensor_data[sensor]]
                closest_idx = self._find_closest_index(timestamps,
                                                       ego_entry[0])
                if closest_idx is not None:
                    matched_filename = self.sensor_filenames[sensor][
                        closest_idx]
                    matched_entry = self.sensor_data[sensor][closest_idx]
                    frame_matches.append((matched_filename, matched_entry[0],
                                          tuple(matched_entry[1:])))
                else:
                    frame_matches.append(None)
                    logging.warning(
                        f"No match found for ego frame {idx} with sensor {sensor}. Timestamp: {ego_entry[0]}"
                    )
            matches.append(frame_matches)

        return matches

    def _find_closest_index(self, sorted_timestamps: List[float],
                            target: float) -> Optional[int]:
        """
        Finds the index of the timestamp closest to the target in a sorted list.

        Args:
            sorted_timestamps (List[float]): Sorted list of timestamps.
            target (float): Timestamp to match.

        Returns:
            Optional[int]: Index of closest timestamp or None if no match within max discrepancy.
        """
        idx = bisect.bisect_left(sorted_timestamps, target)
        candidates = []
        if idx < len(sorted_timestamps):
            candidates.append((abs(sorted_timestamps[idx] - target), idx))
        if idx > 0:
            candidates.append(
                (abs(sorted_timestamps[idx - 1] - target), idx - 1))

        if not candidates:
            return None

        best_diff, best_idx = min(candidates)
        return best_idx if best_diff <= self.max_timestamp_discrepancy else None

    def count_fully_matched_ego_frames(self) -> int:
        """
        Counts how many ego frames have matches from all sensors.

        Returns:
            int: Number of fully matched ego frames.
        """
        return sum(1 for match in self.matched_frames if None not in match)

    def count_partially_or_unmatched_ego_frames(self) -> int:
        """
        Counts how many ego frames are missing at least one sensor match.

        Returns:
            int: Number of partially or fully unmatched ego frames.
        """
        return sum(1 for match in self.matched_frames if None in match)

    def get_unmatched_indices(self) -> List[int]:
        """
        Returns indices of ego frames with missing sensor matches.

        Returns:
            List[int]: List of indices with incomplete matches.
        """
        return [
            i for i, match in enumerate(self.matched_frames) if None in match
        ]

    def get_absolute_match_for_ego_index(
            self,
            index: int) -> Optional[List[Optional[Tuple[str, float, Tuple]]]]:
        """
        Retrieves the match data for a specific ego frame.

        Args:
            index (int): Index of the ego frame.

        Returns:
            Optional[List[Optional[Tuple[str, float, Tuple]]]]: List of matches from 
                each other sensor. Each match is a tuple of 
                (filename, timestamp, pose) or None if no match was found.
        """
        if 0 <= index < len(self.matched_frames):
            frame = self.matched_frames[index]
            if frame is None:
                logging.warning(f"No matches found for ego frame {index}.")
                return None

            return frame

        logging.error(f"Index {index} out of range for matched frames.")
        return None

    def get_all_matches(
            self) -> List[List[Optional[Tuple[str, float, Tuple]]]]:
        """
        Returns all matched frame data.

        Returns:
            List[List[Optional[Tuple[str, float, Tuple]]]]: All match data.
                The outer list contains entries for each ego frame.
                The inner list contains that frame's matches for each other sensor.
                Each match is a tuple of (filename, timestamp, pose) or None if no match was found.
        """
        return self.matched_frames

    def get_relative_match_for_ego_index(
            self,
            index: int) -> Optional[List[Optional[Tuple[str, float, Tuple]]]]:
        """
        Retrieves the relative pose match data for a specific ego frame.
        Ego sensor is at origin: (0, 0, 0, 0, 0, 0, 1)
        
        Args:
            index (int): Index of the ego frame.

        Returns:
            Optional[List[Optional[Tuple[str, float, Tuple]]]]: List of matches 
                from each other sensor. Each match is a tuple of 
                (filename, timestamp, pose) or None if no match was found.
        """
        if 0 <= index < len(self.matched_frames):
            frame = self.matched_frames[index]
            if frame is None:
                logging.warning(f"No matches found for ego frame {index}.")
                return None
            ego_filename, ego_timestamp, ego_pose = frame[0]

            # first 3 are translation (x, y, z), last 4 are quaternion (qx, qy, qz, qw)
            ego_translation = ego_pose[:3]
            ego_quat = ego_pose[3:]

            rel_frame = [(ego_filename, ego_timestamp, (0.0, 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 1.0))]

            for match in frame[1:]:
                if match is None:
                    rel_frame.append(None)
                else:
                    match_filename, match_timestamp, match_pose = match
                    match_translation = match_pose[:3]
                    match_quat = match_pose[3:]

                    delta = np.array(match_translation) - np.array(
                        ego_translation)
                    R_ego_inv = Rotation.from_quat(ego_quat).inv()
                    rel_translation = tuple(R_ego_inv.apply(delta))
                    rel_rotation = quaternion_multiply(
                        quaternion_inverse(ego_quat), match_quat)

                    rel_tum_entry = rel_translation + rel_rotation
                    rel_frame.append(
                        (match_filename, match_timestamp, rel_tum_entry))
                    print(f"rel: \t{rel_tum_entry}")
            return rel_frame
        logging.error(f"Index {index} out of range for matched frames.")
        return None

    def get_all_relative_matches(
            self) -> List[List[Optional[Tuple[str, float, Tuple]]]]:
        """
        Returns all matched frame data with poses relative to ego sensor origin.

        Returns:
            List[List[Optional[Tuple[str, float, Tuple]]]]: All relative match data.
                The outer list contains entries for each ego frame.
                The inner list contains that frame's matches for each other sensor.
                Each match is a tuple of (filename, timestamp, pose) or None if no match was found.
        """
        return [
            self.get_relative_match_for_ego_index(i)
            for i in range(len(self.matched_frames))
        ]

    def print_summary(self):
        """
        Prints a summary of matching statistics.
        """
        total = len(self.matched_frames)
        matched = self.count_fully_matched_ego_frames()
        unmatched = self.count_partially_or_unmatched_ego_frames()
        print(f"Total ego frames: {total}")
        print(f"Fully matched ego frames: {matched}")
        print(f"Partially/unmatched ego frames: {unmatched}")

    def save_merged_ply_at_index(self,
                                 index: int,
                                 output_file: str,
                                 relative_match: bool = False,
                                 colored: bool = False):
        """
        Merges point clouds from the given index into a combined point cloud,
        then saves this new point cloud to the given file 
        'output_file'. 

        For relative matches: The centers of the matched plys are at their 
        respective absolute world coordinates (read from respective TUM files). 
        This is the default mode.

        For absolute matches: The ego ply origin remains at 0 0 0, and the other
        plys are shifted relative to the absolute poses in their respective TUM
        files.

        Args:
            output_point_cloud_file (str): Path to save the merged point cloud.
            index (int): Index of the ego frame to merge.
            relative_match (bool): If True, use relative matches instead of 
                absolute.
            colored (bool): If True, assign random colors to each point cloud 
                for better differentiation (generally for debugging).
        """
        if relative_match:
            match = self.get_relative_match_for_ego_index(index)
        else:
            match = self.get_absolute_match_for_ego_index(index)
        if match:
            clouds = []
            for entry in match:
                if entry is not None:
                    filename, _, pose = entry
                    clouds.append({
                        'file': filename,
                        'pose': pose,
                    })
                    if colored:
                        color = random_bright_color()
                        clouds[-1]['color'] = color
            combine_point_clouds_with_poses(clouds, output_file)

        else:
            logging.error(
                f"No matches found for ego frame {index}. Cannot save merged point cloud."
            )
            raise ValueError(
                f"No matches found for ego frame {index}. Cannot save merged point cloud."
            )

    def save_all_merged_plys(self,
                             output_dir: str,
                             relative_match: bool = False):
        """
        Merges all point cloud matches to their respective combined point 
        clouds, then saves these new point cloud frames to the given directory 
        'output_dir'. The filenames of this new point cloud sequence are 0.ply,
        1.ply, 2.ply, etc. (according to the indices). The centers of the 
        matched plys are at their respective absolute world coordinates (read
        from respective TUM files).

        For relative matches: The centers of the matched plys are at their 
        respective absolute world coordinates (read from respective TUM files). 
        This is the default mode.

        For absolute matches: The ego ply origin remains at 0 0 0, and the other
        plys are shifted relative to the absolute poses in their respective TUM
        files.

        Args:
            output_dir (str): Directory to save the merged point clouds.
            relative_match (bool): If True, use relative matches instead of absolute.
        """
        os.makedirs(output_dir, exist_ok=True)
        for index in range(len(self.matched_frames)):
            output_file = os.path.join(output_dir, f"{index}.ply")
            self.save_merged_ply_at_index(index,
                                          output_file,
                                          relative_match=relative_match)
