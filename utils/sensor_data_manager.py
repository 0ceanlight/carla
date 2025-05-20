import os
import bisect
from typing import List, Tuple, Dict, Optional
from natsort import natsorted
from .tum_file_parser import load_tum_file
from .quaternion_utils import quaternion_inverse, quaternion_multiply

class SensorDataManager:
    """
    A class to manage and synchronize LIDAR sensor data and their associated TUM pose files.

    Attributes:
        base_dir (str): Root directory containing sensor data.
        sensors (List[str]): List of sensor names. The first is assumed to be the reference (ego) sensor.
        max_timestamp_discrepancy (float): Maximum allowed timestamp difference in seconds for matching.
        sensor_data (Dict[str, List[Tuple]]): Parsed TUM entries for each sensor.
        sensor_filenames (Dict[str, List[str]]): Corresponding sorted .ply filenames for each sensor.
        matched_frames (List[List[Optional[Tuple[str, Tuple]]]]): Matched frames across sensors.
    """

    def __init__(self, base_dir: str, sensors: List[str], max_timestamp_discrepancy: float = 1.0):
        """
        Initialize the SensorDataManager.

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
                key=lambda x: int(x.split('.')[0])
            )
            full_paths = [os.path.join(frames_path, f) for f in filenames]

            if len(tum_data) != len(filenames):
                raise ValueError(f"Mismatch between TUM entries and .ply files for sensor {sensor}. There are {len(tum_data)} TUM entries and {len(filenames)} .ply files.")

            self.sensor_data[sensor] = tum_data
            self.sensor_filenames[sensor] = full_paths

    def _match_frames(self) -> List[List[Optional[Tuple[str, Tuple]]]]:
        """
        Matches each ego frame to the closest frames from other sensors based on timestamp.

        Returns:
            List of lists, each containing tuples of (filename, TUM entry) or None if no match found.
        """
        matches = []
        ego_timestamps = [entry[0] for entry in self.sensor_data[self.ego_sensor]]

        for idx, ego_entry in enumerate(self.sensor_data[self.ego_sensor]):
            frame_matches = []
            ego_filename = self.sensor_filenames[self.ego_sensor][idx]
            frame_matches.append((ego_filename, ego_entry))

            for sensor in self.other_sensors:
                timestamps = [entry[0] for entry in self.sensor_data[sensor]]
                closest_idx = self._find_closest_index(timestamps, ego_entry[0])
                if closest_idx is not None:
                    matched_filename = self.sensor_filenames[sensor][closest_idx]
                    matched_entry = self.sensor_data[sensor][closest_idx]
                    frame_matches.append((matched_filename, matched_entry))
                else:
                    frame_matches.append(None)
            matches.append(frame_matches)

        return matches

    def _find_closest_index(self, sorted_timestamps: List[float], target: float) -> Optional[int]:
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
            candidates.append((abs(sorted_timestamps[idx - 1] - target), idx - 1))

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
        return [i for i, match in enumerate(self.matched_frames) if None in match]

    def get_match_for_ego_index(self, index: int) -> Optional[List[Optional[Tuple[str, Tuple]]]]:
        """
        Retrieves the match data for a specific ego frame.

        Args:
            index (int): Index of the ego frame.

        Returns:
            Optional[List[Optional[Tuple[str, Tuple]]]]: List of matches or None.
        """
        if 0 <= index < len(self.matched_frames):
            return self.matched_frames[index]
        return None

    def get_all_matches(self) -> List[List[Optional[Tuple[str, Tuple]]]]:
        """
        Returns all matched frame data.

        Returns:
            List[List[Optional[Tuple[str, Tuple]]]]: All match data.
        """
        return self.matched_frames

    def get_relative_match_for_ego_index(self, index: int) -> Optional[List[Optional[Tuple[str, Tuple]]]]:
        """
        Retrieves the relative pose match data for a specific ego frame.
        Ego sensor is at origin: (0, 0, 0, 0, 0, 0, 1)
        
        Args:
            index (int): Index of the ego frame.

        Returns:
            Optional[List[Optional[Tuple[str, Tuple]]]]: List of relative matches or None.
        """
        if 0 <= index < len(self.matched_frames):
            frame = self.matched_frames[index]
            if frame is None:
                return None
            ego_filename, ego_pose = frame[0]
            relative_frame = [(ego_filename, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))]

            ego_translation = ego_pose[1:4]
            ego_quat = ego_pose[4:]

            for match in frame[1:]:
                if match is None:
                    relative_frame.append(None)
                else:
                    filename, pose = match
                    rel_translation = tuple(p - e for p, e in zip(pose[1:4], ego_translation))
                    rel_rotation = quaternion_multiply(quaternion_inverse(ego_quat), pose[4:])
                    relative_pose = rel_translation + rel_rotation
                    relative_frame.append((filename, relative_pose))
            return relative_frame
        return None

    def get_all_relative_matches(self) -> List[List[Optional[Tuple[str, Tuple]]]]:
        """
        Returns all matched frame data with poses relative to ego sensor origin.

        Returns:
            List[List[Optional[Tuple[str, Tuple]]]]: All relative match data.
        """
        return [self.get_relative_match_for_ego_index(i) for i in range(len(self.matched_frames))]

    def summary(self):
        """
        Prints a summary of matching statistics.
        """
        total = len(self.matched_frames)
        matched = self.count_fully_matched_ego_frames()
        unmatched = self.count_partially_or_unmatched_ego_frames()
        print(f"Total ego frames: {total}")
        print(f"Fully matched ego frames: {matched}")
        print(f"Partially/unmatched ego frames: {unmatched}")

