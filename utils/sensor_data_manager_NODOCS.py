import os
import bisect
from typing import List, Tuple, Dict, Optional
from natsort import natsorted
from .tum_file_parser import load_tum_file

class SensorDataManager:
    def __init__(self, base_dir: str, sensors: List[str], max_timestamp_discrepancy: float = 2.0):
        self.base_dir = base_dir
        self.ego_sensor = sensors[0]
        self.other_sensors = sensors[1:]
        self.max_timestamp_discrepancy = max_timestamp_discrepancy

        self.sensor_data = {}  # sensor -> list of TUM tuples
        self.sensor_filenames = {}  # sensor -> list of .ply filenames

        self._load_all_data()
        self.matched_frames = self._match_frames()

    def _load_all_data(self):
        for sensor in [self.ego_sensor] + self.other_sensors:
            sensor_path = os.path.join(self.base_dir, sensor)
            tum_path = os.path.join(sensor_path, 'ground_truth_poses_tum.txt')
            frames_path = os.path.join(sensor_path, 'lidar_frames')

            tum_data = load_tum_file(tum_path)
            filenames = natsorted([f for f in os.listdir(frames_path) if f.endswith('.ply')], key=lambda x: int(x.split('.')[0]))

            if len(tum_data) != len(filenames):
                raise ValueError(f"Mismatch between TUM entries and .ply files for sensor {sensor}.")

            self.sensor_data[sensor] = tum_data
            self.sensor_filenames[sensor] = filenames

    def _match_frames(self) -> List[List[Optional[Tuple[str, Tuple]]]]:
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
        return sum(1 for match in self.matched_frames if None not in match)

    def count_partially_or_unmatched_ego_frames(self) -> int:
        return sum(1 for match in self.matched_frames if None in match)

    def get_unmatched_indices(self) -> List[int]:
        return [i for i, match in enumerate(self.matched_frames) if None in match]

    def get_match_for_ego_index(self, index: int) -> Optional[List[Optional[Tuple[str, Tuple]]]]:
        if 0 <= index < len(self.matched_frames):
            return self.matched_frames[index]
        return None

    def get_all_matches(self) -> List[List[Optional[Tuple[str, Tuple]]]]:
        return self.matched_frames

    def summary(self):
        total = len(self.matched_frames)
        matched = self.count_fully_matched_ego_frames()
        unmatched = self.count_partially_or_unmatched_ego_frames()
        print(f"Total ego frames: {total}")
        print(f"Fully matched ego frames: {matched}")
        print(f"Partially/unmatched ego frames: {unmatched}")

