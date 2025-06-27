#!/bin/bash

# Path to the base build directory
BASE_DIR="build"

# Loop over each matching frames directory
for frames_dir in "$BASE_DIR"/sim_*_output/ego_lidar/frames; do
  # Get absolute path to the sim_x_output directory
  sim_output_dir="$(dirname "$(dirname "$frames_dir")")"
  sim_name="$(basename "$sim_output_dir")"

  # Replace _output with _pin_slam_output
  pin_slam_output="${sim_name/_output/_pin_slam_output}"

  echo "Running PIN-SLAM on $frames_dir..."

  docker run -it --rm --ipc host --privileged --network host -p 8080:8081 --gpus all \
    -v /carla_hard_disk/carla-git/build:/storage/ \
    pinslam:localbuild \
    /usr/bin/python3 pin_slam.py -sm \
    -i "/storage/${sim_name}/ego_lidar/frames/" \
    -o "/storage/${pin_slam_output}"
done