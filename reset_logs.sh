#!/bin/sh

sudo rm -rf ./build/lidar_output_frames/*
sudo rm -rf ../PIN_SLAM/tmp_storage/lidar_output_frames/*
sudo rm -rf ../PIN_SLAM/tmp_storage/lidar_results/*

/home/bike/miniforge3/envs/carla/bin/python /home/bike/ocean/carla-git/lidar_save.py --nframes 64 -s 5
cp -r build/lidar_output_frames ../PIN_SLAM/tmp_storage

# in docker
# python3 pin_slam.py -i /storage/lidar_output_frames/ -vsm -o /storage/lidar_results/
