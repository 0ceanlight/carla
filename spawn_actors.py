"""This script contains methods to spawn various types of actors, including 
sensors and vehicles. These methods should be called before starting the
main loop of the CARLA simulator. Each method returns one or multiple actors,
which should be appended to an actor list, and despawned after the game loop is
finished."""

import argparse
import collections
import datetime
import logging
import math
import numpy as np
import numpy.random as random
import os
import pygame
import re
import sys
import weakref
import logging
import time

from global_config_parser import load_global_config
from utils.tum_file_parser import append_right_handed_tum_pose

global_config = load_global_config('global_config.ini')


def save_right_handed_ply(lidar_data, filename):
    """
    Save the LiDAR data to a PLY file in right-handed coordinate system.
    :param lidar_data: LiDAR data to save
    :param filename: Name of the file to save the data
    """
    # Convert the point cloud to a numpy array
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).copy()
    points = points.reshape((-1, 4))

    # Flip Y-axis to convert to right-handed coordinate system
    points[:, 1] *= -1

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save to ASCII .ply file
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float32 x\n')
        f.write('property float32 y\n')
        f.write('property float32 z\n')
        f.write('property float32 I\n')
        f.write('end_header\n')
        np.savetxt(f, points, fmt='%.4f %.4f %.4f %.4f', delimiter=' ')


def create_lidar_callback(data_dir):
    """
    Creates a LiDAR callback function that saves point cloud data and corresponding poses
    to disk, maintaining a separate frame counter per callback instance.

    Args:
        data_dir (str): Path to the directory where LiDAR frames and pose files will be saved.

    Returns:
        function: A callback function that takes a `point_cloud` object as input and
                  performs the following:
                  - Increments and tracks its own frame count.
                  - Saves the point cloud to a .ply file.
                  - Appends the vehicle pose to a TUM-format pose file.
    """
    frame_counter = 0

    def lidar_callback(point_cloud):
        """
        Callback function for processing and saving LiDAR point cloud data.

        Args:
            point_cloud: An object representing a LiDAR point cloud. Expected to have the following attributes:
                         - frame_number (int): Identifier for the frame.
                         - timestamp (float): Timestamp of the frame.
                         - transform (matrix or pose object): The vehicle's pose at this timestamp.

        Side Effects:
            - Writes a PLY file for the point cloud to `<data_dir>/lidar_frames/<frame_number>.ply`.
            - Appends a line to `<data_dir>/ground_truth_poses_tum.txt` in TUM pose format.
        """
        # increment frame number
        nonlocal frame_counter
        frame_counter += 1

        frame_number = point_cloud.frame_number

        logging.debug(
            f"Saving LiDAR data for: {frame_number}, point count: {(point_cloud)}"
        )

        ply_path = os.path.join(data_dir, 'lidar_frames',
                                f'{frame_number}.ply')
        # Save the point cloud to a file in PLY format
        save_right_handed_ply(point_cloud, ply_path)

        filename = os.path.join(data_dir, f'ground_truth_poses_tum.txt')
        timestamp = point_cloud.timestamp
        # Save the vehicle pose to a file in TUM format
        append_right_handed_tum_pose(filename, point_cloud.transform,
                                     timestamp)


def create_camera_callback(data_dir):
    """
    Creates a camera callback function that TODO
    """
    frame_counter = 0

    def camera_callback(point_cloud):
        # TODO
        pass


def spawn_lidar(output_dir, world, transform, attach_to=None):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('channels', global_config.carla_lidar.channels)
    lidar_bp.set_attribute('range', global_config.carla_lidar.range)
    lidar_bp.set_attribute('points_per_second',
                           global_config.carla_lidar.points_per_second)
    lidar_bp.set_attribute('rotation_frequency',
                           global_config.carla_lidar.rotation_frequency)
    lidar_bp.set_attribute('upper_fov', global_config.carla_lidar.upper_fov)
    lidar_bp.set_attribute('lower_fov', global_config.carla_lidar.lower_fov)
    lidar_bp.set_attribute('horizontal_fov',
                           global_config.carla_lidar.horizontal_fov)
    lidar_bp.set_attribute(
        'atmosphere_attenuation_rate',
        global_config.carla_lidar.atmosphere_attenuation_rate)
    lidar_bp.set_attribute('dropoff_general_rate',
                           global_config.carla_lidar.dropoff_general_rate)
    lidar_bp.set_attribute('dropoff_intensity_limit',
                           global_config.carla_lidar.dropoff_intensity_limit)
    lidar_bp.set_attribute('dropoff_zero_intensity',
                           global_config.carla_lidar.dropoff_zero_intensity)
    lidar_bp.set_attribute('sensor_tick', global_config.carla_lidar.sensor_tick)
    lidar_bp.set_attribute('noise_stddev',
                           global_config.carla_lidar.noise_stddev)

    lidar = world.spawn_actor(lidar_bp, transform, attach_to=attach_to)
    lidar.listen(create_lidar_callback(output_dir))

    return lidar


def spawn_camera(output_dir, world, transform, attach_to=None):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    camera_bp.set_attribute('bloom_intensity',
                            global_config.carla_camera.bloom_intensity)
    camera_bp.set_attribute('fov', global_config.carla_camera.fov)
    camera_bp.set_attribute('fstop', global_config.carla_camera.fstop)
    camera_bp.set_attribute('image_size_x',
                            global_config.carla_camera.image_size_x)
    camera_bp.set_attribute('image_size_y',
                            global_config.carla_camera.image_size_y)
    camera_bp.set_attribute('iso', global_config.carla_camera.iso)
    camera_bp.set_attribute('gamma', global_config.carla_camera.gamma)
    camera_bp.set_attribute('lens_flare_intensity',
                            global_config.carla_camera.lens_flare_intensity)
    camera_bp.set_attribute('sensor_tick',
                            global_config.carla_camera.sensor_tick)
    camera_bp.set_attribute('shutter_speed',
                            global_config.carla_camera.shutter_speed)

    # Attach camera to the vehicle
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    camera.listen(create_camera_callback(output_dir))

    return camera


def spawn_vehicles(world,
                   traffic_manager,
                   transform=None,
                   filter="vehicle.*.*",
                   type="car",
                   seed=None,
                   number=1):
    """
    TODO: docs... basically dis spawns vehicle @ location & sets autopilot
    """
    if seed:
        random.seed(seed)
    else:
        # TODO: dis a good idea?
        random.seed(int(time.time()))

    actor_list = []

    for _ in range(number):
        vehicle_bp = None
        bp_library = world.get_blueprint_library()
        vehicle_bps = []
        if filter:
            if not filter.startswith("vehicle"):
                raise "Filter must be a vehicle"
            vehicle_bps = bp_library.filter(filter)
        else:
            vehicle_bps = bp_library.filter("vehicle.*.*")
        if type:
            vehicle_bps = [i for i in vehicle_bps if i.base_type == type]

        if not vehicle_bps:
            raise f"No valid blueprints found with filter={filter} and type={type}"

        vehicle_bp = random.choice(vehicle_bps)

        if transform is None:
            # Get a random spawn point from the map
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in the map.")
            transform = random.choice(spawn_points)

        vehicle = world.spawn_actor(vehicle_bp, transform)
        # set to automatic control
        vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.update_vehicle_lights(vehicle, True)

        actor_list.append(vehicle)

    return (actor_list)
