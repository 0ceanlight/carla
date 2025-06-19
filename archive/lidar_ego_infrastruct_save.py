#!/usr/bin/env python

# Copyright (c) 2024 Technical University of Munich
# Author: 0ceanlight
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This script spawns a vehicle in the CARLA simulator and attaches a camera to
it. The camera captures images and displays them in a Pygame window. The vehicle 
is set to autopilot mode."""

import argparse
import collections
import datetime
import logging
import numpy as np
import numpy.random as random
import os
import pygame
import re
import sys
import weakref
import time

from utils.math_utils import euler_to_quaternion
from utils.tum_file_parser import append_tum_poses, append_right_handed_tum_pose


# ==============================================================================
# -- Basic functions -----------------------------------------------------------
# ==============================================================================


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


# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================


try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


actor_list = []
frame_counter = 0

# camera callback
def ego_camera_callback(image, display, clock):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    clock.tick(30)

def lidar_callback(point_cloud, data_dir):
    # increment frame number
    global frame_counter
    frame_counter += 1

    frame_number = point_cloud.frame_number

    print(f"Saving LiDAR data for: {frame_number}, point count: {(point_cloud)}")

    ply_path = os.path.join(data_dir, 'lidar_frames', f'{frame_number}.ply')
    # Save the point cloud to a file in PLY format
    save_right_handed_ply(point_cloud, ply_path)

    filename = os.path.join(data_dir, f'ground_truth_poses_tum.txt')
    timestamp = point_cloud.timestamp
    # Save the vehicle pose to a file in TUM format
    append_right_handed_tum_pose(point_cloud.transform, timestamp, filename)

def imu_callback(imu_data, data_dir):
    # Get current unix timestamp in ms as int
    timestamp = int(datetime.datetime.now().timestamp() * 1000)

    frame_number = str(timestamp)
    # TODO: save IMU data to disk


# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles spawning/teardown of actors,
    ticking the agent and the world (if needed), etc.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        world = client.get_world()

        if args.asynch:
            print("You are currently in asynchronous mode, and traffic might experience some issues")
        else:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        display = None
        clock = None
        if not args.headless:        
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            clock = pygame.time.Clock()


        # AGENT ----------------------------------------------------------------
        vehicle_bp = world.get_blueprint_library().find("vehicle.dodge.charger")
        spawn_points = world.get_map().get_spawn_points()
        ego_destination = random.choice(spawn_points).location
        ego_transform = carla.Transform(ego_destination)
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_transform)
        # set to automatic control
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.update_vehicle_lights(ego_vehicle, True)


        ego_vehicle.get_transform()

        # EGO CAMERA -----------------------------------------------------------
        # right now, camera is only used for pygame visualization
        if not args.headless:
            # spawn a camera 
            ego_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            ego_camera_bp.set_attribute("image_size_x", str(args.width))
            ego_camera_bp.set_attribute("image_size_y", str(args.height))
            ego_camera_bp.set_attribute("fov", str(args.fov))
            # TODO: set transform
            ego_camera_transform = carla.Transform(carla.Location(x=-0.7, z=1.7))

            # Attach camera to the vehicle
            ego_camera = world.spawn_actor(ego_camera_bp, ego_camera_transform, attach_to=ego_vehicle)
            ego_camera.listen(lambda image: ego_camera_callback(image, display, clock))


        # EGO LIDAR SENSOR -----------------------------------------------------
        ego_lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        ego_lidar_bp.set_attribute('channels',str(32))
        ego_lidar_bp.set_attribute('points_per_second',str(700000)) # TODO: Density too high?
        ego_lidar_bp.set_attribute('rotation_frequency',str(40))
        ego_lidar_bp.set_attribute('range',str(60)) # TODO: Needs to be pushed higher
        ego_lidar_bp.set_attribute('sensor_tick',str(args.sensor_tick))
        ego_lidar_location = carla.Location(0,0,2)
        ego_lidar_rotation = carla.Rotation(0,0,0)
        ego_lidar_transform = carla.Transform(ego_lidar_location, ego_lidar_rotation)
        ego_lidar = world.spawn_actor(ego_lidar_bp, ego_lidar_transform, attach_to=ego_vehicle)
        ego_path = os.path.join(args.data_dir, 'ego_lidar')
        ego_lidar.listen(lambda point_cloud: lidar_callback(point_cloud, ego_path))

        # consider NORTH to be the courthouse
        # west middle part of square
        # Location(x=-52.452820, y=22.877516, z=0.045138)
        # middle of square
        # Location(x=-51.755508, y=-1.344367, z=0.076584)

        # INFRASTRUCT LIDAR SENSOR (SE) ----------------------------------------
        infrastruct_lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        infrastruct_lidar_bp.set_attribute('channels',str(32))
        infrastruct_lidar_bp.set_attribute('points_per_second',str(700000)) # TODO: Density too high?
        infrastruct_lidar_bp.set_attribute('rotation_frequency',str(40))
        infrastruct_lidar_bp.set_attribute('range',str(60)) # TODO: Needs to be pushed higher
        infrastruct_lidar_bp.set_attribute('upper_fov',str(0))
        # fov needs to be lower down (othwerwise omits large circle near base of
        # sensor pole, which cuts into crossing)
        infrastruct_lidar_bp.set_attribute('lower_fov',str(-80)) 
        infrastruct_lidar_bp.set_attribute('sensor_tick',str(args.sensor_tick))
        infrastruct_lidar_location = carla.Location(-61.2, 36.8, 7.6) # SE corner
        infrastruct_lidar_rotation = carla.Rotation(0,0,0) # TODO: set to correct rotation + decrease FOV
        infrastruct_lidar_transform = carla.Transform(infrastruct_lidar_location, infrastruct_lidar_rotation)
        infrastruct_lidar = world.spawn_actor(infrastruct_lidar_bp, infrastruct_lidar_transform)
        infrastruct_path = os.path.join(args.data_dir, 'infrastruct_lidar')
        infrastruct_lidar.listen(lambda point_cloud: lidar_callback(point_cloud, infrastruct_path))

        # IMU SENSOR -----------------------------------------------------------
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', str(args.sensor_tick))
        imu_location = carla.Location(0,0,2)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location, imu_rotation)
        ego_imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle)
        ego_imu.listen(lambda imu_data: imu_callback(imu_data, args.data_dir))


        # MISC. SETUP ----------------------------------------------------------
        # keep tracking of actors to remove at the end
        global actor_list

        actor_list.append(ego_vehicle)
        if not args.headless:
            actor_list.append(ego_camera)
        actor_list.append(ego_lidar)
        actor_list.append(infrastruct_lidar)
        actor_list.append(ego_imu)

        # TRAFFIC MANAGER --------------------------------
        # # set vehicle to drive 20% of speed limit

        # Examples of how to use Traffic Manager parameters
        # traffic_manager.global_percentage_speed_difference(30.0)
        # traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 15)


        # LOOP -----------------------------------------------------------------
        logging.info(f'Capturing {args.nframes} LiDAR frames...')
        global frame_counter
        while True:
            if args.asynch:
                world.wait_for_tick()
            else:
                world.tick()

            if not args.headless:
                # TODO: I have no idea what this does
                # world.render(display)
                # Updates pygame window, doesn't work if it's in camera callback
                pygame.display.flip()

            if frame_counter >= args.nframes:
                logging.info('Finished capturing LiDAR frames. Exiting...')
                break

    finally:
        logging.info('Cleaning up...')

        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        if world is not None and not args.asynch:
            settings = world.get_settings()
            settings.synchronous_mode = False
            # settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)


        time.sleep(0.5)

        logging.info("Cleanup finished. Exiting.")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fov', default='60.0',
        help='FOV for camera')
    argparser.add_argument(
        '--nframes', default=256, type=int,
        help='Number of data snapshots to record (default: 256)')
    argparser.add_argument(
        '--sensor_tick', default=0.1, type=float,
        help='Simulation seconds between sensor captures (default: 0.1)')
    argparser.add_argument(
        '--data_dir',
        help='Directory to save sensor data and ground truth poses')
    argparser.add_argument(
        '--asynch', action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '-s', '--seed', default=None, type=int,
        help='Set seed for repeating executions (default: None)')
    argparser.add_argument(
        '--headless', action='store_true',
        help='Disable rendering (headless mode)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nInterrupted by user. Bye!')


if __name__ == '__main__':
    main()

# TODO: remove unneded args and imports
# lidar_save.py --data_dir <output dir> -s 5 --headless --sensor_tick 0.1 