#!/usr/bin/env python

# Copyright (c) 2024 Technical University of Munich
# Author: 0ceanlight
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This script spawns a vehicle in the CARLA simulator and attaches a camera and
LiDAR to it. Both record to given data directories + TODO: infrastruct 
The vehicle is set to autopilot mode."""

import argparse
import logging
import numpy.random as random
import os
import re
import sys
import time

from spawn_actors import *
from utils.math_utils import euler_to_quaternion
from utils.tum_file_parser import append_tum_poses
from utils.misc import clear_directory
from global_config_parser import load_global_config
from sim_config_parser import load_sim_config

# ==============================================================================
# -- Basic functions -----------------------------------------------------------
# ==============================================================================

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


# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles spawning/teardown of actors,
    ticking the agent and the world (if needed), etc.
    """

    global_config = load_global_config("global_config.ini")
    sim_config = load_sim_config("sim_config_0.ini")

    # Clear output data directory
    clear_directory(sim_config.general.output_dir)

    # Keep track of actors and world to despawn at the end
    actor_list = []
    world = None

    try:
        if sim_config.general.seed is not None:
            random.seed(sim_config.general.seed)

        # WORLD ----------------------------------------------------------------
        client = carla.Client(global_config.carla_world.host, global_config.carla_world.simulator_port)
        # optional TODO: move timeout and delta seconds to global_config?
        client.set_timeout(60.0)

        world = client.get_world()
        if world is None:
            logging.critical("Failed to get CARLA world")
            raise RuntimeError("Failed to get CARLA world")
        traffic_manager = client.get_trafficmanager()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # TRAFFIC MANAGER ------------------------------------------------------
        # # set vehicle to drive 20% of speed limit

        # Examples of how to use Traffic Manager parameters
        # traffic_manager.global_percentage_speed_difference(30.0)
        # traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 15)
        # optional TODO: move traffic manager options to sim_config?
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        traffic_manager = client.get_trafficmanager(global_config.carla_world.traffic_manager_port)


        traffic_manager.set_synchronous_mode(True)

        if sim_config.general.seed is not None:
            traffic_manager.set_random_device_seed(sim_config.general.seed)


        # ACTORS ---------------------------------------------------------------
        # Keep track of actors to despawn at the end

        # AGENT
        ego_vehicle = spawn_vehicle(
            world, 
            traffic_manager, 
            transform=sim_config.ego_vehicle.transform,
            filter=sim_config.ego_vehicle.filter, 
            type=sim_config.ego_vehicle.type,
            seed=sim_config.general.seed, 
            number=1)
        actor_list.append(ego_vehicle)

        # OTHER CARS
        actor_list += spawn_vehicles(
            world, 
            traffic_manager, 
            seed=sim_config.general.seed, 
            number=sim_config.other_vehicles.n_vehicles)

        # EGO CAMERA
        actor_list.append(spawn_camera(
            sim_config.ego_camera.data_dir, 
            world, 
            sim_config.ego_camera.transform, 
            attach_to=ego_vehicle))

        # EGO LIDAR SENSOR
        actor_list.append(spawn_lidar(
            sim_config.ego_lidar.data_dir, 
            world, 
            sim_config.ego_lidar.transform, 
            attach_to=ego_vehicle))

        # consider NORTH to be the courthouse
        # west middle part of square
        # Location(x=-52.452820, y=22.877516, z=0.045138)
        # middle of square
        # Location(x=-51.755508, y=-1.344367, z=0.076584)

        # INFRASTRUCT LIDAR SENSOR (NE)
        actor_list.append(spawn_lidar(
            sim_config.ne_lidar.data_dir, 
            world, 
            sim_config.ne_lidar.transform, 
            attach_to=None))


        # LOOP -----------------------------------------------------------------
        logging.info(f'Running simulation for {sim_config.general.n_ticks} ticks...')
        global frame_counter
        tick_ctr = 0
        while True:
            if tick_ctr >= sim_config.general.n_ticks:
                logging.info('BOOM! Finished capturing data. Exiting...')
                break

            # Synchronous mode tick
            world.tick()
            tick_ctr += 1
            logging.info(f'TICK {tick_ctr} / {sim_config.general.n_ticks}')

    finally:
        logging.info('Cleaning up...')

        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            # settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # TODO: needed?
        time.sleep(0.5)

        logging.info("Cleanup finished. Exiting.")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Simulation Loop')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--data_dir',
        help='Directory to save sensor data and ground truth poses')
    # TODO: add argument for which simulation to run

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    print('Log level:', logging.getLevelName(log_level))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', global_config.carla_world.host, global_config.carla_world.simulator_port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nInterrupted by user. Bye!')


if __name__ == '__main__':
    main()

# TODO: remove unneded args and imports
# main_loop.py --data_dir <output dir> --verbose --simulation 0