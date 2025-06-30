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

from utils.math_utils import euler_to_quaternion
from utils.tum_file_parser import tum_append_tuples
from utils.misc import clear_directory
import config.global_config_parser as global_config_parser
import config.sim_config_parser as sim_config_parser

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
    global_config_parser.init_config(args.global_config)
    global_config = global_config_parser.get_config()


    sim_config_parser.init_config(args.sim_config)
    sim_config = sim_config_parser.get_config()

    # Requires config to be loaded first
    from spawn_actors import spawn_vehicle, spawn_vehicles, spawn_camera, spawn_lidar


    # Keep track of actors and world to despawn at the end
    actor_list = []
    world = None

    try:
        # TODO: is this used?
        if sim_config.general.seed is not None:
            random.seed(sim_config.general.seed)

        # WORLD ----------------------------------------------------------------
        client = carla.Client(global_config.carla_world.host, global_config.carla_world.simulator_port)
        # optional TODO: move timeout and delta seconds to global_config?
        client.set_timeout(15.0)

        try:
            world = client.get_world()
            if world is None:
                raise RuntimeError("Failed to get CARLA world, although no exception was raised.")
        except RuntimeError as e:
            logging.critical(f"Failed to get CARLA world: {e}. Is the CARLA server running?")
            raise RuntimeError("Failed to get CARLA world") from e

        logging.info('Listening to server %s:%s', global_config.carla_world.host, global_config.carla_world.simulator_port)

        # Clear output data directory
        if not args.no_save:
            clear_directory(sim_config.general.output_dir, noconfirm=args.noconfirm)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # TRAFFIC MANAGER ------------------------------------------------------
        # Examples of how to use Traffic Manager parameters
        # # Set vehicle to drive 20% of speed limit
        # traffic_manager.global_percentage_speed_difference(20.0)
        # traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 15)
        # Optional TODO: move traffic manager options to sim_config?

        traffic_manager = client.get_trafficmanager(global_config.carla_world.traffic_manager_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)

        if sim_config.general.seed is not None:
            traffic_manager.set_random_device_seed(sim_config.general.seed)

        # ACTORS ---------------------------------------------------------------
        # Keep track of actors to despawn at the end

        agents = {}

        # ---+ SINGLE AGENTS +---
        for agent_config in sim_config.agents:
            # Spawn each agent, appending a pointer to the CARLA object with
            # its name to the `agents` dict, so that sensors can be attached.
            spawned_agent = spawn_vehicle(
                world,
                traffic_manager, 
                transform=agent_config.transform,
                filter=agent_config.filter, 
                type=agent_config.type,
                seed=sim_config.general.seed, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
                autopilot=agent_config.autopilot)
            if spawned_agent is None:
                logging.error(f"Failed to spawn agent {agent_config.name}.")
                continue
            agents[agent_config.name] = spawned_agent
            actor_list.append(spawned_agent)

        # ---+ OTHER VEHICLES +---
        actor_list += spawn_vehicles(
            world, 
            traffic_manager, 
            seed=sim_config.general.seed, 
            number=sim_config.other_vehicles.n_vehicles)

        # ---+ SENSORS +---
        # Only spawn sensors if data saving is enabled
        if not args.no_save:
            for sensor_config in sim_config.sensors:
                # Attach to an agent if specified
                attach_actor = agents.get(sensor_config.attach_to) if sensor_config.attach_to else None
                if sensor_config.name.endswith("_lidar"):
                    actor_list.append(spawn_lidar(
                        sensor_config.data_dir, 
                        world, 
                        sensor_config.transform, 
                        attach_to=attach_actor))
                elif sensor_config.name.endswith("_camera"):
                    actor_list.append(spawn_camera(
                        sensor_config.data_dir, 
                        world, 
                        sensor_config.transform, 
                        attach_to=attach_actor))


        # MAIN SIM LOOP --------------------------------------------------------
        logging.info(f'Running simulation for {sim_config.general.n_ticks} ticks...')
        global frame_counter
        tick_ctr = 0
        while True:
            if tick_ctr >= sim_config.general.n_ticks:
                logging.info('BOOM! Finished capturing data. Exiting...')
                # wait until all sensors have finished saving data... 
                # omitting this can lead to incomplete callbacks
                if not args.no_save:
                    logging.info('Waiting for sensors to finish saving data...')

                    # Wait for up to 60 seconds for the sensors to finish saving data
                    max_wait_time = 60
                    while True:
                        # Verify whether the number of TUM entries (number of lines) and number of files is equal in each data dir
                        # We know that the simulation is done when the number of files in the respective `frames` directory is equal to the number of lines in the `ground_truth_poses_tum.txt` file
                        directories = [s.data_dir for s in sim_config.sensors if s.data_dir is not None]

                        finished = True
                        for directory in directories:
                            if not os.path.exists(directory):
                                logging.warning(f"Directory {directory} does not exist.")
                                continue

                            # Count the number of files in the `frames` directory
                            frames_dir = os.path.join(directory, 'frames')
                            if not os.path.exists(frames_dir):
                                logging.warning(f"Frames directory {frames_dir} does not exist.")
                                # Skip this sensor if the frames directory is missing
                                continue
                            num_files = len([f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))])
                            logging.debug(f"Number of files in {frames_dir}: {num_files}")

                            # Count the number of lines in the `ground_truth_poses_tum.txt` file
                            tum_file = os.path.join(directory, 'ground_truth_poses_tum.txt')
                            if not os.path.exists(tum_file):
                                logging.warning(f"TUM file {tum_file} does not exist.")
                                # Skip this sensor if the TUM file is missing
                                continue # Sensor directories loop
                            with open(tum_file, 'r') as f:
                                num_lines = sum(1 for line in f if line.strip() and not line.startswith('#'))
                            logging.debug(f"Number of lines in {tum_file}: {num_lines}")

                            if num_files != num_lines:
                                finished = False
                                logging.debug(f"Number of files ({num_files}) does not match number of lines ({num_lines}) in {tum_file}. Waiting for sensors to finish saving data...")
                        # Loop post condition: If all numbers are equal, finished = True else False

                        if finished:
                            logging.info('All sensors finished saving data.')
                            break # Wait time loop
                        elif max_wait_time <= 0:
                            logging.warning('Max wait time reached. Exiting...')
                            break # Wait time loop
                        else:
                            logging.info(f'Waiting for sensors to finish saving data... {max_wait_time} seconds left.')
                            time.sleep(1)
                            max_wait_time -= 1
                    
                    time.sleep(3) # Wait a bit more just for good measure

                break # Main sim loop
            # END MAIN SIM LOOP ----------------------------------------------------

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
        '-n', '--no-save', action='store_true', dest='no_save',
        help='Don\'t save any data to disk, only run simulation')
    argparser.add_argument(
        '-g', '--global-config', type=str, default='config/global_config.ini',
        help='Path to the global configuration file (default: config/global_config.ini)')
    argparser.add_argument(
        '-s', '--sim-config', type=str, default='config/sim_config_0.ini',
        help='Path to the simulation configuration file (default: config/sim_config_0.ini)')
    argparser.add_argument(
        '--noconfirm', action='store_true', dest='noconfirm',
        help='Do not ask for confirmation before clearing the output directory'
    )
        
    args = argparser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    print('Log level:', logging.getLevelName(log_level))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nInterrupted by user. Bye!')


if __name__ == '__main__':
    main()

# TODO: remove unneded args and imports
# main_loop.py --data_dir <output dir> --verbose --simulation 0