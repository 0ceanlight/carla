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
import math
import numpy as np
import numpy.random as random
import os
import pygame
import re
import sys
import weakref


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

# camera callback
def ego_camera_callback(image, display, clock):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    clock.tick(30)

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


        # EGO CAMERA -----------------------------------------------------------
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


        # consider NORTH to be the courthouse
        # west middle part of square
        # Location(x=-52.452820, y=22.877516, z=0.045138)
        # middle of square
        # Location(x=-51.755508, y=-1.344367, z=0.076584)


        # MISC. SETUP ----------------------------------------------------------
        # keep tracking of actors to remove at the end
        actor_list.append(ego_vehicle)
        actor_list.append(ego_camera)

        # TRAFFIC MANAGER --------------------------------
        # # set vehicle to drive 20% of speed limit

        # Examples of how to use Traffic Manager parameters
        # traffic_manager.global_percentage_speed_difference(30.0)
        # traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 15)


    # LOOP ---------------------------------------------------------------------
        while True:
            if args.asynch:
                world.wait_for_tick()
            else:
                world.tick()

            # TODO: I have no idea what this does
            # world.render(display)
            # Updates pygame window, doesn't work if it's in camera callback
            pygame.display.flip()


    finally:

        for actor in actor_list:
            actor.destroy()

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        print("Cleanup finished. Exiting.")


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
        '--asynch', action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '-l', '--loop', action='store_true', dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str, choices=["Behavior", "Basic", "Constant"], default="Behavior",
        help="select which agent to run")
    argparser.add_argument(
        '-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"], default='normal',
        help='Choose one of the possible agent behaviors (default: normal)')
    argparser.add_argument(
        '-s', '--seed', default=None, type=int,
        help='Set seed for repeating executions (default: None)')

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