import argparse
import collections
import datetime
import logging
import math
import numpy as np
import numpy.random as random
import os
import pyglet
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

# Setup Pyglet for windowing
pyglet.options['shadow_window'] = False  # Prevent OpenGL context issues

# camera callback for ego_camera
def ego_camera_callback(image, window):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    texture = pyglet.image.Texture.create(image.width, image.height, pyglet.gl.GL_RGB)
    texture.blit_into(pyglet.image.ImageData(image.width, image.height, 'RGB', array.tobytes()), 0, 0)
    window.switch_to()
    window.dispatch_events()
    window.clear()
    texture.blit(0, 0)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles spawning/teardown of actors,
    ticking the agent and the world (if needed), etc.
    """
    logging.info("Initializing CARLA...")

    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        world = client.get_world()

        if args.asynch:
            logging.warning("You are currently in asynchronous mode, and traffic might experience some issues")
        else:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # Create two pyglet windows
        ego_window = pyglet.window.Window(args.width, args.height, "Ego Camera")
        second_window = pyglet.window.Window(args.width, args.height, "Static Camera")
        clock = pyglet.clock.Clock()

        # AGENT ----------------------------------------------------------
        vehicle_bp = world.get_blueprint_library().find("vehicle.dodge.charger")
        spawn_points = world.get_map().get_spawn_points()
        ego_destination = random.choice(spawn_points).location
        ego_transform = carla.Transform(ego_destination)
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_transform)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.update_vehicle_lights(ego_vehicle, True)

        # EGO CAMERA -----------------------------------------------------
        ego_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        ego_camera_bp.set_attribute("image_size_x", str(args.width))
        ego_camera_bp.set_attribute("image_size_y", str(args.height))
        ego_camera_bp.set_attribute("fov", str(args.fov))
        ego_camera_transform = carla.Transform(carla.Location(x=2.5, z=1.7))
        ego_camera = world.spawn_actor(ego_camera_bp, ego_camera_transform, attach_to=ego_vehicle)
        ego_camera.listen(lambda image: ego_camera_callback(image, ego_window))

        # STATIC CAMERA --------------------------------------------------
        static_camera_transform = carla.Transform(carla.Location(x=-52.452820, y=22.877516, z=40))
        static_camera = world.spawn_actor(ego_camera_bp, static_camera_transform)
        static_camera.listen(lambda image: ego_camera_callback(image, second_window))

        # MISC. SETUP ----------------------------------------------------
        actor_list.append(ego_vehicle)
        actor_list.append(ego_camera)
        actor_list.append(static_camera)

        # TRAFFIC MANAGER --------------------------------
        # Example of how to use Traffic Manager parameters
        # traffic_manager.global_percentage_speed_difference(30.0)

        # LOOP -----------------------------------------------------------
        while not ego_window.has_exit and not second_window.has_exit:
            try:
                if args.asynch:
                    world.wait_for_tick()
                else:
                    world.tick()

                pyglet.app.platform_event_loop.dispatch_events()

                # Render Pyglet windows
                pyglet.clock.tick()
                ego_window.flip()
                second_window.flip()

                # Print location of ego_vehicle every tick
                logging.debug(f"ego_vehicle location: {ego_vehicle.get_transform().location}")

            except Exception as e:
                logging.error(f"Error during simulation tick: {e}")
                break

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        logging.info("Cleaning up actors...")

        # Cleanup actors
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()

        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        logging.info("Cleanup finished. Exiting.")

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug', help='Print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720', help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fov', default='60.0', help='FOV for camera')
    argparser.add_argument(
        '--asynch', action='store_true', help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*', help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All', help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '-l', '--loop', action='store_true', dest='loop', help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str, choices=["Behavior", "Basic", "Constant"], default="Behavior", help="select which agent to run")
    argparser.add_argument(
        '-b', '--behavior', type=str, choices=["cautious", "normal", "aggressive"], default='normal', help='Choose one of the possible agent behaviors (default: normal)')
    argparser.add_argument(
        '-s', '--seed', default=None, type=int, help='Set seed for repeating executions (default: None)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")

if __name__ == '__main__':
    main()