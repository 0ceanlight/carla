"""This script contains methods to spawn various types of actors, including 
sensors and vehicles. These methods should be called before starting the
main loop of the CARLA simulator. Each method returns one or multiple actors,
which should be appended to an actor list, and despawned after the game loop is
finished."""

import logging
import numpy as np
import numpy.random as random
import os
import logging
import time
import shutil
import carla

from utils.tum_file_parser import append_right_handed_tum_pose
from utils.misc import clear_directory

from config.global_config_parser import load_global_config
global_config = load_global_config('config/global_config.ini')

def valid_spawn_transform(world, transform, lane_type=carla.LaneType.Any):
    """
    Projects the given transform to the nearest valid spawn point on the road

    Note that the following values are set to fixed standard values for valid spawns:
    - Z coordinate: 0.6
    - pitch: 0.0
    - roll: 0.0

    X and Y are taken from the given transform, but projected to specified lane type.
    Yaw is taken as-is from the given transform (free choice of direction).

    Args:
        world (carla.World): The CARLA world object
        lane_type (carla.LaneType): The type of lane to project to. Available lane types: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
        transform (carla.Transform): The transform to project
    Returns:
        carla.Transform: A valid spawn point transform projected to the road
    """
    # map-specific z coordinate that avoids collision (with ground, for example) at spawn
    fixed_z = 0.6
    fixed_pitch = 0.0
    fixed_roll = 0.0

    waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=lane_type)

    if waypoint == None:
        raise RuntimeError(f'no valid spawn position found near given coordinates with LaneType={lane_type}')

    spawn_point = carla.Transform(
        carla.Location(x=waypoint.transform.location.x, 
                       y=waypoint.transform.location.y, 
                       z=fixed_z),
        carla.Rotation(pitch=fixed_pitch, 
                       roll=fixed_roll, 
                       yaw=transform.rotation.yaw)
    )

    return spawn_point

def _try_get_spawn_points(world, number=1, seed=None):
    """
    Get a list of spawn points from the CARLA world. 
    
    If the number of requested spawn points is less than the available spawn points, a warning is logged and all available spawn points are returned. If no spawn points are available, a critical error is logged and an empty list is returned.

    Args:
        world (carla.World): The CARLA world object
        number (int): Number of spawn points to return
        seed (int, optional): Random seed for reproducibility

    Returns:    
        list: A list of spawn points (carla.Transform objects)
    """
    if seed is not None:
        random.seed(seed)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logging.critical("No spawn points available in the map.")
        return []

    number_of_spawn_points = len(spawn_points)
    
    # If we have enough spawn points, randomly select the requested number
    # Essentially shuffles points if n_points is equal to number_of_spawn_points
    if number <= number_of_spawn_points:
        # Use replace=False to ensure unique spawn points
        spawn_points = random.choice(
            spawn_points, size=number, replace=False).tolist()
    # If we have fewer spawn points than requested, return all available spawn 
    # points and log a warning
    elif number > number_of_spawn_points:
        logging.warning(
            f"Requested {number} spawn points, but only {number_of_spawn_points} are available. Returning all available spawn points."
        )

    return spawn_points


def _try_get_random_vehicles(world, number=1, filter="vehicle.*.*", type="car", seed=None):
    """
    Get a list of random vehicles from the CARLA world.

    If the number of requested vehicles is less than the available blueprints, a warning is logged and all available blueprints are returned. If no blueprints are available, a critical error is logged and an empty list is returned.

    Args:
        world (carla.World): The CARLA world object
        number (int): Number of vehicles to return
        filter (str): Blueprint filter for vehicles
        type (str): Type of vehicle to spawn (e.g., "car")
        seed (int, optional): Random seed for reproducibility

    Returns:
        list: A list of vehicle blueprints (carla.Blueprint objects)
    """
    if seed is not None:
        random.seed(seed)

    bp_library = world.get_blueprint_library()
    vehicle_bps = bp_library.filter(filter)
    
    if type:
        vehicle_bps = [i for i in vehicle_bps if i.get_attribute("base_type") == type]

    if not vehicle_bps:
        logging.critical(f"No valid blueprints found with filter={filter} and type={type}")
        return []


    logging.debug(f"Returning {number} random vehicles from {len(vehicle_bps)} blueprints matching filter '{filter}' and type '{type}'.")

    # Duplicates are OK
    return random.choice(vehicle_bps, size=number).tolist()


def _try_spawn_vehicle(world, vehicle_bp, transform, traffic_manager, project_to_road=False, lane_type=carla.LaneType.Any):
    """
    Attempts to spawn a vehicle in the CARLA world at the specified transform.
    If the vehicle cannot be spawned, a critical error is logged and None is returned.

    Args:
        world (carla.World): The CARLA world object
        vehicle_bp (carla.Blueprint): The vehicle blueprint to spawn
        transform (carla.Transform): The transform to spawn the vehicle at
        traffic_manager (carla.TrafficManager): The traffic manager object
        project_to_road (bool): Whether to project the spawn point to the road
        lane_type (carla.LaneType): The type of lane to project to (if project_to_road is True)
    Returns:
        carla.Actor: The spawned vehicle actor, or None if spawning failed.
    """
    if project_to_road:
        # Project the transform to the nearest valid spawn point on the road
        transform = valid_spawn_transform(world, transform, lane_type)

    vehicle = world.try_spawn_actor(vehicle_bp, transform)
    if vehicle is None:
        logging.critical(f"Vehicle {vehicle_bp.id} could not be spawned at location {transform}")
        return None
    # Set to automatic control
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.update_vehicle_lights(vehicle, True)
    logging.debug(f"Vehicle {vehicle_bp.id} was spawned at {transform.location} and {transform.rotation}")
    return vehicle


def _save_right_handed_ply(lidar_data, filename):
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

    ply_dir = os.path.join(data_dir, 'lidar_frames')
    os.makedirs(ply_dir, exist_ok=True)

    tum_file = os.path.join(data_dir, f'ground_truth_poses_tum.txt')

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

        nonlocal frame_counter
        nonlocal ply_dir
        nonlocal tum_file

        frame_number = f"{frame_counter:06d}"

        logging.debug(
            f"Saving LiDAR data for frame: {frame_number}, point count: {(point_cloud)}"
        )

        # Format with zero-padded frame number
        ply_path = os.path.join(ply_dir, f'{frame_number}.ply')
        # Save the point cloud to a file in PLY format
        _save_right_handed_ply(point_cloud, ply_path)

        # number of seconds since Unix epoch, according to TUM format
        timestamp = time.time()
        # Save the vehicle pose to a file in TUM format
        append_right_handed_tum_pose(tum_file, point_cloud.transform,
                                     timestamp)

        # increment frame number
        frame_counter += 1

    return lidar_callback


def create_camera_callback(data_dir):
    """
    Creates a camera callback function that saves camera images and corresponding poses.

    Args:
        data_dir (str): Path to the directory where camera frames and pose files will be saved.
    
    Returns:
        function: A callback function that takes a `camera_image` object as input and
                  performs the following:
                  - Saves the camera image to a file.
                  - Appends the vehicle pose to a TUM-format pose file.
    """

    frame_counter = 0

    camera_dir = os.path.join(data_dir, 'camera_frames')
    os.makedirs(camera_dir, exist_ok=True)

    tum_file = os.path.join(data_dir, f'ground_truth_poses_tum.txt')

    def camera_callback(camera_image):
        """
        Callback function for processing and saving camera images.

        Args:
            camera_image: An object representing a camera image. Expected to have the following attributes:
                          - frame_number (int): Identifier for the frame.
                          - timestamp (float): Timestamp of the frame.
                          - transform (matrix or pose object): The vehicle's pose at this timestamp.
                          - raw_data (bytes): The raw image data.

        Side Effects:
            - Saves the camera image to `<data_dir>/camera_frames/<frame_number>.jpg`.
            - Appends a line to `<data_dir>/ground_truth_poses_tum.txt` in TUM pose format.
        """

        nonlocal frame_counter
        nonlocal camera_dir
        nonlocal tum_file

        frame_number = f"{frame_counter:06d}"

        logging.debug(
            f"Saving camera image for frame: {frame_number}, timestamp: {camera_image.timestamp}"
        )

        # Save the camera image to a file
        image_path = os.path.join(camera_dir, f'{frame_number}.png')
        camera_image.save_to_disk(image_path)

        # number of seconds since Unix epoch, according to TUM format
        timestamp = time.time()
        # Save the vehicle pose to a file in TUM format
        append_right_handed_tum_pose(tum_file, camera_image.transform,
                                     timestamp)

        # increment frame number
        frame_counter += 1

    return camera_callback


def spawn_lidar(output_dir, world, transform, attach_to=None):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('channels', 
                           str(global_config.carla_lidar.channels))
    lidar_bp.set_attribute('range', 
                           str(global_config.carla_lidar.range))
    lidar_bp.set_attribute('points_per_second',
                           str(global_config.carla_lidar.points_per_second))
    lidar_bp.set_attribute('rotation_frequency',
                           str(global_config.carla_lidar.rotation_frequency))
    lidar_bp.set_attribute('upper_fov', 
                           str(global_config.carla_lidar.upper_fov))
    lidar_bp.set_attribute('lower_fov', 
                           str(global_config.carla_lidar.lower_fov))
    lidar_bp.set_attribute('horizontal_fov',
                           str(global_config.carla_lidar.horizontal_fov))
    lidar_bp.set_attribute('atmosphere_attenuation_rate', 
                           str(global_config.carla_lidar.atmosphere_attenuation_rate))
    lidar_bp.set_attribute('dropoff_general_rate',
                           str(global_config.carla_lidar.dropoff_general_rate))
    lidar_bp.set_attribute('dropoff_intensity_limit',
                           str(global_config.carla_lidar.dropoff_intensity_limit))
    lidar_bp.set_attribute('dropoff_zero_intensity',
                           str(global_config.carla_lidar.dropoff_zero_intensity))
    lidar_bp.set_attribute('sensor_tick', 
                           str(global_config.carla_lidar.sensor_tick))
    lidar_bp.set_attribute('noise_stddev',
                           str(global_config.carla_lidar.noise_stddev))

    lidar = world.spawn_actor(lidar_bp, transform, attach_to=attach_to)
    lidar.listen(create_lidar_callback(output_dir))

    return lidar


def spawn_camera(output_dir, world, transform, attach_to=None):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    camera_bp.set_attribute('bloom_intensity',
                            str(global_config.carla_camera.bloom_intensity))
    camera_bp.set_attribute('fov', 
                            str(global_config.carla_camera.fov))
    camera_bp.set_attribute('fstop', 
                            str(global_config.carla_camera.fstop))
    camera_bp.set_attribute('image_size_x',
                            str(global_config.carla_camera.image_size_x))
    camera_bp.set_attribute('image_size_y',
                            str(global_config.carla_camera.image_size_y))
    camera_bp.set_attribute('iso', 
                            str(global_config.carla_camera.iso))
    camera_bp.set_attribute('gamma', 
                            str(global_config.carla_camera.gamma))
    camera_bp.set_attribute('lens_flare_intensity',
                            str(global_config.carla_camera.lens_flare_intensity))
    camera_bp.set_attribute('sensor_tick',
                            str(global_config.carla_camera.sensor_tick))
    camera_bp.set_attribute('shutter_speed',
                            str(global_config.carla_camera.shutter_speed))

    # Attach camera to the vehicle
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    camera.listen(create_camera_callback(output_dir))

    return camera


def spawn_vehicles(world,
                   traffic_manager,
                   transforms=None,
                   filter="vehicle.*.*",
                   type="car",
                   seed=None,
                   number=1,
                   project_to_road=False,
                   lane_type=carla.LaneType.Any):
    """
    Spawns vehicles in the CARLA world at specified transforms or random spawn points.

    Args:
        world: The CARLA world object.
        traffic_manager: The CARLA traffic manager object.
        transforms (list): List of transforms to spawn vehicles at. If None, random spawn points are used.
        filter (str): Blueprint filter for vehicles.
        type (str): Type of vehicle to spawn (e.g., "car").
        seed (int): Random seed for reproducibility.
        number (int): Number of vehicles to spawn.
        project_to_road (bool): Whether to project the spawn points to the road.
        lane_type (carla.LaneType): The type of lane to project to (if project_to_road is True).

    Returns:
        list: A list of spawned vehicle actors.
    """

    actor_list = []

    if transforms is None:
        transforms = _try_get_spawn_points(world, number=number, seed=seed)

    if len(transforms) < number:
        logging.warning(
            f"Requested {number} vehicles, but only {len(transforms)} spawn points available. Spawning at available spawn points."
        )
        number = len(transforms)

    vehicle_bps = _try_get_random_vehicles(world, number=number, filter=filter, type=type, seed=seed)

    for vehicle_bp, transform in zip(vehicle_bps, transforms):
        vehicle = _try_spawn_vehicle(world, vehicle_bp, transform, traffic_manager, project_to_road=project_to_road, lane_type=lane_type)

        if vehicle is not None:
            actor_list.append(vehicle)
            logging.debug(f"Vehicle {vehicle_bp.id} was spawned at {transform.location}")

    return actor_list


def spawn_vehicle(world,
                   traffic_manager,
                   transform=None,
                   filter="vehicle.*.*",
                   type="car",
                   seed=None,
                   project_to_road=False,
                   lane_type=carla.LaneType.Any):
    """
    Spawns a single vehicle in the CARLA world at the specified transform random spawn point.

    Args:
        world: The CARLA world object.
        traffic_manager: The CARLA traffic manager object.
        transform: Transform to spawn the vehicle at. If None, a random spawn point is used.
        filter (str): Blueprint filter for the vehicle.
        type (str): Type of vehicle to spawn (e.g., "car").
        seed (int): Random seed for reproducibility.
        project_to_road (bool): Whether to project the spawn point to the road.
        lane_type (carla.LaneType): The type of lane to project to (if project_to_road is True).
    
    Returns:
        vehicle: The spawned vehicle actor, or None if spawning failed.
    """

    if transform is None:
        transforms = _try_get_spawn_points(world, number=1, seed=seed)
        if not transforms:
            return None
        transform = transforms[0]

    vehicle_bps = _try_get_random_vehicles(world, number=1, filter=filter, type=type, seed=seed)
    if not vehicle_bps:
        return None
    vehicle_bp = vehicle_bps[0]

    return _try_spawn_vehicle(world, vehicle_bp, transform, traffic_manager, project_to_road=project_to_road, lane_type=lane_type)