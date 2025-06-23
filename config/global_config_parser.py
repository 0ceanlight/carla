import configparser
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict


class ConfigError(Exception):
    """Raised when the configuration is invalid."""
    pass


def _cast(value: str, cast_type: Callable) -> Any:
    try:
        return cast_type(value)
    except ValueError as e:
        raise ConfigError(f"Failed to cast value '{value}' to {cast_type.__name__}: {e}")


def _validate_section(config: configparser.ConfigParser, section: str, expected: Dict[str, Callable]) -> SimpleNamespace:
    if not config.has_section(section):
        raise ConfigError(f"Missing required section: [{section}]")

    data = {}
    for key, cast_func in expected.items():
        if not config.has_option(section, key):
            raise ConfigError(f"Missing key '{key}' in section [{section}]")
        raw_value = config.get(section, key)
        data[key] = _cast(raw_value, cast_func)

    return SimpleNamespace(**data)


def load_global_config(path: str) -> SimpleNamespace:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    parser = configparser.ConfigParser()
    parser.read(path)

    # Define expected schema
    schema = {
        "carla_world": {
            "host": str,
            "simulator_port": int,
            "traffic_manager_port": int,
            "random_seed": int,
            "output_dir": str,
        },
        "carla_lidar": {
            "channels": int,
            "range": float,
            "points_per_second": int,
            "rotation_frequency": float,
            "upper_fov": float,
            "lower_fov": float,
            "horizontal_fov": float,
            "atmosphere_attenuation_rate": float,
            "dropoff_general_rate": float,
            "dropoff_intensity_limit": float,
            "dropoff_zero_intensity": float,
            "sensor_tick": float,
            "noise_stddev": float,
        },
        "carla_camera": {
            "bloom_intensity": float,
            "fov": float,
            "fstop": float,
            "image_size_x": int,
            "image_size_y": int,
            "iso": float,
            "gamma": float,
            "lens_flare_intensity": float,
            "sensor_tick": float,
            "shutter_speed": float,
        },
        "registration": {
            "voxel_size": float,
            "max_iterations": int,
        }
    }

    # Validate and load sections
    config_ns = {}
    for section, expected_fields in schema.items():
        config_ns[section] = _validate_section(parser, section, expected_fields)

    return SimpleNamespace(**config_ns)

# Example usage:
# import carla_config_parser
# 
# config = carla_config_parser.load_config("path/to/your/config.ini")
# 
# print(config.carla_lidar.range)          # 80.0
# print(config.carla_camera.image_size_x)  # 800
# print(config.registration.voxel_size)    # 1.0
