import configparser
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict

class ConfigError(Exception):
    """Raised when the configuration is invalid."""
    pass


def _cast(value: str, cast_type: Callable) -> Any:
    """
    Attempts to cast a string value to a specified type.

    Args:
        value: The raw string value to cast.
        cast_type: A callable that converts the string to the desired type.

    Returns:
        The casted value.

    Raises:
        ConfigError: If the casting fails.
    """
    try:
        return cast_type(value)
    except ValueError as e:
        raise ConfigError(f"Failed to cast value '{value}' to {cast_type.__name__}: {e}")


def _validate_section(config: configparser.ConfigParser, section: str, expected: Dict[str, Callable]) -> SimpleNamespace:
    """
    Validates a config section against an expected schema and returns a SimpleNamespace.

    Args:
        config: The loaded configparser object.
        section: The name of the section to validate.
        expected: A dictionary mapping expected keys to their expected types.

    Returns:
        A SimpleNamespace containing the validated and casted values.

    Raises:
        ConfigError: If the section or keys are missing, or casting fails.
    """
    if not config.has_section(section):
        raise ConfigError(f"Missing required section: [{section}]")

    data = {}
    for key, cast_func in expected.items():
        if not config.has_option(section, key):
            raise ConfigError(f"Missing key '{key}' in section [{section}]")
        raw_value = config.get(section, key)
        data[key] = _cast(raw_value, cast_func)

    return SimpleNamespace(**data)


_config = None  # Global singleton instance


def init_config(path: str):
    """
    Initializes the global configuration by loading and validating the config file.

    Args:
        path: Path to the config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If validation fails.
    """
    global _config
    _config = _load_and_validate_config(path)


def get_config() -> SimpleNamespace:
    """
    Returns the global configuration object.

    Returns:
        A SimpleNamespace containing all validated configuration sections.

    Raises:
        ConfigError: If config has not been initialized.
    """
    if _config is None:
        raise ConfigError("Configuration not initialized. Call init_config(path) first.")
    return _config


def _load_and_validate_config(path: str) -> SimpleNamespace:
    """
    Loads and validates the config file at the given path.

    Args:
        path: Path to the .ini config file.

    Returns:
        A SimpleNamespace with nested namespaces for each validated section.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If any section or field is invalid.
    """
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

# -----------------------------------
# Example usage:

# import config.global_config_parser as global_config_parser
#
# global_config_parser.init_config("config/config1.ini") # Can be omitted if already initialized
# config = global_config_parser.get_config()
#
# print(config.carla_lidar.range)          # e.g. 80.0
# print(config.carla_camera.image_size_x)  # e.g. 800
# print(config.registration.voxel_size)    # e.g. 1.0
