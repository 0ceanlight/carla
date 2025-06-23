import configparser
import os
from types import SimpleNamespace
from typing import Optional

import carla


class ConfigError(Exception):
    """Raised when the simulation configuration is invalid."""
    pass


def _parse_vector(s: str) -> list[float]:
    """
    Parses a comma-separated string into a list of floats.

    Args:
        s: A string like "1.0, 2.0, 3.0"

    Returns:
        A list of floats.

    Raises:
        ConfigError: If parsing fails.
    """
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError:
        raise ConfigError(f"Invalid vector string: '{s}'")


def _make_transform(location_str: str, rotation_str: Optional[str] = None) -> carla.Transform:
    """
    Creates a CARLA Transform object from location and optional rotation strings.

    Args:
        location_str: String of comma-separated floats for location.
        rotation_str: Optional string of comma-separated floats for rotation.

    Returns:
        A carla.Transform object.
    """
    loc = carla.Location(*_parse_vector(location_str))
    rot = carla.Rotation(*_parse_vector(rotation_str)) if rotation_str else carla.Rotation(0.0, 0.0, 0.0)
    return carla.Transform(loc, rot)


def _load_actor(config, section: str, base_output_dir: str) -> SimpleNamespace:
    """
    Loads a vehicle or actor section from the config.

    Args:
        config: The configparser object.
        section: The section name in the config file.
        base_output_dir: The root directory for output data.

    Returns:
        A SimpleNamespace with actor properties.
    """
    is_random_actor = section == "other_vehicles"

    if not is_random_actor:
        if not config.has_option(section, "location"):
            raise ConfigError(f"[{section}] must have a 'location' field")
        location_str = config.get(section, "location")
        rotation_str = config.get(section, "rotation") if config.has_option(section, "rotation") else None
        transform = _make_transform(location_str, rotation_str)
    else:
        transform = None  # Randomized spawn → no transform

    data_dir = config.get(section, "data_dir")
    full_data_path = os.path.join(base_output_dir, data_dir)

    info = {
        "data_dir": full_data_path,
        "transform": transform,
    }

    for opt_key in ["filter", "type", "n_vehicles"]:
        if config.has_option(section, opt_key):
            value = config.getint(section, opt_key) if opt_key == "n_vehicles" else config.get(section, opt_key)
            info[opt_key] = value

    return SimpleNamespace(**info)


_sim_config = None  # Singleton instance


def init_config(path: str):
    """
    Initializes the simulation config from the given path.

    Args:
        path: Path to the config.ini file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ConfigError: If parsing or validation fails.
    """
    global _sim_config
    _sim_config = _load_sim_config(path)


def get_config() -> SimpleNamespace:
    """
    Returns the initialized simulation config.

    Returns:
        A SimpleNamespace with `.general`, `.ego_vehicle`, etc.

    Raises:
        ConfigError: If config has not been initialized.
    """
    if _sim_config is None:
        raise ConfigError("Simulation config not initialized. Call init_config(path) first.")
    return _sim_config


def _load_sim_config(path: str) -> SimpleNamespace:
    """
    Loads and validates the full simulation config file.

    Args:
        path: Path to the config.ini file.

    Returns:
        A SimpleNamespace containing simulation parameters and actor configs.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If required fields are missing or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)

    if "general" not in config:
        raise ConfigError("Missing [general] section")

    seed = config.getint("general", "seed")
    n_ticks = config.getint("general", "n_ticks")
    output_dir = config.get("general", "output_dir")

    actors = {}
    for section in config.sections():
        if section == "general":
            continue
        actors[section] = _load_actor(config, section, output_dir)

    return SimpleNamespace(
        general=SimpleNamespace(
            seed=seed,
            n_ticks=n_ticks,
            output_dir=output_dir
        ),
        **actors
    )

# -----------------------------------
# Example usage:

# import sim_config_parser
#
# sim_config_parser.init_config("config/sim_config.ini") # Can be omitted if already initialized
# config = sim_config_parser.get_config()
#
# print(config.ego_vehicle.filter)          # e.g. "vehicle.dodge.charger"
# print(config.ego_vehicle.type)            # e.g. "car"
# print(config.ego_vehicle.transform.location)  # e.g. carla.Location(x=0.0, y=0.0, z=0.0)
