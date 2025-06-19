import configparser
import os
from types import SimpleNamespace
from typing import Optional

import carla


class ConfigError(Exception):
    """Raised when the simulation configuration is invalid."""
    pass


def _parse_vector(s: str) -> list[float]:
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError:
        raise ConfigError(f"Invalid vector string: '{s}'")


def _make_transform(location_str: str, rotation_str: Optional[str] = None) -> carla.Transform:
    loc = carla.Location(*_parse_vector(location_str))
    if rotation_str:
        rot = carla.Rotation(*_parse_vector(rotation_str))
    else:
        rot = carla.Rotation(0.0, 0.0, 0.0)
    return carla.Transform(loc, rot)


def _load_actor(config, section: str, base_output_dir: str) -> SimpleNamespace:
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


def load_sim_config(path: str) -> SimpleNamespace:
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
            output_dir=output_dir),
        **actors
    )

# Example usage: 
# from sim_config_parser import load_sim_config
# config = load_sim_config("config.ini")

# print(config.ego_vehicle.filter)  # "vehicle.dodge.charger"
# print(config.ego_vehicle.type)    # "car"
# print(config.ego_vehicle.transform.location)  # carla.Location(x=0.0, ...)
