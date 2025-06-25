import configparser
import os
from types import SimpleNamespace
from typing import Optional

import carla


class ConfigError(Exception):
    """Raised when the simulation configuration is invalid."""
    pass


_config = None


def _parse_vector(s: str) -> list[float]:
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError:
        raise ConfigError(f"Invalid vector string: {s}")


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


def _get_attach_to(sensor_name: str, attach_to: Optional[str], agents: list[SimpleNamespace]) -> Optional[str]:
    """
    Gets the attach_to value for a sensor, ensuring it matches an existing agent.
    
    Args:
        attach_to: The name of the agent to attach to, or None.
        agents: List of existing agents."""

    if attach_to is None or attach_to.strip().lower() == "none":
        return None
    if not any(agent.name == attach_to for agent in agents):
        raise ConfigError(f"""Sensor {sensor_name} specified to attach to 
                            {attach_to}, but no such agent exists. This field 
                            should be either a valid agent name, or 
                            'None'""")
    return attach_to


class DotDict(SimpleNamespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def init_config(path: str):
    """
    Initializes the simulation config from the given path.

    Args:
        path: Path to the config.ini file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ConfigError: If parsing or validation fails.
    """
    global _config

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    parser = configparser.ConfigParser()
    parser.read(path)

    general = parser["general"]
    general_config = DotDict(
        name="general",
        seed=int(general["seed"]) if general["seed"].strip().lower() != "none" else None,
        n_ticks=int(general["n_ticks"]),
        output_dir=general["output_dir"],
    )

    agents = []
    sensors = []
    other_vehicles = None

    for section in parser.sections():
        if section == "general":
            continue

        item = parser[section]
        name = section

        if name.endswith("_vehicle"):
            # single agent
            transform = _make_transform(item["location"], item.get("rotation"))
            agent = DotDict(
                name=name,
                transform=transform,
                location=item["location"],
                rotation=item.get("rotation", "0.0,0.0,0.0"),
                filter=item["filter"],
                type=item["type"],
                autopilot=item.getboolean("autopilot")
            )
            agents.append(agent)
            if name == "ego_vehicle":
                ego_vehicle = agent

        elif name.endswith("_lidar") or name.endswith("_camera"):
            transform = _make_transform(item["location"], item.get("rotation"))
            attach_to = _get_attach_to(name, item.get("attach_to"), agents)
            # Join output_dir with data_dir
            abs_dir = os.path.join(general_config.output_dir, item.get("data_dir", name))
            sensor = DotDict(
                name=name,
                attach_to=attach_to,
                location=item["location"],
                rotation=item.get("rotation", "0.0,0.0,0.0"),
                transform=transform,
                data_dir=abs_dir,
            )
            sensors.append(sensor)

        elif name == "other_vehicles":
            other_vehicles = DotDict(
                name=name,
                n_vehicles=int(item["n_vehicles"]),
                filter=item["filter"],
                type=item["type"]
            )

    _config = DotDict(
        general=general_config,
        sensors=sensors,
        agents=agents,
        other_vehicles=other_vehicles,
        ego_vehicle=ego_vehicle
    )


def get_config():
    """
    Returns the initialized simulation config.

    Returns:
        The simulation config object.

    Raises:
        ConfigError: If the config has not been initialized.
    """
    if _config is None:
        raise ConfigError("Simulation config not initialized. Call init_config(path) first.")
    return _config

# -----------------------------------
# Example usage:

# import sim_config_parser
#
# sim_config_parser.init_config("config/sim_config.ini") # Can be omitted if already initialized
# config = sim_config_parser.get_config()
# 
# for vehicle in config.vehicles:
#     print(vehicle.filter)          # e.g. "vehicle.dodge.charger"
#     print(vehicle.type)            # e.g. "car"
#     print(vehicle.transform.location)  # e.g. carla.Location(x=0.0, y=0.0, z=0.0)
