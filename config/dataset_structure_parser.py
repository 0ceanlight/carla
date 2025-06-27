import json
from typing import Dict
from utils.misc import read_jsonc_file, strip_jsonc_comments


def load_simulation_config(path: str) -> Dict[str, Dict]:
    """
    Loads simulation configuration from a JSONC file.

    Args:
        path (str): Path to the JSONC config.

    Returns:
        Dict[str, Dict]: Parsed configuration.
    """
    raw = read_jsonc_file(path)
    clean_json = strip_jsonc_comments(raw)
    config = json.loads(clean_json)

    simulations = {}
    for sim_entry in config["simulations"]:
        for sim_name, sim_config in sim_entry.items():
            simulations[sim_name] = {
                "ego": sim_config["ego"],
                "sensor_permutations": sim_config["sensor_permutations"]
            }

    return simulations
