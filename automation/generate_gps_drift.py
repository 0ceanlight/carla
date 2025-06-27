import os
import numpy as np
import logging

import config.dataset_structure_parser as dataset_parser
from utils.misc import simulate_gps_drift
from utils.tum_file_parser import load_tum_file, save_tum_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Paths relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
SIM_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "build", "sim_output")

def generate_gps_drift_for_sim(sim_name: str, ego_sensor: str):
    sensor_dir = os.path.join(SIM_OUTPUT_DIR, sim_name, ego_sensor)
    gt_file = os.path.join(sensor_dir, "ground_truth_poses_tum.txt")
    gps_file = os.path.join(sensor_dir, "gps_poses_tum.txt")

    if not os.path.exists(gt_file):
        logging.warning(f"⚠️ Ground truth file missing: {gt_file}")
        return

    logging.info(f"📡 Generating GPS-drifted poses for {sim_name}/{ego_sensor}")

    try:
        raw_data = load_tum_file(gt_file)
        timestamps = np.array([entry[0] for entry in raw_data])
        poses = np.array([entry[1:] for entry in raw_data])  # shape (N, 7)

        drifted_poses = simulate_gps_drift(poses)

        drifted_data = [(ts,) + tuple(pose) for ts, pose in zip(timestamps, drifted_poses)]
        save_tum_file(gps_file, drifted_data)

        logging.info(f"✅ Saved GPS-drifted poses to {gps_file}")
    except Exception as e:
        logging.exception(f"❌ Failed to generate GPS poses for {sim_name}: {e}")


def main():
    logging.info("🔍 Loading simulation config...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)
    logging.info(f"📁 Found {len(sim_configs)} simulations in config")

    for sim_name, sim_data in sim_configs.items():
        ego_sensor = sim_data["ego"]
        generate_gps_drift_for_sim(sim_name, ego_sensor)


if __name__ == "__main__":
    main()
# TODO: untested