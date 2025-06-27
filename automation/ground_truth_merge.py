import os
import shutil
import logging

import config.dataset_structure_parser as dataset_parser
from utils.sensor_data_merger import SensorDataMerger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Absolute base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
SIM_INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "build", "sim_output")
MERGED_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "build", "gt_merged_sim_output")


def merge_permutation(sim_name: str, permutation_name: str, sensor_list: list[str]):
    sim_input_dir = os.path.join(SIM_INPUT_DIR, sim_name)
    output_dir = os.path.join(MERGED_OUTPUT_DIR, sim_name, permutation_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"🔄 Merging sensors for {sim_name}/{permutation_name}")

    try:
        merger = SensorDataMerger(
            base_dir=sim_input_dir,
            sensors=sensor_list,
            max_timestamp_discrepancy=0.2
        )
        merger.save_all_merged_plys(os.path.join(output_dir, "frames"), relative_match=True)

        # Copy ground truth pose file from first sensor
        reference_pose_file = os.path.join(sim_input_dir, sensor_list[0], "ground_truth_poses_tum.txt")
        if os.path.exists(reference_pose_file):
            shutil.copy(reference_pose_file, os.path.join(output_dir, "ground_truth_poses_tum.txt"))
            logging.info(f"📌 Saved GT poses from {sensor_list[0]} to merged folder.")
        else:
            logging.warning(f"⚠️ No ground truth pose file found for {sensor_list[0]}")
    except Exception as e:
        logging.exception(f"❌ Merge failed for {sim_name}/{permutation_name}: {e}")


def main():
    logging.info("📁 Loading dataset structure...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)

    for sim_name, sim_data in sim_configs.items():
        permutations = sim_data.get("sensor_permutations", {})
        for perm_name, sensors in permutations.items():
            merge_permutation(sim_name, perm_name, sensors)

    logging.info("✅ All merges completed.")


if __name__ == "__main__":
    main()
