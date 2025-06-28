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
    """
    Merge point cloud frames from multiple sensors using timestamp alignment.
    Output will be stored in a temporary `.part` directory, and renamed on success.

    Args:
        sim_name (str): Simulation name, e.g., "sim_0"
        permutation_name (str): Sensor group name, e.g., "2_agent"
        sensor_list (list[str]): List of sensors to merge
    """
    sim_input_dir = os.path.join(SIM_INPUT_DIR, sim_name)
    output_dir_final = os.path.join(MERGED_OUTPUT_DIR, sim_name, permutation_name)
    output_dir_tmp = output_dir_final + ".part"

    if os.path.exists(output_dir_final):
        logging.info(f"✅ Skipping {sim_name}/{permutation_name} - already completed.")
        return
    if os.path.exists(output_dir_tmp):
        # Delete the temporary directory if it exists
        logging.warning(f"⚠️ Overwriting .part directory for {sim_name}/{permutation_name} - probably left over from an incomplete run. ")
        shutil.rmtree(output_dir_tmp)

    os.makedirs(os.path.join(output_dir_tmp, "frames"), exist_ok=True)
    logging.info(f"🔄 Merging sensors for {sim_name}/{permutation_name}...")

    try:
        merger = SensorDataMerger(
            base_dir=sim_input_dir,
            sensors=sensor_list,
            max_timestamp_discrepancy=0.2
        )
        merger.save_all_merged_plys(os.path.join(output_dir_tmp, "frames"), relative_match=True)

        # Copy GT pose file from first sensor
        reference_pose_file = os.path.join(sim_input_dir, sensor_list[0], "ground_truth_poses_tum.txt")
        if os.path.exists(reference_pose_file):
            shutil.copy(reference_pose_file, os.path.join(output_dir_tmp, "ground_truth_poses_tum.txt"))
            logging.info(f"📌 Saved GT poses from {sensor_list[0]}")
        else:
            logging.warning(f"⚠️ No ground truth pose file found for {sensor_list[0]}")

        # Finalize output
        os.rename(output_dir_tmp, output_dir_final)
        logging.info(f"✅ Finished merging {sim_name}/{permutation_name}")

    except Exception as e:
        logging.exception(f"❌ Merge failed for {sim_name}/{permutation_name}: {e}")
        # Leave .part folder for inspection


def main():
    """
    Main script entry point. Loads simulation config and performs sensor merging
    for all sensor permutations.
    """
    logging.info("📁 Loading dataset structure...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)

    for sim_name, sim_data in sim_configs.items():
        permutations = sim_data.get("sensor_permutations", {})
        for perm_name, sensors in permutations.items():
            merge_permutation(sim_name, perm_name, sensors)

    logging.info("✅ All merges completed.")


if __name__ == "__main__":
    main()
