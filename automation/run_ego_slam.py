import os
import subprocess
import logging
import config.dataset_structure_parser as dataset_parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Paths relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "dataset_structure.jsonc")
BASE_INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "build", "sim_output")
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "build", "slam_output_ego_only")
DOCKER_MOUNT_DIR = os.path.join(SCRIPT_DIR, "..", "build")


def run_slam_on_sim(sim_name: str, ego_sensor_name: str):
    input_path = f"/storage/sim_output/{sim_name}/{ego_sensor_name}/frames"
    output_path_container = f"/storage/slam_output_ego_only/{sim_name}"
    output_path_host = os.path.join(BASE_OUTPUT_DIR, sim_name)

    os.makedirs(output_path_host, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm", "--ipc", "host", "--privileged", "--network", "host",
        "-p", "8080:8081", "--gpus", "all",
        "-v", f"{DOCKER_MOUNT_DIR}:/storage/",
        "pinslam:localbuild",
        "/usr/bin/python3", "pin_slam.py", "-sm",
        "-i", input_path,
        "-o", output_path_container
    ]

    logging.info(f"Running SLAM for '{sim_name}' using ego sensor '{ego_sensor_name}'")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Finished SLAM for '{sim_name}'")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ SLAM failed for '{sim_name}' with error: {e}")
    except Exception as ex:
        logging.exception(f"Unexpected error during SLAM for '{sim_name}': {ex}")


def main():
    logging.info("🔍 Loading simulation config...")
    sim_configs = dataset_parser.load_simulation_config(CONFIG_PATH)
    logging.info(f"🔧 Loaded {len(sim_configs)} simulations")

    for sim_name, sim_data in sim_configs.items():
        ego_sensor = sim_data["ego"]
        run_slam_on_sim(sim_name, ego_sensor)

    # Change ownership of the output directory after all processing
    uid = os.getuid()
    gid = os.getgid()
    logging.info(f"🔑 Changing ownership of {BASE_OUTPUT_DIR} to UID:{uid} GID:{gid}")
    try:
        subprocess.run(["sudo", "chown", "-R", f"{uid}:{gid}", BASE_OUTPUT_DIR], check=True)
        logging.info("✅ Ownership update complete.")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to update ownership: {e}")
    except Exception as ex:
        logging.exception(f"Unexpected error during chown: {ex}")


if __name__ == "__main__":
    main()