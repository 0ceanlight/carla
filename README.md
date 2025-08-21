<p align="center">
<h1 align="center">Multi-Modal Collaborative Perception for V2X in CARLA</h1>

  <p align="center">
    <a href="https://github.com/0ceanlight/carla#starting-the-container"><img src="https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=Docker&logoColor=white" /></a>
    <a href="https://github.com/0ceanlight/carla#python-packages"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://drive.google.com/file/d/1GWYTYjibJONULrEsY3bM9CpGix62-yYt/view?usp=sharing"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/0ceanlight/carla/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>

  <p align="center"><a href="https://www.tum.de/"><strong>Technical University of Munich</strong></a>
  <h3 align="center"><a href="https://drive.google.com/file/d/1GWYTYjibJONULrEsY3bM9CpGix62-yYt/view?usp=sharing">Project Paper</a></h3>
  <div align="center"></div>
</p>

This project presents a 3D point cloud alignment approach for the localization of a vehicle in an intersection with sub-decimeter accuracy using multi-modal collaborative perception. 



## Table of Contents
- [Table of Contents](#table-of-contents)
- [Getting started](#getting-started)
  - [Starting the Container](#starting-the-container)
  - [Installing packages](#installing-packages)
    - [System Packages](#system-packages)
    - [Python Packages](#python-packages)
  - [PythonAPI Test Run (optional)](#pythonapi-test-run-optional)
- [Next Steps](#next-steps)
- [1. Running a Custom Simulation](#1-running-a-custom-simulation)
  - [CARLA Configuration](#carla-configuration)
  - [Running the CARLA Simulation](#running-the-carla-simulation)
- [2. Recreating the Data Collection Process](#2-recreating-the-data-collection-process)

## Getting started

First, we need to set up the CARLA simulator and install the required packages to run the scripts in this repository. This guide assumes you have Docker installed and are familiar with basic command line operations.

### Starting the Container

Prerequisites: See [CARLA's documentation](https://carla-ue5.readthedocs.io/en/latest/). We will be using the [Docker setup](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/#running-carla-using-a-docker-container) for CARLA 0.10.0.

First, make sure you are in this repository's directory. This will be relevant during the docker mounting process.

```bash
cd /path/to/this/repo
```

First, run CARLA's 0.10.0 DockerHub image. The following command does several things:
- Downloads DockerHub image `carla:0.10.0`, if it isn't already downloaded.
- Runs `./CarlaUnreal.sh` and starts the CARLA server in headless mode (this may take a few seconds to load, depending on your GPU). To run CARLA in a windowed mode, you can remove the `-RenderOffScreen` flag, but this is not recommended due to the exceptionally high resource usage. Instead, instantiate a spectator using the `spectator_mode.py` script.
- Mounts this repository at `/mnt` inside the container. This gives us access from within the container to run our Python scripts with the simulator. <br />
Note: If you are running the command from somewhere else, replace `$(pwd)` with the path to this repository.
- Gives our container the name `carla_server`. You can use either this or the container ID to attach terminals in the subsequent commands.

```bash
docker run -d \
    --mount type=bind,src=$(pwd),dst=/mnt \
    --name carla_server \
    --runtime=nvidia \
    --net=host \
    --user=$(id -u):$(id -g) \
    --env=DISPLAY=$DISPLAY \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    carlasim/carla:0.10.0 bash CarlaUnreal.sh -nosound -RenderOffScreen
```

<br />

### Installing packages

Next, we will install the packages required to run scripts that use CARLA's PythonAPI inside of our container.

#### System Packages

Attach a root terminal to the running container (we need root to run `apt-get`).

```bash
docker exec -u root -it carla_server bash
```

Install `pip` and `git`.

```bash
apt-get update && apt-get upgrade -y
apt-get install -y python3-pip git
```

Type `exit` in your root container terminal, open a new terminal. 

#### Python Packages

Next, attach a non-root user terminal to the running container.

```bash
docker exec -it carla_server bash
```

Install Python dependencies and CARLA's PythonAPI.

```bash
cd ~/PythonAPI/examples
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install ../carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install shapely networkx
```

If you're not running scripts from within the carla `PythonAPI` directory, you will need to add it to your `PYTHONPATH`.

```bash
export PYTHONPATH=/home/carla/PythonAPI/carla:$PYTHONPATH
```

Or, to keep this in your path when you open a new terminal, add it to `.bashrc`.

```bash
echo 'export PYTHONPATH=/home/carla/PythonAPI/carla:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

<br />

<!-- TODO: Schmeiß alles in Dockerfile -->

### PythonAPI Test Run (optional)

Now you're all set to try out any one of the scripts in the examples folder!

Let's try running `generate_traffic.py`.

```bash
python3 generate_traffic.py
```

This should spawn multiple pedestrians and cars driving around the CARLA map.


## Next Steps

Now that we have CARLA up and running, we can start collecting and evaluating data for our multi-modal collaborative perception system. 

First, install the required Python packages for this repository. Make sure you are in the root directory of this repository and run:

```bash
pip install -r requirements.txt
```

We now have two options:

1. Running and evaluating a custom simulation in CARLA.
2. Recreating the data collection process from the original paper.

## 1. Running a Custom Simulation

### CARLA Configuration

In order to run a custom simulation, you will need to create two configuration files: a **simulation** configuration file (`*.ini`) and a **global** CARLA configuration file (`*.ini`). When creating your own configuration files, make sure to instantiate the same fields as in the provided example files.

- The **global** configuration file defines the CARLA server settings and sensor configuration, such as the ports to connect to, resolution of sensors, and other global parameters. It is recommended to use the default settings provided in the `global_config.ini` file in the `config` directory, but you can modify it as needed for your simulation.

- The **simulation** configuration file defines the specific parameters for your simulation, such as the number of vehicles, their initial positions, and the position of sensors. This file will be used by the `main_loop.py` script to set up the simulation environment. For an example configuration, you can refer to `sim_config_0.ini` in the `config` directory.

### Running the CARLA Simulation

Before continuing, make sure you have the CARLA server running on the port specified in your global configuration file (default is 2000). You can start the CARLA server using the Docker command provided above. It can take a few seconds for the server to load. If it is not running, the following scripts will throw a `RuntimeError: Connection Refused`.

To run the simulation, use the `main_loop.py` script. This will start the CARLA server, spawn vehicles and sensors, and collect data based on your configuration. You can use the following arguments:
- `--verbose`, `-v`: Enable verbose output for debugging purposes.
- `--sim_config`, `-s`: Path to your simulation configuration file (e.g., `config/sim_config_0.ini`).
- `--global_config`, `-g`: Path to your global configuration file (e.g., `config/global_config.ini`).
- `--no-save`, `-n`: Disable saving the collected data to disk. This is especially useful for visualizing the simulation without saving data, and runs a heck of a lot faster. You can run `spectator_mode.py` before running `main_loop.py` to visualize the simulation in real-time.

## 2. Recreating the Data Collection Process

For the recreation of the data collection process from the original paper, you can use the provided scripts in the `automation` directory. These scripts will automatically set up the CARLA environment, spawn vehicles and sensors, collect data based on the original paper's configuration, and evaluate the results.

In order to do this, please run the scripts in order of their numbering, as in the following example:

```bash
# Make sure you are in the root directory of this repository.
chmod +x automation/0_collect_carla_data.sh
# This script assumes the CARLA server is running, will spawn vehicles and sensors, and collect data based on the original paper's configuration. Output is saved in the `build/sim_output` directory.
./automation/0_collect_carla_data.sh
# This script simulates GPS drift, generating a `gps_poses_tum.txt` file in the `build/sim_output` directory of each simulation's ego_lidar.
python3 1_generate_gps_drift.py
# This script merges point cloud data for sensors other than the ego lidar, placing results in `build/gt_merged_sim_output`.
python3 1_merge_gt_clouds.py
# This script runs PIN-SLAM on each ego lidar point cloud, placing results in `build/slam_output_ego_only`.
# This script requires sudo permissions to make SLAM data accessible to the current user due to the use of docker as root.
python3 1_run_ego_slam.py
# This script registers the ego lidar point cloud with the merged point clouds from other sensors, placing results in `build/registered_sim_output`.
python3 2_register_ego_with_merged.py
# This script runs GTSAM pose graph estimation on previous results, creating final trajectory estimates in `build/fused_output`.
python3 3_fuse_slam_reg_poses.py
# This script calculates average error in GPS, SLAM, registration, and fused trajectories, printing averages over each simulation and permuation to stdout.
python3 print_avg_results.py
````