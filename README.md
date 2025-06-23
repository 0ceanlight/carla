# CARLA

Multi-Modal Collaborative Perception for V2X in CARLA: A Deep 3D Point Cloud Alignment Approach for Intersection Localization.

## Step 0: Getting started

### Starting the Container

Prerequisites: See [CARLA's documentation](https://carla-ue5.readthedocs.io/en/latest/). We will be using the [Docker setup](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/#running-carla-using-a-docker-container) for CARLA 0.10.0.

First, make sure you are in this repository's directory. This will be relevant during the docker mounting process.

```
cd /path/to/this/repo
```

First, run CARLA's 0.10.0 DockerHub image. The following command does several things:
- Downloads DockerHub image `carla:0.10.0`, if it isn't already downloaded.
- Runs `./CarlaUnreal.sh` and starts the CARLA server in headless mode (this may take a few seconds to load, depending on your GPU). To run CARLA in a windowed mode, you can remove the `-RenderOffScreen` flag, but this is not recommended due to the exceptionally high resource usage. Instead, instantiate a spectator using the `spectator_mode.py` script.
- Mounts this repository at `/mnt` inside the container. This gives us access from within the container to run our Python scripts with the simulator. <br />
Note: If you are running the command from somewhere else, replace `$(pwd)` with the path to this repository.
- Gives our container the name `carla_server`. You can use either this or the container ID to attach terminals in the subsequent commands.

```
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

```
docker exec -u root -it carla_server bash
```

Install `pip` and `git`.

```
apt-get update && apt-get upgrade -y
apt-get install -y python3-pip git
```

Type `exit` in your root container terminal, open a new terminal. 

#### Python Packages

Next, attach a non-root user terminal to the running container.

```
docker exec -it carla_server bash
```

Install Python dependencies and CARLA's PythonAPI.

```
cd ~/PythonAPI/examples
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install ../carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install shapely networkx
```

If you're not running scripts from within the carla `PythonAPI` directory, you will need to add it to your `PYTHONPATH`.

```
export PYTHONPATH=/home/carla/PythonAPI/carla:$PYTHONPATH
```

Or, to keep this in your path when you open a new terminal, add it to `.bashrc`.

```
echo 'export PYTHONPATH=/home/carla/PythonAPI/carla:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

<br />

# TODO: Schmeiß alles in Dockerfile

### PythonAPI Test Run (optional)

Now you're all set to try out any one of the scripts in the examples folder!

Let's try running `generate_traffic.py`.

```
python3 generate_traffic.py
```

This should spawn multiple pedestrians and cars driving around the CARLA map.


## Step 1: Collecting Data from CARLA

### Configuration

In order to run a custom simulation, you will need to create two configuration files: a **simulation** configuration file (`*.ini`) and a **global** CARLA configuration file (`*.ini`). When creating your own configuration files, make sure to instantiate the same fields as in the provided example files.

- The **global** configuration file defines the CARLA server settings and sensor configuration, such as the ports to connect to, resolution of sensors, and other global parameters. It is recommended to use the default settings provided in the `global_config.ini` file in the `config` directory, but you can modify it as needed for your simulation.

- The **simulation** configuration file defines the specific parameters for your simulation, such as the number of vehicles, their initial positions, and the position of sensors. This file will be used by the `main_loop.py` script to set up the simulation environment. For an example configuration, you can refer to `sim_config_0.ini` in the `config` directory.

### Running the Simulation

Before continuing, make sure you have the CARLA server running on the port specified in your global configuration file (default is 2000). You can start the CARLA server using the Docker command provided above. It can take a few seconds for the server to load. If it is not running, the following scripts will throw a `RuntimeError: Connection Refused`.

To run the simulation, use the `main_loop.py` script. This will start the CARLA server, spawn vehicles and sensors, and collect data based on your configuration. You can use the following arguments:
- `--verbose`, `-v`: Enable verbose output for debugging purposes.
- `--sim_config`, `-s`: Path to your simulation configuration file (e.g., `config/sim_config_0.ini`).
- `--global_config`, `-g`: Path to your global configuration file (e.g., `config/global_config.ini`).
- `--no-save`, `-n`: Disable saving the collected data to disk. This is especially useful for visualizing the simulation without saving data, and runs a heck of a lot faster. You can run `spectator_mode.py` before running `main_loop.py` to visualize the simulation in real-time.
