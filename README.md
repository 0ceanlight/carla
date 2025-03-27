# CARLA

Multi-Modal Collaborative Perception for V2X in CARLA: A Deep 3D Point Cloud Alignment Approach for Intersection Localization.

## Getting started

### Starting the Container

Prerequisites: See [CARLA's documentation](https://carla-ue5.readthedocs.io/en/latest/). We will be using the [Docker setup](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/#running-carla-using-a-docker-container) for CARLA 0.10.0.

First, make sure you are in this repository's directory. This will be relevant during the docker mounting process.

```
cd /path/to/this/repo
```

First, run CARLA's 0.10.0 DockerHub image. The following command does several things:
- Downloads DockerHub image `carla:0.10.0`, if it isn't already downloaded.
- Runs `./CarlaUnreal.sh` to start the CARLA server in a new window (this may take a few seconds to load, depending on your GPU).
- Note: You can also run it with flags like `-dx11` to use DirectX, or `-quality-level=Low` to use less resources
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
    carlasim/carla:0.10.0 bash CarlaUnreal.sh -nosound
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

### PythonAPI Test Run

Now you're all set to try out any one of the scripts in the examples folder!

Let's try running `generate_traffic.py`.

```
python3 generate_traffic.py
```

This should spawn multiple pedestrians and cars driving around the CARLA map.
