import configparser

class CarlaConfig:
    def __init__(self):
        self.host = ""
        self.simulator_port = 0
        self.traffic_manager_port = 0
        self.random_seed = 0
        self.output_dir = ""

        self.channels = 0
        self.range = 0.0
        self.points_per_second = 0
        self.rotation_frequency = 0.0
        self.upper_fov = 0.0
        self.lower_fov = 0.0
        self.horizontal_fov = 0.0
        self.atmosphere_attenuation_rate = 0.0
        self.dropoff_general_rate = 0.0
        self.dropoff_intensity_limit = 0.0
        self.dropoff_zero_intensity = 0.0
        self.sensor_tick = 0.0
        self.noise_stddev = 0.0

        self.voxel_size = 0.0

    def read(self, file):
        config = configparser.ConfigParser()
        try:
            config.read(file)
        except:
            raise FileNotFoundError(f"Given config file {file} does not exist")

        try:
            self.carla_host = config["carla-world"]["host"]
            self.carla_simulator_port = int(config["carla-world"]["simulator_port"])
            self.carla_traffic_manager_port = int(config["carla-world"]["traffic_manager_port"])
            self.carla_random_seed = int(config["carla-world"]["random_seed"])
            self.carla_output_dir = config["carla-world"]["output_dir"]

            self.lidar_channels = int(config["carla-lidar"]["channels"])
            self.lidar_range = float(config["carla-lidar"]["range"])
            self.lidar_points_per_second = int(config["carla-lidar"]["points_per_second"])
            self.lidar_rotation_frequency = float(config["carla-lidar"]["rotation_frequency"])
            self.lidar_upper_fov = float(config["carla-lidar"]["upper_fov"])
            self.lidar_lower_fov = float(config["carla-lidar"]["lower_fov"])
            self.lidar_horizontal_fov = float(config["carla-lidar"]["horizontal_fov"])
            self.lidar_atmosphere_attenuation_rate = float(config["carla-lidar"]["atmosphere_attenuation_rate"])
            self.lidar_dropoff_general_rate = float(config["carla-lidar"]["dropoff_general_rate"])
            self.lidar_dropoff_intensity_limit = float(config["carla-lidar"]["dropoff_intensity_limit"])
            self.lidar_dropoff_zero_intensity = float(config["carla-lidar"]["dropoff_zero_intensity"])
            self.lidar_sensor_tick = float(config["carla-lidar"]["sensor_tick"])
            self.lidar_noise_stddev = float(config["carla-lidar"]["noise_stddev"])

            self.registration_voxel_size = float(config["registration"]["voxel_size"])
        except KeyError as e:
            raise ValueError(f"Missing key in config file: {e}")

        return self