import carla
import numpy as np
import cv2
import pyglet
from pynput import keyboard
from threading import Lock
import math
import time
import argparse

# Constants
BASE_SPEED = 100.0  # m/s
BOOST_MULTIPLIER = 1.3
SNAIL_MULTIPLIER = 0.3
ROTATE_SPEED = 30.0  # deg/s
MOUSE_SENSITIVITY = 0.1  # deg/pixel

# CARLA window size
CARLA_WINDOW_WIDTH = 900 * 2
CARLA_WINDOW_HEIGHT = 440 * 2

# Mouse control window size
MOUSE_WINDOW_WIDTH = 800
MOUSE_WINDOW_HEIGHT = 600

SHOW_MOUSE_WINDOW = False # Set to False to hide mouse control window

# Key configs (note that these can lead to collisions with other keys)
# # Movement
# forward_key = 'w'
# backward_key = 's'
# left_key = 'a'
# right_key = 'd'
# up_key = 'space'
# down_key = 'shift'
# fast_key = 'ctrl'
# slow_key = 'alt'
# # Rotation
# look_up_key = 'up'
# look_down_key = 'down'
# look_right_key = 'right'
# look_left_key = 'left'

# Movement - shifted minecraft
forward_key = 'e'
backward_key = 'd'
left_key = 's'
right_key = 'f'
up_key = 'space'
down_key = 'shift'
fast_key = 'a'
slow_key = 'z'
# Rotation - vim movement keybinds
look_up_key = 'k'
look_down_key = 'j'
look_right_key = 'l'
look_left_key = 'h'

class KeyState:
    """
    Tracks currently pressed keyboard keys using pynput in a thread-safe way.
    """
    def __init__(self):
        self.pressed_keys = set()
        self.lock = Lock()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key):
        with self.lock:
            self.pressed_keys.discard(key)

    def is_pressed(self, key_str):
        """
        Check if a key is currently pressed.

        Args:
            key_str (str): The key character or name (e.g., 'w', 'space', 'ctrl').

        Returns:
            bool: True if the key is pressed, False otherwise.
        """
        with self.lock:
            for key in self.pressed_keys:
                if hasattr(key, 'char') and key.char == key_str:
                    return True
                if hasattr(key, 'name') and key.name == key_str:
                    return True
        return False

class MouseRotationWindow(pyglet.window.Window):
    """
    Pyglet window used to track mouse movement for camera rotation.
    """
    def __init__(self, width=800, height=600, visible=True):
        super().__init__(width=width, height=height, visible=visible, caption="Camera Mouse Control")
        self.set_mouse_visible(False)
        self.dx = 0
        self.dy = 0
        self.mouse_cx = self.width // 2
        self.mouse_cy = self.height // 2
        self.set_mouse_position(self.mouse_cx, self.mouse_cy)
        pyglet.clock.schedule_interval(self.clear_mouse_delta, 1/60)

        # Make window capture mouse. This is activated when window is focused.
        # This can be exited by using a system shortcut like Super+Tab to focus
        # another window.
        self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        self.dx += dx
        self.dy += dy

    def get_mouse_delta(self):
        """
        Get the accumulated mouse motion since last call.

        Returns:
            Tuple[int, int]: (dx, dy)
        """
        return self.dx, self.dy

    def clear_mouse_delta(self, dt):
        self.dx = 0
        self.dy = 0

def clamp_pitch(pitch):
    """
    Clamp pitch rotation to [-90, 90] degrees.

    Args:
        pitch (float): Desired pitch.

    Returns:
        float: Clamped pitch.
    """
    return max(-90.0, min(90.0, pitch))

def main():
    """
    Connects to CARLA, spawns a floating spectator-attached camera, and allows
    Minecraft-like control via WASD/arrow keys and mouse.
    """

    # Argparser for display options
    parser = argparse.ArgumentParser(description="CARLA Spectator Mode")
    parser.add_argument(
        '-s', '--show_mouse_window', type=bool, default=SHOW_MOUSE_WINDOW, 
        dest='show_mouse_window', help='Show mouse control window (default: False)')
    parser.add_argument(
        '-w', '--window_size', type=int, nargs=2, default=(MOUSE_WINDOW_WIDTH, MOUSE_WINDOW_HEIGHT),
        dest='window_size', 
        help=f'Set mouse control window size: width height (default: {MOUSE_WINDOW_WIDTH} {MOUSE_WINDOW_HEIGHT})')
    parser.add_argument(
        '-c', '--carla_window_size', type=int, nargs=2, default=(CARLA_WINDOW_WIDTH, CARLA_WINDOW_HEIGHT),
        dest='carla_window_size',
        help=f'Set CARLA window size: width height (default: {CARLA_WINDOW_WIDTH} {CARLA_WINDOW_HEIGHT})')
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Create camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(args.carla_window_size[0]))
    camera_bp.set_attribute('image_size_y', str(args.carla_window_size[1]))
    camera_bp.set_attribute('fov', '90')

    # Set spectator position and orientation
    spectator = world.get_spectator()
    start_transform = carla.Transform(carla.Location(x=-50, y=20, z=50), carla.Rotation(pitch=-90.0))
    spectator.set_transform(start_transform)

    # Spawn camera sensor attached to spectator
    camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=spectator)

    image_data = {"frame": None}

    def process_img(image):
        """
        Callback for incoming camera frames.

        Args:
            image (carla.Image): The image from the sensor.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        image_data["frame"] = array

    camera.listen(process_img)

    keys = KeyState()
    mouse_window = MouseRotationWindow(width=args.window_size[0], height=args.window_size[1], visible=args.show_mouse_window)

    pitch = -90.0
    yaw = 0.0
    prev_time = time.time()

    try:
        while not mouse_window.has_exit:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            # Determine movement speed
            speed = BASE_SPEED * (BOOST_MULTIPLIER if keys.is_pressed(fast_key) else 1.0)
            speed = BASE_SPEED * (SNAIL_MULTIPLIER if keys.is_pressed(slow_key) else 1.0)

            # Handle rotation input
            if keys.is_pressed(look_up_key):
                pitch += ROTATE_SPEED * dt
            if keys.is_pressed(look_down_key):
                pitch -= ROTATE_SPEED * dt
            if keys.is_pressed(look_left_key):
                yaw -= ROTATE_SPEED * dt
            if keys.is_pressed(look_right_key):
                yaw += ROTATE_SPEED * dt

            dx, dy = mouse_window.get_mouse_delta()
            yaw += dx * MOUSE_SENSITIVITY
            pitch += dy * MOUSE_SENSITIVITY
            pitch = clamp_pitch(pitch)

            # Compute direction vectors (horizontal plane)
            yaw_rad = math.radians(yaw)
            forward = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad))
            right = carla.Vector3D(x=-math.sin(yaw_rad), y=math.cos(yaw_rad))

            # Compute movement vector
            move = carla.Location()

            if keys.is_pressed(forward_key):
                move += carla.Location(x=forward.x * speed * dt, y=forward.y * speed * dt)
            if keys.is_pressed(backward_key):
                move -= carla.Location(x=forward.x * speed * dt, y=forward.y * speed * dt)
            if keys.is_pressed(left_key):
                move -= carla.Location(x=right.x * speed * dt, y=right.y * speed * dt)
            if keys.is_pressed(right_key):
                move += carla.Location(x=right.x * speed * dt, y=right.y * speed * dt)
            if keys.is_pressed(up_key):
                move.z += speed * dt
            if keys.is_pressed(down_key):
                move.z -= speed * dt

            # Update spectator transform
            current = spectator.get_transform()
            loc = current.location + move
            rot = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
            spectator.set_transform(carla.Transform(loc, rot))

            # Debug output
            print(f"Location: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}) | Pitch: {pitch:.1f}, Yaw: {yaw:.1f}")

            # Show camera image
            if image_data["frame"] is not None:
                cv2.imshow("Spectator Camera", image_data["frame"])
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or keys.is_pressed('q'):
                    break

            pyglet.clock.tick()
            mouse_window.dispatch_events()

    finally:
        print("Cleaning up...")
        camera.stop()
        camera.destroy()
        cv2.destroyAllWindows()
        mouse_window.close()

if __name__ == "__main__":
    main()

