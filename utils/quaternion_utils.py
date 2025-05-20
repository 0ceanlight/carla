import math

def quaternion_inverse(self, q):
    """Returns the inverse of a quaternion."""
    x, y, z, w = q
    return (-x, -y, -z, w)

def quaternion_multiply(self, q1, q2):
    """Multiplies two quaternions q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    :param roll: Roll angle in degrees
    :param pitch: Pitch angle in degrees
    :param yaw: Yaw angle in degrees
    :return: Quaternion as a tuple (qx, qy, qz, qw)
    """

    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return (qx, qy, qz, qw)