"""
This file contains utility functions
"""
from scipy.spatial.transform import Rotation

def normalize_gripper_val(x, min_val=-10, max_val=850):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def unnormalize_gripper_val(normalized_x, min_val=-10, max_val=850):
    return ((normalized_x + 1) / 2) * (max_val - min_val) + min_val

def euler_to_quaternion(roll, pitch, yaw):
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_quat()

def quaternion_to_euler(qx, qy, qz, qw):
    rot = Rotation.from_quat([qx, qy, qz, qw])
    return rot.as_euler('xyz')
