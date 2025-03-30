"""
a list of rotation utils
"""
import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_quaternion(roll, pitch, yaw):
    # Assuming degrees for the xArm, as in your original code
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    quat = rot.as_quat()
    # print(quat)
    return quat


def construct_robot_pose_with_rot_6d(robot_pose, rotation_transformer):
    """
    construct robot pose in 6d rotation format
    """
    pos = robot_pose[:3]
    quat = robot_pose[3:7]
    # switch from x y z w to w x y z
    quat = [quat[3], quat[0], quat[1], quat[2]]
    gripper = robot_pose[7]
    
    rot = rotation_transformer.forward(np.array(quat))
    
    # combine the position, orientation and gripper
    robot_state = np.concatenate([pos, rot, [gripper]])
    
    return robot_state