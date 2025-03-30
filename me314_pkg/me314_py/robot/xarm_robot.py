"""
xarm robot implementation for policy learning.
"""
from termcolor import cprint
from xarm.wrapper import XArmAPI

from me314_py.robot.robot_abc import RobotABC

class XArm(RobotABC):
    """
    Abstract base class for robot implementations.
    """

    def __init__(self, name: str, ip: str):
        """
        Initialize the robot with a name.
        """
        self.name = name
        # Initialize the robot
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(4)
        self.arm.set_state(0)
        cprint(f"Connected to robot at {ip}.", "green")

    def start(self):
        """
        Start the robot.
        """
        pass

    def stop(self):
        """
        Stop the robot.
        """
        cprint("Stopping the robot.", "red")
        self.arm.reset(wait=True)
        
    def goto_reset(self):
        self.arm.arm.set_mode(0)
        self.arm.arm.set_state(0)
        self.arm.set_position(x=245.6, y=1.8, z=164, roll=179.1, pitch=0.8, yaw=1.2,
                                speed=25, is_radian=False, wait=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(4)
        self.arm.set_state(0)

    def command_pose(self, pose: dict):
        """
        Move the robot to a given pose.
        :param pose: Dictionary containing the pose information.
        """
        x, y, z = pose["position"]
        roll, pitch, yaw = pose["orientation"]
        # XArm must be in mode 0 to move to a new position
        self.arm.arm.set_mode(0)
        self.arm.arm.set_state(0)
        self.arm.set_position(x, y, z, roll, pitch, yaw, speed=25, is_radian=False, wait=True)
        self.arm.set_mode(4)
        self.arm.set_state(0)

    def get_proprioception(self):
        """
        Get the proprioception of the robot.
        :return: Dictionary containing the proprioception information.
        """
        # this line says position, but it is pose
        status, pose = self.arm.get_position(is_radian=True)

        # convert xyz from mm to m
        pose[0:3] = [x / 1000 for x in pose[0:3]]

        # get gripper position
        status, gripper = self.arm.get_gripper_position()

        # m and radians for pose
        proprio = {
            "pose": pose,
            "gripper": gripper,
        }

        return proprio