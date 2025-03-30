#!/usr/bin/env python3

import numpy as np
from me314_py.constants import ROBOT_IP
from me314_py.robot.xarm_robot import XArm

def main():
    robot = XArm("xarm", ROBOT_IP)
    
    # Reset robot position
    input("Press Enter to initialize and reset the robot position...")
    robot.goto_reset()
    
    # Get current pose: position is returned in mm and orientation in degrees
    curr_pose = robot.arm.get_position(is_radian=False)[1]

    curr_position = np.array(curr_pose[:3]) 
    curr_orientation_deg = np.array(curr_pose[3:])
    
    print(f"Current position: {curr_position.tolist()}")
    print(f"Current orientation: {curr_orientation_deg.tolist()}")

    # Position A: lift current position by 50mm
    pos_A = curr_position + np.array([0, 0, 50])
    print(f"Moving to position A: {pos_A.tolist()}")
    pose_A = {
        "position": pos_A.tolist(),
        "orientation": curr_orientation_deg.tolist()
    }
    input("Press Enter to move to position A (lift 50mm upward)...")
    robot.command_pose(pose_A)
    
    # Position B: translate position A by 50mm in the X direction
    print(f"Moving to position B: {pos_A.tolist()} + [50, 0, 0]")
    pos_B = pos_A + np.array([50, 0, 0])
    pose_B = {
        "position": pos_B.tolist(),
        "orientation": curr_orientation_deg.tolist()
    }
    input("Press Enter to move to position B (translate 50mm in X direction)...")
    robot.command_pose(pose_B)

    input("Press Enter to return the robot to its home position...")
    robot.goto_reset()

if __name__ == '__main__':
    main()
