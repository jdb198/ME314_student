#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R
import tf2_ros
from rclpy.duration import Duration
import time

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

CAMERA_OFFSET = 0.058

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pickandplace_node')

        self.bridge = CvBridge()

        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        self.realsense_sub = self.create_subscription(Image, '/color/image_raw', self.realsense_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # self.realsense_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/color/image_raw', self.realsense_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        self.depth_image = None
        self.detected_cube_world_pose = None
        self.search_complete = False

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def realsense_callback(self, msg: Image):
        if self.search_complete or self.depth_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_mask = cv2.erode(red_mask, None, iterations=2)
        red_mask = cv2.dilate(red_mask, None, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                self.get_logger().info(f"Red cube detected at image center: ({cx}, {cy})")

                # Depth lookup
                z_meters = float(self.depth_image[cy, cx]) / 1000.0
                if z_meters == 0 or np.isnan(z_meters):
                    self.get_logger().warn("Invalid depth value. Skipping.")
                    return

                # Projection with approximate intrinsics (assumes fx=fy=600, cx=320, cy=240)
                fx = fy = 640.5
                cx_d = 640.0
                cy_d = 360.0

                x_meters = (cx - cx_d) * z_meters / fx + (0.15)
                y_meters = (cy - cy_d) * z_meters / fy + (-0.07)

                # z_grasp = z_meters + 0.01  # adjusted to be just above cube surface
                z_grasp = 0.005  # adjusted to be just above cube surface

                self.detected_cube_world_pose = [x_meters, y_meters, z_grasp, 1.0, 0.0, 0.0, 0.0]

    def publish_pose(self, pose_array: list):
        """
        Publishes a pose command to the command queue using an array format.
        pose_array format: [x, y, z, qx, qy, qz, qw]
        """
        # Create a CommandQueue message containing a single pose command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the pose command
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        # Populate the pose_command with the values from the pose_array
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        """
        Publishes a gripper command to the command queue.
        For example:
          0.0 is "fully open"
          1.0 is "closed"
        """
        # Create a CommandQueue message containing a single gripper command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the gripper command
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = Example()

    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # Give time for detection
    timeout_sec = 10
    start_time = node.get_clock().now().nanoseconds
    while node.detected_cube_world_pose is None:
        rclpy.spin_once(node, timeout_sec=0.5)
        current_time = node.get_clock().now().nanoseconds
        if (current_time - start_time) > timeout_sec * 1e9:
            break

    
    # fallback test pose
    if node.detected_cube_world_pose is None:
        node.get_logger().warn("No red cube detected, using fallback pose.")
        node.detected_cube_world_pose = [0.4, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]

    approach_pose = node.detected_cube_world_pose.copy()
    lift_pose = node.detected_cube_world_pose.copy()

    green_plate_pose = [approach_pose[0], approach_pose[1] + 0.35, approach_pose[2], 1.0, 0.0, 0.0, 0.0]

    approach_pose[2] += 0.10
    node.get_logger().info("Moving above cube...")
    node.publish_pose(approach_pose)

    node.get_logger().info("Moving to cube to grasp...")
    node.publish_pose(node.detected_cube_world_pose)

    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    lift_pose[2] += 0.15
    node.get_logger().info("Lifting cube...")
    node.publish_pose(lift_pose)

    node.get_logger().info("Moving cube to green pad...")
    node.publish_pose(green_plate_pose)

    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    node.get_logger().info("Shutting down.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
