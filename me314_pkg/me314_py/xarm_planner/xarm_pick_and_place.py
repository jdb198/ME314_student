#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper


class Example(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        
        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        self.current_RGB = None
        self.RGB_sub = self.create_subscription(Image, '/color/image_raw', self.realsense_color_callback, 10)

        self.current_depth = None
        self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.realsense_depth_callback, 10)

        self.current_info = None
        self.info_sub = self.create_subscription(CameraInfo, '/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def realsense_color_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.current_RGB = cv_image

    def realsense_depth_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.current_depth = cv_image
    
    def camera_info_callback(self, msg: CameraInfo):
        self.current_info = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

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

    def generate_pc(self, masked_depth):
        self.cam_model.fromCameraInfo(self.current_info)
        out = np.zeros([np.sum(masked_depth > 1), 3])
        
        i = 0
        for v in range(masked_depth.shape[0]):
            for u in range(masked_depth.shape[1]):
                depth = masked_depth[v, u] 

                if depth > 0.001:
                    depth = depth * 0.001
                    x, y, z = self.cam_model.projectPixelTo3dRay((u, v))
                    x *= depth
                    y *= depth
                    z *= depth
                    out[i, 0] = x
                    out[i, 1] = y
                    out[i, 2] = z
                    i += 1
                    #self.get_logger().info(f"3D point at ({u}, {v}): ({x:.3f}, {y:.3f}, {z:.3f})")
        return out   
    
    def transform_camera_to_world(self, point_camera_frame):
        # converts to a point stamped, which is some ros format.
        if type(point_camera_frame) == np.ndarray:
            camera_point_ros = PointStamped()

            camera_point_ros.header.frame_id = 'camera_color_optical_frame'
            camera_point_ros.point.x = float(point_camera_frame[0])
            camera_point_ros.point.y = float(point_camera_frame[1])
            camera_point_ros.point.z = float(point_camera_frame[2])
        else:
            camera_point_ros = point_camera_frame
        transform = self.tf_buffer.lookup_transform(
                'world',
                'camera_color_optical_frame',
                rclpy.time.Time()
            )

        # transform the point
        point_in_world = tf2_geometry_msgs.do_transform_point(camera_point_ros, transform)
        if type(point_camera_frame) == np.ndarray:
            return np.array([point_in_world.point.x, point_in_world.point.y, point_in_world.point.z])
        return point_in_world

def main(args=None):
    rclpy.init(args=args)
    node = Example()

    # Define poses using the array format [x, y, z, qx, qy, qz, qw]
    p0 = [0.3, -0.15, 0.3029, 1.0, 0.0, 0.0, 0.0]

    poses = [p0]

    # Let's first open the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # Move the arm to each pose
    for i, pose in enumerate(poses):
        node.get_logger().info(f"Publishing Pose {i+1}...")
        node.publish_pose(pose)

    # Now look for the cube
    while node.current_RGB is None or node.current_depth is None or node.current_info is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    # get the current RGBD info
    rgb_image = node.current_RGB.copy()
    depth_image = node.current_depth.copy()

    # Create a color mask to filter for the cube
    mask = cv2.inRange(rgb_image, (0, 0, 100), (10, 10, 255))

    # apply this mask to the depth image
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    # then create a pointcloud from the isolated depth image of the cube
    pc = node.generate_pc(masked_depth)

    # find the centroid of these points
    centroid = np.mean(pc, axis=0)
    print("centroid", centroid)


    while not node.tf_buffer.can_transform("world", "camera_color_optical_frame", rclpy.time.Time()):
        rclpy.spin_once(node, timeout_sec=0.1)
    world_centroid = node.transform_camera_to_world(centroid)
    print("world space centroid", world_centroid)
    block_pose = [world_centroid[0], world_centroid[1], world_centroid[2]-0.003, 1.0, 0.0, 0.0, 0.0]
    
    node.get_logger().info(f"Publishing Pose Block...")
    node.publish_pose(block_pose)

    # Now close the gripper.
    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    node.get_logger().info("All actions done. Shutting down.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()