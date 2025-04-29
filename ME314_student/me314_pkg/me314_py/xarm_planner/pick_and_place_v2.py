#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import tf2_ros
from rclpy.duration import Duration
import time

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

CAMERA_OFFSET = 0.058
REAL = True

class PickPlace(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        self.init_arm_pose = None

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)
        
        self.bridge = CvBridge()

        if REAL:
            color_sub = '/camera/realsense2_camera_node/color/image_raw'
            depth_sub = '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw'
        else:
            color_sub = '/color/image_raw'
            depth_sub = '/aligned_depth_to_color/image_raw'

        self.subscription = self.create_subscription(
            Image,
            color_sub,
            self.camera_listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription_depth = self.create_subscription(
            Image,
            depth_sub,
            self.depth_listener_callback,
            10)
        self.subscription_depth  # prevent unused variable warning

        # Intrinsics for RGB and Depth cameras
        if REAL:
            self.rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)
        else:
            self.rgb_K = (640.5098266601562, 640.5098266601562, 640.0, 360.0)

        self.red_center_coordinates = None
        self.red_depth = None
        self.green_center_coordinates = None
        self.green_depth = None

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.found = False
        self.gotDepth = False

        self.pose_to_box = []
        self.pose_to_square = []

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

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


    def camera_listener_callback(self, msg):
        self.get_logger().info('Received an image')

        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return
        
        # If there are no center coordinates for red OR green:
        # 1) raise camera
        # 2) look for objects
        # 3) if both are found, set their coordinates
        # 4) if both are not found, raise camera return empty
        
        if self.red_center_coordinates is None or self.green_center_coordinates is None :
            if self.current_arm_pose is not None:
                # Raise the camera
                pose = self.current_arm_pose
                if self.init_arm_pose is None:
                    self.init_arm_pose = pose

                # Extract position and orientation as a list and apply z-offset
                new_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z + 0.1,  # Add 1.0 offset to Z
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ]
                
                self.publish_pose(new_pose)
                self.get_logger().info("Raising camera to look for objects...")
                # Look for objects
                # cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                # cv_ColorImage_rgb = cv2.cvtColor(cv_ColorImage, cv2.COLOR_BGR2RGB)

                # Mask for red and green objects
                masked_image_red, red_center = self.mask_red_object(cv_ColorImage)
                masked_image_green, green_center = self.mask_green_object(cv_ColorImage)
                
                if red_center != (None, None):
                    self.get_logger().info(f"Found red object at: {red_center}")
                    self.red_center_coordinates = red_center

                if green_center != (None, None):
                    self.get_logger().info(f"Found green object at: {green_center}")
                    self.green_center_coordinates = green_center

    def depth_listener_callback(self, msg):
        """
        When a new depth image arrives, if we have a valid banana center from YOLO,
        we align depth to the color frame, read the depth at that pixel, and compute
        the 3D coords in the base frame. Finally, publish a PoseStamped.
        """
        if self.red_center_coordinates is None or self.green_center_coordinates is None:
            # Red box and green square not detected yet
            return
        else:
            # Both red box and green square detected
            self.get_logger().info("Both red box and green square detected.")
            aligned_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            rx, ry = self.red_center_coordinates
            self.red_depth = aligned_depth[ry, rx]  # Get depth at red box center
            gx, gy = self.green_center_coordinates
            self.green_depth = aligned_depth[gy, gx]  # Get depth at green square center

    # ---------------------------------------------------------------------
    #  MASK RED BOX
    # ---------------------------------------------------------------------
    def mask_red_object(self, frame: np.ndarray):
        """
        Detect the red object in the frame using HSV masking.
        Returns the annotated image and the (x, y) center of the largest red contour.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 100, 90])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks and combine
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame, (None, None)

        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return frame, (None, None)

        # Calculate center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Annotate image
        result = frame.copy()
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
        return result, (cx, cy)

    def mask_green_object(self, frame: np.ndarray):
        """
        Detect the green object in the frame using HSV masking.
        Returns the annotated image and the (x, y) center of the largest green contour.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define HSV range for green color
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame, (None, None)

        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return frame, (None, None)

        # Calculate center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Annotate image
        result = frame.copy()
        # cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)  # Green center dot
        return result, (cx, cy)


    # ---------------------------------------------------------------------
    # COORDINATE TRANSFORMS, ALIGNMENTS, ETC.
    # ---------------------------------------------------------------------
    def pixel_to_camera_frame(self, pixel_coords, depth_m):
        """
        Convert pixel coordinates + depth (meters) to camera coordinates.
        """
        fx, fy, cx, cy = self.rgb_K  # (fx, fy, cx, cy)
        u, v = pixel_coords
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m + CAMERA_OFFSET
        return (X, Y, Z)

    def camera_to_base_tf(self, camera_coords, frame_name: str):
        """
        Use TF to transform from 'frame_name' to 'world'.
        Returns a 4x1 array [x, y, z, 1] in base frame, or None on error.
        """
        try:
            if self.tf_buffer.can_transform('world',
                                            frame_name,
                                            rclpy.time.Time()):
                transform_camera_to_base = self.tf_buffer.lookup_transform(
                    'world',
                    frame_name,
                    rclpy.time.Time())

                tf_geom = transform_camera_to_base.transform

                trans = np.array([tf_geom.translation.x,
                                  tf_geom.translation.y,
                                  tf_geom.translation.z], dtype=float)
                rot = np.array([tf_geom.rotation.x,
                                tf_geom.rotation.y,
                                tf_geom.rotation.z,
                                tf_geom.rotation.w], dtype=float)

                transform_mat = self.create_transformation_matrix(rot, trans)
                print(f"tranform_mat: {transform_mat}")
                camera_coords_homogenous = np.array([[camera_coords[0]],
                                                     [camera_coords[1]],
                                                     [camera_coords[2]],
                                                     [1]])
                base_coords = transform_mat @ camera_coords_homogenous
                return base_coords
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to convert camera->base transform: {str(e)}")
            return None

    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """ Create a 4x4 homogeneous transform from (x, y, z, w) quaternion and (tx, ty, tz). """
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()

    while not node.found or not node.gotDepth:
        rclpy.spin_once(node)
        if node.red_center_coordinates is not None and node.green_center_coordinates is not None:
            node.found = True
        if node.red_depth is not None and node.green_depth is not None:
            node.gotDepth = True

    # Let's first open the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # Convert pixel coords + depth to camera coordinates
    # camera_coords = self.pixel_to_camera_frame(self.center_coordinates, depth_at_center_m)
    camera_coords_red = node.pixel_to_camera_frame(node.red_center_coordinates, node.red_depth/1000.0)
    camera_coords_green = node.pixel_to_camera_frame(node.green_center_coordinates, node.green_depth/1000.0)
    # Transform camera coords to the robot arm frame
    world_coords_red = node.camera_to_base_tf(camera_coords_red, 'camera_color_optical_frame')
    world_coords_green = node.camera_to_base_tf(camera_coords_green, 'camera_color_optical_frame')
    node.get_logger().info(f"Red world coords: {world_coords_red}")
    node.get_logger().info(f"Green world coords: {world_coords_green}")

    # Create pose for red box
    pose_above_red = [world_coords_red[0, 0], world_coords_red[1, 0], world_coords_red[2, 0] + 0.1,
                1.0, 0.0, 0.0, 0.0]  # Assuming no rotation needed
    pose_red = [world_coords_red[0, 0], world_coords_red[1, 0], world_coords_red[2, 0] + 0.01,
                1.0, 0.0, 0.0, 0.0]  # Assuming no rotation needed
    node.pose_to_box.append(pose_above_red)
    node.pose_to_box.append(pose_red)
    node.get_logger().info(f"Red box pose: {pose_red}")

    # Move the arm to each pose
    for i, pose in enumerate(node.pose_to_box):
        node.get_logger().info(f"Publishing Pose {i+1}...")
        node.publish_pose(pose)

    # Now close the gripper.
    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    # Move the arm to the green square
    pose_above_green = [world_coords_green[0, 0], world_coords_green[1, 0] + 0.06, world_coords_green[2, 0] + 0.1,
                  1.0, 0.0, 0.0, 0.0]  # Assuming no rotation needed
    node.pose_to_square.append(pose_above_green)

    pose_green = [world_coords_green[0, 0], world_coords_green[1, 0] + 0.06, world_coords_green[2, 0] + 0.0127,
                  1.0, 0.0, 0.0, 0.0]  # Assuming no rotation needed
    node.pose_to_square.append(pose_green)
    node.get_logger().info(f"Green square pose: {pose_green}")

    node.get_logger().info("Moving to green square...")
    for i, pose in enumerate(node.pose_to_square):
        node.get_logger().info(f"Publishing Pose {i+1}...")
        node.publish_pose(pose)

    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    node.get_logger().info("Moving back to initial position...")
    node.publish_pose(pose_above_green)
    init_pose = [
                    node.init_arm_pose.position.x,
                    node.init_arm_pose.position.y,
                    node.init_arm_pose.position.z,
                    node.init_arm_pose.orientation.x,
                    node.init_arm_pose.orientation.y,
                    node.init_arm_pose.orientation.z,
                    node.init_arm_pose.orientation.w
                ]
    node.publish_pose(init_pose)

    node.get_logger().info("All actions done. Shutting down.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()