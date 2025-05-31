#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped, WrenchStamped
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
from rclpy.duration import Duration
import time

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

CAMERA_OFFSET = 0.058

class DollarBill(Node):
    def __init__(self):
        super().__init__('dollarbill_node')

        self.bridge = CvBridge()

        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        self.init_arm_pose = None

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        # Simulation Subscribers
        # self.realsense_sub = self.create_subscription(Image, '/color/image_raw', self.realsense_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Real Subscribers
        self.realsense_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/color/image_raw', self.realsense_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Force/Torque sensor for surface detection
        self.FT_force_x = 0.0
        self.FT_force_y = 0.0
        self.FT_force_z = 0.0
        self.FT_torque_x = 0.0
        self.FT_torque_y = 0.0
        self.FT_torque_z = 0.0
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_ext_state_cb, 10)

        #Execution state monitoring
        self.arm_executing_sub = self.create_subscription(Bool, '/me314_xarm_is_executing', self.execution_state_callback, 10)
        self.arm_executing = True

        self.dollar_bill_center = None
        self.dollar_bill_depth = None
        self.dollar_bill_angle = None  # Orientation angle
        self.target_center = None
        self.target_depth = None

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.found = False
        self.gotDepth = False

        # Force thresholds
        self.surface_force_threshold = 1.0  # Threshold for table contact detection

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

    def ft_ext_state_cb(self, msg: WrenchStamped):
        """Callback for force/torque sensor data"""
        self.FT_force_x = msg.wrench.force.x
        self.FT_force_y = msg.wrench.force.y
        self.FT_force_z = msg.wrench.force.z
        self.FT_torque_x = msg.wrench.torque.x
        self.FT_torque_y = msg.wrench.torque.y
        self.FT_torque_z = msg.wrench.torque.z

    # ---------------------------------------------------------------------
    #  NEW CODE
    # ---------------------------------------------------------------------

    def execution_state_callback(self, msg: Bool):
        self.arm_executing = msg.data

    def depth_callback(self, msg: Image):
        if self.dollar_bill_center is None or self.target_center is None:
            return
        else:
            self.get_logger().info("Both dollar bill and target detected.")
            aligned_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            dx, dy = self.dollar_bill_center
            self.dollar_bill_depth = aligned_depth[dy, dx]
            gx, gy = self.target_center
            self.target_depth = aligned_depth[gy, gx]

    def realsense_callback(self, msg: Image):
        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return
        
        self.get_logger().info('Received an image')

        # If there are no center coordinates for red OR green:
        # 1) raise camera
        # 2) look for objects
        # 3) if both are found, set their coordinates
        # 4) if both are not found, raise camera return empty

        if self.dollar_bill_center is None or self.target_center is None:
            if self.current_arm_pose is not None:
                pose = self.current_arm_pose
                if self.init_arm_pose is None:
                    self.init_arm_pose = pose

                # Raise camera to look for objects
                new_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z + 0.1,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ]

                self.publish_pose(new_pose)
                self.get_logger().info("Raising camera to look for objects...")

                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

                # Look for dollar bill and target
                masked_image_dollar, dollar_center, dollar_angle = self.mask_dollar_bill(cv_image)
                masked_image_green, green_center = self.mask_green_object(cv_image)
                
                if dollar_center != (None, None):
                    self.get_logger().info(f"Found dollar bill at: {dollar_center}, angle: {dollar_angle:.1f}°")
                    self.dollar_bill_center = dollar_center
                    self.dollar_bill_angle = dollar_angle

                if green_center != (None, None):
                    self.get_logger().info(f"Found target at: {green_center}")
                    self.target_center = green_center


    def mask_dollar_bill(self, image: np.ndarray):
        """
        Detect dollar bill using darker green color detection and determine its orientation.
        Returns the annotated image, center coordinates, and orientation angle.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Darker green range for fake dollar bill
        lower_green = np.array([35, 50, 20])   # Darker, less saturated green
        upper_green = np.array([85, 180, 120])  # Allow for various lighting conditions

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations to clean the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, (None, None), None
        
        # Find the largest contour (should be the dollar bill)
        largest = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle to determine orientation
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # Calculate center
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return image, (None, None), None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Get the angle of the rectangle (orientation of dollar bill)
        angle = rect[2]
        
        # Adjust angle to represent rotation from horizontal
        # OpenCV's angle is between -90 and 0 degrees
        if rect[1][0] < rect[1][1]:  # width < height
            angle += 90
            
        # Normalize angle to 0-180 range
        if angle < 0:
            angle += 180

        # Annotate image
        result = image.copy()
        cv2.drawContours(result, [box], 0, (0, 255, 255), 2)  # Yellow rectangle
        cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)  # Red center dot
        
        # Draw orientation line
        length = 50
        end_x = int(cx + length * np.cos(np.radians(angle)))
        end_y = int(cy + length * np.sin(np.radians(angle)))
        cv2.line(result, (cx, cy), (end_x, end_y), (255, 0, 255), 2)  # Magenta line

        return result, (cx, cy), angle

    def mask_green_object(self, image: np.ndarray):
        """
        Detect the green object in the frame using HSV masking.
        Returns the annotated image and the (x, y) center of the largest green contour.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define HSV range for green color
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, (None, None)

        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return image, (None, None)

        # Calculate center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Annotate image
        result = image.copy()
        # cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)  # Green center dot
        return result, (cx, cy)
    
    def find_surface_contact(self, xy_pose, start_height, step_size=0.002, max_steps=50):
        """
        Gradually lower the gripper until contact with table surface is detected.
        Returns (success, contact_z_height)
        """
        self.get_logger().info("FINDING TABLE SURFACE HEIGHT USING FORCE FEEDBACK...")
        
        x, y = xy_pose[0], xy_pose[1]
        qx, qy, qz, qw = xy_pose[3], xy_pose[4], xy_pose[5], xy_pose[6]
        
        current_z = start_height
        
        for step in range(max_steps):
            test_pose = [x, y, current_z, qx, qy, qz, qw]
            
            self.get_logger().info(f"Testing height Z={current_z:.4f}...")
            self.publish_pose(test_pose)
            
            # Wait for movement and force reading
            time.sleep(0.3)
            
            # Check for contact with table (force above threshold)
            if abs(self.FT_force_z) > self.surface_force_threshold:
                self.get_logger().info(f"TABLE CONTACT DETECTED AT Z={current_z:.4f}, FORCE_Z={self.FT_force_z:.2f}N")
                return True, current_z
            
            current_z -= step_size
        
        self.get_logger().warn(f"NO TABLE CONTACT DETECTED AFTER {max_steps} STEPS")
        return False, None
    
    def rapid_grasp_and_lift(self, grasp_pose, lift_height=0.0127):
        """
        Rapidly close gripper and lift to avoid crushing sensors on table.
        This is a coordinated motion to grasp the dollar bill effectively.
        Uses incremental lifting (2mm steps) coordinated with gripper closing (0.01 steps).
        """
        self.get_logger().info("EXECUTING COORDINATED GRASP AND LIFT...")
        
        # Parameters for coordinated motion
        height_increment = 0.002  # 2mm per step
        gripper_increment = 0.01  # Gripper closing increment
        target_gripper_pos = 0.974  # Final gripper position
        
        # Calculate number of steps
        height_steps = int(lift_height / height_increment)  # e.g., 100mm / 2mm = 50 steps
        gripper_steps = int(target_gripper_pos / gripper_increment)  # e.g., 0.8 / 0.01 = 80 steps
        
        # Use the larger number of steps for smoother motion
        num_steps = max(height_steps, gripper_steps)
        
        # Recalculate increments to distribute evenly over num_steps
        actual_height_increment = lift_height / num_steps
        actual_gripper_increment = target_gripper_pos / num_steps
        
        self.get_logger().info(f"Coordinated motion: {num_steps} steps, "
                               f"height increment: {actual_height_increment*1000:.1f}mm, "
                               f"gripper increment: {actual_gripper_increment:.3f}")
        
        # Starting positions
        current_gripper_pos = 0.0
        start_z = grasp_pose[2]
        
        # Execute coordinated motion loop
        for step in range(num_steps):
            # Calculate new positions
            current_height = start_z + (step + 1) * actual_height_increment
            current_gripper_pos = (step + 1) * actual_gripper_increment
            
            # Ensure we don't exceed target values
            current_gripper_pos = min(current_gripper_pos, target_gripper_pos)
            
            # Create pose for current step
            current_pose = [
                grasp_pose[0],
                grasp_pose[1], 
                current_height,
                grasp_pose[3],
                grasp_pose[4],
                grasp_pose[5],
                grasp_pose[6]
            ]
            
            # Execute coordinated commands
            self.publish_pose(current_pose)
            self.publish_gripper_position(current_gripper_pos)
            
            # Log progress every 10 steps
            if step % 10 == 0 or step == num_steps - 1:
                self.get_logger().info(f"Step {step+1}/{num_steps}: "
                                       f"height={current_height:.4f}m, "
                                       f"gripper={current_gripper_pos:.2f}")
            
            # Small delay between steps for smooth motion
            time.sleep(0.05)  # 50ms delay per step
        
        # Final hold to ensure motion completes
        time.sleep(0.5)
        
        self.get_logger().info("COORDINATED GRASP AND LIFT COMPLETED")

    def create_oriented_pose(self, world_coords, angle_degrees):
        """
        Create a pose that aligns the gripper with the long side of the dollar bill.
        """
        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)
        
        # Create rotation around Z-axis to align with dollar bill orientation
        # We want to rotate the gripper to be parallel to the long side
        rotation = R.from_euler('z', angle_rad)
        
        # Base orientation (pointing down)
        base_quat = R.from_euler('xyz', [np.pi, 0, 0])  # Point gripper down
        
        # Combine rotations
        final_rotation = rotation * base_quat
        final_quat = final_rotation.as_quat()  # [x, y, z, w]
        
        return [
            world_coords[0, 0],
            world_coords[1, 0], 
            world_coords[2, 0],
            final_quat[0],  # qx
            final_quat[1],  # qy  
            final_quat[2],  # qz
            final_quat[3]   # qw
        ]

    # Coordinate transformation from image to camera in world frame
    def img_pixel_to_cam(self, pixel_coords, depth_m):
        # Simulation Intrinsics
        # rgb_K = (640.5098266601562, 640.5098266601562, 640.0, 360.0)

        # Real Intrinsics
        # rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)
        # rgb_K = (428.16888427734375, 426.84771728515625, 428.16888427734375, 232.67684936523438)
        rgb_K = (908.6455078125, 909.2957153320312, 646.2830810546875, 373.0643615722656)

        fx, fy, cx, cy = rgb_K
        u, v = pixel_coords
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m + CAMERA_OFFSET
        return (X, Y, Z)

    # Coordinate transform from camera frame to base frame
    def camera_to_base_tf(self, camera_coords, frame_name: str):
        """
        Use TF to transform from 'frame_name' to 'world'.
        Returns a 4x1 array [x, y, z, 1] in base frame, or None on error.
        """
        try:
            if self.tf_buffer.can_transform('world', frame_name, rclpy.time.Time()):
                transform_camera_to_base = self.tf_buffer.lookup_transform('world', frame_name,  rclpy.time.Time())

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
    node = DollarBill()

    # Wait until both objects are found and depth is available
    while not node.found or not node.gotDepth:
        rclpy.spin_once(node)
        if node.dollar_bill_center is not None and node.target_center is not None:
            node.found = True
        if node.dollar_bill_depth is not None and node.target_depth is not None:
            node.gotDepth = True

    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # Convert pixel coordinates to world coordinates
    camera_coords_dollar = node.img_pixel_to_cam(node.dollar_bill_center, node.dollar_bill_depth/1000.0)
    camera_coords_target = node.img_pixel_to_cam(node.target_center, node.target_depth/1000.0)
    
    world_coords_dollar = node.camera_to_base_tf(camera_coords_dollar, 'camera_color_optical_frame')
    world_coords_target = node.camera_to_base_tf(camera_coords_target, 'camera_color_optical_frame')
    
    node.get_logger().info(f"Dollar bill world coords: {world_coords_dollar}")
    node.get_logger().info(f"Target world coords: {world_coords_target}")
    node.get_logger().info(f"Dollar bill orientation: {node.dollar_bill_angle:.1f}°")

    # Create oriented pose aligned with dollar bill's long side
    oriented_pose_above = node.create_oriented_pose(world_coords_dollar, node.dollar_bill_angle)
    oriented_pose_above[2] += 0.15  # 15cm above for approach

    # Move above dollar bill with correct orientation
    node.get_logger().info("Moving above dollar bill with correct orientation...")
    node.publish_pose(oriented_pose_above)
    time.sleep(2)

    # Find table surface using force feedback
    node.get_logger().info("Detecting table surface...")
    surface_found, surface_z = node.find_surface_contact(
        oriented_pose_above, 
        world_coords_dollar[2, 0] + 0.1,  # Start 10cm above estimated position
        step_size=0.002,  # 2mm steps for precision
        max_steps=50
    )

    if not surface_found:
        node.get_logger().error("Failed to detect table surface - aborting operation.")
        node.publish_gripper_position(0.0)
        time.sleep(1)
        if node.init_arm_pose is not None:
            init_pose = [
                node.init_arm_pose.position.x, node.init_arm_pose.position.y, node.init_arm_pose.position.z,
                node.init_arm_pose.orientation.x, node.init_arm_pose.orientation.y, 
                node.init_arm_pose.orientation.z, node.init_arm_pose.orientation.w
            ]
            node.publish_pose(init_pose)
        node.destroy_node()
        rclpy.shutdown()
        return

    # Create grasp pose at detected surface height
    grasp_pose = node.create_oriented_pose(world_coords_dollar, node.dollar_bill_angle)
    grasp_pose[2] = surface_z + 0.002  # Just 2mm above surface to touch dollar bill

    node.get_logger().info(f"Moving to grasp position at surface height: {surface_z:.4f}")
    node.publish_pose(grasp_pose)
    time.sleep(1)

    # Execute rapid grasp and lift
    node.rapid_grasp_and_lift(grasp_pose, lift_height=0.1)

    # Move to target location
    target_pose = [world_coords_target[0, 0], world_coords_target[1, 0], world_coords_target[2, 0] + 0.1,
                   1.0, 0.0, 0.0, 0.0]
    
    node.get_logger().info("Moving to target location...")
    node.publish_pose(target_pose)
    time.sleep(2)

    # Release dollar bill
    node.get_logger().info("Releasing dollar bill...")
    node.publish_gripper_position(0.0)
    time.sleep(1)

    # Return to initial position
    if node.init_arm_pose is not None:
        node.get_logger().info("Returning to initial position...")
        init_pose = [
            node.init_arm_pose.position.x, node.init_arm_pose.position.y, node.init_arm_pose.position.z,
            node.init_arm_pose.orientation.x, node.init_arm_pose.orientation.y, 
            node.init_arm_pose.orientation.z, node.init_arm_pose.orientation.w
        ]
        node.publish_pose(init_pose)

    node.get_logger().info("Dollar bill pickup completed successfully!")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
