#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped, WrenchStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
import tf2_geometry_msgs
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration
import time
import torch

# Import the command queue message types
from me314_msgs.msg import CommandQueue, CommandWrapper

CAMERA_OFFSET = 0.058

class PegAndHole(Node):
    def __init__(self):
        super().__init__('pegandhole_node')

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        self.init_arm_pose = None

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        # Simulation Subscribers (uncomment as needed)
        # self.realsense_sub = self.create_subscription(Image, '/color/image_raw', self.realsense_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        # self.camera_info_sub = self.create_subscription(CameraInfo, '/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # Real Subscribers
        self.realsense_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/color/image_raw', self.realsense_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/realsense2_camera_node/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # Initialize variables to store the latest force/torque data
        self.FT_force_x = 0.0
        self.FT_force_y = 0.0
        self.FT_force_z = 0.0
        self.FT_torque_x = 0.0
        self.FT_torque_y = 0.0
        self.FT_torque_z = 0.0
        
        # Create a subscription to the force/torque sensor topic
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_ext_state_cb, 10)

        self.cyl_center = None
        self.cyl_depth = None
        self.hole_center = None
        self.hole_depth = None
        self.camera_info = None

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.found_cylinder = False
        self.found_hole = False
        self.gotDepth = False
        self.gotInfo = False

        self.force = None
        self.force_x = 0
        self.force_y = 0
        self.force_z = 0

        # Force thresholds for detecting contact with the surface and holes
        self.hole_force_threshold = 0.5
        self.surface_force_threshold = 1.0

        self.get_logger().info("Peg and Hole Node Initialized")

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.cam_model.fromCameraInfo(msg)
        self.gotInfo = True
        self.get_logger().info("Received camera info")

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
        """
        Callback function that runs whenever a new force/torque message is received.
        
        This function extracts the force and torque data from the message
        and stores it for later use.
        
        Args:
            msg (WrenchStamped): The force/torque sensor message
        """
        # Extract force components from the message
        self.FT_force_x = msg.wrench.force.x
        self.FT_force_y = msg.wrench.force.y
        self.FT_force_z = msg.wrench.force.z
        
        # Extract torque components from the message
        self.FT_torque_x = msg.wrench.torque.x
        self.FT_torque_y = msg.wrench.torque.y
        self.FT_torque_z = msg.wrench.torque.z

    # ---------------------------------------------------------------------
    #  SENSING AND DETECTION
    # ---------------------------------------------------------------------

    def depth_callback(self, msg: Image):
        aligned_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.current_depth = aligned_depth
        
        if self.cyl_center is not None:
            rx, ry = self.cyl_center
            self.cyl_depth = aligned_depth[ry, rx]  # Get depth at red cylinder center
            self.get_logger().info(f"Red cylinder depth: {self.cyl_depth}")
        
        if self.hole_center is not None:
            hx, hy = self.hole_center
            self.hole_depth = aligned_depth[hy, hx]  # Get depth at hole center
            self.get_logger().info(f"Hole depth: {self.hole_depth}")
            
        if self.cyl_depth is not None and self.hole_depth is not None:
            self.gotDepth = True

    def realsense_callback(self, msg: Image):
        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.current_RGB = cv_image
        
        # If there's no detection for the cylinder or hole yet
        if not self.found_cylinder or not self.found_hole:
            if self.current_arm_pose is not None:
                # Store initial pose if not stored yet
                if self.init_arm_pose is None:
                    self.init_arm_pose = self.current_arm_pose
                    self.get_logger().info("Initial arm pose recorded")
                
                # Raise the camera to get a better view if needed
                if not self.found_cylinder or not self.found_hole:
                    pose = self.current_arm_pose
                    new_pose = [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z + 0.1,  # Raise by 10cm
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w
                    ]
                    self.publish_pose(new_pose)
                    self.get_logger().info("Raising camera to look for objects...")
                    time.sleep(2)  # Give time to move
            
            # Look for the cylinder
            if not self.found_cylinder:
                _, red_center = self.mask_red_object(cv_image)
                if red_center != (None, None):
                    self.cyl_center = red_center
                    self.found_cylinder = True
                    self.get_logger().info(f"Found red cylinder at: {red_center}")
            
            # Look for the hole using the enhanced hole detection
            if not self.found_hole:
                hole_center = self.detect_hole_in_blue_object(cv_image)
                if hole_center != (None, None):
                    self.hole_center = hole_center
                    self.found_hole = True
                    self.get_logger().info(f"Found hole center at: {hole_center}")

    # Mask for red object (cylinder)
    def mask_red_object(self, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

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

        if not contours:
            return image, (None, None)
        else:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return image, (None, None)

            # Calculate center
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Annotate image
            result = image.copy()
            cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
            return result, (cx, cy)

    # Enhanced hole detection - combines color detection with contour analysis
    def detect_hole_in_blue_object(self, image: np.ndarray):
        """
        Advanced method to detect the hole in a blue object.
        Uses color segmentation, contour analysis, and edge detection.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define HSV range for blue
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask_clean = cv2.morphologyEx(blue_mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of the blue object
        contours, _ = cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (None, None)
        
        # Get the largest blue contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask just for this contour
        blue_object_mask = np.zeros_like(blue_mask_clean)
        cv2.drawContours(blue_object_mask, [largest_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        blue_object = cv2.bitwise_and(image, image, mask=blue_object_mask)
        
        # Convert the masked image to grayscale
        gray_blue = cv2.cvtColor(blue_object, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_blue, (5, 5), 0)
        
        # Use adaptive thresholding to identify darker regions (potential holes)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Clean up the binary image
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the binary image (potential holes)
        hole_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find the most likely hole
        hole_candidates = []
        for contour in hole_contours:
            area = cv2.contourArea(contour)
            # Filter by area (not too small, not too large)
            if 50 < area < 5000:
                # Get the circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Holes tend to be circular
                    if circularity > 0.6:
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # Check if center is inside the blue object
                            if blue_object_mask[cy, cx] > 0:
                                hole_candidates.append((contour, circularity, (cx, cy)))
        
        # Sort candidates by circularity
        if hole_candidates:
            # Sort by circularity (highest first)
            hole_candidates.sort(key=lambda x: x[1], reverse=True)
            best_hole = hole_candidates[0]
            return best_hole[2]  # Return the center coordinates
            
        # If no good hole candidates, use the centroid of the blue object as fallback
        if len(contours) > 0:
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        return (None, None)
    

    def find_contact_height(self, xy_pose, start_height, step_size=0.005, max_steps=40):
        """
        Gradually lower the end effector until contact with the block surface is detected.
        
        Args:
            xy_pose: [x, y, initial_z, qx, qy, qz, qw] - The pose with x,y coordinates to use
            start_height: The height to start from (should be safely above the surface)
            step_size: How much to lower z each step in meters
            max_steps: Maximum number of downward steps before giving up
            
        Returns:
            tuple: (success, z_height) - Whether surface was found and the surface height
        """
        self.get_logger().info("FINDING BLOCK SURFACE HEIGHT USING FORCE FEEDBACK...")
        
        # Extract position and orientation
        x, y = xy_pose[0], xy_pose[1]
        qx, qy, qz, qw = xy_pose[3], xy_pose[4], xy_pose[5], xy_pose[6]
        
        # Start from a safe height
        current_z = start_height
        
        for step in range(max_steps):
            # Create pose at current height
            test_pose = [x, y, current_z, qx, qy, qz, qw]
            
            # Move to test position
            self.get_logger().info(f"TESTING HEIGHT Z={current_z:.4f}...")
            self.publish_pose(test_pose)
            time.sleep(0.5)  # Give time for motion and force reading
            
            # Check if contact with surface is detected (force above threshold)
            if abs(self.FT_force_z) > self.surface_force_threshold:
                self.get_logger().info(f"SURFACE CONTACT DETECTED AT Z={current_z:.4f}, FORCE_Z={self.FT_force_z:.2f}N")
                # Move back up slightly for spiral search
                surface_z = current_z + step_size
                return True, surface_z
            
            # Lower position for next test
            current_z -= step_size
        
        # If we reach here, we didn't detect the surface
        self.get_logger().warn(f"NO SURFACE CONTACT DETECTED AFTER {max_steps} STEPS")
        return False, None
    
    # Check if we're over the hole based on force feedback
    def check_insertion_complete(self):
        """
        Check if we're positioned over the hole based on force feedback.
        When over the hole, we should see LESS force than when on the surface.
        
        Returns:
            bool: True if we're over the hole (low force), False if we're on surface (high force)
        """
        # Check if z-force is below the threshold (indicates hole)
        if abs(self.FT_force_z) < self.hole_force_threshold:
            self.get_logger().info(f"DETECTED LOW FORCE: {self.FT_force_z:.2f} N - LIKELY OVER HOLE")
            return True
        else:
            self.get_logger().info(f"HIGH FORCE: {self.FT_force_z:.2f} N - STILL ON SURFACE")
            return False

    # Perform a spiral search pattern around a center point
    def spiral_search(self, center_pose, surface_z, lift_height=0.05, max_radius=0.02, step_size=0.002, max_attempts=30):
        """
        Performs a spiral search pattern around a center point to find the hole.
        For each point in the spiral:
        1. Lifts up
        2. Moves to the next position
        3. Lowers down to the contact height
        4. Checks if it's over a hole
        
        Args:
            center_pose: The initial pose (center of the spiral) as [x, y, z, qx, qy, qz, qw]
            surface_z: The z-coordinate of the detected surface
            lift_height: How high to lift between moves
            max_radius: Maximum radius of the spiral in meters
            step_size: Step size between spiral points in meters
            max_attempts: Maximum number of attempts before giving up
            
        Returns:
            bool: True if hole was found, False otherwise
        """
        self.get_logger().info("STARTING IMPROVED SPIRAL SEARCH PATTERN FOR HOLE DETECTION...")
        
        # Extract position from the center pose
        cx, cy, _ = center_pose[0], center_pose[1], center_pose[2]
        
        # Keep orientation the same
        qx, qy, qz, qw = center_pose[3], center_pose[4], center_pose[5], center_pose[6]
        
        # Define the contact height (just 1mm above detected surface)
        contact_z = surface_z + 0.001
        
        # Define lifted height (5cm above surface by default)
        lifted_z = surface_z + lift_height
        
        # Try at the center position first
        # 1. First move to lifted position above center
        self.get_logger().info("MOVING TO LIFTED POSITION ABOVE CENTER...")
        lifted_pose = [cx, cy, lifted_z, qx, qy, qz, qw]
        self.publish_pose(lifted_pose)
        time.sleep(1.0)
        
        # 2. Lower to contact height
        self.get_logger().info("LOWERING TO CONTACT HEIGHT AT CENTER POSITION...")
        contact_pose = [cx, cy, contact_z, qx, qy, qz, qw]
        self.publish_pose(contact_pose)
        time.sleep(1.0)
        
        # 3. Check if we're over the hole (indicated by LOW force)
        if self.check_insertion_complete():
            self.get_logger().info("HOLE FOUND AT CENTER POSITION!")
            
            # 4. Lower into the hole
            self.get_logger().info("LOWERING INTO HOLE...")
            insertion_pose = [cx, cy, contact_z - 0.05, qx, qy, qz, qw]  # Go 5cm down
            self.publish_pose(insertion_pose)
            time.sleep(2.0)
            
            return True
        
        # Start spiral search if initial attempt fails
        angle = 0.0
        radius = 0.0
        
        for i in range(max_attempts):
            # Increase radius for next point in spiral
            radius += step_size
            
            # Stop if we've reached the maximum radius
            if radius > max_radius:
                self.get_logger().warn(f"REACHED MAXIMUM SEARCH RADIUS ({max_radius} m) WITHOUT FINDING HOLE")
                return False
            
            # Calculate next position in spiral pattern
            angle += np.pi / 4  # 45 degree increment
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            
            # 1. LIFT UP before moving laterally
            self.get_logger().info(f"LIFTING UP TO MOVE TO NEXT POSITION...")
            lifted_pose = [cx + dx, cy + dy, lifted_z, qx, qy, qz, qw]
            self.publish_pose(lifted_pose)
            time.sleep(1.0)
            
            # 2. LOWER DOWN to contact height at new position
            self.get_logger().info(f"LOWERING TO CONTACT HEIGHT AT POSITION OFFSET: ({dx:.4f}, {dy:.4f})")
            contact_pose = [cx + dx, cy + dy, contact_z, qx, qy, qz, qw]
            self.publish_pose(contact_pose)
            time.sleep(1.0)
            
            # 3. Check if we're over the hole at this position
            if self.check_insertion_complete():
                self.get_logger().info(f"HOLE FOUND AT POSITION OFFSET: ({dx:.4f}, {dy:.4f})")
                
                # 4. Lower into the hole
                self.get_logger().info("LOWERING INTO HOLE...")
                insertion_pose = [cx + dx, cy + dy, contact_z - 0.05, qx, qy, qz, qw]  # Go 5cm down
                self.publish_pose(insertion_pose)
                time.sleep(2.0)
                
                return True
            
            # If not over hole, lift up again before moving to next position
            self.get_logger().info("NOT OVER HOLE, CONTINUING SEARCH...")
        
        self.get_logger().warn(f"SPIRAL SEARCH COMPLETED {max_attempts} ATTEMPTS WITHOUT FINDING HOLE")
        return False

    # Coordinate transformation from image to camera in world frame
    def img_pixel_to_cam(self, pixel_coords, depth_m):
        # Simulation Intrinsics
        # rgb_K = (640.5098266601562, 640.5098266601562, 640.0, 360.0)

        # Real Intrinsics
        # rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)
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

    # Transform camera coordinates to world coordinates
    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """ Create a 4x4 homogeneous transform from (x, y, z, w) quaternion and (tx, ty, tz). """
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

def main(args=None):
    rclpy.init(args=args)
    node = PegAndHole()
    
    # First, find the objects
    node.get_logger().info("Searching for objects using SAM2 for hole detection...")
    
    # Wait until both objects are found and depth is available
    while not (node.found_cylinder and node.found_hole and node.gotDepth and node.gotInfo):
        rclpy.spin_once(node)
        if node.found_cylinder:
            node.get_logger().info("Cylinder found!")
        if node.found_hole:
            node.get_logger().info("Hole found with SAM2!")
        if node.gotDepth:
            node.get_logger().info("Depth information available")
        if node.gotInfo:
            node.get_logger().info("Camera calibration available")
    
    node.get_logger().info("Objects located successfully!")
    
    # Open gripper at start
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)
    time.sleep(1)  # Wait for gripper to open
    
    # Get 3D coordinates in camera frame
    camera_coords_red = node.img_pixel_to_cam(node.cyl_center, node.cyl_depth/1000.0)
    node.get_logger().info(f"Cylinder in camera frame: {camera_coords_red}")
    
    # Get 3D coordinates in camera frame for the hole detected by SAM2
    camera_coords_hole = node.img_pixel_to_cam(node.hole_center, node.hole_depth/1000.0)
    
    # Transform camera coordinates to world coordinates
    world_coords_red = node.camera_to_base_tf(camera_coords_red, 'camera_color_optical_frame')
    world_coords_hole = node.camera_to_base_tf(camera_coords_hole, 'camera_color_optical_frame')

    node.get_logger().info(f"Cylinder world coordinates: {world_coords_red}")
    node.get_logger().info(f"Hole world coordinates: {world_coords_hole}")

    # Check if coordinates were successfully obtained
    if world_coords_red is None or world_coords_hole is None:
        node.get_logger().error("Failed to get world coordinates - aborting operation.")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Create poses for the motion sequence
    # Position slightly above the cylinder
    pose_above_red = [
        world_coords_red[0, 0], 
        world_coords_red[1, 0], 
        world_coords_red[2, 0] + 0.30,  # 50cm above
        1.0, 0.0, 0.0, 0.0  # Downward orientation
    ]

    # Position at the cylinder for grasping
    pose_red = [
        world_coords_red[0, 0], 
        world_coords_red[1, 0], 
        world_coords_red[2, 0] + 0.10,  # Slightly above to avoid collision
        1.0, 0.0, 0.0, 0.0
    ]

    # Position above the hole
    pose_above_hole = [
        world_coords_hole[0, 0], 
        world_coords_hole[1, 0], 
        world_coords_hole[2, 0] + 0.30,  # 20cm above
        1.0, 0.0, 0.0, 0.0
    ]
    
    # Execute the pick and place sequence
    
    # 1. Move above the cylinder
    node.get_logger().info("Moving above cylinder...")
    node.publish_pose(pose_above_red)
    time.sleep(2)  # Wait for motion to complete
    
    # 2. Move down to the cylinder
    node.get_logger().info("Moving to cylinder for pickup...")
    node.publish_pose(pose_red)
    time.sleep(2)
    
    # 3. Grasp the cylinder
    node.get_logger().info("Grasping cylinder...")
    node.publish_gripper_position(1.0)  # Close gripper
    time.sleep(1)
    
    # 4. Lift the cylinder
    node.get_logger().info("Lifting cylinder...")
    node.publish_pose(pose_above_red)
    time.sleep(2)
    
    # 5. Move above the hole
    node.get_logger().info("Moving above hole...")
    node.publish_pose(pose_above_hole)
    time.sleep(2)
    
    # Force feedback to detect the surface height
    # 6. Find the surface height of the block using force feedback
    node.get_logger().info("FINDING BLOCK SURFACE HEIGHT USING FORCE FEEDBACK...")
    # Create a starting pose for height detection
    surface_detection_pose = [
        world_coords_hole[0, 0],
        world_coords_hole[1, 0],
        world_coords_hole[2, 0] + 0.15,  # Start 15cm above estimated position
        1.0, 0.0, 0.0, 0.0
    ]
    
    surface_found, surface_z = node.find_contact_height(
        surface_detection_pose, 
        world_coords_hole[2, 0] + 0.15,  # Start height
        step_size=0.005,  # 5mm steps
        max_steps=40
    )
    
    # Handle the case where surface was not found
    if not surface_found:
        node.get_logger().error("FAILED TO DETECT BLOCK SURFACE - ABORTING.")
        # Release cylinder and return to start position
        node.publish_gripper_position(0.0)
        node.publish_pose(pose_above_hole)
        time.sleep(2)
        if node.init_arm_pose is not None:
            node.publish_pose([
                node.init_arm_pose.position.x,
                node.init_arm_pose.position.y,
                node.init_arm_pose.position.z,
                node.init_arm_pose.orientation.x,
                node.init_arm_pose.orientation.y,
                node.init_arm_pose.orientation.z,
                node.init_arm_pose.orientation.w
            ])
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Use the detected surface height for the spiral search
    # 7. Set up starting pose for spiral search
    node.get_logger().info(f"BLOCK SURFACE FOUND AT Z={surface_z:.4f}, STARTING SPIRAL SEARCH...")
    
    # Initial pose at center of hole location
    center_pose = [
        world_coords_hole[0, 0], 
        world_coords_hole[1, 0], 
        surface_z + 0.05,  # 5cm above surface for safety
        1.0, 0.0, 0.0, 0.0
    ]
    
    # 8. Perform improved spiral search for the hole
    node.get_logger().info("STARTING IMPROVED SPIRAL SEARCH WITH LIFT-MOVE-LOWER APPROACH...")
    hole_found = node.spiral_search(
        center_pose,
        surface_z,
        lift_height=0.05,  # Lift 5cm between positions
        max_radius=0.03,  # 3cm radius search
        step_size=0.003,  # 3mm steps
        max_attempts=40
    )
    
    # 9. Release the cylinder (whether or not hole was found)
    node.get_logger().info("Releasing cylinder...")
    node.publish_gripper_position(0.0)  # Open gripper
    time.sleep(1)
    
    # 10. Move back up
    node.get_logger().info("Moving back up...")
    node.publish_pose(pose_above_hole)
    time.sleep(2)
    
    # 11. Return to initial position
    if node.init_arm_pose is not None:
        node.get_logger().info("Returning to initial position...")
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
    
    # ADDED SUCCESS REPORTING BASED ON HOLE DETECTION
    if hole_found:
        node.get_logger().info("PICK AND PLACE OPERATION COMPLETED SUCCESSFULLY! CYLINDER INSERTED INTO HOLE.")
    else:
        node.get_logger().warn("PICK AND PLACE OPERATION COMPLETED, BUT HOLE WAS NOT FOUND DURING SPIRAL SEARCH.")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
