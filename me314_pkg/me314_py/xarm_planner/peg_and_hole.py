#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped
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

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

CAMERA_OFFSET = 0.058

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pickandplace_node')

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        self.init_arm_pose = None

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        # Simulation Subscribers
        # self.realsense_sub = self.create_subscription(Image, '/color/image_raw', self.realsense_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        # self.camera_info_sub = self.create_subscription(CameraInfo, '/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # Real Subscribers
        self.realsense_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/color/image_raw', self.realsense_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/realsense2_camera_node/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        self.ft_sensor_sub = self.create_subscription(Float64, '/xarm/uf_ftsensor_ext_states', self.ft_callback, 10)

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

        self.get_logger().info("Enhanced Pick and Place node initialized")

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

    # ---------------------------------------------------------------------
    #  NEW CODE
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

    def ft_callback (self, msg: Float64):
        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return
        else:
            self.force = msg.data

    # Mask for red object
    # Returns the annotated image and the (x, y) center of the largest red contour
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

    # Coordinate transformation from image to camera in world frame
    def img_pixel_to_cam(self, pixel_coords, depth_m):
        # Simulation Intrinsics
        # rgb_K = (640.5098266601562, 640.5098266601562, 640.0, 360.0)

        # Real Intrinsics
        rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)

        fx, fy, cx, cy = rgb_K
        u, v = pixel_coords
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m + CAMERA_OFFSET
        return (X, Y, Z)
    
        # if self.gotInfo and self.cam_model is not None:
        #         # Use the calibrated camera model for accurate projection
        #         ray = self.cam_model.projectPixelTo3dRay(pixel_coords)
        #         x = ray[0] * depth_m
        #         y = ray[1] * depth_m
        #         z = ray[2] * depth_m
        #         return (x, y, z)
        # else:
        #     # Fallback to manual calculation using approximate intrinsics
        #     # Real Intrinsics (fallback)
        #     rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)
        #     fx, fy, cx, cy = rgb_K
        #     u, v = pixel_coords
        #     x = (u - cx) * depth_m / fx
        #     y = (v - cy) * depth_m / fy
        #     z = depth_m
        #     return (x, y, z)

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
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to convert camera->base transform: {str(e)}")
            return None

    # Create a 4x4 transformation matrix from quaternion and translation
    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """ Create a 4x4 homogeneous transform from (x, y, z, w) quaternion and (tx, ty, tz). """
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix
    
    # Generate point cloud from depth image
    def generate_pc(self, masked_depth):
        if not self.gotInfo:
            self.get_logger().error("No camera info available for point cloud generation")
            return None
            
        out = np.zeros([np.sum(masked_depth > 1), 3])
        
        i = 0
        for v in range(masked_depth.shape[0]):
            for u in range(masked_depth.shape[1]):
                depth = masked_depth[v, u] 

                if depth > 0.001:
                    depth = depth * 0.001  # Convert to meters
                    ray = self.cam_model.projectPixelTo3dRay((u, v))
                    x = ray[0] * depth
                    y = ray[1] * depth
                    z = ray[2] * depth
                    out[i, 0] = x
                    out[i, 1] = y
                    out[i, 2] = z
                    i += 1
        return out
    
    # Advanced hole detection using point cloud
    def detect_hole_center_3d(self):
        """
        Use point cloud analysis to find the center of the hole in 3D space.
        Returns world coordinates of the hole center.
        """
        if self.current_depth is None or not self.found_hole:
            self.get_logger().error("Cannot detect hole in 3D: missing depth or hole detection")
            return None
            
        # Create a mask for the blue object
        hsv = cv2.cvtColor(self.current_RGB, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply the mask to the depth image
        masked_depth = cv2.bitwise_and(self.current_depth, self.current_depth, mask=blue_mask)
        
        # Generate point cloud
        pc = self.generate_pc(masked_depth)
        if pc is None or len(pc) == 0:
            self.get_logger().error("Failed to generate point cloud for hole detection")
            return None
            
        # Use the already detected hole center as guidance
        if self.hole_center is not None and self.hole_depth is not None:
            # Convert 2D+depth to 3D in camera frame
            hole_3d = self.img_pixel_to_cam(self.hole_center, self.hole_depth/1000.0)
            # Transform to world frame
            world_hole = self.camera_to_base_tf(hole_3d, 'camera_color_optical_frame')
            return world_hole
            
        return None

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()

    # First, find the objects
    node.get_logger().info("Searching for objects...")
    
    # Wait until both objects are found and depth is available
    while not (node.found_cylinder and node.found_hole and node.gotDepth and node.gotInfo):
        rclpy.spin_once(node)
        if node.found_cylinder:
            node.get_logger().info("Cylinder found!")
        if node.found_hole:
            node.get_logger().info("Hole found!")
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
    
    # Get 3D coordinates of hole center using point cloud analysis (more accurate)
    world_coords_hole = node.detect_hole_center_3d()

    if world_coords_hole is None:
        # Fallback to basic method
        camera_coords_hole = node.img_pixel_to_cam(node.hole_center, node.hole_depth/1000.0)
        world_coords_hole = node.camera_to_base_tf(camera_coords_hole, 'camera_color_optical_frame')

    # Transform camera coordinates to world coordinates
    world_coords_red = node.camera_to_base_tf(camera_coords_red, 'camera_color_optical_frame')

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
        world_coords_red[2, 0] + 0.1,  # 10cm above
        1.0, 0.0, 0.0, 0.0  # Downward orientation
    ]

    # Position at the cylinder for grasping
    pose_red = [
        world_coords_red[0, 0], 
        world_coords_red[1, 0], 
        world_coords_red[2, 0] + 0.005,  # Slightly above to avoid collision
        1.0, 0.0, 0.0, 0.0
    ]

    # Position above the hole
    pose_above_hole = [
        world_coords_hole[0, 0], 
        world_coords_hole[1, 0], 
        world_coords_hole[2, 0] + 0.1,  # 10cm above
        1.0, 0.0, 0.0, 0.0
    ]

    # Position at the hole for placement
    pose_hole = [
        world_coords_hole[0, 0], 
        world_coords_hole[1, 0], 
        world_coords_hole[2, 0] + 0.02,  # Position for placement, slightly above
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
    
    # 6. Lower to the hole
    node.get_logger().info("Lowering to hole...")
    node.publish_pose(pose_hole)
    time.sleep(2)
    
    # 7. Release the cylinder
    node.get_logger().info("Releasing cylinder...")
    node.publish_gripper_position(0.0)  # Open gripper
    time.sleep(1)
    
    # 8. Move back up
    node.get_logger().info("Moving back up...")
    node.publish_pose(pose_above_hole)
    time.sleep(2)
    
    # 9. Return to initial position
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
    
    node.get_logger().info("Pick and place operation completed successfully!")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
