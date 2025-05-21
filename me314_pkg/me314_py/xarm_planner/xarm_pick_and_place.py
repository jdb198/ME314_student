#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
import time

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

        # Create the subscriber
        self.arm_executing_sub = self.create_subscription(Bool, '/me314_xarm_is_executing', self.execution_state_callback, 10)

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.arm_executing = True

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg
    
    def execution_state_callback(self, msg: Bool):
        print(msg.data)
        self.arm_executing = msg.data

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
        out = []
        i = 0
        for v in range(masked_depth.shape[0]):
            for u in range(masked_depth.shape[1]):
                depth = masked_depth[v, u] 

                if depth > 0.001:

                    if np.random.rand() < 0.97:
                        continue
                    depth = depth * 0.001
                    x, y, z = self.cam_model.projectPixelTo3dRay((u, v))
                    x *= depth
                    y *= depth
                    z *= depth

                    out.append([x, y, z])
                    i += 1
                    
        print("pointcloud generated")
        return np.array(out)   
    
    def transform_camera_to_world(self, point_camera_frame:np.ndarray, rclpy_time):
        # converts to a point stamped, which is some ros format.
         
        transform = self.tf_buffer.lookup_transform(
                'world',
                'camera_color_optical_frame',
                rclpy_time
            )
        all_points = np.zeros(point_camera_frame.shape)


        for i in range(point_camera_frame.shape[0]):

            camera_point_ros = PointStamped()

            camera_point_ros.header.frame_id = 'camera_color_optical_frame'
            camera_point_ros.point.x = float(point_camera_frame[i, 0])
            camera_point_ros.point.y = float(point_camera_frame[i, 1])
            camera_point_ros.point.z = float(point_camera_frame[i, 2])

            # transform the point
            point_in_world = tf2_geometry_msgs.do_transform_point(camera_point_ros, transform)
            all_points[i, 0] = point_in_world.point.x
            all_points[i, 1] = point_in_world.point.y
            all_points[i, 2] = point_in_world.point.z

        print("transform done")
        return all_points

def get_object_points(node: Example, mask_bounds:list):
    # Now look for the cube
    while node.current_RGB is None or node.current_depth is None or node.current_info is None or not node.tf_buffer.can_transform("world", "camera_color_optical_frame", rclpy.time.Time()):
        rclpy.spin_once(node, timeout_sec=0.1)

    rclpy_time = rclpy.time.Time()

    # get the current RGBD info
    rgb_image = node.current_RGB.copy()
    depth_image = node.current_depth.copy()

    # Create a color mask to filter for the cube
    #mask = cv2.inRange(rgb_image, (0, 0, 100), (10, 10, 255))
    mask = cv2.inRange(rgb_image, mask_bounds[0], mask_bounds[1])

    if np.sum(mask) == 0:
        return np.empty((1))
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    # apply this mask to the depth image
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    # then create a pointcloud from the isolated depth image of the cube
    pc = node.generate_pc(masked_depth)

    # transform to world space
    world_points = node.transform_camera_to_world(pc, rclpy_time)

    return world_points

def main(args=None):
    rclpy.init(args=args)
    node = Example()

    cube_mask_bounds = [(0, 0, 100), (10, 10, 255)]
    goal_mask_bounds = [(0, 100, 0), (10, 255, 10)]

    # Define poses using the array format [x, y, z, qx, qy, qz, qw]
    
    search_positions = [[0.15, -0.15, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.25, -0.15, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.35, -0.15, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.15, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.35, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.15, 0.15, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.15, 0.4, 1.0, 0.0, 0.0, 0.0],
                        [0.35, 0.15, 0.4, 1.0, 0.0, 0.0, 0.0]]
    

    # Let's first open the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # We want to map out the space, move through a pattern and collect locations of relevant objects
    cube_pts = []
    goal_pts = []
    
    for i in range(len(search_positions)):
        
        new_pose = search_positions[i]
        print("moving to new_pose")
        node.publish_pose(new_pose)

        while node.arm_executing:
            rclpy.spin_once(node)

        print("waiting complete, taking snapshot.")
        
        # only add if centroid is visible
        new_cube_pts = get_object_points(node, cube_mask_bounds)
        if new_cube_pts.size > 1:
            cube_pts.append(new_cube_pts)
        else:
            print("cube not seen")

        new_goal_pts = get_object_points(node, goal_mask_bounds)
        if new_goal_pts.size > 1:
            goal_pts.append(new_goal_pts)
        else:
            print("goal not seen")


    cube_pts = np.concatenate(cube_pts, axis=0)
    goal_pts = np.concatenate(goal_pts, axis=0)
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(cube_pts[:, 0])
    plt.title("cube x")
    plt.figure()
    plt.hist(cube_pts[:, 1])
    plt.title("cube y")
    plt.figure()
    plt.hist(cube_pts[:, 2])
    plt.title("cube z")

    plt.figure()
    plt.hist(goal_pts[:, 0])
    plt.title("goal x")
    plt.figure()
    plt.hist(goal_pts[:, 1])
    plt.title("goal y")
    plt.figure()
    plt.hist(goal_pts[:, 2])
    plt.title("goal z")
    plt.show(block=True)
    """
    print(cube_pts.shape)
    print(goal_pts.shape)

    cube_world_centroid = np.mean(cube_pts, 0)  
    goal_world_centroid = np.mean(goal_pts, 0)    

    print("cube world space centroid", cube_world_centroid)
    print("goal world space centroid", goal_world_centroid)

    block_pose = [cube_world_centroid[0], cube_world_centroid[1], cube_world_centroid[2]-0.025, 1.0, 0.0, 0.0, 0.0]
    goal_pose = [goal_world_centroid[0], goal_world_centroid[1], goal_world_centroid[2]-0.01, 1.0, 0.0, 0.0, 0.0]

    node.get_logger().info(f"Publishing Pose Block...")
    node.publish_pose(block_pose)

    # Now close the gripper.
    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    # Now move to goal
    node.publish_pose(goal_pose)

    # Now open the gripper.
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)
    
    node.get_logger().info("All actions done. Shutting down.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
