    def rapid_grasp_and_lift(self, grasp_pose, lift_height=0.1):
        """
        Rapidly close gripper and lift to avoid crushing sensors on table.
        This is a coordinated motion to grasp the dollar bill effectively.
        Uses incremental lifting (2mm steps) coordinated with gripper closing (0.01 steps).
        """
        self.get_logger().info("EXECUTING COORDINATED GRASP AND LIFT...")
        
        # Parameters for coordinated motion
        height_increment = 0.002  # 2mm per step
        gripper_increment = 0.01  # Gripper closing increment
        target_gripper_pos = 0.8  # Final gripper position
        
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
