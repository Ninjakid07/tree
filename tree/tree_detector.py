#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class BrownPaperDetector(Node):
    def __init__(self):
        super().__init__('brown_paper_detector')
        
        # Initialize CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Create publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Adjust topic name if different
            self.image_callback,
            10
        )
        
        # Movement parameters
        self.linear_speed = 0.2  # Forward speed (m/s)
        self.is_moving = False
        
        # Specific brown color #895129 in HSV
        # RGB(137, 81, 41) -> HSV(13, 179, 137)
        # Precise detection range for this exact brown color
        self.brown_lower = np.array([8, 149, 107])   # Lower HSV bounds for #895129
        self.brown_upper = np.array([18, 209, 167])  # Upper HSV bounds for #895129
        
        # Minimum contour area to consider as valid detection
        self.min_area = 1000
        
        self.get_logger().info('Brown Paper Detector Node Started')
        self.get_logger().info('Looking for brown colored paper #895129...')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detect brown paper in the image
            brown_detected = self.detect_brown_paper(cv_image)
            
            # Control robot movement based on detection
            if brown_detected and not self.is_moving:
                self.start_moving()
            elif not brown_detected and self.is_moving:
                self.stop_moving()
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_brown_paper(self, image):
        """
        Detect the specific brown colored paper #895129 in the image
        Returns True if the exact brown paper is detected, False otherwise
        """
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for the specific brown color #895129
        brown_mask = cv2.inRange(hsv, self.brown_lower, self.brown_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any significant brown areas are detected
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Calculate percentage of image covered by brown
                total_pixels = image.shape[0] * image.shape[1]
                brown_percentage = (area / total_pixels) * 100
                
                self.get_logger().info(f'Brown paper #895129 detected! Area: {area:.0f} pixels ({brown_percentage:.1f}% of image)')
                return True
        
        return False

    def start_moving(self):
        """Start moving the robot forward"""
        if not self.is_moving:
            self.get_logger().info('Brown paper #895129 detected - Moving forward!')
            self.is_moving = True
            
            # Create and publish forward movement command
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

    def stop_moving(self):
        """Stop the robot movement"""
        if self.is_moving:
            self.get_logger().info('Brown paper #895129 lost - Stopping!')
            self.is_moving = False
            
            # Create and publish stop command
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

    def shutdown_callback(self):
        """Ensure robot stops when node is shutdown"""
        self.stop_moving()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and run the brown paper detector node
        detector = BrownPaperDetector()
        
        # Handle shutdown gracefully
        try:
            rclpy.spin(detector)
        except KeyboardInterrupt:
            detector.get_logger().info('Shutting down...')
        finally:
            detector.shutdown_callback()
            detector.destroy_node()
            
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
