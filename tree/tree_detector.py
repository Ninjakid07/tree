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
            '/oak/rgb/image_raw',  # Correct topic for ROSbot 3 Pro OAK camera
            self.image_callback,
            10
        )
        
        # Movement parameters
        self.linear_speed = 0.2  # Forward speed (m/s)
        self.is_moving = False
        
        # Frame counter for debugging
        self.frame_count = 0
        self.detection_count = 0
        
        # Brown color #895129 detection - MUCH more restrictive to avoid false positives
        # RGB(137, 81, 41) -> HSV(13, 179, 137)
        # Tighter ranges to avoid detecting furniture/wooden surfaces
        self.brown_lower = np.array([10, 120, 80])   # More restrictive bounds
        self.brown_upper = np.array([18, 255, 180])  # More restrictive bounds
        
        # Minimum contour area to consider as valid detection (higher to avoid small brown spots)
        self.min_area = 2000  # Increased from 500 to avoid furniture detection
        
        # Debug mode - set to True to see detection masks
        self.debug_mode = True  # Enabled by default to help with setup
        
        self.get_logger().info('Brown Paper Detector Node Started')
        self.get_logger().info('Looking for brown colored PAPER #895129 with shape filtering...')
        self.get_logger().info('This will ignore brown furniture/surfaces and only detect paper-like shapes')
        self.get_logger().info(f'Subscribing to camera topic: /oak/rgb/image_raw')
        self.get_logger().info(f'Publishing to velocity topic: /cmd_vel')
        if self.debug_mode:
            self.get_logger().info('Debug mode enabled - showing detection windows')
        
        # Timer to check if we're receiving images
        self.timer = self.create_timer(5.0, self.status_check)

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            
            # Log every 30 frames (about once per second at 30fps)
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'Processing frame {self.frame_count}... Camera is working!')
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Log image dimensions occasionally
            if self.frame_count % 60 == 0:
                h, w = cv_image.shape[:2]
                self.get_logger().info(f'Image size: {w}x{h} pixels')
            
            # Detect brown paper in the image
            brown_detected = self.detect_brown_paper(cv_image)
            
            # Control robot movement based on detection
            if brown_detected and not self.is_moving:
                self.detection_count += 1
                self.start_moving()
            elif not brown_detected and self.is_moving:
                self.stop_moving()
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_brown_paper(self, image):
        """
        Detect brown colored paper #895129 with shape filtering to avoid false positives
        Returns True if brown paper (rectangular shape) is detected, False otherwise
        """
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for the specific brown color (more restrictive)
        brown_mask = cv2.inRange(hsv, self.brown_lower, self.brown_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check each contour for paper-like properties
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Must be above minimum area
            if area < self.min_area:
                continue
                
            # Check if shape is somewhat rectangular (paper-like)
            if self.is_paper_like_shape(contour, area):
                valid_detections.append(area)
        
        # Debug visualization (optional)
        if self.debug_mode:
            self.show_debug_image(image, brown_mask, sum(valid_detections))
        
        # Only consider it a detection if we have valid paper-like shapes
        if valid_detections:
            total_area = sum(valid_detections)
            total_pixels = image.shape[0] * image.shape[1]
            brown_percentage = (total_area / total_pixels) * 100
            
            self.get_logger().info(f'✓ BROWN PAPER DETECTED! Area: {total_area:.0f} pixels ({brown_percentage:.1f}% of image)')
            return True
        else:
            # Log occasionally that we're looking but not finding valid paper
            if self.frame_count % 60 == 0:  # Every 2 seconds
                total_brown = sum(cv2.contourArea(c) for c in contours)
                self.get_logger().info(f'Scanning... Brown areas: {total_brown:.0f}, Valid paper shapes: 0')
        
        return False

    def is_paper_like_shape(self, contour, area):
        """
        Check if a contour looks like a piece of paper
        Returns True if the shape is paper-like, False otherwise
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = float(w) / h
        
        # Paper should have reasonable aspect ratio (not too thin or too square)
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            return False
        
        # Calculate how much of the bounding rectangle is filled
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Paper should fill a good portion of its bounding rectangle
        if extent < 0.5:  # At least 50% filled
            return False
        
        # Check if contour can be approximated as a quadrilateral (4-sided shape)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Paper should be somewhat rectangular (3-6 sides when approximated)
        if len(approx) < 3 or len(approx) > 8:
            return False
            
        return True

    def status_check(self):
        """Periodic status check to ensure everything is working"""
        self.get_logger().info(f'Status: Processed {self.frame_count} frames, {self.detection_count} brown detections')
        if self.frame_count == 0:
            self.get_logger().warn('⚠️  No camera frames received! Check camera connection and topic name.')
            self.get_logger().info('💡 Try running: ros2 topic echo /oak/rgb/image_raw --once')
            self.get_logger().info('💡 Available camera topics: /oak/rgb/image_raw, /oak/rgb/image_rect')

    def show_debug_image(self, original, mask, area):
        """Show debug visualization of color detection and shape filtering"""
        try:
            # Create debug window showing original and mask
            debug_img = cv2.hconcat([original, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            
            # Add text overlay with detection info
            cv2.putText(debug_img, f'Brown Area: {area:.0f} (need >{self.min_area})', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, 'Original Image | Brown Mask', 
                       (10, debug_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Brown Paper Detection Debug', debug_img)
            cv2.waitKey(1)
        except:
            pass  # Ignore if display not available

    def start_moving(self):
        """Start moving the robot forward"""
        if not self.is_moving:
            self.get_logger().info('🟤 BROWN DETECTED → MOVING FORWARD! 🚀')
            self.is_moving = True
            
            # Create and publish forward movement command
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

    def stop_moving(self):
        """Stop the robot movement"""
        if self.is_moving:
            self.get_logger().info('❌ BROWN LOST → STOPPING! 🛑')
            self.is_moving = False
            
            # Create and publish stop command
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

    def shutdown_callback(self):
        """Ensure robot stops when node is shutdown"""
        self.get_logger().info('Shutting down brown detector...')
        self.stop_moving()
        if hasattr(self, 'timer'):
            self.timer.cancel()

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
