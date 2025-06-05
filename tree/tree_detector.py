#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class BrownColorDetector(Node):
    def __init__(self):
        super().__init__('brown_color_detector')
        
        # Initialize CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',  # ROSbot 3 Pro OAK camera
            self.image_callback,
            10
        )
        
        # Create publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Movement parameters
        self.linear_speed = 0.2  # Forward speed (m/s)
        self.is_moving = False
        
        # Detection threshold
        self.percentage_threshold = 11.0  # Move forward if brown > 11%
        
        # Frame counter for debugging
        self.frame_count = 0
        self.detection_count = 0
        
        # BROAD BROWN SPECTRUM DETECTION
        # Covers all brown colors: light brown, dark brown, reddish brown, yellowish brown
        
        # Range 1: Reddish browns (like mahogany, chestnut)
        self.brown_lower1 = np.array([0, 30, 30])      # Very broad range
        self.brown_upper1 = np.array([15, 255, 200])   
        
        # Range 2: Classic browns (like chocolate, coffee)
        self.brown_lower2 = np.array([8, 40, 20])  
        self.brown_upper2 = np.array([25, 255, 180])
        
        # Range 3: Yellowish browns (like tan, beige, khaki)
        self.brown_lower3 = np.array([15, 20, 40])
        self.brown_upper3 = np.array([35, 200, 220])
        
        # Minimum area to consider as detection (lowered for broader detection)
        self.min_area = 500
        
        # Debug mode - disabled by default to avoid GTK errors
        self.debug_mode = False  # Set to True only if you have display
        
        self.get_logger().info('🟤 Brown Color Detector Started')
        self.get_logger().info('🎯 Target: ENTIRE BROWN SPECTRUM (all brown colors)')
        self.get_logger().info(f'🚀 MOVEMENT MODE - Forward if brown > {self.percentage_threshold}%')
        self.get_logger().info(f'📷 Camera topic: /oak/rgb/image_raw')
        self.get_logger().info(f'🔧 Publishing to: /cmd_vel')
        if self.debug_mode:
            self.get_logger().info('🐛 Debug mode enabled - showing detection windows')
        
        # Timer for status updates
        self.timer = self.create_timer(10.0, self.status_update)

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Log camera status occasionally
            if self.frame_count % 60 == 0:
                h, w = cv_image.shape[:2]
                self.get_logger().info(f'📷 Camera working! Frame {self.frame_count}, Size: {w}x{h}')
            
            # Detect brown color and get percentage
            brown_percentage, detection_info = self.detect_brown_color(cv_image)
            
            # Movement logic based on percentage threshold
            if brown_percentage > self.percentage_threshold:
                if not self.is_moving:
                    self.start_moving()
                self.get_logger().info(f'{brown_percentage:.2f}% detected brown')
            else:
                if self.is_moving:
                    self.stop_moving()
                self.get_logger().info(f'{brown_percentage:.2f}% NOT DETECTED')
                
        except Exception as e:
            self.get_logger().error(f'❌ Error processing image: {str(e)}')

    def detect_brown_color(self, image):
        """
        Detect ANY brown color across the entire brown spectrum
        Returns (percentage: float, info: str)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Try multiple detection ranges
        mask1 = cv2.inRange(hsv, self.brown_lower1, self.brown_upper1)
        mask2 = cv2.inRange(hsv, self.brown_lower2, self.brown_upper2)  
        mask3 = cv2.inRange(hsv, self.brown_lower3, self.brown_upper3)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        combined_mask = cv2.bitwise_or(combined_mask, mask3)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate total brown area
        total_brown_pixels = cv2.countNonZero(combined_mask)
        
        # Find contours for detailed analysis
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_area = 0
        if contours:
            largest_contour_area = max(cv2.contourArea(contour) for contour in contours)
        
        # Calculate percentage of image (keep this calculation exactly as it is)
        total_pixels = image.shape[0] * image.shape[1]
        brown_percentage = (total_brown_pixels / total_pixels) * 100
        
        # Debug visualization (only if enabled and display available)
        if self.debug_mode:
            self.show_debug_image(image, combined_mask, total_brown_pixels, largest_contour_area, brown_percentage)
        
        # Create info string
        info = f"Total: {total_brown_pixels} pixels ({brown_percentage:.2f}%), Largest blob: {largest_contour_area:.0f}"
        
        return brown_percentage, info

    def show_debug_image(self, original, mask, total_pixels, largest_area, percentage):
        """Show debug visualization - handles display errors gracefully"""
        try:
            # Resize images for display if too large
            h, w = original.shape[:2]
            if w > 800:
                scale = 800 / w
                new_w, new_h = int(w * scale), int(h * scale)
                original = cv2.resize(original, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
            
            # Create side-by-side display
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.hconcat([original, mask_colored])
            
            # Add text overlays
            cv2.putText(debug_img, f'Brown Pixels: {total_pixels}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, f'Percentage: {percentage:.2f}%', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, f'Threshold: {self.percentage_threshold}%', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(debug_img, 'Original | Brown Mask', 
                       (10, debug_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show detection status
            status_color = (0, 255, 0) if percentage > self.percentage_threshold else (0, 0, 255)
            status_text = "MOVING!" if percentage > self.percentage_threshold else "Scanning..."
            cv2.putText(debug_img, status_text, 
                       (debug_img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.imshow('Brown Spectrum Detection', debug_img)
            cv2.waitKey(1)
            
        except Exception as e:
            # Silently disable debug mode if display not available
            if self.debug_mode:
                self.get_logger().warn(f'Disabling debug display - no GUI available: {str(e)[:50]}...')
                self.debug_mode = False

    def start_moving(self):
        """Start moving the robot forward"""
        self.get_logger().info('🚀 BROWN > 11% → MOVING FORWARD!')
        self.is_moving = True
        
        # Create and publish forward movement command
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def stop_moving(self):
        """Stop the robot movement"""
        self.get_logger().info('🛑 BROWN < 11% → STOPPING!')
        self.is_moving = False
        
        # Create and publish stop command
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def status_update(self):
        """Print periodic status"""
        movement_status = "MOVING" if self.is_moving else "STOPPED"
        self.get_logger().info(f'📊 Status: {self.frame_count} frames processed, Robot: {movement_status}')
        
        if self.frame_count == 0:
            self.get_logger().warn('⚠️  No camera frames received!')
            self.get_logger().info('💡 Check: ros2 topic echo /oak/rgb/image_raw --once')

    def shutdown_callback(self):
        """Ensure robot stops when node is shutdown"""
        self.get_logger().info('🛑 Shutting down brown detector...')
        if self.is_moving:
            self.stop_moving()
        if hasattr(self, 'timer'):
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = BrownColorDetector()
        print('🚀 Starting brown spectrum detection with movement...')
        print('👀 Robot will move forward when brown > 11%!')
        
        try:
            rclpy.spin(detector)
        except KeyboardInterrupt:
            detector.get_logger().info('🛑 Stopping detection...')
        finally:
            detector.shutdown_callback()
            detector.destroy_node()
            
    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # Ignore if no windows or no display

if __name__ == '__main__':
    main()
