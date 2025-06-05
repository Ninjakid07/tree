#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        
        # Debug mode
        self.debug_mode = True
        
        self.get_logger().info('🟤 Brown Color Detector Started')
        self.get_logger().info('🎯 Target: ENTIRE BROWN SPECTRUM (all brown colors)')
        self.get_logger().info('🔍 DETECTION ONLY MODE - No movement')
        self.get_logger().info(f'📷 Camera topic: /oak/rgb/image_raw')
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
            
            # Detect brown color
            brown_detected, detection_info = self.detect_brown_color(cv_image)
            
            # Print detection results
            if brown_detected:
                self.detection_count += 1
                self.get_logger().info(f'🟤 BROWN COLOR DETECTED! {detection_info}')
            else:
                # Print scan status every 3 seconds when not detecting
                if self.frame_count % 90 == 0:
                    self.get_logger().info(f'🔍 Scanning for brown colors... {detection_info}')
                
        except Exception as e:
            self.get_logger().error(f'❌ Error processing image: {str(e)}')

    def detect_brown_color(self, image):
        """
        Detect ANY brown color across the entire brown spectrum
        Returns (detected: bool, info: str)
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
        
        # Calculate percentage of image
        total_pixels = image.shape[0] * image.shape[1]
        brown_percentage = (total_brown_pixels / total_pixels) * 100
        
        # Debug visualization
        if self.debug_mode:
            self.show_debug_image(image, combined_mask, total_brown_pixels, largest_contour_area)
        
        # Create info string
        info = f"Total: {total_brown_pixels} pixels ({brown_percentage:.2f}%), Largest blob: {largest_contour_area:.0f}"
        
        # Detection logic - either total area OR largest contour must be significant
        detected = (total_brown_pixels > self.min_area) or (largest_contour_area > self.min_area)
        
        return detected, info

    def show_debug_image(self, original, mask, total_pixels, largest_area):
        """Show debug visualization"""
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
            cv2.putText(debug_img, f'Largest Blob: {largest_area:.0f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, f'Threshold: {self.min_area}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(debug_img, 'Original | Brown Mask', 
                       (10, debug_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show detection status
            status_color = (0, 255, 0) if (total_pixels > self.min_area or largest_area > self.min_area) else (0, 0, 255)
            status_text = "DETECTED!" if (total_pixels > self.min_area or largest_area > self.min_area) else "Scanning..."
            cv2.putText(debug_img, status_text, 
                       (debug_img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.imshow('Brown Spectrum Detection', debug_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Debug display error: {e}')

    def status_update(self):
        """Print periodic status"""
        detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
        self.get_logger().info(f'📊 Status: {self.frame_count} frames processed, {self.detection_count} detections ({detection_rate:.1f}%)')
        
        if self.frame_count == 0:
            self.get_logger().warn('⚠️  No camera frames received!')
            self.get_logger().info('💡 Check: ros2 topic echo /oak/rgb/image_raw --once')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = BrownColorDetector()
        print('🚀 Starting brown spectrum detection...')
        print('👀 Point camera at ANY brown object to test!')
        
        try:
            rclpy.spin(detector)
        except KeyboardInterrupt:
            detector.get_logger().info('🛑 Stopping detection...')
        finally:
            if hasattr(detector, 'timer'):
                detector.timer.cancel()
            detector.destroy_node()
            
    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
