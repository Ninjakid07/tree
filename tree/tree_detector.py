#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class TreeDetector(Node):
    def __init__(self):
        super().__init__('tree_detector')
        
        # Subscribe to LiDAR data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Standard LiDAR topic for ROSbot
            self.scan_callback,
            10
        )
        
        self.get_logger().info("Tree Detector Node Started")
        
    def scan_callback(self, msg):
        """Process LiDAR scan data to detect trees"""
        
        # Step 1: Convert LiDAR data to cartesian points
        points = self.lidar_to_cartesian(msg)
        
        # Step 2: Filter points within 1m range
        filtered_points = self.filter_points_by_distance(points, max_distance=1.0)
        
        # Step 3: Group points into clusters (potential trees)
        clusters = self.cluster_points(filtered_points, cluster_threshold=0.1)
        
        # Step 4: Validate clusters as trees
        trees = self.validate_tree_clusters(clusters)
        
        # Step 5: Calculate tree coordinates and distances
        self.print_tree_results(trees)
    
    def lidar_to_cartesian(self, scan_msg):
        """Convert LiDAR polar data to cartesian coordinates"""
        points = []
        
        for i, distance in enumerate(scan_msg.ranges):
            # Skip invalid readings
            if math.isinf(distance) or math.isnan(distance):
                continue
            if distance < scan_msg.range_min or distance > scan_msg.range_max:
                continue
                
            # Calculate angle for this point
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Convert to cartesian coordinates
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            points.append((x, y))
        
        return points
    
    def filter_points_by_distance(self, points, max_distance):
        """Keep only points within specified distance range"""
        filtered = []
        
        for x, y in points:
            distance = math.sqrt(x**2 + y**2)
            if 0.1 <= distance <= max_distance:  # 0.1m minimum to filter noise
                filtered.append((x, y))
        
        return filtered
    
    def cluster_points(self, points, cluster_threshold):
        """Group nearby points into clusters"""
        if not points:
            return []
        
        clusters = []
        current_cluster = [points[0]]
        
        for i in range(1, len(points)):
            # Calculate distance to previous point
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            if distance <= cluster_threshold:
                # Add to current cluster
                current_cluster.append(points[i])
            else:
                # Start new cluster
                if len(current_cluster) >= 2:  # Minimum 2 points per cluster
                    clusters.append(current_cluster)
                current_cluster = [points[i]]
        
        # Add the last cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def validate_tree_clusters(self, clusters):
        """Validate that clusters represent trees"""
        valid_trees = []
        
        for cluster in clusters:
            # Tree validation criteria
            if len(cluster) < 2 or len(cluster) > 15:  # Reasonable point count
                continue
            
            # Calculate cluster span (diameter)
            x_coords = [point[0] for point in cluster]
            y_coords = [point[1] for point in cluster]
            
            span_x = max(x_coords) - min(x_coords)
            span_y = max(y_coords) - min(y_coords)
            max_span = max(span_x, span_y)
            
            # Check if cluster size is reasonable for a tree (0.1m to 0.3m diameter)
            if 0.05 <= max_span <= 0.4:
                valid_trees.append(cluster)
        
        return valid_trees
    
    def print_tree_results(self, trees):
        """Calculate and print tree coordinates and distances"""
        if not trees:
            self.get_logger().info("No trees detected within 1m range")
            return
        
        print("\n" + "="*50)
        print(f"DETECTED {len(trees)} TREES:")
        print("="*50)
        
        for i, tree_cluster in enumerate(trees, 1):
            # Calculate tree center (average of all points in cluster)
            x_coords = [point[0] for point in tree_cluster]
            y_coords = [point[1] for point in tree_cluster]
            
            tree_x = sum(x_coords) / len(x_coords)
            tree_y = sum(y_coords) / len(y_coords)
            
            # Calculate distance from robot (0,0) to tree center
            distance = math.sqrt(tree_x**2 + tree_y**2)
            
            # Print results
            print(f"Tree {i}: Coordinates ({tree_x:.2f}, {tree_y:.2f}), Distance: {distance:.2f}m")
        
        print("="*50 + "\n")

def main(args=None):
    rclpy.init(args=args)
    
    tree_detector = TreeDetector()
    
    try:
        rclpy.spin(tree_detector)
    except KeyboardInterrupt:
        pass
    finally:
        tree_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()