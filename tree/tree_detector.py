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
        
        print("\n" + "="*60)
        print("DEBUG: NEW LIDAR SCAN RECEIVED")
        print("="*60)
        
        # Step 1: Convert LiDAR data to cartesian points
        points = self.lidar_to_cartesian(msg)
        print(f"DEBUG: Raw LiDAR data - Total points: {len(msg.ranges)}")
        print(f"DEBUG: Valid cartesian points: {len(points)}")
        
        # Step 2: Filter points within 1m range
        filtered_points = self.filter_points_by_distance(points, max_distance=1.0)
        print(f"DEBUG: Points within 1m range: {len(filtered_points)}")
        
        if filtered_points:
            print("DEBUG: Sample filtered points (first 5):")
            for i, (x, y) in enumerate(filtered_points[:5]):
                dist = math.sqrt(x**2 + y**2)
                print(f"  Point {i+1}: ({x:.3f}, {y:.3f}) - Distance: {dist:.3f}m")
        
        # Step 3: Group points into clusters (potential trees)
        clusters = self.cluster_points(filtered_points, cluster_threshold=0.1)
        print(f"DEBUG: Number of clusters found: {len(clusters)}")
        
        # Step 4: Validate clusters as trees
        trees = self.validate_tree_clusters(clusters)
        print(f"DEBUG: Valid tree clusters: {len(trees)}")
        
        # Step 5: Calculate tree coordinates and distances
        self.print_tree_results(trees)
    
    def lidar_to_cartesian(self, scan_msg):
        """Convert LiDAR polar data to cartesian coordinates"""
        points = []
        invalid_count = 0
        out_of_range_count = 0
        
        print(f"DEBUG: LiDAR scan info:")
        print(f"  - Total ranges: {len(scan_msg.ranges)}")
        print(f"  - Angle range: {math.degrees(scan_msg.angle_min):.1f}° to {math.degrees(scan_msg.angle_max):.1f}°")
        print(f"  - Range limits: {scan_msg.range_min:.2f}m to {scan_msg.range_max:.2f}m")
        
        for i, distance in enumerate(scan_msg.ranges):
            # Skip invalid readings
            if math.isinf(distance) or math.isnan(distance):
                invalid_count += 1
                continue
            if distance < scan_msg.range_min or distance > scan_msg.range_max:
                out_of_range_count += 1
                continue
                
            # Calculate angle for this point
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Convert to cartesian coordinates
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            points.append((x, y))
        
        print(f"  - Invalid readings (inf/nan): {invalid_count}")
        print(f"  - Out of range readings: {out_of_range_count}")
        print(f"  - Valid points converted: {len(points)}")
        
        return points
    
    def filter_points_by_distance(self, points, max_distance):
        """Keep only points within specified distance range"""
        filtered = []
        too_close = 0
        too_far = 0
        
        for x, y in points:
            distance = math.sqrt(x**2 + y**2)
            if distance < 0.1:
                too_close += 1
            elif distance > max_distance:
                too_far += 1
            else:
                filtered.append((x, y))
        
        print(f"DEBUG: Distance filtering results:")
        print(f"  - Points too close (<0.1m): {too_close}")
        print(f"  - Points too far (>{max_distance}m): {too_far}")
        print(f"  - Points in valid range: {len(filtered)}")
        
        if filtered:
            distances = [math.sqrt(x**2 + y**2) for x, y in filtered]
            print(f"  - Distance range: {min(distances):.3f}m to {max(distances):.3f}m")
        
        return filtered
    
    def cluster_points(self, points, cluster_threshold):
        """Group nearby points into clusters"""
        if not points:
            print("DEBUG: No points to cluster")
            return []
        
        print(f"DEBUG: Starting clustering with {len(points)} points")
        print(f"DEBUG: Cluster threshold: {cluster_threshold}m")
        
        clusters = []
        current_cluster = [points[0]]
        cluster_count = 0
        
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
                    cluster_count += 1
                    print(f"DEBUG: Cluster {cluster_count} created with {len(current_cluster)} points")
                current_cluster = [points[i]]
        
        # Add the last cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
            cluster_count += 1
            print(f"DEBUG: Final cluster {cluster_count} created with {len(current_cluster)} points")
        
        print(f"DEBUG: Total clusters created: {len(clusters)}")
        return clusters
    
    def validate_tree_clusters(self, clusters):
        """Validate that clusters represent trees"""
        valid_trees = []
        
        print(f"DEBUG: Validating {len(clusters)} clusters...")
        
        for i, cluster in enumerate(clusters, 1):
            print(f"DEBUG: --- Cluster {i} Validation ---")
            print(f"DEBUG: Cluster has {len(cluster)} points")
            
            # Tree validation criteria
            if len(cluster) < 2:
                print(f"DEBUG: Cluster {i} REJECTED - Too few points ({len(cluster)} < 2)")
                continue
            if len(cluster) > 15:
                print(f"DEBUG: Cluster {i} REJECTED - Too many points ({len(cluster)} > 15)")
                continue
            
            # Calculate cluster span (diameter)
            x_coords = [point[0] for point in cluster]
            y_coords = [point[1] for point in cluster]
            
            span_x = max(x_coords) - min(x_coords)
            span_y = max(y_coords) - min(y_coords)
            max_span = max(span_x, span_y)
            
            print(f"DEBUG: Cluster {i} span - X: {span_x:.3f}m, Y: {span_y:.3f}m, Max: {max_span:.3f}m")
            
            # Calculate cluster center
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            center_dist = math.sqrt(center_x**2 + center_y**2)
            
            print(f"DEBUG: Cluster {i} center: ({center_x:.3f}, {center_y:.3f}), Distance: {center_dist:.3f}m")
            
            # Check if cluster size is reasonable for a tree (0.05m to 0.4m diameter)
            if max_span < 0.05:
                print(f"DEBUG: Cluster {i} REJECTED - Too small ({max_span:.3f}m < 0.05m)")
                continue
            if max_span > 0.4:
                print(f"DEBUG: Cluster {i} REJECTED - Too large ({max_span:.3f}m > 0.4m)")
                continue
            
            print(f"DEBUG: Cluster {i} ACCEPTED as tree!")
            valid_trees.append(cluster)
        
        print(f"DEBUG: Validation complete - {len(valid_trees)} valid trees found")
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
