import time
import math
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.cluster import DBSCAN

# Currently using emuilate stream to simulate LiDAR data, need to test with real LiDAR data

# Cutoff the height at 1m?
VERTICAL_CUTOFF = 1

def emulate_stream():
    """
    This function emulates a stream of LiDAR data by generating random points.
    It creates a list of dictionaries, each representing a point with x, y, z coordinates and intensity.
    Need to add logic for handling height cutoff.
    """
    import random

    # Create a buffer to store points
    buffer = []

    # Generate a list of 1000 random points and add them to the buffer
    for _ in range(1000):
        point = {
            'x': round(random.uniform(-10.0, 10.0), 4),
            'y': round(random.uniform(-10.0, 10.0), 4),
            'z': round(random.uniform(-10.0, 10.0), 4),
            'intensity': random.randint(0, 255)
        }
        buffer.append(point)

    return buffer

def async_get_scan():
    """
    Main function used to get the LiDAR scan data and call other helper functions.
    """
    emulated_buffer = emulate_stream()
    # Simulate processing the buffer
    for point in emulated_buffer:
        x = point['x']
        y = point['y']
        z = point['z']
        intensity = point['intensity']
        print(f"Point {emulated_buffer.index(point) + 1}: x={x}, y={y}, z={z}, intensity={intensity}")
    
def listen(duration: int):
    """
    Simulate listening for data with small delay
    """
    for i in range(duration):
        print(f"Listening... {i + 1}/{duration}")
        async_get_scan() # call the simulated scan function
        time.sleep(0.1) # simulate small delay

def circular_shift(data: List[float]) -> List[float]:
    """
    Perform a circular shift on the input list.
    The first element moves to the end of the list.
    """
    if not data:
        return data
    return data[1:] + [data[0]]

def calculate_angle_difference(start_angle: int, end_angle: int) -> float:
    """
    Calculate the angle in radians from start_angle to end_angle.
    The angles are given in hundredths of a degree.
    """
    start_angle_rad = math.radians(start_angle / 100.0)
    end_angle_rad = math.radians(end_angle / 100.0)
    return end_angle_rad - start_angle_rad

def get_distance(data: List[float]) -> float:
    """
    Calculate the distance from the data list.
    The first element is the distance, the second is the strength.
    """
    if len(data) < 2:
        raise ValueError("Data list must contain at least two elements.")
    distance = data[0]
    strength = data[1]
    return distance, strength

def filter(point_cloud: List[dict], cutoff_height: float) -> List[dict]:
    """
    Filter the point cloud data based on a height cutoff.
    """
    filtered_point_cloud = []
    for point in point_cloud:
        if point['z'] < cutoff_height:
            filtered_point_cloud.append(point)

    # Handle noise and preprocessing, intensity logic?

    
    return filtered_point_cloud

def detect_obstacles(point_cloud: List[dict], threshold: float) -> List[dict]:
    """
    Detect obstacles in the point cloud data based on a distance threshold.
    """
    obstacles = []
    for point in point_cloud:
        distance = math.sqrt(point['x']**2 + point['y']**2 + point['z']**2)
        point['distance'] = distance
        if point['distance'] < threshold:
            obstacles.append(point)
    
    # Handle logic for detections. Set flags to indicate type of object?

    return obstacles

def cluster_points(point_cloud: List[dict], neighbor_threshold: float):
    """
    This needs more work, rough implementation of clustering.
    Using DBSCAN for clustering based on distance threshold.
    DO NOT USE. USELSS AS OF NOW.
    """
    points_array = np.array([[point['x'], point['y'], point['z']] for point in point_cloud])
    
    clustering = DBSCAN(eps=neighbor_threshold, min_samples=5).fit(points_array)
    labels = clustering.labels_
    clusters = {} # Dictionary to hold clusters
    for label in set(labels):
        if label == -1:
            continue  # Ignore noise
        clusters[label] = [point for i, point in enumerate(point_cloud) if labels[i] == label]
    return clusters

def create_occupancy_grid(point_cloud, resolution=0.1, x_range=(-10, 10), y_range=(-10, 10)):
    """
    Creates a 2D occupancy grid from LiDAR point cloud
    """
    
    grid_width = int((x_range[1] - x_range[0]) / resolution)
    grid_height = int((y_range[1] - y_range[0]) / resolution)
    
    grid = np.zeros((grid_height, grid_width))
    
    # Fill grid based on point cloud data
    for point in point_cloud:
        if x_range[0] <= point['x'] < x_range[1] and y_range[0] <= point['y'] < y_range[1]:
            # Calculate grid indices
            grid_x = int((point['x'] - x_range[0]) / resolution)
            grid_y = int((point['y'] - y_range[0]) / resolution)
            
            # Ensure indices are within bounds
            grid_x = min(grid_x, grid_width - 1)
            grid_y = min(grid_y, grid_height - 1)
            
            grid[grid_y, grid_x] = 1
            
    return grid


if __name__ == "__main__":
    print("Running LiDAR simulation...")

    #listen(3) # Uncomment to simulate listening for data

    point_cloud = emulate_stream()
    grid = create_occupancy_grid(point_cloud, resolution=0.1, x_range=(-10, 10), y_range=(-10, 10))

    
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='binary', origin='lower')
    plt.colorbar(label='Occupancy (0=free, 1=occupied)')
    
    # Add grid lines
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set labels and title
    plt.xlabel('X Grid Cells')
    plt.ylabel('Y Grid Cells')
    plt.title('LiDAR Occupancy Grid')
    
    # Add coordinate information in the corner
    plt.text(0.02, 0.02, f'Resolution: {0.1}m\nRange: X=(-10,10), Y=(-10,10)', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

    
