"""
Flower of Life Pattern

This module implements the Flower of Life sacred geometry pattern.
The Flower of Life consists of multiple overlapping circles arranged in a flower-like pattern,
forming a model of creation with important geometric properties.

Author: Soul Development Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='flower_of_life.log'
)
logger = logging.getLogger('flower_of_life')

class FlowerOfLife:
    """
    Implementation of the Flower of Life sacred geometry pattern.
    
    The Flower of Life is formed from multiple Seed of Life patterns arranged
    in a hexagonal grid, creating a complex interconnected pattern that holds
    all geometric information for creation.
    """
    
    def __init__(self, radius=1.0, resolution=64, iterations=3):
        """
        Initialize a new Flower of Life pattern.
        
        Args:
            radius (float): Radius of each circle forming the pattern
            resolution (int): Resolution of the generated pattern matrices
            iterations (int): Number of iterations to expand the pattern
        """
        self.radius = radius
        self.resolution = resolution
        self.iterations = iterations
        
        # Calculate circle centers
        self.circle_centers = self._calculate_circle_centers()
        
        # Generate pattern matrices
        self.pattern_2d = None
        self.pattern_3d = None
        
        logger.info(f"Flower of Life initialized with radius {radius}, resolution {resolution}, iterations {iterations}")
    
    def _calculate_circle_centers(self):
        """
        Calculate the centers for all circles in the Flower of Life pattern.
        
        Returns:
            list: List of (x, y) circle center coordinates
        """
        centers = [(0, 0)]  # Central circle
        
        # First ring is at 60-degree intervals
        for i in range(6):
            angle = i * np.pi / 3
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            centers.append((x, y))
        
        # Additional iterations for larger Flower of Life
        if self.iterations > 1:
            current_centers = centers.copy()
            for iteration in range(1, self.iterations):
                new_centers = []
                for center in current_centers:
                    cx, cy = center
                    # Add 6 surrounding circles
                    for i in range(6):
                        angle = i * np.pi / 3
                        x = cx + self.radius * np.cos(angle)
                        y = cy + self.radius * np.sin(angle)
                        
                        # Check if this center is already in our list (within tolerance)
                        is_duplicate = False
                        for existing_center in centers:
                            ex, ey = existing_center
                            if np.sqrt((x - ex)**2 + (y - ey)**2) < self.radius * 0.1:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            new_centers.append((x, y))
                            centers.append((x, y))
                
                current_centers = new_centers
        
        logger.info(f"Calculated {len(centers)} circle centers for Flower of Life")
        return centers
    
    def generate_2d_pattern(self):
        """
        Generate a 2D matrix representation of the Flower of Life pattern.
        
        This creates a 2D array where higher values represent the
        overlapping circles, with the highest values at intersection points.
        
        Returns:
            ndarray: 2D matrix representation of the Flower of Life
        """
        # Create the bounds of the pattern
        max_bound = self.radius * (2 + self.iterations)
        x = np.linspace(-max_bound, max_bound, self.resolution)
        y = np.linspace(-max_bound, max_bound, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Initialize pattern matrix
        self.pattern_2d = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        
        # Add contribution from each circle
        for center_x, center_y in self.circle_centers:
            # Calculate distance from center for each point
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # Add 1 to the pattern where points are inside the circle
            self.pattern_2d += (distance <= self.radius).astype(float)
        
        logger.info(f"2D Flower of Life pattern generated with shape {self.pattern_2d.shape}")
        return self.pattern_2d
    
    def generate_3d_pattern(self, height=1.0):
        """
        Generate a 3D matrix representation of the Flower of Life pattern.
        
        This extends the 2D pattern into 3D space by creating a volume with
        the specified height, centered at z=0.
        
        Args:
            height (float): Height of the 3D pattern
            
        Returns:
            ndarray: 3D matrix representation of the Flower of Life
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
        
        # Create z coordinate array
        z = np.linspace(-height/2, height/2, self.resolution)
        
        # Initialize 3D pattern
        self.pattern_3d = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float64)
        
        # Extend 2D pattern to 3D with Gaussian falloff from central plane
        for k, z_val in enumerate(z):
            # Calculate falloff based on distance from central plane (z=0)
            falloff = np.exp(-(z_val**2) / (0.2 * height)**2)
            
            # Set this layer's pattern to 2D pattern multiplied by falloff
            self.pattern_3d[:, :, k] = self.pattern_2d * falloff
        
        logger.info(f"3D Flower of Life pattern generated with shape {self.pattern_3d.shape}")
        return self.pattern_3d
    
    def get_2d_pattern(self):
        """
        Get the 2D pattern matrix.
        
        Returns:
            ndarray: 2D matrix representation of the Flower of Life
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
        return self.pattern_2d
    
    def get_3d_pattern(self, height=1.0):
        """
        Get the 3D pattern matrix.
        
        Args:
            height (float): Height of the 3D pattern
            
        Returns:
            ndarray: 3D matrix representation of the Flower of Life
        """
        if self.pattern_3d is None:
            self.generate_3d_pattern(height)
        return self.pattern_3d
    
    def get_intersection_nodes(self, min_overlap=3):
        """
        Find the intersection nodes in the Flower of Life pattern.
        
        These nodes are points where multiple circles intersect and represent
        important energy nodes in the pattern.
        
        Args:
            min_overlap (int): Minimum number of overlapping circles to consider a node
            
        Returns:
            list: List of node coordinates and their overlap count
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
            
        # Find local maxima in the pattern that have at least min_overlap circles
        nodes = []
        
        # Skip the border pixels
        for i in range(1, self.resolution-1):
            for j in range(1, self.resolution-1):
                value = self.pattern_2d[i, j]
                
                # Check if this is a local maximum with sufficient overlap
                if value >= min_overlap:
                    # Check if it's a local maximum
                    neighborhood = self.pattern_2d[i-1:i+2, j-1:j+2]
                    if value >= np.max(neighborhood):
                        # Convert grid coordinates to pattern coordinates
                        max_bound = self.radius * (2 + self.iterations)
                        x = -max_bound + 2 * max_bound * j / self.resolution
                        y = -max_bound + 2 * max_bound * i / self.resolution
                        
                        nodes.append({
                            'position': (x, y),
                            'overlap': int(value)
                        })
        
        # Sort nodes by overlap count (descending)
        nodes.sort(key=lambda x: x['overlap'], reverse=True)
        
        logger.info(f"Found {len(nodes)} intersection nodes with {min_overlap}+ overlapping circles")
        return nodes
    
    def visualize_2d(self, show_centers=True, show_nodes=True, min_node_overlap=3, 
                    show=True, save_path=None):
        """
        Visualize the 2D Flower of Life pattern.
        
        Args:
            show_centers (bool): Whether to show circle centers
            show_nodes (bool): Whether to show intersection nodes
            min_node_overlap (int): Minimum circle overlap to consider a node
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            if self.pattern_2d is None:
                self.generate_2d_pattern()
                
            plt.figure(figsize=(10, 8))
            
            # Create the bounds of the pattern for visualization
            max_bound = self.radius * (2 + self.iterations)
            
            # Plot the pattern as a heatmap
            plt.imshow(self.pattern_2d, extent=[-max_bound, max_bound, -max_bound, max_bound],
                      cmap='viridis', origin='lower')
            plt.colorbar(label='Overlapping Circles')
            
            # Add circle outlines
            if show_centers:
                ax = plt.gca()
                for center_x, center_y in self.circle_centers:
                    circle = plt.Circle((center_x, center_y), self.radius, 
                                      fill=False, color='white', linestyle='-', linewidth=0.5)
                    ax.add_patch(circle)
                    
                    # Add center points
                    plt.plot(center_x, center_y, 'wo', markersize=3)
            
            # Add intersection nodes
            if show_nodes:
                nodes = self.get_intersection_nodes(min_overlap=min_node_overlap)
                node_x = [node['position'][0] for node in nodes]
                node_y = [node['position'][1] for node in nodes]
                plt.plot(node_x, node_y, 'r*', markersize=5)
            
            plt.title(f"Flower of Life Pattern ({len(self.circle_centers)} circles)")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"2D visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in 2D visualization: {str(e)}")
            return False
    
    def get_geometric_properties(self):
        """
        Get the key geometric properties of the Flower of Life pattern.
        
        Returns:
            dict: Dictionary of geometric properties
        """
        # Calculate the number of circles
        num_circles = len(self.circle_centers)
        
        # Estimate total area
        circle_area = np.pi * self.radius**2
        
        # Get intersection nodes
        nodes = self.get_intersection_nodes(min_overlap=2)
        num_intersections = len(nodes)
        
        # Calculate special proportions
        vesica_piscis_ratio = np.sqrt(3) / 2  # Ratio in the vesica piscis formed by overlapping circles
        
        properties = {
            'radius': self.radius,
            'num_circles': num_circles,
            'num_iterations': self.iterations,
            'circle_area': circle_area,
            'approximate_total_area': num_circles * circle_area * 0.6,  # 0.6 accounts for overlaps
            'num_intersections': num_intersections,
            'vesica_piscis_ratio': vesica_piscis_ratio,
            'sacred_geometry_connections': [
                'Tree of Life',
                'Metatron\'s Cube',
                'Platonic Solids',
                'Fruit of Life'
            ]
        }
        
        return properties
    
    def __str__(self):
        """String representation of the Flower of Life pattern."""
        props = self.get_geometric_properties()
        return (f"Flower of Life Pattern\n"
                f"Radius: {props['radius']}\n"
                f"Number of Circles: {props['num_circles']}\n"
                f"Iterations: {props['num_iterations']}\n"
                f"Resolution: {self.resolution}x{self.resolution}")
