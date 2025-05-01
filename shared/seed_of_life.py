"""
Seed of Life Pattern

This module implements the Seed of Life sacred geometry pattern.
The Seed of Life consists of seven circles arranged with six-fold symmetry,
one circle in the center and six surrounding circles placed so that their
centers lie on the circumference of the central circle.

This pattern is a fundamental template of creation and forms the basis
for the Flower of Life pattern when extended further.

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
    filename='seed_of_life.log'
)
logger = logging.getLogger('seed_of_life')

class SeedOfLife:
    """
    Implementation of the Seed of Life sacred geometry pattern.
    
    The Seed of Life consists of seven circles: one circle in the center and six
    circles arranged around it with their centers on the circumference of the
    central circle. This forms a perfect template for creation and cellular division.
    """
    
    def __init__(self, radius=1.0, resolution=64):
        """
        Initialize a new Seed of Life pattern.
        
        Args:
            radius (float): Radius of each circle forming the pattern
            resolution (int): Resolution of the generated pattern matrices
        """
        self.radius = radius
        self.resolution = resolution
        
        # Calculate circle centers
        self.circle_centers = self._calculate_circle_centers()
        
        # Generate pattern matrices
        self.pattern_2d = None
        self.pattern_3d = None
        self.generate_2d_pattern()
        
        logger.info(f"Seed of Life initialized with radius {radius} and resolution {resolution}")
    
    def _calculate_circle_centers(self):
        """
        Calculate the centers for all circles in the Seed of Life pattern.
        
        Returns:
            list: List of (x, y) circle center coordinates
        """
        # Central circle at origin
        centers = [(0, 0)]
        
        # Six surrounding circles (hexagonal arrangement)
        for i in range(6):
            angle = i * np.pi / 3  # 60 degrees apart
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            centers.append((x, y))
        
        logger.info(f"Calculated {len(centers)} circle centers for Seed of Life")
        return centers
    
    def generate_2d_pattern(self):
        """
        Generate a 2D matrix representation of the Seed of Life pattern.
        
        This creates a 2D array where higher values represent the
        overlapping circles, with the highest values at intersection points.
        
        Returns:
            ndarray: 2D matrix representation of the Seed of Life
        """
        # Create the bounds of the pattern
        max_bound = 2 * self.radius
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
        
        logger.info(f"2D Seed of Life pattern generated with shape {self.pattern_2d.shape}")
        return self.pattern_2d
    
    def generate_3d_pattern(self, height=1.0):
        """
        Generate a 3D matrix representation of the Seed of Life pattern.
        
        This extends the 2D pattern into 3D space by creating a volume with
        the specified height, centered at z=0.
        
        Args:
            height (float): Height of the 3D pattern
            
        Returns:
            ndarray: 3D matrix representation of the Seed of Life
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
        
        logger.info(f"3D Seed of Life pattern generated with shape {self.pattern_3d.shape}")
        return self.pattern_3d
    
    def get_2d_pattern(self):
        """
        Get the 2D pattern matrix.
        
        Returns:
            ndarray: 2D matrix representation of the Seed of Life
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
            ndarray: 3D matrix representation of the Seed of Life
        """
        if self.pattern_3d is None:
            self.generate_3d_pattern(height)
        return self.pattern_3d
    
    def get_intersection_nodes(self, min_overlap=3):
        """
        Find the intersection nodes in the Seed of Life pattern.
        
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
                        max_bound = 2 * self.radius
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
        Visualize the 2D Seed of Life pattern.
        
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
            max_bound = 2 * self.radius
            
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
            
            plt.title(f"Seed of Life Pattern ({len(self.circle_centers)} circles)")
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
    
    def visualize_3d(self, threshold=3, opacity=0.3, show=True, save_path=None):
        """
        Create a 3D visualization of the Seed of Life pattern.
        
        Args:
            threshold (float): Overlap threshold for visualization
            opacity (float): Opacity of the volume rendering
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            if self.pattern_3d is None:
                self.generate_3d_pattern()
                
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get the data
            x, y, z = np.indices(self.pattern_3d.shape)
            
            # Create a mask for thresholding
            mask = self.pattern_3d > threshold
            
            # Plot the voxels
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=threshold, vmax=np.max(self.pattern_3d))
            colors = cmap(norm(self.pattern_3d))
            
            ax.voxels(mask, facecolors=colors, alpha=opacity)
            
            # Add circle centers
            centers_3d = [(cx, cy, 0) for cx, cy in self.circle_centers]
            center_x = [c[0] for c in centers_3d]
            center_y = [c[1] for c in centers_3d]
            center_z = [c[2] for c in centers_3d]
            
            # Scale the centers to match the 3D grid
            max_bound = 2 * self.radius
            center_x = [(cx + max_bound) * self.resolution / (2 * max_bound) for cx in center_x]
            center_y = [(cy + max_bound) * self.resolution / (2 * max_bound) for cy in center_y]
            center_z = [self.resolution // 2 for _ in center_z]  # Middle of z-axis
            
            ax.scatter(center_x, center_y, center_z, color='red', s=50, marker='o')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Seed of Life 3D Pattern")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"3D visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in 3D visualization: {str(e)}")
            return False
    
    def get_geometric_properties(self):
        """
        Get the key geometric properties of the Seed of Life pattern.
        
        Returns:
            dict: Dictionary of geometric properties
        """
        # Distance between adjacent circle centers
        center_distance = self.radius
        
        # Calculate the central hexagon dimensions
        inner_radius = center_distance / 2  # Radius of inscribed circle
        outer_radius = center_distance  # Radius of circumscribed circle
        
        # Calculate intersection regions
        inner_area = np.pi * self.radius**2  # Area of one circle
        
        # Rough approximation of total pattern area
        # Central circle + partial area of surrounding circles
        total_area = inner_area + 6 * (inner_area * 0.7)  # 70% of each outer circle adds to total
        
        properties = {
            'radius': self.radius,
            'num_circles': len(self.circle_centers),
            'center_distance': center_distance,
            'inner_hexagon_radius': inner_radius,
            'outer_hexagon_radius': outer_radius,
            'circle_area': inner_area,
            'approximate_total_area': total_area
        }
        
        return properties
    
    def __str__(self):
        """String representation of the Seed of Life pattern."""
        props = self.get_geometric_properties()
        return (f"Seed of Life Pattern\n"
                f"Radius: {props['radius']}\n"
                f"Number of Circles: {props['num_circles']}\n"
                f"Circle Area: {props['circle_area']:.4f}\n"
                f"Inner Hexagon Radius: {props['inner_hexagon_radius']:.4f}\n"
                f"Outer Hexagon Radius: {props['outer_hexagon_radius']:.4f}\n"
                f"Resolution: {self.resolution}x{self.resolution}")


if __name__ == "__main__":
    # Example usage
    seed = SeedOfLife(radius=1.0, resolution=256)
    print(seed)
    
    # Visualize
    seed.visualize_2d(save_path="seed_of_life_2d.png")
    
    # Get pattern matrices for use in field systems
    pattern_2d = seed.get_2d_pattern()
    pattern_3d = seed.get_3d_pattern(height=0.5)
    
    # Get intersection nodes
    nodes = seed.get_intersection_nodes(min_overlap=3)
    print(f"Found {len(nodes)} intersection nodes")