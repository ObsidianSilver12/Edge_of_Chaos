"""
Flower of Life Pattern Generator

This module provides functions to generate the Flower of Life sacred geometry pattern
for use in the Soul Development Framework's field systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

class FlowerOfLife:
    """
    Generator for the Flower of Life sacred geometry pattern.
    
    The Flower of Life is a geometric pattern composed of multiple evenly-spaced, 
    overlapping circles arranged in a flower-like pattern with six-fold symmetry.
    """
    
    def __init__(self, radius: float = 1.0, resolution: int = 256, iterations: int = 3, seed_rings: int = 1):
        """
        Initialize Flower of Life pattern generator.
        
        Args:
            radius: Base radius for the initial circle
            resolution: Resolution of the generated pattern array
            iterations: Number of iterations to build pattern complexity
            seed_rings: Number of initial seed rings
        """
        self.radius = radius
        self.resolution = resolution
        self.iterations = max(1, min(5, iterations))  # Cap iterations for performance
        self.seed_rings = max(1, min(3, seed_rings))
        logger.debug(f"Initialized Flower of Life generator: r={radius}, res={resolution}, iter={iterations}")
    
    def _create_circle_mask(self, center_x: float, center_y: float, radius: float, array_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a circular mask in a 2D array.
        
        Args:
            center_x, center_y: Center coordinates
            radius: Circle radius
            array_shape: Shape of the target array
            
        Returns:
            np.ndarray: 2D array with circle mask
        """
        y, x = np.ogrid[:array_shape[0], :array_shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        return mask.astype(np.float32)
    
    def _get_circle_centers(self) -> List[Tuple[float, float]]:
        """
        Calculate circle centers for the Flower of Life pattern.
        
        Returns:
            List[Tuple[float, float]]: List of (x,y) center coordinates
        """
        centers = [(0, 0)]  # Start with center point
        
        # Calculate position of first ring
        first_ring = []
        for i in range(6):
            angle = i * np.pi / 3  # 60-degree increments
            x = 2 * self.radius * np.cos(angle)
            y = 2 * self.radius * np.sin(angle)
            first_ring.append((x, y))
        
        centers.extend(first_ring)
        
        if self.iterations > 1:
            # Calculate subsequent iterations of circles
            for _ in range(self.iterations - 1):
                new_centers = []
                for center in centers:
                    for i in range(6):
                        angle = i * np.pi / 3
                        x = center[0] + 2 * self.radius * np.cos(angle)
                        y = center[1] + 2 * self.radius * np.sin(angle)
                        # Check if center is already in the list (within tolerance)
                        is_duplicate = any(
                            np.sqrt((x - cx)**2 + (y - cy)**2) < 0.1 * self.radius
                            for cx, cy in centers + new_centers
                        )
                        if not is_duplicate:
                            new_centers.append((x, y))
                
                centers.extend(new_centers)
        
        return centers
    
    def generate_2d_pattern(self) -> np.ndarray:
        """
        Generate a 2D pattern array of the Flower of Life.
        
        Returns:
            np.ndarray: 2D array containing the pattern
        """
        # Create empty pattern array
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Calculate proper offset to center the pattern
        center_x = self.resolution // 2
        center_y = self.resolution // 2
        
        # Get all circle centers
        centers = self._get_circle_centers()
        
        # Find overall pattern size to scale properly
        max_distance = max(
            np.sqrt(x**2 + y**2) + self.radius 
            for x, y in centers
        )
        
        # Scale factor to fit pattern within the resolution
        scale = min(center_x, center_y) / max_distance * 0.95
        
        # Draw each circle
        for cx, cy in centers:
            # Scale and translate coordinates
            scaled_x = center_x + cx * scale
            scaled_y = center_y + cy * scale
            scaled_radius = self.radius * scale
            
            # Create and add circle mask
            circle_mask = self._create_circle_mask(
                scaled_x, scaled_y, scaled_radius, pattern.shape
            )
            pattern += circle_mask
        
        # Apply sigmoid function to enhance edges
        pattern = 1.0 / (1.0 + np.exp(-10 * (pattern - 0.5)))
        
        # Normalize to 0-1 range
        if np.max(pattern) > 0:
            pattern /= np.max(pattern)
        
        logger.debug(f"Generated Flower of Life pattern with {len(centers)} circles")
        return pattern
    
    def generate_3d_pattern(self, z_depth: int = 32) -> np.ndarray:
        """
        Generate a 3D pattern array based on the Flower of Life.
        
        Args:
            z_depth: Depth of the 3D pattern
            
        Returns:
            np.ndarray: 3D array containing the pattern
        """
        # Generate 2D base pattern
        pattern_2d = self.generate_2d_pattern()
        
        # Create 3D pattern with z-axis falloff
        pattern_3d = np.zeros((self.resolution, self.resolution, z_depth), dtype=np.float32)
        center_z = z_depth // 2
        
        for z in range(z_depth):
            # Calculate z-axis falloff (gaussian)
            z_falloff = np.exp(-((z - center_z)**2) / (2 * (z_depth//6)**2))
            pattern_3d[:, :, z] = pattern_2d * z_falloff
        
        logger.debug(f"Generated 3D Flower of Life pattern with {z_depth} depth layers")
        return pattern_3d
    
    def visualize_pattern(self, is_3d: bool = False) -> None:
        """
        Visualize the generated pattern.
        
        Args:
            is_3d: Whether to visualize as 3D pattern
        """
        if is_3d:
            pattern = self.generate_3d_pattern()
            # Visualize middle slice and 3D view
            plt.figure(figsize=(12, 6))
            
            plt.subplot(121)
            plt.imshow(pattern[:, :, pattern.shape[2]//2], cmap='viridis')
            plt.title("Middle Z-slice")
            plt.colorbar()
            
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.subplot(122, projection='3d')
            x, y, z = np.where(pattern > 0.3)
            colors = pattern[x, y, z]
            ax.scatter(x, y, z, c=colors, alpha=0.1, s=1)
            plt.title("3D View (Threshold 0.3)")
            
        else:
            pattern = self.generate_2d_pattern()
            plt.figure(figsize=(8, 8))
            plt.imshow(pattern, cmap='viridis')
            plt.title("Flower of Life Pattern")
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the generator."""
        return f"FlowerOfLife(r={self.radius}, res={self.resolution}, iter={self.iterations})"


# Provide a function that matches the expected interface for the embedding system
def generate_flower_pattern(size: float) -> np.ndarray:
    """
    Generate a Flower of Life pattern for embedding in fields.
    
    Args:
        size: Size parameter for the pattern
        
    Returns:
        np.ndarray: 2D pattern array
    """
    # Size is used to determine radius (size/6 gives good results)
    fol = FlowerOfLife(radius=size/6, resolution=min(512, int(size*2)), iterations=3)
    return fol.generate_2d_pattern()

def get_base_glyph_elements(self) -> Dict[str, Any]:
    """
    Returns the geometric elements (circles) for a simple line art
    representation of the Flower of Life.
    """
    # Use _get_circle_centers which you confirmed exists
    all_circle_centers = self._get_circle_centers() 

    circles_data = []
    for center_pos_fol in all_circle_centers: # Renamed center_pos
        circles_data.append({'center': tuple(center_pos_fol), 'radius': self.radius})

    if not all_circle_centers:
        return {'circles': [], 'projection_type': '2d', 'bounding_box': {'xmin':-1.0,'xmax':1.0,'ymin':-1.0,'ymax':1.0}}

    all_x_fol = [c[0] for c in all_circle_centers]; all_y_fol = [c[1] for c in all_circle_centers] # Renamed
    padding_fol = self.radius * 0.2 # Renamed

    min_x_coord = min(all_x_fol) - self.radius
    max_x_coord = max(all_x_fol) + self.radius
    min_y_coord = min(all_y_fol) - self.radius
    max_y_coord = max(all_y_fol) + self.radius

    return {
        'circles': circles_data,
        'projection_type': '2d',
        'bounding_box': {
            'xmin': float(min_x_coord - padding_fol), 'xmax': float(max_x_coord + padding_fol),
            'ymin': float(min_y_coord - padding_fol), 'ymax': float(max_y_coord + padding_fol),
        }
    }

# Example usage
if __name__ == "__main__":
    # Test the pattern generator
    flower = FlowerOfLife(radius=10, resolution=256, iterations=3)
    pattern = flower.generate_2d_pattern()
    
    print(f"Pattern shape: {pattern.shape}")
    print(f"Pattern min: {np.min(pattern)}, max: {np.max(pattern)}")
    
    # Visualize
    flower.visualize_pattern(is_3d=False)
    
    # Also test 3D
    flower.visualize_pattern(is_3d=True)

