"""
Vesica Piscis Module

This module implements the fundamental vesica piscis sacred geometry pattern.
The vesica piscis is created by the intersection of two circles of the same radius,
where the center of each circle lies on the circumference of the other.

Key functions:
- Generate precise vesica piscis geometry
- Calculate sacred ratios within the pattern
- Establish duality functions
- Calculate energy at intersection points
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional

def generate_vesica_piscis_2d(center1: Tuple[float, float], 
                              center2: Tuple[float, float],
                              radius: float, 
                              resolution: int = 100) -> Dict[str, Any]:
    """
    Generate a 2D vesica piscis pattern with precise geometric properties.
    
    Args:
        center1: The (x, y) center of the first circle
        center2: The (x, y) center of the second circle
        radius: The radius of both circles
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and calculated properties
    """
    # Verify that centers are correctly positioned for vesica piscis
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    if not np.isclose(distance, radius):
        raise ValueError("For a valid vesica piscis, the distance between centers must equal the radius")
    
    # Create a grid for calculating the pattern
    x = np.linspace(min(center1[0], center2[0]) - radius, 
                    max(center1[0], center2[0]) + radius, 
                    resolution)
    y = np.linspace(min(center1[1], center2[1]) - radius, 
                    max(center1[1], center2[1]) + radius, 
                    resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distances from each point to the circle centers
    dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2)
    dist2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2)
    
    # Create the circles and vesica piscis intersection
    circle1 = dist1 <= radius
    circle2 = dist2 <= radius
    vesica = circle1 & circle2
    
    # Calculate sacred ratios within the vesica piscis
    # Height to width ratio (approximates sqrt(3))
    width = 2 * radius
    height = np.sqrt(3) * radius
    ratio = height / width
    
    # Calculate intersection points
    # The top and bottom points of the vesica piscis
    mid_x = (center1[0] + center2[0]) / 2
    mid_y = (center1[1] + center2[1]) / 2
    
    # Calculate direction vector from center1 to center2
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length, dy/length
    
    # Calculate perpendicular vector
    px, py = -dy, dx
    
    # Intersection points are along perpendicular line through midpoint
    # Calculate distance from midpoint to the intersection
    d = np.sqrt(radius**2 - (length/2)**2)
    
    # Calculate intersection points
    point1 = (mid_x + px * d, mid_y + py * d)
    point2 = (mid_x - px * d, mid_y - py * d)
    
    return {
        'pattern': vesica,
        'circle1': circle1,
        'circle2': circle2,
        'centers': (center1, center2),
        'radius': radius,
        'width': width,
        'height': height,
        'ratio': ratio,
        'intersection_points': (point1, point2),
        'midpoint': (mid_x, mid_y),
        'grid_x': X,
        'grid_y': Y
    }

def generate_vesica_piscis_3d(center1: Tuple[float, float, float], 
                             center2: Tuple[float, float, float],
                             radius: float, 
                             resolution: int = 50) -> Dict[str, Any]:
    """
    Generate a 3D vesica piscis pattern (intersection of two spheres).
    
    Args:
        center1: The (x, y, z) center of the first sphere
        center2: The (x, y, z) center of the second sphere
        radius: The radius of both spheres
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and calculated properties
    """
    # Verify that centers are correctly positioned for vesica piscis
    distance = np.sqrt(sum((center1[i] - center2[i])**2 for i in range(3)))
    if not np.isclose(distance, radius):
        raise ValueError("For a valid vesica piscis, the distance between centers must equal the radius")
    
    # Create a grid for calculating the pattern
    min_x = min(center1[0], center2[0]) - radius
    max_x = max(center1[0], center2[0]) + radius
    min_y = min(center1[1], center2[1]) - radius
    max_y = max(center1[1], center2[1]) + radius
    min_z = min(center1[2], center2[2]) - radius
    max_z = max(center1[2], center2[2]) + radius
    
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    z = np.linspace(min_z, max_z, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Calculate distances from each point to the sphere centers
    dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2 + (Z - center1[2])**2)
    dist2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2)
    
    # Create the spheres and vesica piscis intersection
    sphere1 = dist1 <= radius
    sphere2 = dist2 <= radius
    vesica = sphere1 & sphere2
    
    # Calculate the circle of intersection
    # Find the midpoint between centers
    midpoint = [(center1[i] + center2[i])/2 for i in range(3)]
    
    # Calculate direction vector from center1 to center2
    direction = [center2[i] - center1[i] for i in range(3)]
    length = np.sqrt(sum(d**2 for d in direction))
    direction = [d/length for d in direction]
    
    # Calculate circle radius in the intersection
    intersection_radius = np.sqrt(radius**2 - (length/2)**2)
    
    return {
        'pattern': vesica,
        'sphere1': sphere1,
        'sphere2': sphere2,
        'centers': (center1, center2),
        'radius': radius,
        'intersection_radius': intersection_radius,
        'midpoint': midpoint,
        'direction': direction,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z
    }

def calculate_vesica_sacred_ratios(vesica_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the sacred ratios and proportions in the vesica piscis.
    
    Args:
        vesica_data: Dictionary containing vesica piscis geometry data
        
    Returns:
        Dictionary of calculated sacred ratios
    """
    radius = vesica_data['radius']
    
    # Height of the vesica is distance from top to bottom intersection
    height = 2 * np.sqrt(radius**2 - (radius/2)**2)
    
    # Width is the distance between centers
    width = radius
    
    # Key sacred ratios
    height_width_ratio = height / width  # Approximates √3
    
    # Approximation to key ratios
    sqrt_3_approx = height_width_ratio / 2  # Should be close to √3
    
    # Area of vesica piscis
    vesica_area = radius**2 * (np.pi - np.sin(np.pi/3) * np.cos(np.pi/3) - np.pi/3)
    
    # Circle area
    circle_area = np.pi * radius**2
    
    # Area ratio
    area_ratio = vesica_area / circle_area
    
    return {
        'height_width_ratio': height_width_ratio,
        'sqrt_3_approximation': sqrt_3_approx,
        'vesica_area': vesica_area,
        'circle_area': circle_area,
        'area_ratio': area_ratio,
    }

def calculate_vesica_energy_distribution(vesica_data: Dict[str, Any], 
                                        base_frequency: float = 432.0) -> Dict[str, Any]:
    """
    Calculate the energy distribution within the vesica piscis.
    
    Args:
        vesica_data: Dictionary containing vesica piscis geometry data
        base_frequency: The base frequency for energy calculations
        
    Returns:
        Dictionary containing energy distribution data
    """
    # Extract grid and pattern data
    X = vesica_data['grid_x']
    Y = vesica_data['grid_y']
    
    if 'grid_z' in vesica_data:  # 3D case
        Z = vesica_data['grid_z']
        center1 = vesica_data['centers'][0]
        center2 = vesica_data['centers'][1]
        midpoint = vesica_data['midpoint']
        
        # Calculate distance from each point to the midpoint
        dist_mid = np.sqrt((X - midpoint[0])**2 + (Y - midpoint[1])**2 + (Z - midpoint[2])**2)
        
        # Calculate distance from each point to the intersection circle
        # Project points onto the plane of the intersection circle
        direction = vesica_data['direction']
        
        # Create array of all points
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Vector from midpoint to each point
        vectors = points - np.array(midpoint)
        
        # Project vectors onto direction (dot product)
        projections = np.sum(vectors * np.array(direction), axis=1).reshape(X.shape)
        
        # Distance from point to plane of intersection
        dist_plane = np.abs(projections)
        
        # Points on the plane have high energy
        # Points at the exact intersection of the spheres have maximum energy
        dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2 + (Z - center1[2])**2)
        dist2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2)
        sphere_boundary = np.abs(dist1 - vesica_data['radius']) + np.abs(dist2 - vesica_data['radius'])
        
        # Energy increases near the intersection circle and decreases with distance
        energy = np.exp(-dist_plane / (vesica_data['radius'] * 0.1)) * \
                 np.exp(-sphere_boundary / (vesica_data['radius'] * 0.05))
                 
    else:  # 2D case
        center1 = vesica_data['centers'][0]
        center2 = vesica_data['centers'][1]
        
        # Calculate distance from each point to the centers
        dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2)
        dist2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2)
        
        # Energy is highest at the intersection points
        # It forms a standing wave pattern based on the distance from each center
        wave1 = np.sin(2 * np.pi * dist1 / vesica_data['radius'] * (base_frequency / 432.0))
        wave2 = np.sin(2 * np.pi * dist2 / vesica_data['radius'] * (base_frequency / 432.0))
        
        # Create interference pattern 
        energy = (wave1 + wave2)**2
        
        # Apply mask for vesica region
        energy = energy * vesica_data['pattern']
    
    # Normalize energy to [0, 1]
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Find points of maximum energy
    # These are sacred points in the vesica piscis
    if 'grid_z' in vesica_data:  # 3D case
        # Flatten arrays for easier processing
        flat_energy = energy.flatten()
        flat_x = X.flatten()
        flat_y = Y.flatten()
        flat_z = Z.flatten()
        
        # Find indices of top energy points
        top_indices = np.argsort(flat_energy)[-10:]  # Get 10 highest points
        
        # Get coordinates of these points
        max_energy_points = [(flat_x[i], flat_y[i], flat_z[i]) for i in top_indices]
        max_energy_values = [flat_energy[i] for i in top_indices]
    else:  # 2D case
        # Flatten arrays for easier processing
        flat_energy = energy.flatten()
        flat_x = X.flatten()
        flat_y = Y.flatten()
        
        # Find indices of top energy points
        top_indices = np.argsort(flat_energy)[-10:]  # Get 10 highest points
        
        # Get coordinates of these points
        max_energy_points = [(flat_x[i], flat_y[i]) for i in top_indices]
        max_energy_values = [flat_energy[i] for i in top_indices]
    
    return {
        'energy': energy,
        'max_energy_points': max_energy_points,
        'max_energy_values': max_energy_values,
        'base_frequency': base_frequency
    }

def calculate_vesica_duality_properties(vesica_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the duality properties of the vesica piscis.
    The vesica piscis represents the union of opposites and balance of dualities.
    
    Args:
        vesica_data: Dictionary containing vesica piscis geometry data
        
    Returns:
        Dictionary containing duality properties
    """
    # Calculate the axis of duality (line connecting the centers)
    c1 = vesica_data['centers'][0]
    c2 = vesica_data['centers'][1]
    
    # Midpoint represents the balance point between dualities
    if len(c1) == 3:  # 3D case
        midpoint = [(c1[i] + c2[i])/2 for i in range(3)]
        axis_vector = [c2[i] - c1[i] for i in range(3)]
    else:  # 2D case
        midpoint = [(c1[i] + c2[i])/2 for i in range(2)]
        axis_vector = [c2[i] - c1[i] for i in range(2)]
    
    # Calculate polarity measure (normalized position along the axis)
    # Points can be measured from -1 (fully in circle 1) to +1 (fully in circle 2)
    
    # Calculate unity ratio (ratio of intersection area to total area)
    if 'pattern' in vesica_data:
        intersection_count = np.sum(vesica_data['pattern'])
        union_count = np.sum(vesica_data['circle1'] | vesica_data['circle2'])
        unity_ratio = intersection_count / union_count if union_count > 0 else 0
    else:
        # Theoretical calculation for 3D
        unity_ratio = 0.39  # Approximate for spheres
    
    # Calculate dualistic properties
    duality_properties = {
        'axis_vector': axis_vector,
        'midpoint': midpoint,
        'unity_ratio': unity_ratio,
        'yin_yang_balance': 0.5,  # Perfect balance by design
        'opposition_strength': 1.0,  # Full opposition by design
        'reconciliation_potential': unity_ratio * 2  # Higher intersection means greater reconciliation
    }
    
    return duality_properties

def visualize_vesica_piscis_2d(vesica_data: Dict[str, Any], 
                              show_energy: bool = False,
                              show_sacred_points: bool = False) -> plt.Figure:
    """
    Create a visualization of the 2D vesica piscis pattern.
    
    Args:
        vesica_data: Dictionary containing vesica piscis geometry data
        show_energy: Whether to overlay energy distribution
        show_sacred_points: Whether to mark sacred points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    X = vesica_data['grid_x']
    Y = vesica_data['grid_y']
    
    # Plot the circles as contours
    ax.contour(X, Y, vesica_data['circle1'].astype(int), levels=[0.5], colors=['blue'], alpha=0.5)
    ax.contour(X, Y, vesica_data['circle2'].astype(int), levels=[0.5], colors=['red'], alpha=0.5)
    
    # Plot the vesica piscis
    if show_energy and 'energy' in vesica_data:
        # Show energy distribution as a heatmap
        im = ax.imshow(vesica_data['energy'], extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', cmap='viridis', alpha=0.7)
        fig.colorbar(im, ax=ax, label='Energy')
    else:
        # Just show the vesica piscis region
        ax.contourf(X, Y, vesica_data['pattern'].astype(int), levels=[0.5, 1.5], 
                   colors=['purple'], alpha=0.3)
    
    # Mark centers of circles
    center1, center2 = vesica_data['centers']
    ax.plot(center1[0], center1[1], 'bo', markersize=8, label='Circle 1 Center')
    ax.plot(center2[0], center2[1], 'ro', markersize=8, label='Circle 2 Center')
    
    # Mark intersection points
    if 'intersection_points' in vesica_data:
        point1, point2 = vesica_data['intersection_points']
        ax.plot(point1[0], point1[1], 'go', markersize=8, label='Intersection Point 1')
        ax.plot(point2[0], point2[1], 'go', markersize=8, label='Intersection Point 2')
    
    # Mark sacred points (high energy points)
    if show_sacred_points and 'max_energy_points' in vesica_data:
        for i, point in enumerate(vesica_data['max_energy_points']):
            if len(point) == 2:  # 2D points
                ax.plot(point[0], point[1], 'yo', markersize=6)
    
    # Mark midpoint
    if 'midpoint' in vesica_data:
        midpoint = vesica_data['midpoint']
        ax.plot(midpoint[0], midpoint[1], 'mo', markersize=8, label='Midpoint')
    
    # Add title and labels
    ax.set_title('Vesica Piscis Sacred Geometry')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    ax.legend(loc='upper right')
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    return fig

def embed_vesica_in_field(field_array: np.ndarray, 
                         center1: Tuple[float, float, float],
                         center2: Tuple[float, float, float],
                         radius: float,
                         strength: float = 1.0) -> np.ndarray:
    """
    Embed a vesica piscis pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center1: Center coordinates of first sphere (x, y, z)
        center2: Center coordinates of second sphere (x, y, z)
        radius: Radius of the spheres
        strength: Strength of the pattern (0.0 to 1.0)
        
    Returns:
        Modified field array with embedded pattern
    """
    field_shape = field_array.shape
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.arange(field_shape[0]),
        np.arange(field_shape[1]),
        np.arange(field_shape[2]),
        indexing='ij'
    )
    
    # Calculate distances from each point to the sphere centers
    dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
    dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)
    
    # Create the spheres (shells rather than solid spheres)
    # Using a Gaussian function to create a smooth shell
    shell_width = radius * 0.1
    shell1 = np.exp(-((dist1 - radius) / shell_width)**2)
    shell2 = np.exp(-((dist2 - radius) / shell_width)**2)
    
    # Combine shells to form vesica piscis pattern
    # Higher values where both shells overlap
    vesica_pattern = shell1 * shell2
    
    # Normalize pattern to [0, 1] range
    vesica_pattern = vesica_pattern / np.max(vesica_pattern)
    
    # Apply pattern to field with given strength
    # This enhances field values where the pattern is strong
    modified_field = field_array * (1.0 + vesica_pattern * strength)
    
    # Normalize field after modification
    modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field


# Example usage
if __name__ == "__main__":
    # Create a 2D vesica piscis
    center1 = (50, 50)
    center2 = (100, 50)
    radius = 50
    
    vesica_2d = generate_vesica_piscis_2d(center1, center2, radius)
    
    # Calculate energy distribution
    energy_data = calculate_vesica_energy_distribution(vesica_2d)
    vesica_2d.update(energy_data)
    
    # Calculate sacred ratios
    ratios = calculate_vesica_sacred_ratios(vesica_2d)
    print("Sacred Ratios:")
    for key, value in ratios.items():
        print(f"{key}: {value}")
    
    # Calculate duality properties
    duality = calculate_vesica_duality_properties(vesica_2d)
    print("\nDuality Properties:")
    for key, value in duality.items():
        if key != 'axis_vector' and key != 'midpoint':
            print(f"{key}: {value}")
    
    # Visualize
    fig = visualize_vesica_piscis_2d(vesica_2d, show_energy=True, show_sacred_points=True)
    plt.show()
