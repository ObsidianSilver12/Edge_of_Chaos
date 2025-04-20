"""
Flower of Life Module

This module implements the Flower of Life sacred geometry pattern.
The Flower of Life consists of multiple evenly-spaced, overlapping circles arranged in a
hexagonal pattern, forming a symmetrical and harmonious pattern.

Key functions:
- Generate precise Flower of Life geometry
- Calculate energy at pattern intersections
- Identify key sacred ratios and points
- Embed the pattern into energy fields
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional

def generate_flower_of_life_2d(center: Tuple[float, float],
                              first_radius: float,
                              num_layers: int = 3,
                              resolution: int = 100) -> Dict[str, Any]:
    """
    Generate a 2D Flower of Life pattern with precise geometric properties.
    
    Args:
        center: The (x, y) center of the pattern
        first_radius: The radius of each circle
        num_layers: Number of layers around the central circle (1-7)
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and calculated properties
    """
    # Validate input
    if num_layers < 1 or num_layers > 7:
        raise ValueError("Number of layers must be between 1 and 7")
    
    # Create a grid for calculating the pattern
    x_min = center[0] - first_radius * (2 * num_layers + 1)
    x_max = center[0] + first_radius * (2 * num_layers + 1)
    y_min = center[1] - first_radius * (2 * num_layers + 1)
    y_max = center[1] + first_radius * (2 * num_layers + 1)
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # First circle at the center
    circle_centers = [(center[0], center[1])]
    
    # Calculate centers for the first layer (6 circles)
    for i in range(6):
        angle = np.pi / 3 * i
        x_c = center[0] + first_radius * np.cos(angle)
        y_c = center[1] + first_radius * np.sin(angle)
        circle_centers.append((x_c, y_c))
    
    # Add additional layers if requested
    if num_layers >= 2:
        # Second layer (12 circles)
        for i in range(12):
            angle = np.pi / 6 * i
            x_c = center[0] + 2 * first_radius * np.cos(angle)
            y_c = center[1] + 2 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    if num_layers >= 3:
        # Third layer (18 circles)
        # This creates the complete Flower of Life with 19 circles total
        for i in range(18):
            angle = np.pi / 9 * i
            x_c = center[0] + 3 * first_radius * np.cos(angle)
            y_c = center[1] + 3 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    if num_layers >= 4:
        # Fourth layer (24 circles)
        for i in range(24):
            angle = np.pi / 12 * i
            x_c = center[0] + 4 * first_radius * np.cos(angle)
            y_c = center[1] + 4 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    if num_layers >= 5:
        # Fifth layer (30 circles)
        for i in range(30):
            angle = np.pi / 15 * i
            x_c = center[0] + 5 * first_radius * np.cos(angle)
            y_c = center[1] + 5 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    if num_layers >= 6:
        # Sixth layer (36 circles)
        for i in range(36):
            angle = np.pi / 18 * i
            x_c = center[0] + 6 * first_radius * np.cos(angle)
            y_c = center[1] + 6 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    if num_layers >= 7:
        # Seventh layer (42 circles)
        for i in range(42):
            angle = np.pi / 21 * i
            x_c = center[0] + 7 * first_radius * np.cos(angle)
            y_c = center[1] + 7 * first_radius * np.sin(angle)
            circle_centers.append((x_c, y_c))
    
    # Calculate distances from each point to each circle center
    circles = []
    combined_pattern = np.zeros_like(X, dtype=bool)
    
    for c_center in circle_centers:
        dist = np.sqrt((X - c_center[0])**2 + (Y - c_center[1])**2)
        circle = dist <= first_radius
        circles.append(circle)
        combined_pattern = combined_pattern | circle
    
    # Identify all intersection points
    intersection_points = []
    
    # Find pairwise intersections between circles
    for i in range(len(circle_centers)):
        for j in range(i+1, len(circle_centers)):
            c1 = circle_centers[i]
            c2 = circle_centers[j]
            
            # Distance between centers
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            
            # Circles intersect if the distance is less than 2*radius
            if dist < 2*first_radius and dist > 0:
                # Calculate the midpoint
                mid_x = (c1[0] + c2[0]) / 2
                mid_y = (c1[1] + c2[1]) / 2
                
                # Calculate direction vector from c1 to c2
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                # Calculate perpendicular vector
                px, py = -dy, dx
                
                # Calculate distance from midpoint to the intersection
                d = np.sqrt(first_radius**2 - (length/2)**2)
                
                # Calculate intersection points
                point1 = (mid_x + px * d, mid_y + py * d)
                point2 = (mid_x - px * d, mid_y - py * d)
                
                intersection_points.append(point1)
                intersection_points.append(point2)
    
    # Find key sacred points (where multiple circles intersect)
    intersection_counts = np.zeros_like(X)
    for circle in circles:
        intersection_counts = intersection_counts + circle.astype(int)
    
    # Points where 3 or more circles intersect are considered sacred points
    sacred_points = []
    for i in range(resolution):
        for j in range(resolution):
            if intersection_counts[i, j] >= 3:
                sacred_points.append((x[j], y[i], intersection_counts[i, j]))
    
    # Group nearby sacred points (they might be the same point due to grid resolution)
    grouped_sacred_points = []
    for point in sacred_points:
        # Check if this point is close to any existing grouped point
        found = False
        threshold = first_radius * 0.1
        for i, group in enumerate(grouped_sacred_points):
            if np.sqrt((point[0] - group[0][0])**2 + (point[1] - group[0][1])**2) < threshold:
                grouped_sacred_points[i].append(point)
                found = True
                break
        
        if not found:
            grouped_sacred_points.append([point])
    
    # Calculate average position for each group
    sacred_points = []
    for group in grouped_sacred_points:
        x_avg = sum(p[0] for p in group) / len(group)
        y_avg = sum(p[1] for p in group) / len(group)
        count_avg = sum(p[2] for p in group) / len(group)
        sacred_points.append((x_avg, y_avg, count_avg))
    
    return {
        'pattern': combined_pattern,
        'circles': circles,
        'circle_centers': circle_centers,
        'radius': first_radius,
        'num_layers': num_layers,
        'intersection_points': intersection_points,
        'sacred_points': sacred_points,
        'grid_x': X,
        'grid_y': Y,
        'intersection_counts': intersection_counts
    }

def generate_flower_of_life_3d(center: Tuple[float, float, float],
                              radius: float,
                              num_layers: int = 2,
                              resolution: int = 30) -> Dict[str, Any]:
    """
    Generate a 3D Flower of Life pattern (using spheres instead of circles).
    
    Args:
        center: The (x, y, z) center of the pattern
        radius: The radius of each sphere
        num_layers: Number of layers around the central sphere
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and calculated properties
    """
    # Validate input
    if num_layers < 1 or num_layers > 4:  # Limit to 4 layers in 3D for computational reasons
        raise ValueError("Number of layers must be between 1 and 4")
    
    # Create a grid for calculating the pattern
    x_min = center[0] - radius * (2 * num_layers + 1)
    x_max = center[0] + radius * (2 * num_layers + 1)
    y_min = center[1] - radius * (2 * num_layers + 1)
    y_max = center[1] + radius * (2 * num_layers + 1)
    z_min = center[2] - radius * (2 * num_layers + 1)
    z_max = center[2] + radius * (2 * num_layers + 1)
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # First sphere at the center
    sphere_centers = [center]
    
    # Calculate centers for the first layer (12 spheres in 3D fibonacci arrangement)
    # Using fibonacci spiral distribution on a sphere
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(12):
        theta = 2 * np.pi * i / phi
        phi_angle = np.arccos(1 - 2 * (i + 0.5) / 12)
        
        x_c = center[0] + radius * np.sin(phi_angle) * np.cos(theta)
        y_c = center[1] + radius * np.sin(phi_angle) * np.sin(theta)
        z_c = center[2] + radius * np.cos(phi_angle)
        
        sphere_centers.append((x_c, y_c, z_c))
    
    # Add additional layers if requested
    if num_layers >= 2:
        # Second layer (42 spheres)
        for i in range(42):
            theta = 2 * np.pi * i / phi
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / 42)
            
            x_c = center[0] + 2 * radius * np.sin(phi_angle) * np.cos(theta)
            y_c = center[1] + 2 * radius * np.sin(phi_angle) * np.sin(theta)
            z_c = center[2] + 2 * radius * np.cos(phi_angle)
            
            sphere_centers.append((x_c, y_c, z_c))
    
    if num_layers >= 3:
        # Third layer (92 spheres)
        for i in range(92):
            theta = 2 * np.pi * i / phi
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / 92)
            
            x_c = center[0] + 3 * radius * np.sin(phi_angle) * np.cos(theta)
            y_c = center[1] + 3 * radius * np.sin(phi_angle) * np.sin(theta)
            z_c = center[2] + 3 * radius * np.cos(phi_angle)
            
            sphere_centers.append((x_c, y_c, z_c))
    
    if num_layers >= 4:
        # Fourth layer (162 spheres)
        for i in range(162):
            theta = 2 * np.pi * i / phi
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / 162)
            
            x_c = center[0] + 4 * radius * np.sin(phi_angle) * np.cos(theta)
            y_c = center[1] + 4 * radius * np.sin(phi_angle) * np.sin(theta)
            z_c = center[2] + 4 * radius * np.cos(phi_angle)
            
            sphere_centers.append((x_c, y_c, z_c))
    
    # Calculate distances from each point to each sphere center
    spheres = []
    combined_pattern = np.zeros_like(X, dtype=bool)
    
    for s_center in sphere_centers:
        dist = np.sqrt((X - s_center[0])**2 + (Y - s_center[1])**2 + (Z - s_center[2])**2)
        sphere = dist <= radius
        spheres.append(sphere)
        combined_pattern = combined_pattern | sphere
    
    # Calculate intersection counts
    intersection_counts = np.zeros_like(X)
    for sphere in spheres:
        intersection_counts = intersection_counts + sphere.astype(int)
    
    return {
        'pattern': combined_pattern,
        'spheres': spheres,
        'sphere_centers': sphere_centers,
        'radius': radius,
        'num_layers': num_layers,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z,
        'intersection_counts': intersection_counts
    }

def calculate_flower_sacred_ratios(flower_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the sacred ratios and proportions in the Flower of Life.
    
    Args:
        flower_data: Dictionary containing Flower of Life geometry data
        
    Returns:
        Dictionary of calculated sacred ratios
    """
    radius = flower_data['radius']
    
    # Key sacred ratios in the Flower of Life
    # Distance between adjacent centers is exactly equal to the radius
    # Distance from center to the outer layer
    max_distance = radius * flower_data['num_layers']
    
    # Vesica piscis ratio within the pattern (height/width)
    vesica_ratio = np.sqrt(3)
    
    # Ratio of the areas of the inner circle to the whole pattern
    inner_area = np.pi * radius**2
    outer_area = np.pi * (max_distance + radius)**2
    area_ratio = inner_area / outer_area
    
    # Golden ratio approximation in the pattern
    phi = (1 + np.sqrt(5)) / 2
    
    # Hexagonal ratio (distance between opposite vertices in a hexagon / side length)
    hex_ratio = 2.0
    
    return {
        'vesica_ratio': vesica_ratio,  # ≈ 1.732, approximates √3
        'area_ratio': area_ratio,
        'golden_ratio': phi,  # ≈ 1.618
        'hexagonal_ratio': hex_ratio,
        'max_radius_ratio': flower_data['num_layers'],
    }

def calculate_flower_energy_distribution(flower_data: Dict[str, Any], 
                                        base_frequency: float = 432.0) -> Dict[str, Any]:
    """
    Calculate the energy distribution within the Flower of Life.
    
    Args:
        flower_data: Dictionary containing Flower of Life geometry data
        base_frequency: The base frequency for energy calculations
        
    Returns:
        Dictionary containing energy distribution data
    """
    # Extract grid and pattern data
    X = flower_data['grid_x']
    Y = flower_data['grid_y']
    
    if 'grid_z' in flower_data:  # 3D case
        Z = flower_data['grid_z']
        center = flower_data['sphere_centers'][0]  # Central sphere
        
        # Calculate distance from each point to the center
        dist_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # Energy is highest at intersection points and decreases with distance from center
        # Use intersection counts for energy weighting
        intersection_counts = flower_data['intersection_counts']
        
        # Create energy distribution
        # Higher energy at intersections and at the center
        # Energy decays with distance from center
        energy = intersection_counts * np.exp(-dist_center / (flower_data['radius'] * flower_data['num_layers']))
        
    else:  # 2D case
        center = flower_data['circle_centers'][0]  # Central circle
        
        # Calculate distance from each point to the center
        dist_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Create wave pattern for each circle
        waves = np.zeros_like(X)
        for c_center in flower_data['circle_centers']:
            dist = np.sqrt((X - c_center[0])**2 + (Y - c_center[1])**2)
            wave = np.sin(2 * np.pi * dist / flower_data['radius'] * (base_frequency / 432.0))
            waves += wave
        
        # Create energy distribution
        # Higher energy at intersections and at the center
        # Energy decays with distance from center and is modulated by wave interference
        intersection_counts = flower_data['intersection_counts']
        energy = intersection_counts * (0.5 + 0.5 * np.sin(waves)**2) * np.exp(-dist_center / (flower_data['radius'] * flower_data['num_layers']))
    
    # Normalize energy to [0, 1]
    energy = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Find points of maximum energy
    if 'grid_z' in flower_data:  # 3D case
        # This would be complex to visualize, so we'll skip detailed analysis
        max_energy_points = []
        max_energy_values = []
    else:  # 2D case
        # Flatten arrays for easier processing
        flat_energy = energy.flatten()
        flat_x = X.flatten()
        flat_y = Y.flatten()
        
        # Find indices of top energy points
        top_indices = np.argsort(flat_energy)[-19:]  # Get top points (matches the 19 in complete Flower of Life)
        
        # Get coordinates of these points
        max_energy_points = [(flat_x[i], flat_y[i]) for i in top_indices]
        max_energy_values = [flat_energy[i] for i in top_indices]
    
    return {
        'energy': energy,
        'max_energy_points': max_energy_points,
        'max_energy_values': max_energy_values,
        'base_frequency': base_frequency
    }

def identify_platonic_templates(flower_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify and extract Platonic solid templates from the Flower of Life pattern.
    
    Args:
        flower_data: Dictionary containing Flower of Life geometry data
        
    Returns:
        Dictionary containing Platonic solid template information
    """
    # This requires the sacred points to be identified
    sacred_points = flower_data.get('sacred_points', [])
    
    # Sort sacred points by their x, y coordinates for consistent indexing
    sorted_points = sorted(sacred_points, key=lambda p: (p[0], p[1]))
    
    # We need at least the full Flower of Life (minimum 3 layers) for this to work
    platonic_templates = {}
    
    if flower_data['num_layers'] >= 3 and len(sorted_points) >= 19:
        # Extract points for tetrahedron (simplest platonic solid)
        # Usually formed by taking specific sacred points
        # This is a simplified approximation - actual implementation would require 
        # geometric calculations to find exact vertices
        tetrahedron_points = sorted_points[:4]
        
        # Cube (hexahedron)
        hexahedron_points = sorted_points[:8]
        
        # Octahedron
        octahedron_points = sorted_points[4:10]
        
        # Dodecahedron (requires full Flower of Life)
        dodecahedron_points = sorted_points[:20] if len(sorted_points) >= 20 else []
        
        # Icosahedron
        icosahedron_points = sorted_points[:12]
        
        platonic_templates = {
            'tetrahedron': {'points': tetrahedron_points, 'faces': 4, 'edges': 6, 'vertices': 4},
            'hexahedron': {'points': hexahedron_points, 'faces': 6, 'edges': 12, 'vertices': 8},
            'octahedron': {'points': octahedron_points, 'faces': 8, 'edges': 12, 'vertices': 6},
            'dodecahedron': {'points': dodecahedron_points, 'faces': 12, 'edges': 30, 'vertices': 20},
            'icosahedron': {'points': icosahedron_points, 'faces': 20, 'edges': 30, 'vertices': 12}
        }
    
    return platonic_templates

def visualize_flower_of_life_2d(flower_data: Dict[str, Any], 
                               show_energy: bool = False,
                               show_sacred_points: bool = False) -> plt.Figure:
    """
    Create a visualization of the 2D Flower of Life pattern.
    
    Args:
        flower_data: Dictionary containing Flower of Life geometry data
        show_energy: Whether to overlay energy distribution
        show_sacred_points: Whether to mark sacred points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    X = flower_data['grid_x']
    Y = flower_data['grid_y']
    
    # Plot each circle
    for i, center in enumerate(flower_data['circle_centers']):
        circle = plt.Circle(center, flower_data['radius'], fill=False, 
                           color='blue', alpha=0.5, linewidth=1)
        ax.add_artist(circle)
    
    if show_energy and 'energy' in flower_data:
        # Show energy distribution as a heatmap
        im = ax.imshow(flower_data['energy'], extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', cmap='viridis', alpha=0.7)
        fig.colorbar(im, ax=ax, label='Energy')
    
    # Mark center
    center = flower_data['circle_centers'][0]
    ax.plot(center[0], center[1], 'ro', markersize=8, label='Center')
    
    # Mark sacred points
    if show_sacred_points and 'sacred_points' in flower_data:
        for point in flower_data['sacred_points']:
            ax.plot(point[0], point[1], 'go', markersize=6)
    
    # Add title and labels
    ax.set_title('Flower of Life Sacred Geometry')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    ax.legend(loc='upper right')
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    return fig

def embed_flower_in_field(field_array: np.ndarray, 
                         center: Tuple[float, float, float],
                         radius: float,
                         num_layers: int = 3,
                         strength: float = 1.0) -> np.ndarray:
    """
    Embed a Flower of Life pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        radius: Radius of each sphere
        num_layers: Number of layers in the pattern
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
    
    # Generate 3D Flower of Life data at lower resolution for embedding
    embed_resolution = min(30, min(field_shape)//4)  # Lower resolution for performance
    flower_3d = generate_flower_of_life_3d(center, radius, num_layers, embed_resolution)
    
    # Get sphere centers
    sphere_centers = flower_3d['sphere_centers']
    
    # Create spherical shells for each center
    combined_pattern = np.zeros_like(field_array, dtype=float)
    
    for s_center in sphere_centers:
        # Calculate distance from each point to sphere center
        dist = np.sqrt((x - s_center[0])**2 + (y - s_center[1])**2 + (z - s_center[2])**2)
        
        # Create a Gaussian shell
        shell_width = radius * 0.1
        shell = np.exp(-((dist - radius) / shell_width)**2)
        
        # Add to combined pattern
        combined_pattern += shell
    
    # Normalize pattern to [0, 1] range
    max_val = np.max(combined_pattern)
    if max_val > 0:
        combined_pattern = combined_pattern / max_val
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + combined_pattern * strength)
    
    # Normalize field after modification
    modified_field = modified_field / np.max(np.abs(modified_field)) if np.max(np.abs(modified_field)) > 0 else modified_field
    
    return modified_field


# Example usage
if __name__ == "__main__":
    # Create a 2D Flower of Life
    center = (100, 100)
    radius = 30
    num_layers = 3
    
    flower_2d = generate_flower_of_life_2d(center, radius, num_layers)
    
    # Calculate energy distribution
    energy_data = calculate_flower_energy_distribution(flower_2d)
    flower_2d.update(energy_data)
    
    # Calculate sacred ratios
    ratios = calculate_flower_sacred_ratios(flower_2d)
    print("Sacred Ratios:")
    for key, value in ratios.items():
        print(f"{key}: {value}")
    
    # Identify Platonic solid templates
    platonic = identify_platonic_templates(flower_2d)
    print("\nPlatonic Templates:")
    for solid, data in platonic.items():
        print(f"{solid}: {data['vertices']} vertices, {data['edges']} edges, {data['faces']} faces")
    
    # Visualize
    fig = visualize_flower_of_life_2d(flower_2d, show_energy=True, show_sacred_points=True)
    plt.show()