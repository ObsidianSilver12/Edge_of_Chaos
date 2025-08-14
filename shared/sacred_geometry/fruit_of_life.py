"""
Fruit of Life Module

This module implements the fruit of life sacred geometry pattern.
The fruit of life consists of 13 circles arranged in a specific pattern,
formed by extending the Flower of Life pattern.

Key functions:
- Generate precise fruit of life geometry
- Calculate sacred ratios within the pattern
- Calculate energy distribution at intersection points
- Generate field embedding for the pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional

def generate_fruit_of_life_2d(center: Tuple[float, float], 
                             radius: float, 
                             resolution: int = 100) -> Dict[str, Any]:
    """
    Generate a 2D fruit of life pattern.
    
    Args:
        center: The (x, y) center of the central circle
        radius: The radius of each circle
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and properties
    """
    # The Fruit of Life consists of 13 circles:
    # 1 central circle
    # 6 circles arranged in a flower pattern around the central circle
    # 6 additional circles to complete the fruit pattern
    
    # Calculate positions of the circles
    # All circles are the same size and adjacent circles touch
    # Distance between centers is 2 * radius
    circle_centers = [center]  # Start with central circle
    
    # Add the inner ring of 6 circles
    inner_ring = []
    for i in range(6):
        angle = i * np.pi / 3  # 60 degree spacing
        x = center[0] + 2 * radius * np.cos(angle)
        y = center[1] + 2 * radius * np.sin(angle)
        inner_ring.append((x, y))
    
    circle_centers.extend(inner_ring)
    
    # Add the outer ring of 6 circles
    # These are positioned between pairs of inner ring circles, but further out
    outer_ring = []
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3  # Offset by 30 degrees
        x = center[0] + 2 * radius * 2 * np.cos(angle)  # Double distance from center
        y = center[1] + 2 * radius * 2 * np.sin(angle)  # Double distance from center
        outer_ring.append((x, y))
    
    circle_centers.extend(outer_ring)
    
    # Create a grid for the pattern
    x_min = min(c[0] for c in circle_centers) - radius
    x_max = max(c[0] for c in circle_centers) + radius
    y_min = min(c[1] for c in circle_centers) - radius
    y_max = max(c[1] for c in circle_centers) + radius
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create individual circles
    circles = []
    for center_point in circle_centers:
        dist = np.sqrt((X - center_point[0])**2 + (Y - center_point[1])**2)
        circle = dist <= radius
        circles.append(circle)
    
    # Combine all circles into one pattern
    combined_pattern = np.zeros((resolution, resolution), dtype=bool)
    for circle in circles:
        combined_pattern = combined_pattern | circle
    
    # Calculate the intersection points between adjacent circles
    # These are the sacred points in the fruit of life
    intersection_points = []
    for i in range(len(circle_centers)):
        for j in range(i + 1, len(circle_centers)):
            c1 = circle_centers[i]
            c2 = circle_centers[j]
            
            # Calculate distance between centers
            distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            
            # If circles intersect (distance < 2*radius but > 0)
            if distance < 2 * radius + 1e-10 and distance > 1e-10:
                # Calculate midpoint between centers
                mid_x = (c1[0] + c2[0]) / 2
                mid_y = (c1[1] + c2[1]) / 2
                
                # Calculate direction vector from c1 to c2
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length, dy/length
                
                # Calculate perpendicular vector
                px, py = -dy, dx
                
                # Calculate distance from midpoint to intersection
                d = np.sqrt(radius**2 - (distance/2)**2)
                
                # Calculate intersection points
                point1 = (mid_x + px * d, mid_y + py * d)
                point2 = (mid_x - px * d, mid_y - py * d)
                
                # Only add if they're not already in the list
                # (within a small tolerance)
                for point in [point1, point2]:
                    if not any(np.sqrt((p[0]-point[0])**2 + (p[1]-point[1])**2) < 1e-10 for p in intersection_points):
                        intersection_points.append(point)
    
    return {
        'pattern': combined_pattern,
        'circles': circles,
        'centers': circle_centers,
        'radius': radius,
        'grid_x': X,
        'grid_y': Y,
        'intersection_points': intersection_points
    }

def generate_fruit_of_life_3d(center: Tuple[float, float, float], 
                            radius: float, 
                            resolution: int = 50) -> Dict[str, Any]:
    """
    Generate a 3D fruit of life pattern.
    
    Args:
        center: The (x, y, z) center of the central sphere
        radius: The radius of each sphere
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the pattern geometry and properties
    """
    # The 3D Fruit of Life consists of 13 spheres - central sphere, inner ring of 6, outer ring of 6
    
    # Calculate positions of the spheres
    # Central sphere
    sphere_centers = [center]
    
    # Inner ring of 6 spheres in the xy-plane
    inner_ring = []
    for i in range(6):
        angle = i * np.pi / 3  # 60 degree spacing
        x = center[0] + 2 * radius * np.cos(angle)
        y = center[1] + 2 * radius * np.sin(angle)
        z = center[2]
        inner_ring.append((x, y, z))
    
    sphere_centers.extend(inner_ring)
    
    # Outer ring of 6 spheres in the xy-plane
    outer_ring = []
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3  # Offset by 30 degrees
        x = center[0] + 2 * radius * 2 * np.cos(angle)  # Double distance from center
        y = center[1] + 2 * radius * 2 * np.sin(angle)  # Double distance from center
        z = center[2]
        outer_ring.append((x, y, z))
    
    sphere_centers.extend(outer_ring)
    
    # Create a grid for the 3D pattern
    x_min = min(c[0] for c in sphere_centers) - radius
    x_max = max(c[0] for c in sphere_centers) + radius
    y_min = min(c[1] for c in sphere_centers) - radius
    y_max = max(c[1] for c in sphere_centers) + radius
    z_min = min(c[2] for c in sphere_centers) - radius
    z_max = max(c[2] for c in sphere_centers) + radius
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create individual spheres (as masks)
    spheres = []
    for sphere_center in sphere_centers:
        dist = np.sqrt((X - sphere_center[0])**2 + 
                       (Y - sphere_center[1])**2 + 
                       (Z - sphere_center[2])**2)
        sphere = dist <= radius
        spheres.append(sphere)
    
    # Combine all spheres into one pattern
    combined_pattern = np.zeros((resolution, resolution, resolution), dtype=bool)
    for sphere in spheres:
        combined_pattern = combined_pattern | sphere
    
    # Calculate the intersection circles between adjacent spheres
    intersection_circles = []
    for i in range(len(sphere_centers)):
        for j in range(i + 1, len(sphere_centers)):
            c1 = sphere_centers[i]
            c2 = sphere_centers[j]
            
            # Calculate distance between centers
            distance = np.sqrt(sum((c1[k] - c2[k])**2 for k in range(3)))
            
            # If spheres intersect (distance < 2*radius but > 0)
            if distance < 2 * radius + 1e-10 and distance > 1e-10:
                # Calculate midpoint between centers
                midpoint = [(c1[k] + c2[k]) / 2 for k in range(3)]
                
                # Calculate direction vector from c1 to c2
                direction = [c2[k] - c1[k] for k in range(3)]
                length = np.sqrt(sum(d**2 for d in direction))
                direction = [d/length for d in direction]
                
                # Calculate radius of intersection circle
                circle_radius = np.sqrt(radius**2 - (length/2)**2)
                
                intersection_circles.append({
                    'center': midpoint,
                    'radius': circle_radius,
                    'normal': direction
                })
    
    return {
        'pattern': combined_pattern,
        'spheres': spheres,
        'centers': sphere_centers,
        'radius': radius,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z,
        'intersection_circles': intersection_circles
    }

def calculate_fruit_sacred_ratios(fruit_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the sacred ratios and proportions in the fruit of life.
    
    Args:
        fruit_data: Dictionary containing fruit of life geometry data
        
    Returns:
        Dictionary of calculated sacred ratios
    """
    radius = fruit_data['radius']
    centers = fruit_data['centers']
    
    # Calculate the distance between center and inner ring circles
    # (should be 2 * radius)
    inner_ring_distances = []
    for i in range(1, 7):  # Indices 1-6 are inner ring
        center_point = centers[i]
        center_distance = np.sqrt(sum((center_point[j] - centers[0][j])**2 
                                    for j in range(len(centers[0]))))
        inner_ring_distances.append(center_distance)
    
    avg_inner_distance = np.mean(inner_ring_distances)
    inner_distance_ratio = avg_inner_distance / radius  # Should be close to 2
    
    # Calculate the distance between center and outer ring circles
    # (should be 4 * radius)
    outer_ring_distances = []
    for i in range(7, 13):  # Indices 7-12 are outer ring
        center_point = centers[i]
        center_distance = np.sqrt(sum((center_point[j] - centers[0][j])**2 
                                    for j in range(len(centers[0]))))
        outer_ring_distances.append(center_distance)
    
    avg_outer_distance = np.mean(outer_ring_distances)
    outer_distance_ratio = avg_outer_distance / radius  # Should be close to 4
    
    # Calculate the ratio between outer and inner ring distances
    # (should be close to 2)
    ring_ratio = avg_outer_distance / avg_inner_distance
    
    # Calculate the overall width and height of the pattern
    if len(centers[0]) == 2:  # 2D
        width = max(c[0] for c in centers) - min(c[0] for c in centers) + 2 * radius
        height = max(c[1] for c in centers) - min(c[1] for c in centers) + 2 * radius
    else:  # 3D
        width = max(c[0] for c in centers) - min(c[0] for c in centers) + 2 * radius
        height = max(c[1] for c in centers) - min(c[1] for c in centers) + 2 * radius
        depth = max(c[2] for c in centers) - min(c[2] for c in centers) + 2 * radius
    
    # Ratios related to the golden mean phi
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate phi-related ratio in the fruit of life pattern
    # The relationship between inner and outer rings approximates phi in certain measurements
    phi_approximation = outer_distance_ratio / (inner_distance_ratio * phi)
    
    # Ratio of overall structure height to width
    if len(centers[0]) == 2:  # 2D
        aspect_ratio = height / width
        return {
            'inner_distance_ratio': inner_distance_ratio,
            'outer_distance_ratio': outer_distance_ratio,
            'ring_ratio': ring_ratio,
            'aspect_ratio': aspect_ratio,
            'phi_approximation': phi_approximation,
            'width': width,
            'height': height
        }
    else:  # 3D
        aspect_ratio_xy = height / width
        aspect_ratio_z = depth / width
        return {
            'inner_distance_ratio': inner_distance_ratio,
            'outer_distance_ratio': outer_distance_ratio,
            'ring_ratio': ring_ratio,
            'aspect_ratio_xy': aspect_ratio_xy,
            'aspect_ratio_z': aspect_ratio_z,
            'phi_approximation': phi_approximation,
            'width': width,
            'height': height,
            'depth': depth
        }

def calculate_fruit_energy_distribution(fruit_data: Dict[str, Any], 
                                     base_frequency: float = 432.0) -> Dict[str, Any]:
    """
    Calculate the energy distribution within the fruit of life.
    
    Args:
        fruit_data: Dictionary containing fruit of life geometry data
        base_frequency: The base frequency for energy calculations
        
    Returns:
        Dictionary containing energy distribution data
    """
    radius = fruit_data['radius']
    centers = fruit_data['centers']
    
    if 'grid_z' in fruit_data:  # 3D case
        X = fruit_data['grid_x']
        Y = fruit_data['grid_y']
        Z = fruit_data['grid_z']
        
        # Calculate distance from each point to each sphere center
        energy = np.zeros_like(X, dtype=float)
        
        for center_point in centers:
            dist = np.sqrt((X - center_point[0])**2 + 
                          (Y - center_point[1])**2 + 
                          (Z - center_point[2])**2)
            
            # Energy is highest at the sphere boundaries
            # Using a Gaussian shell around each sphere
            shell_energy = np.exp(-((dist - radius) / (radius * 0.05))**2)
            energy += shell_energy
        
        # Enhancement at intersection circles
        for circle in fruit_data['intersection_circles']:
            circle_center = circle['center']
            circle_radius = circle['radius']
            circle_normal = circle['normal']
            
            # Calculate vectors from circle center to each grid point
            vectors = np.stack([(X - circle_center[0]), 
                               (Y - circle_center[1]), 
                               (Z - circle_center[2])], axis=-1)
            
            # Calculate component along normal (distance from the circle's plane)
            normal_vec = np.array(circle_normal)
            normal_dist = np.abs(np.sum(vectors * normal_vec, axis=-1))
            
            # Calculate distance in circle plane
            plane_dist = np.sqrt(np.sum(vectors**2, axis=-1) - normal_dist**2)
            
            # Energy is highest near the circle itself
            circle_energy = np.exp(-((plane_dist - circle_radius) / (radius * 0.05))**2) * \
                           np.exp(-(normal_dist / (radius * 0.05))**2)
            
            energy += circle_energy * 2.0  # Double weight for intersections
    
    else:  # 2D case
        X = fruit_data['grid_x']
        Y = fruit_data['grid_y']
        
        # Calculate distance from each point to each circle center
        energy = np.zeros_like(X, dtype=float)
        
        for center_point in centers:
            dist = np.sqrt((X - center_point[0])**2 + (Y - center_point[1])**2)
            
            # Energy is highest at the circle boundaries
            # Create a thin shell of energy at the boundary of each circle
            shell_energy = np.exp(-((dist - radius) / (radius * 0.05))**2)
            energy += shell_energy
        
        # Enhancement at intersection points
        intersection_points = fruit_data['intersection_points']
        for point in intersection_points:
            point_energy = np.exp(-((X - point[0])**2 + (Y - point[1])**2) / (radius * 0.05)**2)
            energy += point_energy * 2.0  # Double weight for intersections
    
    # Normalize energy to [0, 1]
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Find points of maximum energy
    # These are sacred points in the fruit of life
    if 'grid_z' in fruit_data:  # 3D case
        # Flatten arrays for easier processing
        flat_energy = energy.flatten()
        flat_x = X.flatten()
        flat_y = Y.flatten()
        flat_z = Z.flatten()
        
        # Find indices of top energy points
        top_indices = np.argsort(flat_energy)[-20:]  # Get 20 highest points
        
        # Get coordinates of these points
        max_energy_points = [(flat_x[i], flat_y[i], flat_z[i]) for i in top_indices]
        max_energy_values = [flat_energy[i] for i in top_indices]
    else:  # 2D case
        # Flatten arrays for easier processing
        flat_energy = energy.flatten()
        flat_x = X.flatten()
        flat_y = Y.flatten()
        
        # Find indices of top energy points
        top_indices = np.argsort(flat_energy)[-20:]  # Get 20 highest points
        
        # Get coordinates of these points
        max_energy_points = [(flat_x[i], flat_y[i]) for i in top_indices]
        max_energy_values = [flat_energy[i] for i in top_indices]
    
    return {
        'energy': energy,
        'max_energy_points': max_energy_points,
        'max_energy_values': max_energy_values,
        'base_frequency': base_frequency
    }

def embed_fruit_in_field(field_array: np.ndarray, 
                        center: Tuple[float, float, float],
                        radius: float,
                        strength: float = 1.0) -> np.ndarray:
    """
    Embed a fruit of life pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        radius: Radius of each sphere in the pattern
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
    
    # Generate 3D fruit of life data (just to get the centers)
    fruit_data = generate_fruit_of_life_3d(center, radius, resolution=20)
    centers = fruit_data['centers']
    
    # Initialize energy field
    energy = np.zeros(field_shape)
    
    # Calculate energy from each sphere
    for sphere_center in centers:
        dist = np.sqrt((x - sphere_center[0])**2 + 
                      (y - sphere_center[1])**2 + 
                      (z - sphere_center[2])**2)
        
        # Energy is highest at the sphere boundaries
        shell_energy = np.exp(-((dist - radius) / (radius * 0.1))**2)
        energy += shell_energy
    
    # Calculate the intersection circles between adjacent spheres
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1 = centers[i]
            c2 = centers[j]
            
            # Calculate distance between centers
            distance = np.sqrt(sum((c1[k] - c2[k])**2 for k in range(3)))
            
            # If spheres intersect (distance < 2*radius but > 0)
            if distance < 2 * radius + 1e-10 and distance > 1e-10:
                # Calculate midpoint between centers
                midpoint = [(c1[k] + c2[k]) / 2 for k in range(3)]
                
                # Calculate direction vector from c1 to c2
                direction = [c2[k] - c1[k] for k in range(3)]
                length = np.sqrt(sum(d**2 for d in direction))
                direction = [d/length for d in direction]
                
                # Calculate radius of intersection circle
                circle_radius = np.sqrt(radius**2 - (distance/2)**2)
                
                # Calculate vectors from circle center to each grid point
                vectors = np.stack([(x - midpoint[0]), 
                                   (y - midpoint[1]), 
                                   (z - midpoint[2])], axis=-1)
                
                # Calculate component along normal (distance from the circle's plane)
                normal_vec = np.array(direction)
                normal_dist = np.abs(np.sum(vectors * normal_vec, axis=-1))
                
                # Calculate distance in circle plane
                plane_dist = np.sqrt(np.sum(vectors**2, axis=-1) - normal_dist**2)
                
                # Energy is highest near the circle itself
                circle_energy = np.exp(-((plane_dist - circle_radius) / (radius * 0.05))**2) * \
                               np.exp(-(normal_dist / (radius * 0.05))**2)
                
                energy += circle_energy * 2.0  # Double weight for intersections
    
    # Normalize energy to [0, 1]
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + energy * strength)
    
    # Normalize field after modification
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def visualize_fruit_of_life_2d(fruit_data: Dict[str, Any], 
                             show_energy: bool = False,
                             show_sacred_points: bool = False) -> plt.Figure:
    """
    Create a visualization of the 2D fruit of life pattern.
    
    Args:
        fruit_data: Dictionary containing fruit of life geometry data
        show_energy: Whether to overlay energy distribution
        show_sacred_points: Whether to mark sacred points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    X = fruit_data['grid_x']
    Y = fruit_data['grid_y']
    circles = fruit_data['circles']
    centers = fruit_data['centers']
    radius = fruit_data['radius']
    
    # Plot each circle as a contour
    for i, circle in enumerate(circles):
        if i == 0:
            color = 'blue'  # Central circle
        elif i <= 6:
            color = 'green'  # Inner ring
        else:
            color = 'red'  # Outer ring
        ax.contour(X, Y, circle.astype(int), levels=[0.5], colors=[color], alpha=0.5)
    
    # Plot the combined pattern
    if show_energy and 'energy' in fruit_data:
        # Show energy distribution as a heatmap
        im = ax.imshow(fruit_data['energy'], extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', cmap='viridis', alpha=0.7)
        fig.colorbar(im, ax=ax, label='Energy')
    else:
        # Just show the pattern region
        ax.contourf(X, Y, fruit_data['pattern'].astype(int), levels=[0.5, 1.5], 
                   colors=['purple'], alpha=0.3)
    
    # Mark centers of circles
    for i, center_point in enumerate(centers):
        if i == 0:
            color = 'blue'  # Central circle
        elif i <= 6:
            color = 'green'  # Inner ring
        else:
            color = 'red'  # Outer ring
        ax.plot(center_point[0], center_point[1], 'o', color=color, markersize=6)
    
    # Mark intersection points
    if 'intersection_points' in fruit_data:
        for point in fruit_data['intersection_points']:
            ax.plot(point[0], point[1], 'go', markersize=4, alpha=0.7)
    
    # Mark sacred points (high energy points)
    if show_sacred_points and 'max_energy_points' in fruit_data:
        for point in fruit_data['max_energy_points']:
            if len(point) == 2:  # 2D points
                ax.plot(point[0], point[1], 'yo', markersize=6)
    
    # Add title and labels
    ax.set_title('Fruit of Life Sacred Geometry')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    return fig

def visualize_fruit_of_life_3d(fruit_data: Dict[str, Any], 
                              show_energy: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the fruit of life pattern.
    
    Args:
        fruit_data: Dictionary containing fruit of life geometry data
        show_energy: Whether to show energy distribution
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract centers and radii
    centers = fruit_data['centers']
    radius = fruit_data['radius']
    
    # Plot spheres as wireframes
    for i, center_point in enumerate(centers):
        # Create a wireframe sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        
        x = center_point[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center_point[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center_point[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Use different colors for different groups of spheres
        if i == 0:
            color = 'blue'  # Central sphere
        elif i <= 6:
            color = 'green'  # Inner ring
        else:
            color = 'red'  # Outer ring
            
        ax.plot_surface(x, y, z, color=color, alpha=0.1, linewidth=0.5, edgecolors=color)
    
    # Plot intersection circles
    if 'intersection_circles' in fruit_data:
        for circle in fruit_data['intersection_circles']:
            circle_center = circle['center']
            circle_radius = circle['radius']
            circle_normal = circle['normal']
            
            # Create a basis for the circle plane
            # First basis vector is the normal
            e1 = np.array(circle_normal)
            
            # Create a second basis vector perpendicular to the normal
            if np.abs(e1[0]) > np.abs(e1[1]):
                e2 = np.array([-e1[2], 0, e1[0]])
            else:
                e2 = np.array([0, -e1[2], e1[1]])
            e2 = e2 / np.linalg.norm(e2)
            
            # Third basis vector from cross product
            e3 = np.cross(e1, e2)
            
            # Create circle points
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_x = circle_center[0] + circle_radius * (np.cos(theta) * e2[0] + np.sin(theta) * e3[0])
            circle_y = circle_center[1] + circle_radius * (np.cos(theta) * e2[1] + np.sin(theta) * e3[1])
            circle_z = circle_center[2] + circle_radius * (np.cos(theta) * e2[2] + np.sin(theta) * e3[2])
            
            ax.plot(circle_x, circle_y, circle_z, 'g-', linewidth=1, alpha=0.7)
    
    # Plot energy distribution if requested
    if show_energy and 'energy' in fruit_data:
        # We can't easily show a 3D energy field in a 3D plot
        # Instead, show high energy points
        if 'max_energy_points' in fruit_data:
            energy_points = fruit_data['max_energy_points']
            x = [p[0] for p in energy_points]
            y = [p[1] for p in energy_points]
            z = [p[2] for p in energy_points]
            ax.scatter(x, y, z, color='yellow', s=30, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Fruit of Life')
    
    # Set equal aspect ratio
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    return fig

def get_base_glyph_elements(center: Tuple[float, float], radius: float) -> Dict[str, Any]:
    """
    Returns the geometric elements (circles) for a simple line art
    representation of the Fruit of Life.
    """
    fruit_data = generate_fruit_of_life_2d(center, radius, resolution=10) # Low res for data only
    
    circles_data = []
    for center_pos_frl in fruit_data['centers']: # Renamed center_pos
        circles_data.append({'center': tuple(center_pos_frl), 'radius': fruit_data['radius']})

    all_x_frl = [c[0] for c in fruit_data['centers']]; all_y_frl = [c[1] for c in fruit_data['centers']] # Renamed
    padding_frl = fruit_data['radius'] * 0.2 # Renamed

    return {
        'circles': circles_data,
        'projection_type': '2d',
        'bounding_box': {
            'xmin': float(min(all_x_frl) - fruit_data['radius'] - padding_frl), 
            'xmax': float(max(all_x_frl) + fruit_data['radius'] + padding_frl),
            'ymin': float(min(all_y_frl) - fruit_data['radius'] - padding_frl), 
            'ymax': float(max(all_y_frl) + fruit_data['radius'] + padding_frl),
        }
    }
# Example usage
if __name__ == "__main__":
    # Create a 2D fruit of life
    center_2d = (0, 0)
    radius_2d = 1.0
    
    fruit_2d = generate_fruit_of_life_2d(center_2d, radius_2d)
    
    # Calculate energy distribution
    energy_data_2d = calculate_fruit_energy_distribution(fruit_2d)
    fruit_2d.update(energy_data_2d)
    
    # Calculate sacred ratios
    ratios_2d = calculate_fruit_sacred_ratios(fruit_2d)
    print("2D Sacred Ratios:")
    for key, value in ratios_2d.items():
        print(f"{key}: {value}")
    
    # Visualize 2D
    fig_2d = visualize_fruit_of_life_2d(fruit_2d, show_energy=True, show_sacred_points=True)
    plt.figure(fig_2d.number)
    plt.tight_layout()
    plt.show()
    
    # Create a 3D fruit of life
    center_3d = (0, 0, 0)
    radius_3d = 1.0
    
    fruit_3d = generate_fruit_of_life_3d(center_3d, radius_3d)
    
    # Calculate energy distribution
    energy_data_3d = calculate_fruit_energy_distribution(fruit_3d)
    fruit_3d.update(energy_data_3d)
    
    # Calculate sacred ratios
    ratios_3d = calculate_fruit_sacred_ratios(fruit_3d)
    print("\n3D Sacred Ratios:")
    for key, value in ratios_3d.items():
        print(f"{key}: {value}")
    
    # Visualize 3D
    fig_3d = visualize_fruit_of_life_3d(fruit_3d, show_energy=True)
    plt.figure(fig_3d.number)
    plt.tight_layout()
    plt.show()