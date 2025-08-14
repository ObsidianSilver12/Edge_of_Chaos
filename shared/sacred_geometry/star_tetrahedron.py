"""
64 Star Tetrahedron Module

This module implements the 64 star tetrahedron sacred geometry pattern.
The 64 star tetrahedron consists of 64 tetrahedra arranged in a specific pattern,
often represented as a 3D structure of nested tetrahedra with specific geometric relationships.

Key functions:
- Generate precise 64 star tetrahedron geometry
- Calculate sacred ratios and energy points
- Calculate energy distribution within the pattern
- Generate field embedding for the pattern
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Any, Optional


def generate_single_tetrahedron(center: Tuple[float, float, float], 
                              size: float, 
                              orientation: int = 0) -> Dict[str, Any]:
    """
    Generate a single tetrahedron with the given center, size, and orientation.
    
    Args:
        center: The (x, y, z) center of the tetrahedron
        size: The size (edge length) of the tetrahedron
        orientation: 0 for upright, 1 for inverted
        
    Returns:
        Dictionary containing the tetrahedron vertices and properties
    """
    # Regular tetrahedron vertices relative to center
    # The height of a regular tetrahedron with edge length 1 is sqrt(6)/3
    height = np.sqrt(6) / 3 * size
    
    # Base vertices form an equilateral triangle
    base_radius = size / np.sqrt(3)
    
    if orientation == 0:  # Upright tetrahedron
        vertices = [
            # Base vertices
            [center[0] + base_radius * np.cos(np.pi/6), 
             center[1] + base_radius * np.sin(np.pi/6), 
             center[2] - height/2],
            [center[0] + base_radius * np.cos(np.pi/6 + 2*np.pi/3), 
             center[1] + base_radius * np.sin(np.pi/6 + 2*np.pi/3), 
             center[2] - height/2],
            [center[0] + base_radius * np.cos(np.pi/6 + 4*np.pi/3), 
             center[1] + base_radius * np.sin(np.pi/6 + 4*np.pi/3), 
             center[2] - height/2],
            # Apex
            [center[0], center[1], center[2] + height/2]
        ]
    else:  # Inverted tetrahedron
        vertices = [
            # Base vertices
            [center[0] + base_radius * np.cos(np.pi/6), 
             center[1] + base_radius * np.sin(np.pi/6), 
             center[2] + height/2],
            [center[0] + base_radius * np.cos(np.pi/6 + 2*np.pi/3), 
             center[1] + base_radius * np.sin(np.pi/6 + 2*np.pi/3), 
             center[2] + height/2],
            [center[0] + base_radius * np.cos(np.pi/6 + 4*np.pi/3), 
             center[1] + base_radius * np.sin(np.pi/6 + 4*np.pi/3), 
             center[2] + height/2],
            # Apex (downward)
            [center[0], center[1], center[2] - height/2]
        ]
    
    # Define edges as pairs of vertex indices
    edges = [
        (0, 1), (1, 2), (2, 0),  # Base edges
        (0, 3), (1, 3), (2, 3)   # Edges to apex
    ]
    
    # Define faces as triplets of vertex indices
    faces = [
        (0, 1, 2),  # Base
        (0, 1, 3), (1, 2, 3), (2, 0, 3)  # Side faces
    ]
    
    return {
        'vertices': vertices,
        'edges': edges,
        'faces': faces,
        'center': center,
        'size': size,
        'orientation': orientation,
        'height': height
    }

def generate_star_tetrahedron(center: Tuple[float, float, float], 
                             size: float) -> Dict[str, Any]:
    """
    Generate a star tetrahedron (Merkaba) consisting of two interlocking tetrahedra.
    
    Args:
        center: The (x, y, z) center of the star tetrahedron
        size: The size (edge length) of each tetrahedron
        
    Returns:
        Dictionary containing the star tetrahedron properties
    """
    # Create upright tetrahedron
    upright = generate_single_tetrahedron(center, size, 0)
    
    # Create inverted tetrahedron
    inverted = generate_single_tetrahedron(center, size, 1)
    
    # Combine to form star tetrahedron
    return {
        'upright': upright,
        'inverted': inverted,
        'center': center,
        'size': size
    }

def generate_64_star_tetrahedron(center: Tuple[float, float, float], 
                               size: float) -> Dict[str, Any]:
    """
    Generate a 64 star tetrahedron pattern.
    
    Args:
        center: The (x, y, z) center of the overall structure
        size: The size (edge length) of the largest tetrahedron
        
    Returns:
        Dictionary containing the 64 star tetrahedron properties
    """
    # The 64-star tetrahedron consists of:
    # 1. Central star tetrahedron (one pair)
    # 2. 6 star tetrahedra on the faces (six pairs)
    # 3. 8 star tetrahedra on the vertices (eight pairs)
    # 4. 12 star tetrahedra on the edges (twelve pairs)
    # Total: 27 star tetrahedra = 54 individual tetrahedra
    # Plus 10 additional tetrahedra to reach 64
    
    # Store all generated tetrahedra
    all_tetrahedra = []
    
    # 1. Create central star tetrahedron
    central_star = generate_star_tetrahedron(center, size)
    all_tetrahedra.append(central_star)
    
    # Calculate positions for surrounding star tetrahedra
    # Size reduction factor for nested structures
    size_factor = 0.5
    
    # 2. Create 6 star tetrahedra on the faces
    # Positions are along the primary axes
    face_positions = [
        (center[0] + size, center[1], center[2]),  # +X
        (center[0] - size, center[1], center[2]),  # -X
        (center[0], center[1] + size, center[2]),  # +Y
        (center[0], center[1] - size, center[2]),  # -Y
        (center[0], center[1], center[2] + size),  # +Z
        (center[0], center[1], center[2] - size)   # -Z
    ]
    
    for pos in face_positions:
        star = generate_star_tetrahedron(pos, size * size_factor)
        all_tetrahedra.append(star)
    
    # 3. Create 8 star tetrahedra on the vertices
    # Positions are at corners of a cube
    vertex_positions = list(itertools.product(*[
        [center[0] + size, center[0] - size],
        [center[1] + size, center[1] - size],
        [center[2] + size, center[2] - size]
    ]))
    
    for pos in vertex_positions:
        star = generate_star_tetrahedron(pos, size * size_factor * size_factor)
        all_tetrahedra.append(star)
    
    # 4. Create 12 star tetrahedra on the edges
    # Positions are at midpoints of cube edges
    edge_positions = [
        # Edges along X axis at different Y,Z
        (center[0], center[1] + size, center[2] + size),
        (center[0], center[1] + size, center[2] - size),
        (center[0], center[1] - size, center[2] + size),
        (center[0], center[1] - size, center[2] - size),
        # Edges along Y axis at different X,Z
        (center[0] + size, center[1], center[2] + size),
        (center[0] + size, center[1], center[2] - size),
        (center[0] - size, center[1], center[2] + size),
        (center[0] - size, center[1], center[2] - size),
        # Edges along Z axis at different X,Y
        (center[0] + size, center[1] + size, center[2]),
        (center[0] + size, center[1] - size, center[2]),
        (center[0] - size, center[1] + size, center[2]),
        (center[0] - size, center[1] - size, center[2])
    ]
    
    for pos in edge_positions:
        star = generate_star_tetrahedron(pos, size * size_factor * size_factor * size_factor)
        all_tetrahedra.append(star)
    
    # 5. Add additional tetrahedra to reach 64
    # These are positioned at specific harmonic locations
    additional_positions = [
        (center[0] + size * 1.5, center[1], center[2]),
        (center[0] - size * 1.5, center[1], center[2]),
        (center[0], center[1] + size * 1.5, center[2]),
        (center[0], center[1] - size * 1.5, center[2]),
        (center[0], center[1], center[2] + size * 1.5),
        (center[0], center[1], center[2] - size * 1.5),
        (center[0] + size * 0.75, center[1] + size * 0.75, center[2] + size * 0.75),
        (center[0] - size * 0.75, center[1] - size * 0.75, center[2] - size * 0.75),
        (center[0] + size * 0.75, center[1] - size * 0.75, center[2] + size * 0.75),
        (center[0] - size * 0.75, center[1] + size * 0.75, center[2] - size * 0.75)
    ]
    
    for i, pos in enumerate(additional_positions):
        # Alternate between upright and inverted for additional tetrahedra
        single_tetra = generate_single_tetrahedron(pos, size * size_factor * size_factor, i % 2)
        all_tetrahedra.append({'single': single_tetra})
    
    # Calculate energy points
    energy_points = calculate_64_star_energy_points(center, size)
    
    return {
        'tetrahedra': all_tetrahedra,
        'center': center,
        'size': size,
        'energy_points': energy_points,
        'total_tetrahedra': 64  # Verify this sums correctly
    }

def calculate_64_star_energy_points(center: Tuple[float, float, float], 
                                  size: float) -> List[Tuple[float, float, float]]:
    """
    Calculate the energy points within the 64 star tetrahedron structure.
    These points occur at key intersections and resonance points.
    
    Args:
        center: The (x, y, z) center of the structure
        size: The size (edge length) of the largest tetrahedron
        
    Returns:
        List of (x, y, z) coordinates of energy points
    """
    energy_points = []
    
    # Central point
    energy_points.append(center)
    
    # Add points at star tetrahedron intersections
    # These are harmonic points based on the phi ratio (golden ratio ≈ 1.618)
    phi = (1 + np.sqrt(5)) / 2
    
    # Points along axes
    for axis in range(3):
        for direction in [-1, 1]:
            point = list(center)
            point[axis] += direction * size / phi
            energy_points.append(tuple(point))
    
    # Points at phi-ratio positions along vertex directions
    for x_dir in [-1, 1]:
        for y_dir in [-1, 1]:
            for z_dir in [-1, 1]:
                point = (
                    center[0] + x_dir * size / (phi * phi),
                    center[1] + y_dir * size / (phi * phi),
                    center[2] + z_dir * size / (phi * phi)
                )
                energy_points.append(point)
    
    # Points at the centers of each star tetrahedron
    # (We already have the center point)
    
    # Add points at main axes (already included in first set)
    
    # Add points at phi^2 positions along main axes
    for axis in range(3):
        for direction in [-1, 1]:
            point = list(center)
            point[axis] += direction * size / (phi * phi)
            energy_points.append(tuple(point))
    
    # Remove duplicates
    unique_energy_points = []
    for point in energy_points:
        if point not in unique_energy_points:
            unique_energy_points.append(point)
    
    return unique_energy_points

def calculate_64_star_sacred_ratios(star_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the sacred ratios and proportions in the 64 star tetrahedron.
    
    Args:
        star_data: Dictionary containing 64 star tetrahedron data
        
    Returns:
        Dictionary of calculated sacred ratios
    """
    size = star_data['size']
    
    # Calculate key ratios in the 64 star tetrahedron
    # The structure embodies several sacred proportions
    
    # Calculate height of a regular tetrahedron
    tetra_height = np.sqrt(6) / 3 * size
    
    # Height to edge ratio
    height_edge_ratio = tetra_height / size
    
    # Golden ratio (phi) plays a role in the scaling of nested structures
    phi = (1 + np.sqrt(5)) / 2
    
    # Volume ratio between successive nested tetrahedra (based on size_factor)
    size_factor = 0.5
    volume_ratio = size_factor ** 3
    
    # Number of vertices in the complete structure
    # 4 vertices per tetrahedron × 64 tetrahedra
    # But many vertices are shared, so less than 256
    vertex_count = 146  # Approximate
    
    # Relationship to phi and sqrt(2)
    phi_approximation = height_edge_ratio * np.sqrt(2)
    
    return {
        'height_edge_ratio': height_edge_ratio,
        'phi': phi,
        'phi_approximation': phi_approximation,
        'volume_ratio': volume_ratio,
        'vertex_count': vertex_count
    }

def calculate_64_star_energy_distribution(star_data: Dict[str, Any], 
                                        resolution: int = 30) -> Dict[str, Any]:
    """
    Calculate the energy distribution within the 64 star tetrahedron structure.
    
    Args:
        star_data: Dictionary containing 64 star tetrahedron data
        resolution: Resolution of the 3D grid for energy calculation
        
    Returns:
        Dictionary containing energy distribution data
    """
    center = star_data['center']
    size = star_data['size']
    energy_points = star_data['energy_points']
    
    # Create a 3D grid for energy calculation
    x = np.linspace(center[0] - size * 2, center[0] + size * 2, resolution)
    y = np.linspace(center[1] - size * 2, center[1] + size * 2, resolution)
    z = np.linspace(center[2] - size * 2, center[2] + size * 2, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize energy field
    energy = np.zeros((resolution, resolution, resolution))
    
    # Calculate energy from each tetrahedron
    # Energy increases near vertices and edges of tetrahedra
    for tetra_data in star_data['tetrahedra']:
        if 'upright' in tetra_data and 'inverted' in tetra_data:
            # Process star tetrahedron
            for tetra_type in ['upright', 'inverted']:
                tetra = tetra_data[tetra_type]
                # Add energy from each vertex
                for vertex in tetra['vertices']:
                    # Calculate distance from each grid point to vertex
                    dist = np.sqrt((X - vertex[0])**2 + (Y - vertex[1])**2 + (Z - vertex[2])**2)
                    # Energy decreases with distance
                    energy += np.exp(-dist / (size * 0.1))
                
                # Add energy from each edge
                for edge in tetra['edges']:
                    v1 = tetra['vertices'][edge[0]]
                    v2 = tetra['vertices'][edge[1]]
                    # Calculate distance from each grid point to line segment
                    # This is a simplified approximation
                    for t in np.linspace(0, 1, 5):
                        point = [v1[i] * (1-t) + v2[i] * t for i in range(3)]
                        dist = np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2)
                        energy += 0.5 * np.exp(-dist / (size * 0.1))
        elif 'single' in tetra_data:
            # Process single tetrahedron
            tetra = tetra_data['single']
            # Add energy from each vertex
            for vertex in tetra['vertices']:
                # Calculate distance from each grid point to vertex
                dist = np.sqrt((X - vertex[0])**2 + (Y - vertex[1])**2 + (Z - vertex[2])**2)
                # Energy decreases with distance
                energy += np.exp(-dist / (size * 0.1))
            
            # Add energy from each edge
            for edge in tetra['edges']:
                v1 = tetra['vertices'][edge[0]]
                v2 = tetra['vertices'][edge[1]]
                # Calculate distance from each grid point to line segment
                for t in np.linspace(0, 1, 5):
                    point = [v1[i] * (1-t) + v2[i] * t for i in range(3)]
                    dist = np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2)
                    energy += 0.5 * np.exp(-dist / (size * 0.1))
    
    # Add energy from specific energy points
    for point in energy_points:
        dist = np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2)
        energy += 2.0 * np.exp(-dist / (size * 0.1))
    
    # Normalize energy to [0, 1]
    energy = energy / np.max(energy)
    
    # Find points of maximum energy
    # Flatten arrays for easier processing
    flat_energy = energy.flatten()
    flat_indices = np.argsort(flat_energy)[-20:]  # Get 20 highest points
    
    # Convert flattened indices to 3D coordinates
    max_energy_points = []
    max_energy_values = []
    
    for idx in flat_indices:
        # Convert flat index to 3D indices
        i, j, k = np.unravel_index(idx, energy.shape)
        point = (X[i, j, k], Y[i, j, k], Z[i, j, k])
        max_energy_points.append(point)
        max_energy_values.append(flat_energy[idx])
    
    return {
        'energy': energy,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z,
        'max_energy_points': max_energy_points,
        'max_energy_values': max_energy_values
    }

def embed_64_star_in_field(field_array: np.ndarray, 
                          center: Tuple[float, float, float],
                          size: float,
                          strength: float = 1.0) -> np.ndarray:
    """
    Embed a 64 star tetrahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        size: Size of the pattern
        strength: Strength of the pattern (0.0 to 1.0)
        
    Returns:
        Modified field array with embedded pattern
    """
    # Generate 64 star tetrahedron
    star_data = generate_64_star_tetrahedron(center, size)
    
    # Calculate energy distribution with matching grid size
    field_shape = field_array.shape
    
    # Create coordinate grids matching field array
    x, y, z = np.meshgrid(
        np.arange(field_shape[0]),
        np.arange(field_shape[1]),
        np.arange(field_shape[2]),
        indexing='ij'
    )
    
    # Initialize energy field
    energy = np.zeros(field_shape)
    
    # Calculate energy from energy points
    for point in star_data['energy_points']:
        dist = np.sqrt((x - point[0])**2 + (y - point[1])**2 + (z - point[2])**2)
        energy += np.exp(-dist / (size * 0.1))
    
    # Add energy from tetrahedra
    # (Simplified to avoid detailed tetrahedron calculations)
    # Create a spherical energy field centered at the structure center
    central_dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    # Create shells of energy at different radii
    for radius_factor in [0.5, 1.0, 1.5]:
        radius = size * radius_factor
        shell = np.exp(-((central_dist - radius) / (size * 0.1))**2)
        energy += shell
    
    # Normalize energy to [0, 1]
    energy = energy / np.max(energy)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + energy * strength)
    
    # Normalize field after modification
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def visualize_64_star_tetrahedron(star_data: Dict[str, Any], 
                                show_energy: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the 64 star tetrahedron pattern.
    
    Args:
        star_data: Dictionary containing 64 star tetrahedron data
        show_energy: Whether to show energy points
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot tetrahedra
    for tetra_data in star_data['tetrahedra'][:10]:  # Limit to 10 for clarity
        if 'upright' in tetra_data and 'inverted' in tetra_data:
            # Plot star tetrahedron
            for tetra_type, color in zip(['upright', 'inverted'], ['b', 'r']):
                tetra = tetra_data[tetra_type]
                
                # Plot edges
                for edge in tetra['edges']:
                    v1 = tetra['vertices'][edge[0]]
                    v2 = tetra['vertices'][edge[1]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=color, alpha=0.7)
        
        elif 'single' in tetra_data:
            # Plot single tetrahedron
            tetra = tetra_data['single']
            color = 'g' if tetra['orientation'] == 0 else 'c'
            
            # Plot edges
            for edge in tetra['edges']:
                v1 = tetra['vertices'][edge[0]]
                v2 = tetra['vertices'][edge[1]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=color, alpha=0.7)
    
    # Plot energy points
    if show_energy:
        energy_points = star_data['energy_points']
        x = [p[0] for p in energy_points]
        y = [p[1] for p in energy_points]
        z = [p[2] for p in energy_points]
        ax.scatter(x, y, z, color='gold', s=50, alpha=0.8)
    
    # Add center point
    center = star_data['center']
    ax.scatter([center[0]], [center[1]], [center[2]], color='black', s=100, marker='*')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('64 Star Tetrahedron')
    
    # Set equal aspect ratio
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    range_max = np.max(limits[:, 1] - limits[:, 0])
    limits_mean = np.mean(limits, axis=1)
    ax.set_xlim3d([limits_mean[0] - range_max/2, limits_mean[0] + range_max/2])
    ax.set_ylim3d([limits_mean[1] - range_max/2, limits_mean[1] + range_max/2])
    ax.set_zlim3d([limits_mean[2] - range_max/2, limits_mean[2] + range_max/2])
    
    return fig

def get_base_glyph_elements(center: Tuple[float, float, float], size: float) -> Dict[str, Any]:
    """
    Returns 3D lines for Star Tetrahedron (Merkaba) base glyph. Size is edge length.
    """
    star_data = generate_star_tetrahedron(center, size)
    lines_data = []
    all_verts_list_st = [] # Renamed

    up_verts_st_np = np.array(star_data['upright']['vertices']) # Renamed
    all_verts_list_st.extend(up_verts_st_np.tolist())
    for v_idx1, v_idx2 in star_data['upright']['edges']:
        lines_data.append((up_verts_st_np[v_idx1].tolist(), up_verts_st_np[v_idx2].tolist()))

    down_verts_st_np = np.array(star_data['inverted']['vertices']) # Renamed
    all_verts_list_st.extend(down_verts_st_np.tolist())
    for v_idx1, v_idx2 in star_data['inverted']['edges']:
        lines_data.append((down_verts_st_np[v_idx1].tolist(), down_verts_st_np[v_idx2].tolist()))
    
    overall_verts_st_np = np.array(all_verts_list_st) # Renamed
    min_coords_st = np.min(overall_verts_st_np, axis=0); max_coords_st = np.max(overall_verts_st_np, axis=0) # Renamed
    padding_st = size * 0.15 # Renamed (using 'size' as it's edge_length)

    return {
        'lines': lines_data,
        'projection_type': '3d',
         'bounding_box': {
            'xmin': float(min_coords_st[0]-padding_st), 'xmax': float(max_coords_st[0]+padding_st),
            'ymin': float(min_coords_st[1]-padding_st), 'ymax': float(max_coords_st[1]+padding_st),
            'zmin': float(min_coords_st[2]-padding_st), 'zmax': float(max_coords_st[2]+padding_st),
        }
    }

# Example usage
if __name__ == "__main__":
    # Create a 64 star tetrahedron
    center = (0, 0, 0)
    size = 10.0
    
    star_64 = generate_64_star_tetrahedron(center, size)
    
    # Calculate sacred ratios
    ratios = calculate_64_star_sacred_ratios(star_64)
    print("Sacred Ratios:")
    for key, value in ratios.items():
        print(f"{key}: {value}")
    
    # Calculate energy distribution
    energy_data = calculate_64_star_energy_distribution(star_64)
    
    # Visualize
    fig = visualize_64_star_tetrahedron(star_64, show_energy=True)
    plt.tight_layout()
    plt.show()