"""
Tetrahedron Module

This module implements the tetrahedron platonic solid.
The tetrahedron represents the fire element and is the simplest platonic solid,
consisting of 4 triangular faces, 6 edges, and 4 vertices.

Key functions:
- Generate precise tetrahedron geometry
- Calculate energy dynamics and harmonics
- Associate with fire element aspects
- Establish resonance patterns for dimensional gateways
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict, Any, List, Optional

def generate_tetrahedron(center: Tuple[float, float, float],
                         edge_length: float) -> Dict[str, Any]:
    """
    Generate a tetrahedron with precise geometric properties.
    
    Args:
        center: The (x, y, z) center of the tetrahedron
        edge_length: The length of each edge
        
    Returns:
        Dictionary containing the tetrahedron geometry and properties
    """
    # Calculate vertex coordinates
    # Using regular tetrahedron coordinates based on edge length
    a = edge_length
    h = np.sqrt(6) * a / 4  # height from base to apex
    
    # Tetrahedron vertices based on equilateral triangle base
    # and apex at calculated height
    vertices = np.array([
        [0, 0, 0],                             # Base vertex 1
        [a, 0, 0],                             # Base vertex 2
        [a/2, a * np.sqrt(3)/2, 0],            # Base vertex 3
        [a/2, a * np.sqrt(3)/6, np.sqrt(6)*a/3]  # Apex
    ])
    
    # Center the tetrahedron at the specified center
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid + np.array(center)
    
    # Define faces using vertex indices
    faces = [
        [0, 1, 2],  # Base
        [0, 1, 3],  # Side 1
        [1, 2, 3],  # Side 2
        [0, 2, 3]   # Side 3
    ]
    
    # Define edges using vertex indices
    edges = [
        [0, 1], [1, 2], [2, 0],  # Base edges
        [0, 3], [1, 3], [2, 3]   # Edges to apex
    ]
    
    # Calculate important measurements
    height = np.sqrt(6) * edge_length / 3
    volume = edge_length**3 * np.sqrt(2) / 12
    surface_area = edge_length**2 * np.sqrt(3)
    
    # Calculate dihedral angle (angle between faces)
    dihedral_angle = np.arccos(-1/3)  # Approximately 70.53 degrees
    
    return {
        'vertices': vertices,
        'faces': faces,
        'edges': edges,
        'edge_length': edge_length,
        'height': height,
        'volume': volume,
        'surface_area': surface_area,
        'dihedral_angle': dihedral_angle,
        'center': center,
        'element': 'Fire'
    }

def calculate_tetrahedron_resonance(tetrahedron: Dict[str, Any], 
                                   base_frequency: float = 396.0) -> Dict[str, Any]:
    """
    Calculate the resonance properties of the tetrahedron.
    
    Args:
        tetrahedron: Dictionary containing tetrahedron geometry
        base_frequency: The base frequency for tetrahedron (fire element)
        
    Returns:
        Dictionary containing resonance properties
    """
    # Fire element base frequency is typically 396 Hz (Earth Tone)
    edge_length = tetrahedron['edge_length']
    
    # Calculate primary frequency based on edge length
    # Using the relationship between frequency and size
    primary_frequency = base_frequency * (1 / edge_length)
    
    # Calculate harmonic frequencies
    harmonics = [primary_frequency * n for n in range(1, 8)]
    
    # Calculate resonance nodes (points of maximum vibration)
    # These correspond to key points on the tetrahedron
    vertices = tetrahedron['vertices']
    face_centers = []
    for face in tetrahedron['faces']:
        face_vertices = [vertices[i] for i in face]
        face_center = np.mean(face_vertices, axis=0)
        face_centers.append(face_center)
    
    edge_centers = []
    for edge in tetrahedron['edges']:
        edge_vertices = [vertices[i] for i in edge]
        edge_center = np.mean(edge_vertices, axis=0)
        edge_centers.append(edge_center)
    
    # Calculate energy distribution for resonance
    # Each vertex has specific energy pattern
    vertex_energies = [1.0, 0.78, 0.64, 0.92]  # Fire element distribution
    
    return {
        'primary_frequency': primary_frequency,
        'harmonics': harmonics,
        'face_centers': face_centers,
        'edge_centers': edge_centers,
        'vertex_energies': vertex_energies,
        'element_frequency': base_frequency,
        'resonance_quality': 0.92  # Fire element has high resonance
    }

def get_tetrahedron_aspects() -> Dict[str, Any]:
    """
    Get the metaphysical and elemental aspects associated with the tetrahedron.
    
    Returns:
        Dictionary containing aspect properties
    """
    # Fire element aspects associated with tetrahedron
    aspects = {
        'element': 'Fire',
        'qualities': [
            'Transformation',
            'Energy',
            'Passion',
            'Creativity',
            'Will',
            'Intuition',
            'Inspiration'
        ],
        'chakra': 'Solar Plexus',
        'direction': 'South',
        'season': 'Summer',
        'state_of_matter': 'Plasma',
        'platonic_number': 4,  # Smallest platonic solid
        'sense': 'Sight',
        'consciousness_state': 'Dream State',
        'color': 'Red',
        'taste': 'Bitter',
        'symbolic_creature': 'Salamander',
        'associated_sephiroth': ['Geburah', 'Netzach', 'Tiphareth'],
        'gateway_key': 'First Gateway',
        'vibration': 'Rapid',
        'musical_note': 'G'
    }
    
    # Physiological properties
    physical_aspects = {
        'body_system': 'Metabolic and Digestive',
        'organs': ['Liver', 'Gallbladder', 'Small Intestine'],
        'glands': ['Adrenals'],
        'psychological_traits': [
            'Ambition',
            'Courage',
            'Drive',
            'Motivation',
            'Decisiveness',
            'Enthusiasm'
        ]
    }
    
    # Combination aspects with other elements
    combined_aspects = {
        'fire_earth': 'Manifestation',
        'fire_air': 'Inspiration',
        'fire_water': 'Emotional Power',
        'fire_aether': 'Divine Will'
    }
    
    # Sacred geometry connections
    geometry_aspects = {
        'primary_shape': 'Triangle',
        'angle_sum': 180,  # Sum of angles in triangle
        'polygon_faces': 'Equilateral Triangles',
        'dual_platonic': 'Tetrahedron',  # Self-dual
        'associated_flower_of_life_points': 4,
        'star_tetrahedron_component': True
    }
    
    return {
        'general': aspects,
        'physical': physical_aspects,
        'combinations': combined_aspects,
        'geometry': geometry_aspects
    }

def encode_tetrahedron_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Encode specific patterns for the tetrahedron based on its aspects.
    
    Args:
        pattern_type: The type of pattern to encode ('resonance', 'gateway', 'element')
        
    Returns:
        Dictionary containing encoded pattern
    """
    patterns = {
        'resonance': {
            'wave_pattern': [3, 6, 9, 12],  # Fibonacci derivatives
            'frequency_ratios': [1, 1.5, 2, 3.5],
            'node_activations': [1, 0, 1, 0],  # Binary activation sequence
            'geometric_sequence': [1, 3, 9, 27]  # Power of 3 sequence
        },
        'gateway': {
            'key_sequence': [4, 3, 2, 1],  # Countdown sequence
            'activation_pattern': [1, 0, 0, 1],
            'harmonic_intervals': ['perfect fourth', 'major third', 'perfect fifth'],
            'symbol_sequence': ['triangle', 'flame', 'spark', 'light']
        },
        'element': {
            'fire_pattern': [1, 3, 2, 4],
            'intensity_levels': [0.4, 0.8, 0.6, 1.0],
            'transformation_sequence': [2, 1, 4, 3],
            'color_spectrum': ['red', 'orange', 'yellow', 'white']
        },
        'consciousness': {
            'dream_state_pattern': [4, 1, 4, 1],
            'awareness_levels': [0.1, 0.4, 0.7, 1.0],
            'thought_sequence': [1, 1, 2, 3, 5],  # Fibonacci
            'activation_thresholds': [0.3, 0.6, 0.9]
        }
    }
    
    if pattern_type in patterns:
        return patterns[pattern_type]
    else:
        return patterns  # Return all patterns if type not specified

def visualize_tetrahedron(tetrahedron: Dict[str, Any],
                         show_aspects: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the tetrahedron.
    
    Args:
        tetrahedron: Dictionary containing tetrahedron geometry
        show_aspects: Whether to show aspect-related coloring
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = tetrahedron['vertices']
    faces = tetrahedron['faces']
    
    # Create face polygons
    polygons = []
    for face in faces:
        polygons.append([vertices[i] for i in face])
    
    # Set colors based on aspects if requested
    face_colors = ['r', 'r', 'r', 'r']  # Default fire color
    alpha = 0.6
    
    if show_aspects:
        aspects = get_tetrahedron_aspects()
        # Create color gradient based on aspects
        face_colors = ['#FF3300', '#FF6600', '#FF9900', '#FFCC00']  # Fire spectrum
        alpha = 0.7
    
    # Add faces
    ax.add_collection3d(Poly3DCollection(polygons, alpha=alpha, facecolors=face_colors, linewidths=1, edgecolors='k'))
    
    # Plot vertices
    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='k', s=50)
    
    # Plot edges
    edges = tetrahedron['edges']
    for edge in edges:
        line = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')
    
    # Add labels if showing aspects
    if show_aspects:
        center = tetrahedron['center']
        ax.text(center[0], center[1], center[2] + tetrahedron['height']/2, 
               "Fire Element", color='red', fontsize=12, ha='center')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Tetrahedron (Fire Element)')
    
    # Set equal aspect ratio
    max_range = max([
        np.max(vertices[:, 0]) - np.min(vertices[:, 0]),
        np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
        np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    ])
    mid_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) * 0.5
    mid_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) * 0.5
    mid_z = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    return fig

def embed_tetrahedron_in_field(field_array: np.ndarray, 
                              center: Tuple[float, float, float],
                              edge_length: float,
                              strength: float = 1.0,
                              pattern_type: str = 'resonance') -> np.ndarray:
    """
    Embed a tetrahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        edge_length: Length of tetrahedron edges
        strength: Strength of the pattern (0.0 to 1.0)
        pattern_type: Type of pattern to embed
        
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
    
    # Generate tetrahedron
    tetrahedron = generate_tetrahedron(center, edge_length)
    vertices = tetrahedron['vertices']
    
    # Get encoded pattern
    pattern = encode_tetrahedron_pattern(pattern_type)
    
    # Create a field influence based on distance to tetrahedron parts
    influence = np.zeros_like(field_array, dtype=float)
    
    # Influence from vertices (strongest)
    for i, vertex in enumerate(vertices):
        # Calculate distance from each point to vertex
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        
        # Create exponential decay influence
        # Pattern intensity varies by vertex
        if pattern_type == 'element':
            intensity = pattern['intensity_levels'][i % len(pattern['intensity_levels'])]
        else:
            intensity = 1.0
            
        # Distance falloff - exponential decay
        falloff = np.exp(-dist / (edge_length * 0.5))
        influence += falloff * intensity
    
    # Influence from edges (medium)
    edges = tetrahedron['edges']
    for i, edge in enumerate(edges):
        # Get edge vertices
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Calculate distance from each point to line segment (edge)
        # This is an approximation for computational efficiency
        edge_vector = v2 - v1
        edge_length_sq = np.sum(edge_vector**2)
        
        # Calculate distance to nearest point on line segment
        # For computational efficiency, sample points along the edge
        num_samples = 5
        for t in np.linspace(0, 1, num_samples):
            point = v1 + t * edge_vector
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2 + (z - point[2])**2)
            
            # Weaker influence from edges
            influence += 0.5 * np.exp(-dist / (edge_length * 0.3))
    
    # Normalize influence to [0, 1] range
    max_val = np.max(influence)
    if max_val > 0:
        influence = influence / max_val
    
    # Apply fire element pattern (rhythmic fluctuation)
    # Create oscillating pattern
    grid_size = max(field_shape)
    pattern_wave = np.sin(2 * np.pi * (x + y + z) / (grid_size * 0.1))
    
    # Combine influence with pattern
    pattern_field = influence * (0.5 + 0.5 * pattern_wave)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern_field * strength)
    
    # Normalize field after modification if needed
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def get_base_glyph_elements(center: Tuple[float, float, float], edge_length: float) -> Dict[str, Any]:
    """
    Returns the geometric elements (vertices, lines) for a simple line art
    representation of a tetrahedron.
    """
    solid_data = generate_tetrahedron(center, edge_length) # Call your existing generator
    
    vertices_np = np.array(solid_data['vertices'])
    lines = []
    for edge_indices in solid_data['edges']:
        p1 = vertices_np[edge_indices[0]].tolist()
        p2 = vertices_np[edge_indices[1]].tolist()
        lines.append((p1, p2))
        
    # Calculate a simple bounding box
    min_coords = np.min(vertices_np, axis=0)
    max_coords = np.max(vertices_np, axis=0)
    padding = edge_length * 0.1 # 10% padding
    
    return {
        'lines': lines,
        'vertices': vertices_np.tolist(), # Optional: if you want to draw small dots for vertices
        'projection_type': '3d',
        'bounding_box': {
            'xmin': min_coords[0] - padding, 'xmax': max_coords[0] + padding,
            'ymin': min_coords[1] - padding, 'ymax': max_coords[1] + padding,
            'zmin': min_coords[2] - padding, 'zmax': max_coords[2] + padding,
        }
    }

# Example usage
if __name__ == "__main__":
    # Create a tetrahedron
    center = (0, 0, 0)
    edge_length = 2.0
    
    tetrahedron = generate_tetrahedron(center, edge_length)
    
    # Calculate resonance properties
    resonance = calculate_tetrahedron_resonance(tetrahedron)
    print(f"Primary Frequency: {resonance['primary_frequency']:.2f} Hz")
    print(f"Harmonics: {[f'{h:.2f}' for h in resonance['harmonics']]}")
    
    # Get aspects
    aspects = get_tetrahedron_aspects()
    print("\nTetrahedron Aspects:")
    print(f"Element: {aspects['general']['element']}")
    print(f"Qualities: {', '.join(aspects['general']['qualities'])}")
    print(f"Chakra: {aspects['general']['chakra']}")
    
    # Visualize
    fig = visualize_tetrahedron(tetrahedron, show_aspects=True)
    plt.show()