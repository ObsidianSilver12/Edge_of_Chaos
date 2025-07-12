"""
Octahedron Module

This module implements the octahedron platonic solid.
The octahedron represents the air element and consists of
8 triangular faces, 12 edges, and 6 vertices.

Key functions:
- Generate precise octahedron geometry
- Calculate energy dynamics and harmonics
- Associate with air element aspects
- Establish resonance patterns for dimensional gateways
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict, Any, List, Optional

def generate_octahedron(center: Tuple[float, float, float],
                       edge_length: float) -> Dict[str, Any]:
    """
    Generate an octahedron with precise geometric properties.
    
    Args:
        center: The (x, y, z) center of the octahedron
        edge_length: The length of each edge
        
    Returns:
        Dictionary containing the octahedron geometry and properties
    """
    # Calculate distance from center to vertices
    vertex_distance = edge_length / np.sqrt(2)
    
    # Define vertices of an octahedron centered at origin
    vertices = np.array([
        [vertex_distance, 0, 0],   # 0: right
        [-vertex_distance, 0, 0],  # 1: left
        [0, vertex_distance, 0],   # 2: front
        [0, -vertex_distance, 0],  # 3: back
        [0, 0, vertex_distance],   # 4: top
        [0, 0, -vertex_distance]   # 5: bottom
    ])
    
    # Translate vertices to the specified center
    vertices = vertices + np.array(center)
    
    # Define faces using vertex indices
    faces = [
        [0, 2, 4],  # Right, front, top
        [2, 1, 4],  # Front, left, top
        [1, 3, 4],  # Left, back, top
        [3, 0, 4],  # Back, right, top
        [0, 2, 5],  # Right, front, bottom
        [2, 1, 5],  # Front, left, bottom
        [1, 3, 5],  # Left, back, bottom
        [3, 0, 5]   # Back, right, bottom
    ]
    
    # Define edges using vertex indices
    edges = [
        [0, 2], [2, 1], [1, 3], [3, 0],  # Equator
        [0, 4], [1, 4], [2, 4], [3, 4],  # Top edges
        [0, 5], [1, 5], [2, 5], [3, 5]   # Bottom edges
    ]
    
    # Calculate important measurements
    height = 2 * vertex_distance  # Distance between top and bottom vertices
    volume = (np.sqrt(2) / 3) * edge_length**3
    surface_area = 2 * np.sqrt(3) * edge_length**2
    
    # Calculate dihedral angle (angle between faces)
    dihedral_angle = np.arccos(-1/3)  # Approximately 109.47 degrees
    
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
        'element': 'Air'
    }

def calculate_octahedron_resonance(octahedron: Dict[str, Any], 
                                  base_frequency: float = 285.0) -> Dict[str, Any]:
    """
    Calculate the resonance properties of the octahedron.
    
    Args:
        octahedron: Dictionary containing octahedron geometry
        base_frequency: The base frequency for octahedron (air element)
        
    Returns:
        Dictionary containing resonance properties
    """
    # Air element base frequency is typically 285 Hz
    edge_length = octahedron['edge_length']
    
    # Calculate primary frequency based on edge length
    # Using the relationship between frequency and size
    primary_frequency = base_frequency * (1 / edge_length)
    
    # Calculate harmonic frequencies
    harmonics = [primary_frequency * n for n in range(1, 8)]
    
    # Calculate resonance nodes (points of maximum vibration)
    # These correspond to key points on the octahedron
    vertices = octahedron['vertices']
    
    face_centers = []
    for face in octahedron['faces']:
        face_vertices = [vertices[i] for i in face]
        face_center = np.mean(face_vertices, axis=0)
        face_centers.append(face_center)
    
    edge_centers = []
    for edge in octahedron['edges']:
        edge_vertices = [vertices[i] for i in edge]
        edge_center = np.mean(edge_vertices, axis=0)
        edge_centers.append(edge_center)
    
    # Calculate energy distribution for resonance
    # Each vertex has specific energy pattern
    vertex_energies = [0.76, 0.80, 0.72, 0.74, 0.78, 0.82]  # Air element distribution
    
    return {
        'primary_frequency': primary_frequency,
        'harmonics': harmonics,
        'face_centers': face_centers,
        'edge_centers': edge_centers,
        'vertex_energies': vertex_energies,
        'element_frequency': base_frequency,
        'resonance_quality': 0.82  # Air element has flowing resonance
    }

def get_octahedron_aspects() -> Dict[str, Any]:
    """
    Get the metaphysical and elemental aspects associated with the octahedron.
    
    Returns:
        Dictionary containing aspect properties
    """
    # Air element aspects associated with octahedron
    aspects = {
        'element': 'Air',
        'qualities': [
            'Communication',
            'Intelligence',
            'Connection',
            'Motion',
            'Thought',
            'Wisdom',
            'Clarity'
        ],
        'chakra': 'Throat',
        'direction': 'East',
        'season': 'Spring',
        'state_of_matter': 'Gas',
        'platonic_number': 8,  # Number of faces
        'sense': 'Hearing',
        'consciousness_state': 'Liminal State',
        'color': 'Yellow',
        'taste': 'Sour',
        'symbolic_creature': 'Eagle',
        'associated_sephiroth': ['Binah', 'Kether', 'Chokmah', 'Chesed', 'Tiphareth', 'Geburah'],
        'gateway_key': 'Second Gateway',
        'vibration': 'Medium',
        'musical_note': 'A'
    }
    
    # Physiological properties
    physical_aspects = {
        'body_system': 'Respiratory and Nervous',
        'organs': ['Lungs', 'Brain', 'Nerves'],
        'glands': ['Thymus'],
        'psychological_traits': [
            'Intellectual',
            'Analytical',
            'Communicative',
            'Social',
            'Adaptable',
            'Quick-thinking'
        ]
    }
    
    # Combination aspects with other elements
    combined_aspects = {
        'air_fire': 'Inspiration',
        'air_earth': 'Practical Knowledge',
        'air_water': 'Emotional Intelligence',
        'air_aether': 'Higher Consciousness'
    }
    
    # Sacred geometry connections
    geometry_aspects = {
        'primary_shape': 'Triangle',
        'angle_sum': 180,  # Sum of angles in triangle
        'polygon_faces': 'Equilateral Triangles',
        'dual_platonic': 'Hexahedron (Cube)',
        'associated_flower_of_life_points': 6,
        'vector_equilibrium_component': True
    }
    
    return {
        'general': aspects,
        'physical': physical_aspects,
        'combinations': combined_aspects,
        'geometry': geometry_aspects
    }

def encode_octahedron_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Encode specific patterns for the octahedron based on its aspects.
    
    Args:
        pattern_type: The type of pattern to encode ('resonance', 'gateway', 'element')
        
    Returns:
        Dictionary containing encoded pattern
    """
    patterns = {
        'resonance': {
            'wave_pattern': [8, 12, 6, 8],  # Octahedral number sequence
            'frequency_ratios': [1, 1.5, 2, 3],  # Musical ratios
            'node_activations': [1, 0, 1, 0, 1, 0],  # Binary activation sequence
            'geometric_sequence': [1, 8, 27, 64]  # Cubic number sequence
        },
        'gateway': {
            'key_sequence': [8, 6, 12, 8],  # Face, vertex, edge, face count sequence
            'activation_pattern': [1, 1, 0, 0, 1, 1, 0, 0],
            'harmonic_intervals': ['perfect fifth', 'major third', 'octave'],
            'symbol_sequence': ['triangle', 'air', 'wind', 'breath']
        },
        'element': {
            'air_pattern': [2, 6, 8, 12],
            'flow_levels': [0.6, 0.7, 0.8, 0.6, 0.7, 0.8],  # Flowing, oscillating pattern
            'communication_sequence': [3, 6, 12, 24, 48],  # Doubling progression
            'color_spectrum': ['sky blue', 'white', 'yellow', 'pale yellow', 'light blue', 'silver']
        },
        'consciousness': {
            'liminal_state_pattern': [8, 2, 8, 2],
            'transition_levels': [0.3, 0.5, 0.7, 0.9, 0.7, 0.5],  # Waxing and waning
            'thought_sequence': [1, 2, 4, 8, 16, 32],  # Powers of 2
            'activation_thresholds': [0.2, 0.4, 0.6, 0.8, 1.0]
        }
    }
    
    if pattern_type in patterns:
        return patterns[pattern_type]
    else:
        return patterns  # Return all patterns if type not specified

def visualize_octahedron(octahedron: Dict[str, Any],
                        show_aspects: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the octahedron.
    
    Args:
        octahedron: Dictionary containing octahedron geometry
        show_aspects: Whether to show aspect-related coloring
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = octahedron['vertices']
    faces = octahedron['faces']
    
    # Create face polygons
    polygons = []
    for face in faces:
        polygons.append([vertices[i] for i in face])
    
    # Set colors based on aspects if requested
    face_colors = ['y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']  # Default air color
    alpha = 0.6
    
    if show_aspects:
        aspects = get_octahedron_aspects()
        # Create color gradient based on aspects
        face_colors = ['#FFFF99', '#FFFF66', '#FFFFCC', '#FFFF33', '#F0E68C', '#EEE8AA', '#FAFAD2', '#FFFFE0']
        alpha = 0.7
    
    # Add faces
    ax.add_collection3d(Poly3DCollection(polygons, alpha=alpha, facecolors=face_colors, linewidths=1, edgecolors='k'))
    
    # Plot vertices
    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='k', s=50)
    
    # Plot edges
    edges = octahedron['edges']
    for edge in edges:
        line = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')
    
    # Add labels if showing aspects
    if show_aspects:
        center = octahedron['center']
        ax.text(center[0], center[1], center[2] + octahedron['edge_length']/2, 
               "Air Element", color='goldenrod', fontsize=12, ha='center')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Octahedron (Air Element)')
    
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

def embed_octahedron_in_field(field_array: np.ndarray, 
                             center: Tuple[float, float, float],
                             edge_length: float,
                             strength: float = 1.0,
                             pattern_type: str = 'resonance') -> np.ndarray:
    """
    Embed an octahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        edge_length: Length of octahedron edges
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
    
    # Generate octahedron
    octahedron = generate_octahedron(center, edge_length)
    vertices = octahedron['vertices']
    
    # Get encoded pattern
    pattern = encode_octahedron_pattern(pattern_type)
    
    # Create a field influence based on distance to octahedron parts
    influence = np.zeros_like(field_array, dtype=float)
    
    # Influence from vertices (strongest for air element)
    for i, vertex in enumerate(vertices):
        # Calculate distance from each point to vertex
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        
        # Create exponential decay influence
        # Pattern intensity varies by vertex - air element has flowing patterns
        if pattern_type == 'element':
            intensity = pattern['flow_levels'][i % len(pattern['flow_levels'])]
        else:
            intensity = 1.0
            
        # Distance falloff - more gradual for air element
        falloff = np.exp(-dist / (edge_length * 0.6))
        influence += falloff * intensity
    
    # Influence from edges (strong for air - represents flow channels)
    edges = octahedron['edges']
    for i, edge in enumerate(edges):
        # Get edge vertices
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Calculate midpoint of edge
        midpoint = (v1 + v2) / 2
        
        # Calculate distance from each point to edge midpoint
        dist = np.sqrt((x - midpoint[0])**2 + (y - midpoint[1])**2 + (z - midpoint[2])**2)
        
        # Add flowing pattern along edge
        # For air element, edges represent channels of energy flow
        edge_influence = np.exp(-dist / (edge_length * 0.5))
        
        # Apply flow pattern - air element has dynamic, flowing patterns
        if pattern_type == 'element' and i < len(pattern['flow_levels']):
            edge_influence *= pattern['flow_levels'][i % len(pattern['flow_levels'])]
        
        influence += edge_influence * 1.2  # Edges have strong influence in air element
    
    # Normalize influence to [0, 1] range
    max_val = np.max(influence)
    if max_val > 0:
        influence = influence / max_val
    
    # Apply air element pattern (flowing, wave-like pattern)
    # Create oscillating pattern characteristic of air element
    grid_size = max(field_shape)
    wave_pattern = np.sin(2 * np.pi * (x + y + z) / (grid_size * 0.15)) * np.cos(2 * np.pi * (x - z) / (grid_size * 0.2))
    wave_pattern = 0.5 + 0.5 * wave_pattern  # Normalize to [0, 1]
    
    # Combine influence with pattern - more dynamic for air element
    pattern_field = influence * (0.7 + 0.3 * wave_pattern)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern_field * strength)
    
    # Normalize field after modification if needed
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def get_base_glyph_elements(center: Tuple[float, float, float], edge_length: float) -> Dict[str, Any]:
    """
    Returns the geometric elements (vertices, lines) for a simple line art
    representation of a octahedron.
    """
    solid_data = generate_octahedron(center, edge_length) # Call your existing generator
    
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
    # Create an octahedron
    center = (0, 0, 0)
    edge_length = 2.0
    
    octahedron = generate_octahedron(center, edge_length)
    
    # Calculate resonance properties
    resonance = calculate_octahedron_resonance(octahedron)
    print(f"Primary Frequency: {resonance['primary_frequency']:.2f} Hz")
    print(f"Harmonics: {[f'{h:.2f}' for h in resonance['harmonics']]}")
    
    # Get aspects
    aspects = get_octahedron_aspects()
    print("\nOctahedron Aspects:")
    print(f"Element: {aspects['general']['element']}")
    print(f"Qualities: {', '.join(aspects['general']['qualities'])}")
    print(f"Chakra: {aspects['general']['chakra']}")
    
    # Visualize
    fig = visualize_octahedron(octahedron, show_aspects=True)
    plt.show()