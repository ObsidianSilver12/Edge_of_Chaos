"""
Hexahedron (Cube) Module

This module implements the hexahedron (cube) platonic solid.
The hexahedron represents the earth element and consists of
6 square faces, 12 edges, and 8 vertices.

Key functions:
- Generate precise hexahedron geometry
- Calculate energy dynamics and harmonics
- Associate with earth element aspects
- Establish resonance patterns for dimensional gateways
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict, Any, List, Optional

def generate_hexahedron(center: Tuple[float, float, float],
                        edge_length: float) -> Dict[str, Any]:
    """
    Generate a hexahedron (cube) with precise geometric properties.
    
    Args:
        center: The (x, y, z) center of the hexahedron
        edge_length: The length of each edge
        
    Returns:
        Dictionary containing the hexahedron geometry and properties
    """
    # Calculate half edge length for vertex placement
    half_edge = edge_length / 2
    
    # Define vertices of a cube centered at origin
    vertices = np.array([
        [half_edge, half_edge, half_edge],    # 0: top front right
        [-half_edge, half_edge, half_edge],   # 1: top front left
        [-half_edge, -half_edge, half_edge],  # 2: top back left
        [half_edge, -half_edge, half_edge],   # 3: top back right
        [half_edge, half_edge, -half_edge],   # 4: bottom front right
        [-half_edge, half_edge, -half_edge],  # 5: bottom front left
        [-half_edge, -half_edge, -half_edge], # 6: bottom back left
        [half_edge, -half_edge, -half_edge]   # 7: bottom back right
    ])
    
    # Translate vertices to the specified center
    vertices = vertices + np.array(center)
    
    # Define faces using vertex indices
    faces = [
        [0, 1, 2, 3],  # Top face
        [4, 5, 6, 7],  # Bottom face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 3, 7, 4],  # Right face
        [1, 2, 6, 5]   # Left face
    ]
    
    # Define edges using vertex indices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    
    # Calculate important measurements
    volume = edge_length**3
    surface_area = 6 * edge_length**2
    diagonal = edge_length * np.sqrt(3)  # Body diagonal
    face_diagonal = edge_length * np.sqrt(2)  # Diagonal of a face
    
    # Calculate dihedral angle (angle between faces)
    dihedral_angle = np.pi/2  # 90 degrees for cube
    
    return {
        'vertices': vertices,
        'faces': faces,
        'edges': edges,
        'edge_length': edge_length,
        'volume': volume,
        'surface_area': surface_area,
        'diagonal': diagonal,
        'face_diagonal': face_diagonal,
        'dihedral_angle': dihedral_angle,
        'center': center,
        'element': 'Earth'
    }

def calculate_hexahedron_resonance(hexahedron: Dict[str, Any], 
                                 base_frequency: float = 174.0) -> Dict[str, Any]:
    """
    Calculate the resonance properties of the hexahedron.
    
    Args:
        hexahedron: Dictionary containing hexahedron geometry
        base_frequency: The base frequency for hexahedron (earth element)
        
    Returns:
        Dictionary containing resonance properties
    """
    # Earth element base frequency is typically 174 Hz (Solfeggio frequency)
    edge_length = hexahedron['edge_length']
    
    # Calculate primary frequency based on edge length
    # Using the relationship between frequency and size
    primary_frequency = base_frequency * (1 / edge_length)
    
    # Calculate harmonic frequencies
    harmonics = [primary_frequency * n for n in range(1, 8)]
    
    # Calculate resonance nodes (points of maximum vibration)
    # These correspond to key points on the hexahedron
    vertices = hexahedron['vertices']
    
    face_centers = []
    for face in hexahedron['faces']:
        face_vertices = [vertices[i] for i in face]
        face_center = np.mean(face_vertices, axis=0)
        face_centers.append(face_center)
    
    edge_centers = []
    for edge in hexahedron['edges']:
        edge_vertices = [vertices[i] for i in edge]
        edge_center = np.mean(edge_vertices, axis=0)
        edge_centers.append(edge_center)
    
    # Calculate energy distribution for resonance
    # Each vertex has specific energy pattern
    vertex_energies = [0.84, 0.92, 0.88, 0.86, 0.90, 0.85, 0.89, 0.87]  # Earth element distribution
    
    return {
        'primary_frequency': primary_frequency,
        'harmonics': harmonics,
        'face_centers': face_centers,
        'edge_centers': edge_centers,
        'vertex_energies': vertex_energies,
        'element_frequency': base_frequency,
        'resonance_quality': 0.89  # Earth element has stable resonance
    }

def get_hexahedron_aspects() -> Dict[str, Any]:
    """
    Get the metaphysical and elemental aspects associated with the hexahedron.
    
    Returns:
        Dictionary containing aspect properties
    """
    # Earth element aspects associated with hexahedron
    aspects = {
        'element': 'Earth',
        'qualities': [
            'Stability',
            'Structure',
            'Manifestation',
            'Grounding',
            'Abundance',
            'Endurance',
            'Strength'
        ],
        'chakra': 'Root',
        'direction': 'North',
        'season': 'Winter',
        'state_of_matter': 'Solid',
        'platonic_number': 6,  # Number of faces
        'sense': 'Touch',
        'consciousness_state': 'Physical Awareness',
        'color': 'Green',
        'taste': 'Sweet',
        'symbolic_creature': 'Bull',
        'associated_sephiroth': ['Malkuth', 'Yesod', 'Hod'],
        'gateway_key': 'Third Gateway',
        'vibration': 'Slow',
        'musical_note': 'F'
    }
    
    # Physiological properties
    physical_aspects = {
        'body_system': 'Structural and Skeletal',
        'organs': ['Bones', 'Muscles', 'Skin'],
        'glands': ['Thyroid', 'Parathyroid'],
        'psychological_traits': [
            'Persistence',
            'Reliability',
            'Practicality',
            'Patience',
            'Methodical',
            'Responsibility'
        ]
    }
    
    # Combination aspects with other elements
    combined_aspects = {
        'earth_fire': 'Creation',
        'earth_air': 'Growth',
        'earth_water': 'Fertility',
        'earth_aether': 'Materialization'
    }
    
    # Sacred geometry connections
    geometry_aspects = {
        'primary_shape': 'Square',
        'angle_sum': 360,  # Sum of angles in square
        'polygon_faces': 'Squares',
        'dual_platonic': 'Octahedron',
        'associated_flower_of_life_points': 8,
        'metatrons_cube_component': True
    }
    
    return {
        'general': aspects,
        'physical': physical_aspects,
        'combinations': combined_aspects,
        'geometry': geometry_aspects
    }

def encode_hexahedron_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Encode specific patterns for the hexahedron based on its aspects.
    
    Args:
        pattern_type: The type of pattern to encode ('resonance', 'gateway', 'element')
        
    Returns:
        Dictionary containing encoded pattern
    """
    patterns = {
        'resonance': {
            'wave_pattern': [4, 8, 12, 16],  # Square number sequence
            'frequency_ratios': [1, 2, 4, 8],  # Power of 2 sequence
            'node_activations': [1, 1, 0, 0, 1, 1, 0, 0],  # Binary activation sequence
            'geometric_sequence': [1, 4, 16, 64]  # Power of 4 sequence
        },
        'gateway': {
            'key_sequence': [6, 12, 8, 6],  # Face, edge, vertex count sequence
            'activation_pattern': [1, 0, 1, 0, 1, 0],  # Alternating pattern
            'harmonic_intervals': ['perfect fifth', 'octave', 'perfect fourth'],
            'symbol_sequence': ['square', 'cube', 'earth', 'matter']
        },
        'element': {
            'earth_pattern': [4, 2, 6, 8],
            'stability_levels': [0.7, 0.8, 0.9, 1.0, 0.9, 0.8],  # Stability fluctuation
            'manifestation_sequence': [1, 2, 3, 4, 5, 6],  # Linear progression
            'color_spectrum': ['brown', 'green', 'black', 'yellow', 'ochre', 'tan']
        },
        'consciousness': {
            'physical_awareness_pattern': [6, 1, 6, 1],
            'grounding_levels': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'material_sequence': [1, 3, 6, 10, 15, 21],  # Triangular numbers
            'activation_thresholds': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
    }
    
    if pattern_type in patterns:
        return patterns[pattern_type]
    else:
        return patterns  # Return all patterns if type not specified

def visualize_hexahedron(hexahedron: Dict[str, Any],
                       show_aspects: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the hexahedron.
    
    Args:
        hexahedron: Dictionary containing hexahedron geometry
        show_aspects: Whether to show aspect-related coloring
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = hexahedron['vertices']
    faces = hexahedron['faces']
    
    # Create face polygons
    polygons = []
    for face in faces:
        polygons.append([vertices[i] for i in face])
    
    # Set colors based on aspects if requested
    face_colors = ['g'] * len(faces)  # Default earth color
    alpha = 0.6
    
    if show_aspects:
        aspects = get_hexahedron_aspects()
        # Create color gradient based on aspects
        greens = [
            '#006400', '#228B22', '#32CD32', '#90EE90', '#98FB98', '#8FBC8F'
        ]
        face_colors = greens
        alpha = 0.7
    
    # Add faces
    ax.add_collection3d(Poly3DCollection(polygons, alpha=alpha, facecolors=face_colors, linewidths=1, edgecolors='k'))
    
    # Plot vertices
    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='k', s=50)
    
    # Plot edges
    edges = hexahedron['edges']
    for edge in edges:
        line = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')
    
    # Add labels if showing aspects
    if show_aspects:
        center = hexahedron['center']
        ax.text(center[0], center[1], center[2] + hexahedron['edge_length']/2, 
               "Earth Element", color='darkgreen', fontsize=12, ha='center')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hexahedron (Earth Element)')
    
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

def embed_hexahedron_in_field(field_array: np.ndarray, 
                             center: Tuple[float, float, float],
                             edge_length: float,
                             strength: float = 1.0,
                             pattern_type: str = 'resonance') -> np.ndarray:
    """
    Embed a hexahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        edge_length: Length of hexahedron edges
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
    
    # Generate hexahedron
    hexahedron = generate_hexahedron(center, edge_length)
    vertices = hexahedron['vertices']
    
    # Get encoded pattern
    pattern = encode_hexahedron_pattern(pattern_type)
    
    # Create a field influence based on distance to hexahedron parts
    influence = np.zeros_like(field_array, dtype=float)
    
    # Influence from vertices
    for i, vertex in enumerate(vertices):
        # Calculate distance from each point to vertex
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        
        # Create exponential decay influence
        # Pattern intensity varies by vertex - earth element has stable patterns
        if pattern_type == 'element':
            intensity = pattern['stability_levels'][i % len(pattern['stability_levels'])]
        else:
            intensity = 1.0
            
        # Distance falloff - sharper for earth element
        falloff = np.exp(-dist / (edge_length * 0.3))
        influence += falloff * intensity
    
    # Influence from faces (strong for earth - represents stability)
    faces = hexahedron['faces']
    for i, face in enumerate(faces):
        # Get face vertices
        face_vertices = [vertices[j] for j in face]
        face_center = np.mean(face_vertices, axis=0)
        
        # Calculate distance from each point to face center
        dist = np.sqrt((x - face_center[0])**2 + (y - face_center[1])**2 + (z - face_center[2])**2)
        
        # Add stable pattern along face
        # For earth element, faces represent planes of stability
        face_influence = np.exp(-dist / (edge_length * 0.4))
        
        # Apply stability pattern - earth element has stable, solid patterns
        if pattern_type == 'element' and i < len(pattern['stability_levels']):
            face_influence *= pattern['stability_levels'][i % len(pattern['stability_levels'])]
        
        influence += face_influence * 1.5  # Faces have strong influence in earth element
    
    # Influence from edges (moderate for earth)
    edges = hexahedron['edges']
    for edge in edges:
        # Get edge vertices
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Calculate midpoint of edge
        midpoint = (v1 + v2) / 2
        
        # Calculate distance from each point to edge midpoint
        dist = np.sqrt((x - midpoint[0])**2 + (y - midpoint[1])**2 + (z - midpoint[2])**2)
        
        # Add stable pattern along edge
        edge_influence = np.exp(-dist / (edge_length * 0.35))
        influence += edge_influence * 0.8
    
    # Normalize influence to [0, 1] range
    max_val = np.max(influence)
    if max_val > 0:
        influence = influence / max_val
    
    # Apply earth element pattern (stable, solid pattern)
    # Create squared pattern characteristic of earth element
    grid_size = max(field_shape)
    stability_pattern = np.cos(np.pi * x / grid_size)**2 * np.cos(np.pi * y / grid_size)**2 * np.cos(np.pi * z / grid_size)**2
    stability_pattern = 0.5 + 0.5 * stability_pattern  # Normalize to [0, 1]
    
    # Combine influence with pattern - more stable for earth element
    pattern_field = influence * (0.8 + 0.2 * stability_pattern)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern_field * strength)
    
    # Normalize field after modification if needed
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field


# Example usage
if __name__ == "__main__":
    # Create a hexahedron
    center = (0, 0, 0)
    edge_length = 2.0
    
    hexahedron = generate_hexahedron(center, edge_length)
    
    # Calculate resonance properties
    resonance = calculate_hexahedron_resonance(hexahedron)
    print(f"Primary Frequency: {resonance['primary_frequency']:.2f} Hz")
    print(f"Harmonics: {[f'{h:.2f}' for h in resonance['harmonics']]}")
    
    # Get aspects
    aspects = get_hexahedron_aspects()
    print("\nHexahedron Aspects:")
    print(f"Element: {aspects['general']['element']}")
    print(f"Qualities: {', '.join(aspects['general']['qualities'])}")
    print(f"Chakra: {aspects['general']['chakra']}")
    
    # Visualize
    fig = visualize_hexahedron(hexahedron, show_aspects=True)
    plt.show()
