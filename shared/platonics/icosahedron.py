"""
Icosahedron Module

This module implements the icosahedron platonic solid.
The icosahedron represents the water element and consists of
20 triangular faces, 30 edges, and 12 vertices.

Key functions:
- Generate precise icosahedron geometry
- Calculate energy dynamics and harmonics
- Associate with water element aspects
- Establish resonance patterns for dimensional gateways
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict, Any, List, Optional

def generate_icosahedron(center: Tuple[float, float, float],
                        edge_length: float) -> Dict[str, Any]:
    """
    Generate an icosahedron with precise geometric properties.
    
    Args:
        center: The (x, y, z) center of the icosahedron
        edge_length: The length of each edge
        
    Returns:
        Dictionary containing the icosahedron geometry and properties
    """
    # Calculate the radius of the circumscribed sphere
    # For an icosahedron with edge length e, the radius is:
    # r = e * sin(2π/5) / (2*sin(π/3))
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    radius = edge_length / (2 * np.sin(np.pi / 5))
    
    # Generate the 12 vertices of the icosahedron
    # These are based on three orthogonal golden rectangles
    vertices = []
    
    # Vertices based on golden rectangles
    for i in [-1, 1]:
        for j in [-1, 1]:
            # Vertices from the xy-plane golden rectangle
            vertices.append([0, i * radius / phi, j * radius * phi / 2])
            # Vertices from the yz-plane golden rectangle
            vertices.append([i * radius / phi, j * radius * phi / 2, 0])
            # Vertices from the xz-plane golden rectangle
            vertices.append([i * radius * phi / 2, 0, j * radius / phi])
    
    vertices = np.array(vertices)
    
    # Scale to match specified edge length
    current_edge = np.linalg.norm(vertices[0] - vertices[2])
    scale_factor = edge_length / current_edge
    vertices = vertices * scale_factor
    
    # Translate vertices to the specified center
    vertices = vertices + np.array(center)
    
    # Define faces manually
    # This is one possible assignment of the 20 triangular faces
    faces = [
        [0, 2, 10], [0, 10, 8], [0, 8, 4], [0, 4, 6], [0, 6, 2],
        [1, 3, 11], [1, 11, 9], [1, 9, 5], [1, 5, 7], [1, 7, 3],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 10],
        [4, 8, 9], [4, 9, 5], [4, 5, 7], [4, 7, 6],
        [8, 10, 11], [8, 11, 9]
    ]
    
    # Define edges (pairs of vertices that form edges)
    # This can be derived from the faces
    edges = set()
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.add(edge)
    edges = list(edges)
    
    # Calculate important measurements
    # For an icosahedron with edge length e:
    volume = (5/12) * (3 + np.sqrt(5)) * edge_length**3
    surface_area = 5 * np.sqrt(3) * edge_length**2
    
    # Calculate dihedral angle (angle between faces)
    dihedral_angle = np.arccos(-np.sqrt(5)/3)  # Approximately 138.19 degrees
    
    return {
        'vertices': vertices,
        'faces': faces,
        'edges': edges,
        'edge_length': edge_length,
        'radius': radius * scale_factor,
        'volume': volume,
        'surface_area': surface_area,
        'dihedral_angle': dihedral_angle,
        'center': center,
        'element': 'Water'
    }

def calculate_icosahedron_resonance(icosahedron: Dict[str, Any], 
                                   base_frequency: float = 417.0) -> Dict[str, Any]:
    """
    Calculate the resonance properties of the icosahedron.
    
    Args:
        icosahedron: Dictionary containing icosahedron geometry
        base_frequency: The base frequency for icosahedron (water element)
        
    Returns:
        Dictionary containing resonance properties
    """
    # Water element base frequency is typically 417 Hz (Solfeggio frequency)
    edge_length = icosahedron['edge_length']
    
    # Calculate primary frequency based on edge length
    # Using the relationship between frequency and size
    primary_frequency = base_frequency * (1 / edge_length)
    
    # Calculate harmonic frequencies
    # Water element has flowing harmonics, based on Fibonacci sequence ratios
    harmonics = [primary_frequency * n for n in [1, 1.618, 2.618, 4.236, 6.854, 11.09, 17.944]]
    
    # Calculate resonance nodes (points of maximum vibration)
    # These correspond to key points on the icosahedron
    vertices = icosahedron['vertices']
    
    face_centers = []
    for face in icosahedron['faces']:
        face_vertices = [vertices[i] for i in face]
        face_center = np.mean(face_vertices, axis=0)
        face_centers.append(face_center)
    
    edge_centers = []
    for edge in icosahedron['edges']:
        edge_vertices = [vertices[i] for i in edge]
        edge_center = np.mean(edge_vertices, axis=0)
        edge_centers.append(edge_center)
    
    # Calculate energy distribution for resonance
    # Each vertex has specific energy pattern
    # Water element has flowing, adaptive energy
    vertex_energies = [0.62, 0.68, 0.71, 0.67, 0.63, 0.69, 0.66, 0.70, 0.64, 0.65, 0.72, 0.61]
    
    return {
        'primary_frequency': primary_frequency,
        'harmonics': harmonics,
        'face_centers': face_centers,
        'edge_centers': edge_centers,
        'vertex_energies': vertex_energies,
        'element_frequency': base_frequency,
        'resonance_quality': 0.77  # Water element has flowing, adaptive resonance
    }

def get_icosahedron_aspects() -> Dict[str, Any]:
    """
    Get the metaphysical and elemental aspects associated with the icosahedron.
    
    Returns:
        Dictionary containing aspect properties
    """
    # Water element aspects associated with icosahedron
    aspects = {
        'element': 'Water',
        'qualities': [
            'Emotion',
            'Intuition',
            'Fluidity',
            'Adaptability',
            'Healing',
            'Purification',
            'Empathy'
        ],
        'chakra': 'Sacral',
        'direction': 'West',
        'season': 'Autumn',
        'state_of_matter': 'Liquid',
        'platonic_number': 20,  # Number of faces
        'sense': 'Taste',
        'consciousness_state': 'Flow State',
        'color': 'Blue',
        'taste': 'Salty',
        'symbolic_creature': 'Dolphin',
        'associated_sephiroth': ['Kether', 'Chesed', 'Geburah'],
        'gateway_key': 'Fourth Gateway',
        'vibration': 'Flowing',
        'musical_note': 'D'
    }
    
    # Physiological properties
    physical_aspects = {
        'body_system': 'Circulatory and Lymphatic',
        'organs': ['Heart', 'Kidneys', 'Bladder', 'Blood'],
        'glands': ['Gonads'],
        'psychological_traits': [
            'Emotional',
            'Intuitive',
            'Empathetic',
            'Receptive',
            'Adaptive',
            'Flowing'
        ]
    }
    
    # Combination aspects with other elements
    combined_aspects = {
        'water_fire': 'Emotional Power',
        'water_earth': 'Growth',
        'water_air': 'Emotional Intelligence',
        'water_aether': 'Spiritual Intuition'
    }
    
    # Sacred geometry connections
    geometry_aspects = {
        'primary_shape': 'Triangle',
        'angle_sum': 180,  # Sum of angles in triangle
        'polygon_faces': 'Equilateral Triangles',
        'dual_platonic': 'Dodecahedron',
        'associated_flower_of_life_points': 12,
        'phi_ratio_component': True,  # Contains golden ratio
        'vesica_pisces_connection': 'Flow pathways'
    }
    
    return {
        'general': aspects,
        'physical': physical_aspects,
        'combinations': combined_aspects,
        'geometry': geometry_aspects
    }

def encode_icosahedron_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Encode specific patterns for the icosahedron based on its aspects.
    
    Args:
        pattern_type: The type of pattern to encode ('resonance', 'gateway', 'element')
        
    Returns:
        Dictionary containing encoded pattern
    """
    patterns = {
        'resonance': {
            'wave_pattern': [1, 1, 2, 3, 5, 8, 13, 21],  # Fibonacci sequence
            'frequency_ratios': [1, 1.618, 2.618, 4.236, 6.854],  # Golden ratio progression
            'node_activations': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],  # Binary activation sequence
            'geometric_sequence': [12, 20, 30, 12]  # Vertex, face, edge, vertex count
        },
        'gateway': {
            'key_sequence': [20, 12, 30, 20],  # Face, vertex, edge, face count
            'activation_pattern': [1, 1, 0, 1, 0, 1, 1, 0],  # Water flow pattern
            'harmonic_intervals': ['perfect fourth', 'major sixth', 'minor third'],
            'symbol_sequence': ['triangle', 'water', 'flow', 'emotion']
        },
        'element': {
            'water_pattern': [3, 6, 9, 12, 20],
            'flow_levels': [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6],  # Flowing wave pattern
            'emotion_sequence': [1, 3, 6, 10, 15, 21, 28],  # Triangular numbers
            'color_spectrum': ['dark blue', 'turquoise', 'aqua', 'teal', 'cyan', 'light blue']
        },
        'consciousness': {
            'flow_state_pattern': [20, 5, 20, 5],
            'intuition_levels': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5],  # Flowing wave
            'feeling_sequence': [3, 5, 8, 13, 21, 34],  # Fibonacci derivatives
            'activation_thresholds': [0.3, 0.4, 0.5, 0.6, 0.7]
        }
    }
    
    if pattern_type in patterns:
        return patterns[pattern_type]
    else:
        return patterns  # Return all patterns if type not specified

def visualize_icosahedron(icosahedron: Dict[str, Any],
                         show_aspects: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the icosahedron.
    
    Args:
        icosahedron: Dictionary containing icosahedron geometry
        show_aspects: Whether to show aspect-related coloring
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = icosahedron['vertices']
    faces = icosahedron['faces']
    
    # Create face polygons
    polygons = []
    for face in faces:
        polygons.append([vertices[i] for i in face])
    
    # Set colors based on aspects if requested
    face_colors = ['b'] * len(faces)  # Default water color
    alpha = 0.6
    
    if show_aspects:
        aspects = get_icosahedron_aspects()
        # Create color gradient based on aspects
        blues = [
            '#00008B', '#0000CD', '#0000FF', '#1E90FF', 
            '#00BFFF', '#87CEEB', '#87CEFA', '#ADD8E6',
            '#B0E0E6', '#5F9EA0', '#00CED1', '#48D1CC',
            '#40E0D0', '#00FFFF', '#E0FFFF', '#AFEEEE',
            '#7FFFD4', '#66CDAA', '#20B2AA', '#008B8B'
        ]
        face_colors = blues
        alpha = 0.7
    
    # Add faces
    ax.add_collection3d(Poly3DCollection(polygons, alpha=alpha, facecolors=face_colors, linewidths=1, edgecolors='k'))
    
    # Plot vertices
    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='k', s=50)
    
    # Plot edges
    edges = icosahedron['edges']
    for edge in edges:
        line = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')
    
    # Add labels if showing aspects
    if show_aspects:
        center = icosahedron['center']
        ax.text(center[0], center[1], center[2] + icosahedron['radius'], 
               "Water Element", color='blue', fontsize=12, ha='center')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Icosahedron (Water Element)')
    
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

def embed_icosahedron_in_field(field_array: np.ndarray, 
                              center: Tuple[float, float, float],
                              edge_length: float,
                              strength: float = 1.0,
                              pattern_type: str = 'resonance') -> np.ndarray:
    """
    Embed an icosahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        edge_length: Length of icosahedron edges
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
    
    # Generate icosahedron
    icosahedron = generate_icosahedron(center, edge_length)
    vertices = icosahedron['vertices']
    radius = icosahedron['radius']
    
    # Get encoded pattern
    pattern = encode_icosahedron_pattern(pattern_type)
    
    # Create a field influence based on distance to icosahedron parts
    influence = np.zeros_like(field_array, dtype=float)
    
    # Influence from vertices
    for i, vertex in enumerate(vertices):
        # Calculate distance from each point to vertex
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        
        # Create flowing wave influence - characteristic of water element
        # Pattern intensity varies by vertex - water has flowing patterns
        if pattern_type == 'element':
            intensity = pattern['flow_levels'][i % len(pattern['flow_levels'])]
        else:
            intensity = 0.7 + 0.3 * np.sin(i * np.pi / 6)  # Sinusoidal variation
            
        # Distance falloff - more gradual with wave-like properties for water
        wave_factor = 0.8 + 0.2 * np.sin(dist * 2 * np.pi / radius)  # Add waviness
        falloff = np.exp(-dist / (edge_length * 0.7)) * wave_factor
        influence += falloff * intensity
    
    # Influence from faces (important for water element - creates flow planes)
    faces = icosahedron['faces']
    for i, face in enumerate(faces):
        # Get face vertices
        face_vertices = [vertices[j] for j in face]
        face_center = np.mean(face_vertices, axis=0)
        
        # Create normal vector to face
        v1 = face_vertices[1] - face_vertices[0]
        v2 = face_vertices[2] - face_vertices[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate distance from each point to face center
        dist_to_center = np.sqrt((x - face_center[0])**2 + (y - face_center[1])**2 + (z - face_center[2])**2)
        
        # Create flowing pattern from face
        face_influence = np.exp(-dist_to_center / (edge_length * 1.0))
        
        # Add wave pattern - characteristic of water
        wave_pattern = 0.7 + 0.3 * np.sin(dist_to_center * 2 * np.pi / (edge_length * 0.5))
        face_influence *= wave_pattern
        
        # Apply flow pattern for water element
        if pattern_type == 'element' and i < len(pattern['flow_levels']):
            cycle_index = i % len(pattern['flow_levels'])
            face_influence *= pattern['flow_levels'][cycle_index]
        
        influence += face_influence * 0.8  # Faces have significant influence in water element
    
    # Normalize influence to [0, 1] range
    max_val = np.max(influence)
    if max_val > 0:
        influence = influence / max_val
    
    # Apply water element pattern (flowing, sinusoidal pattern)
    # Create flowing, wave-like pattern characteristic of water element
    grid_size = max(field_shape)
    flow_pattern = np.sin(2 * np.pi * (x + y) / (grid_size * 0.2)) * np.cos(2 * np.pi * (y + z) / (grid_size * 0.25))
    flow_pattern = 0.5 + 0.5 * flow_pattern  # Normalize to [0, 1]
    
    # Combine influence with flowing pattern - characteristic of water
    pattern_field = influence * (0.6 + 0.4 * flow_pattern)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern_field * strength)
    
    # Normalize field after modification if needed
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def get_base_glyph_elements(center: Tuple[float, float, float], edge_length: float) -> Dict[str, Any]:
    """
    Returns the geometric elements (vertices, lines) for a simple line art
    representation of a icosahedron.
    """
    solid_data = generate_icosahedron(center, edge_length) # Call your existing generator
    
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
    # Create an icosahedron
    center = (0, 0, 0)
    edge_length = 2.0
    
    icosahedron = generate_icosahedron(center, edge_length)
    
    # Calculate resonance properties
    resonance = calculate_icosahedron_resonance(icosahedron)
    print(f"Primary Frequency: {resonance['primary_frequency']:.2f} Hz")
    print(f"Harmonics: {[f'{h:.2f}' for h in resonance['harmonics']]}")
    
    # Get aspects
    aspects = get_icosahedron_aspects()
    print("\nIcosahedron Aspects:")
    print(f"Element: {aspects['general']['element']}")
    print(f"Qualities: {', '.join(aspects['general']['qualities'])}")
    print(f"Chakra: {aspects['general']['chakra']}")
    
    # Visualize
    fig = visualize_icosahedron(icosahedron, show_aspects=True)
    plt.show()