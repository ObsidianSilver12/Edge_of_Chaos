"""
Dodecahedron Module

This module implements the dodecahedron platonic solid.
The dodecahedron represents the aether (spirit) element and consists of
12 pentagonal faces, 30 edges, and 20 vertices.

Key functions:
- Generate precise dodecahedron geometry
- Calculate energy dynamics and harmonics
- Associate with aether element aspects
- Establish resonance patterns for dimensional gateways
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict, Any, List, Optional

def generate_dodecahedron(center: Tuple[float, float, float],
                         edge_length: float) -> Dict[str, Any]:
    """
    Generate a dodecahedron with precise geometric properties.
    
    Args:
        center: The (x, y, z) center of the dodecahedron
        edge_length: The length of each edge
        
    Returns:
        Dictionary containing the dodecahedron geometry and properties
    """
    # Golden ratio for dodecahedron construction
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate the radius of the circumscribed sphere
    # For a dodecahedron with edge length e, the radius is:
    # r = e * sqrt(3) * sqrt(5 + 2*sqrt(5)) / 4
    radius = edge_length * np.sqrt(3) * np.sqrt(5 + 2*np.sqrt(5)) / 4
    
    # Generate the 20 vertices of the dodecahedron
    vertices = []
    
    # Vertices from cube coordinates (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices.append([x, y, z])
    
    # Vertices from golden rectangles
    for i in [-1, 1]:
        for j in [-phi, phi]:
            # Cyclic permutation of coordinates
            vertices.append([0, i, j])
            vertices.append([i, j, 0])
            vertices.append([j, 0, i])
    
    vertices = np.array(vertices)
    
    # Scale to match specified edge length
    # First calculate the current edge length between adjacent vertices
    # and scale accordingly
    current_edge = np.min([
        np.linalg.norm(vertices[0] - vertices[16]),
        np.linalg.norm(vertices[12] - vertices[18])
    ])
    scale_factor = edge_length / current_edge
    vertices = vertices * scale_factor
    
    # Translate vertices to the specified center
    vertices = vertices + np.array(center)
    
    # Define the 12 pentagonal faces
    # Each face is defined by 5 vertex indices
    faces = [
        [0, 12, 14, 4, 8],    # Top face 1
        [0, 8, 10, 2, 16],    # Top face 2
        [0, 16, 18, 6, 12],   # Top face 3
        [1, 13, 15, 5, 9],    # Bottom face 1
        [1, 9, 11, 3, 17],    # Bottom face 2
        [1, 17, 19, 7, 13],   # Bottom face 3
        [2, 10, 11, 3, 19],   # Middle face 1
        [2, 19, 7, 6, 18],    # Middle face 2
        [3, 17, 1, 13, 15],   # Middle face 3
        [4, 14, 15, 5, 9],    # Middle face 4
        [4, 9, 11, 10, 8],    # Middle face 5
        [6, 7, 13, 15, 14]    # Middle face 6
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
    # For a dodecahedron with edge length e:
    volume = (15 + 7 * np.sqrt(5)) * edge_length**3 / 4
    surface_area = 3 * np.sqrt(25 + 10 * np.sqrt(5)) * edge_length**2
    
    # Calculate dihedral angle (angle between faces)
    dihedral_angle = np.arccos(-np.sqrt(5)/5)  # Approximately 116.57 degrees
    
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
        'element': 'Aether'
    }

def calculate_dodecahedron_resonance(dodecahedron: Dict[str, Any], 
                                    base_frequency: float = 528.0) -> Dict[str, Any]:
    """
    Calculate the resonance properties of the dodecahedron.
    
    Args:
        dodecahedron: Dictionary containing dodecahedron geometry
        base_frequency: The base frequency for dodecahedron (aether element)
        
    Returns:
        Dictionary containing resonance properties
    """
    # Aether element base frequency is typically 528 Hz (DNA repair frequency)
    edge_length = dodecahedron['edge_length']
    
    # Calculate primary frequency based on edge length
    # Using the relationship between frequency and size
    primary_frequency = base_frequency * (1 / edge_length)
    
    # Calculate harmonic frequencies
    # Aether element has transcendent harmonics, based on phi (golden ratio) powers
    phi = (1 + np.sqrt(5)) / 2
    harmonics = [primary_frequency * phi**n for n in range(7)]
    
    # Calculate resonance nodes (points of maximum vibration)
    # These correspond to key points on the dodecahedron
    vertices = dodecahedron['vertices']
    
    face_centers = []
    for face in dodecahedron['faces']:
        face_vertices = [vertices[i] for i in face]
        face_center = np.mean(face_vertices, axis=0)
        face_centers.append(face_center)
    
    edge_centers = []
    for edge in dodecahedron['edges']:
        edge_vertices = [vertices[i] for i in edge]
        edge_center = np.mean(edge_vertices, axis=0)
        edge_centers.append(edge_center)
    
    # Calculate energy distribution for resonance
    # Each vertex has specific energy pattern
    # Aether element has transcendent, balanced energy
    vertex_energies = [0.94, 0.98, 0.96, 0.95, 0.99, 0.97, 0.93, 0.96, 0.98, 0.99,
                       0.95, 0.94, 0.97, 0.98, 0.96, 0.95, 0.99, 0.97, 0.93, 0.98]
    
    return {
        'primary_frequency': primary_frequency,
        'harmonics': harmonics,
        'face_centers': face_centers,
        'edge_centers': edge_centers,
        'vertex_energies': vertex_energies,
        'element_frequency': base_frequency,
        'resonance_quality': 0.98  # Aether element has highest resonance quality
    }

def get_dodecahedron_aspects() -> Dict[str, Any]:
    """
    Get the metaphysical and elemental aspects associated with the dodecahedron.
    
    Returns:
        Dictionary containing aspect properties
    """
    # Aether element aspects associated with dodecahedron
    aspects = {
        'element': 'Aether',
        'qualities': [
            'Spirit',
            'Transcendence',
            'Unity',
            'Consciousness',
            'Enlightenment',
            'Divine Connection',
            'Integration'
        ],
        'chakra': 'Crown',
        'direction': 'Center/Above',
        'season': 'Eternity',
        'state_of_matter': 'Plasma/Energy',
        'platonic_number': 12,  # Number of faces
        'sense': 'Intuition',
        'consciousness_state': 'Aware State',
        'color': 'Violet/White',
        'taste': None,  # Transcends physical taste
        'symbolic_creature': 'Phoenix',
        'associated_sephiroth': ['Hod', 'Netzach', 'Chesed', 'Daath', 'Geburah'],
        'gateway_key': 'Fifth Gateway',
        'vibration': 'Highest',
        'musical_note': 'B'
    }
    
    # Physiological properties
    physical_aspects = {
        'body_system': 'Energetic and Consciousness',
        'organs': ['Pineal Gland', 'Brain', 'Nervous System', 'Etheric Body'],
        'glands': ['Pineal'],
        'psychological_traits': [
            'Spiritual',
            'Expansive',
            'Intuitive',
            'Transcendent',
            'Integrated',
            'Aware',
            'Enlightened'
        ]
    }
    
    # Combination aspects with other elements
    combined_aspects = {
        'aether_fire': 'Divine Will',
        'aether_earth': 'Materialization',
        'aether_air': 'Higher Consciousness',
        'aether_water': 'Spiritual Intuition'
    }
    
    # Sacred geometry connections
    geometry_aspects = {
        'primary_shape': 'Pentagon',
        'angle_sum': 540,  # Sum of angles in pentagon
        'polygon_faces': 'Regular Pentagons',
        'dual_platonic': 'Icosahedron',
        'associated_flower_of_life_points': 20,
        'phi_ratio_component': True,  # Contains golden ratio
        'metatrons_cube_component': True,
        'transcendent_qualities': 'Unity of All Elements'
    }
    
    return {
        'general': aspects,
        'physical': physical_aspects,
        'combinations': combined_aspects,
        'geometry': geometry_aspects
    }

def encode_dodecahedron_pattern(pattern_type: str) -> Dict[str, Any]:
    """
    Encode specific patterns for the dodecahedron based on its aspects.
    
    Args:
        pattern_type: The type of pattern to encode ('resonance', 'gateway', 'element')
        
    Returns:
        Dictionary containing encoded pattern
    """
    patterns = {
        'resonance': {
            'wave_pattern': [1.0, 1.618, 2.618, 4.236, 6.854, 11.09, 17.944],  # Phi (golden ratio) powers
            'frequency_ratios': [1.0, 1.618, 2.618, 4.236, 6.854],  # Golden ratio progression
            'node_activations': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],  # Binary activation sequence
            'geometric_sequence': [12, 30, 20, 12]  # Face, edge, vertex, face count
        },
        'gateway': {
            'key_sequence': [12, 20, 30, 12],  # Face, vertex, edge, face count
            'activation_pattern': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # Transcendent pattern
            'harmonic_intervals': ['perfect fifth', 'major seventh', 'octave'],
            'symbol_sequence': ['pentagon', 'aether', 'spirit', 'cosmos']
        },
        'element': {
            'aether_pattern': [5, 12, 20, 30],
            'unity_levels': [0.92, 0.94, 0.96, 0.98, 1.0, 0.98, 0.96, 0.94, 0.92, 0.94, 0.96, 0.98],  # Near-perfect harmony
            'transcendence_sequence': [1, 5, 12, 20, 30, 42],  # Dimensional progression
            'color_spectrum': ['violet', 'white', 'gold', 'silver', 'platinum', 'iridescent', 'ultraviolet', 'crystal', 'rainbow', 'clear', 'starlight', 'cosmic']
        },
        'consciousness': {
            'aware_state_pattern': [12, 7, 12, 7],
            'enlightenment_levels': [0.85, 0.88, 0.91, 0.94, 0.97, 1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.88],  # Unity consciousness
            'awareness_sequence': [1, 3, 7, 12, 20, 33, 54],  # Fibonacci variants
            'activation_thresholds': [0.7, 0.8, 0.9, 0.95, 1.0]
        }
    }
    
    if pattern_type in patterns:
        return patterns[pattern_type]
    else:
        return patterns  # Return all patterns if type not specified

def visualize_dodecahedron(dodecahedron: Dict[str, Any],
                         show_aspects: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the dodecahedron.
    
    Args:
        dodecahedron: Dictionary containing dodecahedron geometry
        show_aspects: Whether to show aspect-related coloring
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = dodecahedron['vertices']
    faces = dodecahedron['faces']
    
    # Create face polygons
    polygons = []
    for face in faces:
        polygons.append([vertices[i] for i in face])
    
    # Set colors based on aspects if requested
    face_colors = ['purple'] * len(faces)  # Default aether color
    alpha = 0.6
    
    if show_aspects:
        aspects = get_dodecahedron_aspects()
        # Create color gradient based on aspects
        purples = [
            '#800080', '#9370DB', '#9932CC', '#BA55D3', '#DDA0DD', 
            '#EE82EE', '#DA70D6', '#FF00FF', '#FF00FF', '#8A2BE2',
            '#9400D3', '#8B008B'
        ]
        face_colors = purples
        alpha = 0.7
    
    # Add faces
    ax.add_collection3d(Poly3DCollection(polygons, alpha=alpha, facecolors=face_colors, linewidths=1, edgecolors='k'))
    
    # Plot vertices
    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='k', s=50)
    
    # Plot edges
    edges = dodecahedron['edges']
    for edge in edges:
        line = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'k-')
    
    # Add labels if showing aspects
    if show_aspects:
        center = dodecahedron['center']
        ax.text(center[0], center[1], center[2] + dodecahedron['radius'], 
               "Aether Element", color='purple', fontsize=12, ha='center')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Dodecahedron (Aether Element)')
    
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

def embed_dodecahedron_in_field(field_array: np.ndarray, 
                               center: Tuple[float, float, float],
                               edge_length: float,
                               strength: float = 1.0,
                               pattern_type: str = 'resonance') -> np.ndarray:
    """
    Embed a dodecahedron pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        edge_length: Length of dodecahedron edges
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
    
    # Generate dodecahedron
    dodecahedron = generate_dodecahedron(center, edge_length)
    vertices = dodecahedron['vertices']
    radius = dodecahedron['radius']
    
    # Get encoded pattern
    pattern = encode_dodecahedron_pattern(pattern_type)
    
    # Create a field influence based on distance to dodecahedron parts
    influence = np.zeros_like(field_array, dtype=float)
    
    # Influence from vertices (transcendent for aether element)
    for i, vertex in enumerate(vertices):
        # Calculate distance from each point to vertex
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        
        # Create harmonious, transcendent influence
        # Pattern intensity varies by vertex - aether has harmonious patterns
        if pattern_type == 'element':
            intensity = pattern['unity_levels'][i % len(pattern['unity_levels'])]
        else:
            intensity = 0.9 + 0.1 * np.sin(i * np.pi / 10)  # Harmonic variation
            
        # Distance falloff - more gradual and harmonious for aether
        # Phi-based falloff creates golden ratio harmonics
        phi = (1 + np.sqrt(5)) / 2
        falloff = np.exp(-dist / (edge_length * phi))
        influence += falloff * intensity
    
    # Influence from faces (pentagonal nature important for aether)
    faces = dodecahedron['faces']
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio for pentagons
    
    for i, face in enumerate(faces):
        # Get face vertices
        face_vertices = [vertices[j] for j in face]
        face_center = np.mean(face_vertices, axis=0)
        
        # Calculate distance from each point to face center
        dist_to_center = np.sqrt((x - face_center[0])**2 + (y - face_center[1])**2 + (z - face_center[2])**2)
        
        # Create transcendent pattern from pentagon faces
        # Use golden ratio (phi) in the calculation for aether resonance
        face_influence = np.exp(-dist_to_center / (edge_length * phi))
        
        # Add fibonacci-based pattern - characteristic of aether
        harmonic_pattern = 0.8 + 0.2 * np.sin(dist_to_center * 2 * np.pi / (edge_length * phi))
        face_influence *= harmonic_pattern
        
        # Apply unity pattern for aether element
        if pattern_type == 'element' and i < len(pattern['unity_levels']):
            cycle_index = i % len(pattern['unity_levels'])
            face_influence *= pattern['unity_levels'][cycle_index]
        
        influence += face_influence * 1.0  # Faces have highest influence in aether element
    
    # Aether has a unified field effect - create harmony throughout the field
    # Calculate distance from each point to center of dodecahedron
    dist_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # Create harmonious field emanating from center
    center_influence = np.exp(-dist_to_center / (radius * 2.0))
    influence += center_influence * 0.5
    
    # Normalize influence to [0, 1] range
    max_val = np.max(influence)
    if max_val > 0:
        influence = influence / max_val
    
    # Apply aether element pattern (transcendent, harmonious pattern)
    # Create phi-based pattern characteristic of aether element
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    grid_size = max(field_shape)
    # Multiple overlapping harmonic patterns for transcendence
    phi_pattern = (
        np.sin(2 * np.pi * x / (grid_size * 0.1 * phi)) * 
        np.sin(2 * np.pi * y / (grid_size * 0.1 * phi**2)) * 
        np.sin(2 * np.pi * z / (grid_size * 0.1 * phi**3))
    )
    phi_pattern = 0.5 + 0.5 * phi_pattern  # Normalize to [0, 1]
    
    # Combine influence with phi pattern - characteristic of aether
    pattern_field = influence * (0.7 + 0.3 * phi_pattern)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern_field * strength)
    
    # Normalize field after modification if needed
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def get_base_glyph_elements(center: Tuple[float, float, float], edge_length: float) -> Dict[str, Any]:
    """
    Returns the geometric elements (vertices, lines) for a simple line art
    representation of a dodecahedron.
    """
    solid_data = generate_dodecahedron(center, edge_length) # Call your existing generator
    
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
    # Create a dodecahedron
    center = (0, 0, 0)
    edge_length = 2.0
    
    dodecahedron = generate_dodecahedron(center, edge_length)
    
    # Calculate resonance properties
    resonance = calculate_dodecahedron_resonance(dodecahedron)
    print(f"Primary Frequency: {resonance['primary_frequency']:.2f} Hz")
    print(f"Harmonics: {[f'{h:.2f}' for h in resonance['harmonics']]}")
    
    # Get aspects
    aspects = get_dodecahedron_aspects()
    print("\nDodecahedron Aspects:")
    print(f"Element: {aspects['general']['element']}")
    print(f"Qualities: {', '.join(aspects['general']['qualities'])}")
    print(f"Chakra: {aspects['general']['chakra']}")
    
    # Visualize
    fig = visualize_dodecahedron(dodecahedron, show_aspects=True)
    plt.show()