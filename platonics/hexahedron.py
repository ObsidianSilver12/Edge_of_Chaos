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
    
    # Create face