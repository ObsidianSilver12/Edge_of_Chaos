"""
Tree of Life (Sephiroth) Module

This module implements the sacred geometry pattern of the Tree of Life (Sephiroth)
with precise geometric proportions based on a 9×15 grid.

Key functions:
- Generate precise Tree of Life geometry with correct proportions
- Calculate energy distribution and flow along paths
- Establish relationships between Sephiroth nodes
- Create visualization with proper symmetry and dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.patches as patches

# Constants for the Tree of Life grid
GRID_WIDTH = 9  # 9 cm horizontal
GRID_HEIGHT = 15  # 15 cm vertical
CENTER_X = 4.5  # Center line at 4.5 cm

# Define Sephiroth positions based on the provided measurements
SEPHIROTH_POSITIONS = {
    # Center column
    'Kether': (CENTER_X, 1),
    'Daath': (CENTER_X, 4.75),  # 3.75 down from Kether
    'Tipareth': (CENTER_X, 9),  # 4.25 down from Daath
    'Yesod': (CENTER_X, 14),  # 5 down from Tipareth
    'Malkuth': (CENTER_X, 16),  # 2 down from Yesod (outside the main grid)
    
    # Left column (4.5 from center)
    'Binah': (0, 3),  # 2 down from Kether
    'Geburah': (0, 7.5),  # 4.5 down from Binah
    'Hod': (0, 12),  # 4.5 down from Geburah
    
    # Right column (4.5 from center)
    'Chokmah': (9, 3),  # 2 down from Kether
    'Chesed': (9, 7.5),  # 4.5 down from Chokmah
    'Netzach': (9, 12),  # 4.5 down from Chesed
}

# Define the paths between Sephiroth
PATHS = [
    # Pillar paths
    ('Kether', 'Tipareth'),
    ('Tipareth', 'Yesod'),
    ('Yesod', 'Malkuth'),
    ('Kether', 'Daath'),
    ('Daath', 'Tipareth'),
    
    # Top triangle
    ('Kether', 'Binah'),
    ('Kether', 'Chokmah'),
    ('Binah', 'Chokmah'),
    
    # Middle paths
    ('Binah', 'Daath'),
    ('Chokmah', 'Daath'),
    ('Binah', 'Tipareth'),
    ('Chokmah', 'Tipareth'),
    
    # Second triangle
    ('Binah', 'Geburah'),
    ('Chokmah', 'Chesed'),
    ('Geburah', 'Chesed'),
    
    # Path from Geburah and Chesed to Tipareth
    ('Geburah', 'Tipareth'),
    ('Chesed', 'Tipareth'),
    
    # Lower triangle
    ('Geburah', 'Hod'),
    ('Chesed', 'Netzach'),
    ('Hod', 'Netzach'),
    
    # Paths to Yesod
    ('Hod', 'Yesod'),
    ('Netzach', 'Yesod'),
    
    # Additional paths
    ('Tipareth', 'Hod'),
    ('Tipareth', 'Netzach')
]

# Sephiroth traditional colors and attributes
SEPHIROTH_ATTRIBUTES = {
    'Kether': {
        'color': 'white',
        'title': 'Crown',
        'element': None,
        'planet': None,
        'energy_level': 1.0
    },
    'Chokmah': {
        'color': 'grey',
        'title': 'Wisdom',
        'element': None,
        'planet': 'Uranus',
        'energy_level': 0.95
    },
    'Binah': {
        'color': 'black',
        'title': 'Understanding',
        'element': None,
        'planet': 'Saturn',
        'energy_level': 0.9
    },
    'Daath': {
        'color': 'lavender',
        'title': 'Knowledge',
        'element': None,
        'planet': None,
        'energy_level': 0.85
    },
    'Chesed': {
        'color': 'blue',
        'title': 'Mercy',
        'element': None,
        'planet': 'Jupiter',
        'energy_level': 0.8
    },
    'Geburah': {
        'color': 'red',
        'title': 'Severity',
        'element': None,
        'planet': 'Mars',
        'energy_level': 0.75
    },
    'Tipareth': {
        'color': 'yellow',
        'title': 'Beauty',
        'element': None,
        'planet': 'Sun',
        'energy_level': 0.7
    },
    'Netzach': {
        'color': 'green',
        'title': 'Victory',
        'element': None,
        'planet': 'Venus',
        'energy_level': 0.65
    },
    'Hod': {
        'color': 'orange',
        'title': 'Glory',
        'element': None,
        'planet': 'Mercury',
        'energy_level': 0.6
    },
    'Yesod': {
        'color': 'purple',
        'title': 'Foundation',
        'element': None,
        'planet': 'Moon',
        'energy_level': 0.55
    },
    'Malkuth': {
        'color': 'brown',
        'title': 'Kingdom',
        'element': 'Earth',
        'planet': 'Earth',
        'energy_level': 0.5
    }
}

# Path colors based on the diagram
PATH_COLORS = {
    # Default for most paths
    'default': 'white',
    
    # Special path colors from the diagram
    ('Kether', 'Chokmah'): 'white',
    ('Kether', 'Binah'): 'white',
    ('Binah', 'Chokmah'): 'white',
    
    ('Kether', 'Tipareth'): 'cyan',
    ('Kether', 'Daath'): 'cyan',
    ('Daath', 'Tipareth'): 'cyan',
    ('Binah', 'Daath'): 'cyan',
    ('Chokmah', 'Daath'): 'cyan',
    
    ('Binah', 'Geburah'): 'cyan',
    ('Geburah', 'Chesed'): 'yellow',
    ('Chokmah', 'Chesed'): 'cyan',
    
    ('Geburah', 'Tipareth'): 'yellow',
    ('Chesed', 'Tipareth'): 'yellow',
    
    ('Geburah', 'Hod'): 'red',
    ('Tipareth', 'Hod'): 'red',
    ('Tipareth', 'Netzach'): 'red',
    ('Chesed', 'Netzach'): 'red',
    ('Hod', 'Netzach'): 'red',
    
    ('Hod', 'Yesod'): 'white',
    ('Netzach', 'Yesod'): 'white',
    ('Tipareth', 'Yesod'): 'white',
    ('Yesod', 'Malkuth'): 'white'
}

def generate_tree_of_life(resolution: int = 100) -> Dict[str, Any]:
    """
    Generate a Tree of Life (Sephiroth) structure with precise geometric proportions.
    
    Args:
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the tree geometry and calculated properties
    """
    # Create a grid with the specified dimensions
    x = np.linspace(0, GRID_WIDTH, resolution)
    y = np.linspace(0, GRID_HEIGHT, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the energy field
    energy_field = np.zeros((resolution, resolution))
    
    # Create a dictionary to store Sephiroth nodes and properties
    sephiroth = {}
    
    # Generate each Sephirah node with its properties
    for name, pos in SEPHIROTH_POSITIONS.items():
        # Scale grid positions to the resolution
        grid_x = int(pos[0] / GRID_WIDTH * (resolution - 1))
        grid_y = int(pos[1] / GRID_HEIGHT * (resolution - 1))
        
        # Create a radial energy field around this Sephirah
        # Radius is proportional to the energy level of the Sephirah
        radius = SEPHIROTH_ATTRIBUTES[name]['energy_level'] * min(GRID_WIDTH, GRID_HEIGHT) / 10
        
        # Calculate distance from this Sephirah to all points in the grid
        distances = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
        
        # Create an energy field based on the distance (exponential decay)
        sephirah_energy = np.exp(-distances / radius)
        
        # Add this Sephirah's energy to the overall field
        energy_field += sephirah_energy
        
        # Store Sephirah properties
        sephiroth[name] = {
            'position': pos,
            'grid_position': (grid_x, grid_y),
            'radius': radius,
            'energy': SEPHIROTH_ATTRIBUTES[name]['energy_level'],
            'color': SEPHIROTH_ATTRIBUTES[name]['color'],
            'title': SEPHIROTH_ATTRIBUTES[name]['title']
        }
    
    # Normalize energy field
    if np.max(energy_field) > 0:
        energy_field = energy_field / np.max(energy_field)
    
    # Calculate paths and their properties
    path_data = []
    
    for start, end in PATHS:
        start_pos = SEPHIROTH_POSITIONS[start]
        end_pos = SEPHIROTH_POSITIONS[end]
        
        # Get the path color
        if (start, end) in PATH_COLORS:
            color = PATH_COLORS[(start, end)]
        elif (end, start) in PATH_COLORS:
            color = PATH_COLORS[(end, start)]
        else:
            color = PATH_COLORS['default']
        
        # Calculate path properties
        path_length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Calculate energy flow along this path
        start_energy = SEPHIROTH_ATTRIBUTES[start]['energy_level']
        end_energy = SEPHIROTH_ATTRIBUTES[end]['energy_level']
        energy_flow = start_energy - end_energy
        
        path_data.append({
            'start': start,
            'end': end,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': path_length,
            'energy_flow': energy_flow,
            'color': color
        })
    
    return {
        'sephiroth': sephiroth,
        'paths': path_data,
        'energy_field': energy_field,
        'grid_x': X,
        'grid_y': Y
    }

def calculate_sephiroth_relationships(tree_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate relationships and energy flows between Sephiroth.
    
    Args:
        tree_data: Tree of Life data from generate_tree_of_life()
        
    Returns:
        Dictionary with relationship data
    """
    sephiroth = tree_data['sephiroth']
    paths = tree_data['paths']
    
    # Calculate direct connections for each Sephirah
    connections = {name: [] for name in sephiroth.keys()}
    for path in paths:
        connections[path['start']].append(path['end'])
        connections[path['end']].append(path['start'])
    
    # Calculate resonance between pairs of Sephiroth
    resonance = {}
    for name1 in sephiroth.keys():
        for name2 in sephiroth.keys():
            if name1 != name2:
                # Positions
                pos1 = SEPHIROTH_POSITIONS[name1]
                pos2 = SEPHIROTH_POSITIONS[name2]
                
                # Calculate distance
                distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                
                # Calculate resonance based on distance and energy levels
                energy1 = SEPHIROTH_ATTRIBUTES[name1]['energy_level']
                energy2 = SEPHIROTH_ATTRIBUTES[name2]['energy_level']
                
                # Direct connections have stronger resonance
                connection_factor = 2.0 if name2 in connections[name1] else 1.0
                
                # Resonance formula: product of energies / distance, enhanced by direct connection
                res = (energy1 * energy2 / distance) * connection_factor
                
                resonance[(name1, name2)] = res
    
    # Calculate energy flow patterns
    pillar_flows = {
        'left': calculate_pillar_flow(['Binah', 'Geburah', 'Hod']),
        'middle': calculate_pillar_flow(['Kether', 'Daath', 'Tipareth', 'Yesod', 'Malkuth']),
        'right': calculate_pillar_flow(['Chokmah', 'Chesed', 'Netzach'])
    }
    
    # Calculate horizontal flow patterns
    horizontal_flows = {
        'top': calculate_horizontal_flow(['Binah', 'Kether', 'Chokmah']),
        'daath': calculate_horizontal_flow(['Binah', 'Daath', 'Chokmah']),
        'middle': calculate_horizontal_flow(['Geburah', 'Tipareth', 'Chesed']),
        'bottom': calculate_horizontal_flow(['Hod', 'Yesod', 'Netzach'])
    }
    
    # Calculate triangle resonances
    triangles = {
        'top': calculate_triangle_resonance(['Kether', 'Binah', 'Chokmah']),
        'second': calculate_triangle_resonance(['Binah', 'Geburah', 'Chesed', 'Chokmah']),
        'third': calculate_triangle_resonance(['Geburah', 'Hod', 'Netzach', 'Chesed']),
        'bottom': calculate_triangle_resonance(['Hod', 'Yesod', 'Netzach'])
    }
    
    return {
        'connections': connections,
        'resonance': resonance,
        'pillar_flows': pillar_flows,
        'horizontal_flows': horizontal_flows,
        'triangles': triangles
    }

def calculate_pillar_flow(sephiroth_list: List[str]) -> float:
    """
    Calculate the energy flow along a pillar.
    
    Args:
        sephiroth_list: List of Sephiroth along the pillar (from top to bottom)
        
    Returns:
        Flow strength value
    """
    if not sephiroth_list:
        return 0.0
    
    # Calculate energy gradient along the pillar
    energy_levels = [SEPHIROTH_ATTRIBUTES[s]['energy_level'] for s in sephiroth_list]
    gradient = sum([energy_levels[i] - energy_levels[i+1] for i in range(len(energy_levels)-1)])
    
    # Average energy level
    avg_energy = sum(energy_levels) / len(energy_levels)
    
    # Flow strength is proportional to gradient and average energy
    return gradient * avg_energy

def calculate_horizontal_flow(sephiroth_list: List[str]) -> float:
    """
    Calculate the energy flow along a horizontal line.
    
    Args:
        sephiroth_list: List of Sephiroth along the horizontal (from left to right)
        
    Returns:
        Flow balance value
    """
    if len(sephiroth_list) < 2:
        return 0.0
    
    # Calculate energy balance
    energy_levels = [SEPHIROTH_ATTRIBUTES[s]['energy_level'] for s in sephiroth_list]
    
    # For horizontal flows, we're more interested in balance than gradient
    if len(energy_levels) == 3:  # If there's a center point
        # Calculate how balanced the left and right sides are
        left_right_diff = abs(energy_levels[0] - energy_levels[2])
        center_avg_diff = abs(energy_levels[1] - (energy_levels[0] + energy_levels[2])/2)
        
        # Lower difference means better balance
        balance = 1.0 - (left_right_diff + center_avg_diff) / 2
    else:
        # Calculate average deviation from mean
        mean = sum(energy_levels) / len(energy_levels)
        deviation = sum([abs(e - mean) for e in energy_levels]) / len(energy_levels)
        
        # Lower deviation means better balance
        balance = 1.0 - deviation
    
    return balance

def calculate_triangle_resonance(sephiroth_list: List[str]) -> float:
    """
    Calculate the resonance within a triangle of Sephiroth.
    
    Args:
        sephiroth_list: List of Sephiroth forming the triangle
        
    Returns:
        Resonance value
    """
    if len(sephiroth_list) < 3:
        return 0.0
    
    # Calculate pairwise resonances
    total_resonance = 0.0
    pairs = 0
    
    for i in range(len(sephiroth_list)):
        for j in range(i+1, len(sephiroth_list)):
            s1 = sephiroth_list[i]
            s2 = sephiroth_list[j]
            
            # Positions
            pos1 = SEPHIROTH_POSITIONS[s1]
            pos2 = SEPHIROTH_POSITIONS[s2]
            
            # Distance
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Energy levels
            energy1 = SEPHIROTH_ATTRIBUTES[s1]['energy_level']
            energy2 = SEPHIROTH_ATTRIBUTES[s2]['energy_level']
            
            # Resonance formula: product of energies / distance
            resonance = energy1 * energy2 / distance
            total_resonance += resonance
            pairs += 1
    
    # Average resonance across all pairs
    if pairs > 0:
        avg_resonance = total_resonance / pairs
    else:
        avg_resonance = 0.0
    
    return avg_resonance

def visualize_tree_of_life(tree_data: Dict[str, Any], 
                         show_energy: bool = False,
                         show_grid: bool = True,
                         show_labels: bool = True) -> plt.Figure:
    """
    Create a visualization of the Tree of Life pattern.
    
    Args:
        tree_data: Tree of Life data from generate_tree_of_life()
        show_energy: Whether to show energy field
        show_grid: Whether to show the grid
        show_labels: Whether to show Sephiroth labels
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 20))
    
    # Extract data
    sephiroth = tree_data['sephiroth']
    paths = tree_data['paths']
    
    # Show energy field if requested
    if show_energy and 'energy_field' in tree_data:
        X = tree_data['grid_x']
        Y = tree_data['grid_y']
        energy = tree_data['energy_field']
        
        im = ax.imshow(energy.T, origin='lower', extent=[0, GRID_WIDTH, 0, GRID_HEIGHT],
                      cmap='viridis', alpha=0.3, aspect='auto')
        fig.colorbar(im, ax=ax, label='Energy')
    
    # Show grid if requested
    if show_grid:
        # Add grid lines
        for i in range(GRID_WIDTH + 1):
            ax.axvline(x=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for i in range(GRID_HEIGHT + 1):
            ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Draw paths
    for path in paths:
        start_pos = path['start_pos']
        end_pos = path['end_pos']
        
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
               color=path['color'], linewidth=2, alpha=0.8)
    
    # Draw Sephiroth nodes
    for name, data in sephiroth.items():
        pos = data['position']
        color = data['color']
        
        # Create circle
        circle = plt.Circle(pos, data['radius'], color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Add label if requested
        if show_labels:
            ax.text(pos[0], pos[1], name, ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=10)
    
    # Set axis limits with a bit of padding
    ax.set_xlim(-0.5, GRID_WIDTH + 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT + 0.5)
    
    # Add title and labels
    ax.set_title('Tree of Life (Sephiroth) Sacred Geometry')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    return fig

def embed_tree_of_life_in_field(field_array: np.ndarray, 
                              center: Tuple[float, float, float],
                              size: float,
                              strength: float = 1.0) -> np.ndarray:
    """
    Embed the Tree of Life pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates of the tree (x, y, z)
        size: Scale factor for the tree
        strength: Strength of the pattern (0.0 to 1.0)
        
    Returns:
        Modified field array with embedded pattern
    """
    field_shape = field_array.shape
    
    # Create a 2D version of the tree
    tree_data = generate_tree_of_life()
    
    # Extract grid positions for easier indexing
    tree_grid_x = np.linspace(0, GRID_WIDTH, field_shape[0])
    tree_grid_y = np.linspace(0, GRID_HEIGHT, field_shape[1])
    
    # Create coordinate grids for the 3D field
    x, y, z = np.meshgrid(
        np.arange(field_shape[0]),
        np.arange(field_shape[1]),
        np.arange(field_shape[2]),
        indexing='ij'
    )
    
    # Transform grid coordinates to be centered and scaled
    cx, cy, cz = center
    tx = (x - cx) / size + GRID_WIDTH / 2
    ty = (y - cy) / size + GRID_HEIGHT / 2
    
    # Get z-plane at center for 2D embedding
    z_center = int(cz)
    if z_center < 0 or z_center >= field_shape[2]:
        z_center = field_shape[2] // 2
    
    # Create an empty pattern array
    pattern = np.zeros_like(field_array, dtype=float)
    
    # Embed Sephiroth nodes
    for name, data in tree_data['sephiroth'].items():
        pos = data['position']
        radius = data['radius'] * size
        energy = data['energy']
        
        # Calculate distance from each point to this Sephirah
        # We're creating a spherical influence around each Sephirah node
        dist = np.sqrt((tx - pos[0])**2 + (ty - pos[1])**2 + ((z - cz) / size)**2)
        
        # Create a smooth field around the Sephirah with exponential decay
        sephirah_field = energy * np.exp(-dist / (radius * 0.5))
        
        # Add to the pattern
        pattern += sephirah_field
    
    # Embed paths
    for path in tree_data['paths']:
        start_pos = path['start_pos']
        end_pos = path['end_pos']
        
        # Create a vector from start to end
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Calculate distance from each point to the line segment
            # Project the vector from start_pos to each point onto the path vector
            vx = tx - start_pos[0]
            vy = ty - start_pos[1]
            
            # Projection length
            proj = (vx * dx + vy * dy)
            
            # Clamp to line segment
            proj = np.clip(proj, 0, length)
            
            # Closest point on the line
            closest_x = start_pos[0] + proj * dx
            closest_y = start_pos[1] + proj * dy
            
            # Distance to the closest point
            dist_to_path = np.sqrt((tx - closest_x)**2 + (ty - closest_y)**2 + ((z - cz) / size)**2)
            
            # Create a tubular field around the path
            path_width = size * 0.1
            path_field = np.exp(-dist_to_path / path_width)
            
            # Add to the pattern with lower intensity than Sephiroth
            pattern += path_field * 0.5
    
    # Normalize pattern to [0, 1] range
    if np.max(pattern) > 0:
        pattern = pattern / np.max(pattern)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + pattern * strength)
    
    # Normalize field
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def calculate_entropy(array: np.ndarray) -> float:
    """
    Calculate the entropy of a distribution.
    
    Args:
        array: Numpy array of values
        
    Returns:
        Entropy value
    """
    # Flatten and normalize
    flat = array.flatten()
    if np.sum(flat) > 0:
        flat = flat / np.sum(flat)
    
    # Calculate entropy
    entropy = 0
    for p in flat:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_tree_of_life_metrics(tree_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate various metrics and properties of the Tree of Life structure.
    
    Args:
        tree_data: Tree of Life data from generate_tree_of_life()
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate relationships
    relationships = calculate_sephiroth_relationships(tree_data)
    
    # Calculate sacred proportions
    proportions = {}
    
    # Calculate the ratio of heights (e.g., Kether to Tipareth vs Tipareth to Yesod)
    kether_pos = SEPHIROTH_POSITIONS['Kether']
    tipareth_pos = SEPHIROTH_POSITIONS['Tipareth']
    yesod_pos = SEPHIROTH_POSITIONS['Yesod']
    
    upper_height = tipareth_pos[1] - kether_pos[1]
    lower_height = yesod_pos[1] - tipareth_pos[1]
    height_ratio = upper_height / lower_height
    
    proportions['upper_lower_height_ratio'] = height_ratio
    
    # Calculate width to height ratio
    binah_pos = SEPHIROTH_POSITIONS['Binah']
    chokmah_pos = SEPHIROTH_POSITIONS['Chokmah']
    malkuth_pos = SEPHIROTH_POSITIONS['Malkuth']
    
    width = chokmah_pos[0] - binah_pos[0]
    height = malkuth_pos[1] - kether_pos[1]
    width_height_ratio = width / height
    
    proportions['width_height_ratio'] = width_height_ratio
    
    # Calculate Golden Ratio approximations
    # The Tree of Life often embodies the golden ratio (φ ≈ 1.618)
    phi = (1 + np.sqrt(5)) / 2
    proportions['phi_approximation'] = abs(height_ratio - phi) / phi
    
    # Calculate energy distribution metrics
    energy_field = tree_data['energy_field']
    energy_metrics = {
        'mean_energy': np.mean(energy_field),
        'max_energy': np.max(energy_field),
        'energy_std': np.std(energy_field),
        'energy_entropy': calculate_entropy(energy_field)
    }
    
    # Calculate path metrics
    paths = tree_data['paths']
    
    path_metrics = {
        'total_paths': len(paths),
        'mean_path_length': np.mean([p['length'] for p in paths]),
        'total_path_length': sum([p['length'] for p in paths]),
        'mean_energy_flow': np.mean([abs(p['energy_flow']) for p in paths])
    }
    
    return {
        'relationships': relationships,
        'proportions': proportions,
        'energy_metrics': energy_metrics,
        'path_metrics': path_metrics
    }

def generate_tree_of_life_3d(resolution: int = 50) -> Dict[str, Any]:
    """
    Generate a 3D representation of the Tree of Life structure.
    
    Args:
        resolution: The resolution of the generated grid
        
    Returns:
        Dictionary containing the 3D tree geometry and calculated properties
    """
    # Create a 3D grid
    x = np.linspace(0, GRID_WIDTH, resolution)
    y = np.linspace(0, GRID_HEIGHT, resolution)
    z = np.linspace(-GRID_WIDTH/2, GRID_WIDTH/2, resolution)  # Z dimension centered at 0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize 3D energy field
    energy_field_3d = np.zeros((resolution, resolution, resolution))
    
    # Create a dictionary to store 3D Sephiroth nodes
    sephiroth_3d = {}
    
    # The z-coordinate for each Sephirah (centered at z=0 by default)
    z_coords = {name: 0.0 for name in SEPHIROTH_POSITIONS.keys()}
    
    # Optionally, we could position specific Sephiroth at different z-coordinates
    # to create a more complex 3D structure
    # For example, we could place Sephiroth in the three pillars at different z depths
    # z_coords['Binah'] = -GRID_WIDTH/4  # Left pillar pushed back
    # z_coords['Geburah'] = -GRID_WIDTH/4
    # z_coords['Hod'] = -GRID_WIDTH/4
    #
    # z_coords['Chokmah'] = GRID_WIDTH/4  # Right pillar pushed forward
    # z_coords['Chesed'] = GRID_WIDTH/4
    # z_coords['Netzach'] = GRID_WIDTH/4
    
    # Generate each 3D Sephirah node
    for name, pos_2d in SEPHIROTH_POSITIONS.items():
        # 3D position (x, y, z)
        pos_3d = (pos_2d[0], pos_2d[1], z_coords[name])
        
        # Create a spherical energy field around this Sephirah
        radius = SEPHIROTH_ATTRIBUTES[name]['energy_level'] * min(GRID_WIDTH, GRID_HEIGHT) / 10
        
        # Calculate distance from this Sephirah to all points in the 3D grid
        distances = np.sqrt((X - pos_3d[0])**2 + (Y - pos_3d[1])**2 + (Z - pos_3d[2])**2)
        
        # Create an energy field based on the distance (exponential decay)
        sephirah_energy = np.exp(-distances / radius)
        
        # Add this Sephirah's energy to the overall field
        energy_field_3d += sephirah_energy
        
        # Store 3D Sephirah properties
        sephiroth_3d[name] = {
            'position': pos_3d,
            'radius': radius,
            'energy': SEPHIROTH_ATTRIBUTES[name]['energy_level'],
            'color': SEPHIROTH_ATTRIBUTES[name]['color'],
            'title': SEPHIROTH_ATTRIBUTES[name]['title']
        }
    
    # Calculate 3D paths
    path_data_3d = []
    
    for start, end in PATHS:
        start_pos_2d = SEPHIROTH_POSITIONS[start]
        end_pos_2d = SEPHIROTH_POSITIONS[end]
        
        # Convert to 3D positions
        start_pos_3d = (start_pos_2d[0], start_pos_2d[1], z_coords[start])
        end_pos_3d = (end_pos_2d[0], end_pos_2d[1], z_coords[end])
        
        # Get path color
        if (start, end) in PATH_COLORS:
            color = PATH_COLORS[(start, end)]
        elif (end, start) in PATH_COLORS:
            color = PATH_COLORS[(end, start)]
        else:
            color = PATH_COLORS['default']
        
        # Calculate 3D path length
        path_length = np.sqrt(
            (end_pos_3d[0] - start_pos_3d[0])**2 + 
            (end_pos_3d[1] - start_pos_3d[1])**2 + 
            (end_pos_3d[2] - start_pos_3d[2])**2
        )
        
        # Calculate energy flow along this path
        start_energy = SEPHIROTH_ATTRIBUTES[start]['energy_level']
        end_energy = SEPHIROTH_ATTRIBUTES[end]['energy_level']
        energy_flow = start_energy - end_energy
        
        # Create tubular energy field along the path
        # Calculate direction vector
        dx = end_pos_3d[0] - start_pos_3d[0]
        dy = end_pos_3d[1] - start_pos_3d[1]
        dz = end_pos_3d[2] - start_pos_3d[2]
        
        if path_length > 0:
            # Normalize direction
            dx, dy, dz = dx/path_length, dy/path_length, dz/path_length
            
            # For every point in the grid, calculate its distance to the line segment
            # Create vectors from start point to each grid point
            vx = X - start_pos_3d[0]
            vy = Y - start_pos_3d[1]
            vz = Z - start_pos_3d[2]
            
            # Project onto the path direction
            proj = vx*dx + vy*dy + vz*dz
            
            # Clamp to the path length
            proj = np.clip(proj, 0, path_length)
            
            # Find the closest point on the path
            closest_x = start_pos_3d[0] + proj * dx
            closest_y = start_pos_3d[1] + proj * dy
            closest_z = start_pos_3d[2] + proj * dz
            
            # Calculate distance to the closest point
            dist_to_path = np.sqrt(
                (X - closest_x)**2 + 
                (Y - closest_y)**2 + 
                (Z - closest_z)**2
            )
            
            # Create tubular field around the path
            path_width = min(GRID_WIDTH, GRID_HEIGHT) / 30
            path_field = np.exp(-dist_to_path / path_width)
            
            # Add to the 3D energy field
            energy_field_3d += path_field * 0.3  # Lower intensity than Sephiroth
        
        path_data_3d.append({
            'start': start,
            'end': end,
            'start_pos': start_pos_3d,
            'end_pos': end_pos_3d,
            'length': path_length,
            'energy_flow': energy_flow,
            'color': color
        })
    
    # Normalize 3D energy field
    if np.max(energy_field_3d) > 0:
        energy_field_3d = energy_field_3d / np.max(energy_field_3d)
    
    return {
        'sephiroth': sephiroth_3d,
        'paths': path_data_3d,
        'energy_field': energy_field_3d,
        'grid_x': X,
        'grid_y': Y,
        'grid_z': Z
    }

def visualize_tree_of_life_3d(tree_data_3d: Dict[str, Any], show_labels: bool = True) -> None:
    """
    Create a 3D visualization of the Tree of Life.
    This function uses matplotlib's 3D plotting capabilities.
    
    Args:
        tree_data_3d: Tree of Life 3D data from generate_tree_of_life_3d()
        show_labels: Whether to show Sephiroth labels
    """
    # Extract 3D data
    sephiroth = tree_data_3d['sephiroth']
    paths = tree_data_3d['paths']
    
    # Create a 3D figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw paths as 3D lines
    for path in paths:
        start_pos = path['start_pos']
        end_pos = path['end_pos']
        
        ax.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            [start_pos[2], end_pos[2]],
            color=path['color'],
            linewidth=2,
            alpha=0.8
        )
    
    # Draw Sephiroth as 3D spheres
    for name, data in sephiroth.items():
        pos = data['position']
        color = data['color']
        radius = data['radius']
        
        # Create a sphere at this position
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + pos[0]
        y = radius * np.sin(u) * np.sin(v) + pos[1]
        z = radius * np.cos(v) + pos[2]
        
        # Plot the sphere
        ax.plot_surface(x, y, z, color=color, alpha=0.7)
        
        # Add label if requested
        if show_labels:
            ax.text(
                pos[0], pos[1], pos[2],
                name,
                color='white',
                fontweight='bold',
                fontsize=10
            )
    
    # Set axis limits
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_zlim(-GRID_WIDTH/2, GRID_WIDTH/2)
    
    # Add title and labels
    ax.set_title('3D Tree of Life (Sephiroth) Sacred Geometry')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    ax.set_zlabel('Z Dimension')
    
    # Use equal aspect ratio for all axes
    # Set the aspect ratio to be equal, using the longest dimension as reference
    max_range = max(
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_WIDTH  # Z range is GRID_WIDTH
    )
    
    mid_x = GRID_WIDTH / 2
    mid_y = GRID_HEIGHT / 2
    mid_z = 0
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_tree_of_life_resonance_frequencies() -> Dict[str, Dict[str, float]]:
    """
    Generate resonance frequencies for the Sephiroth based on sacred ratios.
    These can be used for sound generation or harmonic analysis.
    
    Returns:
        Dictionary of Sephiroth frequencies and harmonic relationships
    """
    # Base frequency (can be adjusted based on musical preferences)
    # 432 Hz is often considered a sacred frequency
    base_frequency = 432.0  # Hz
    
    # Calculate frequencies for each Sephirah based on its energy level
    # Higher Sephiroth have higher frequencies
    frequencies = {}
    
    for name, attrs in SEPHIROTH_ATTRIBUTES.items():
        # Scale frequencies based on energy level
        # This creates a harmonic relationship between Sephiroth
        energy = attrs['energy_level']
        
        # Calculate primary frequency for this Sephirah
        # Here we're using a logarithmic scale to create octave relationships
        freq = base_frequency * (2 ** (energy - 0.5))  # Centering around base_frequency
        
        # Calculate harmonic overtones (first 5 harmonics)
        harmonics = [freq * (i + 1) for i in range(5)]
        
        frequencies[name] = {
            'primary': freq,
            'harmonics': harmonics,
            'energy_level': energy
        }
    
    # Calculate resonance between pairs of Sephiroth
    resonance = {}
    
    for name1, data1 in frequencies.items():
        resonance[name1] = {}
        
        for name2, data2 in frequencies.items():
            if name1 != name2:
                # Calculate frequency ratio
                ratio = data1['primary'] / data2['primary']
                
                # Normalize to be within octave (1.0 to 2.0)
                while ratio < 1.0:
                    ratio *= 2.0
                while ratio >= 2.0:
                    ratio /= 2.0
                
                # Calculate consonance (how "pleasing" the ratio is)
                # Simple ratios (1:1, 3:2, 4:3, etc.) are more consonant
                # Here we check how close the ratio is to common musical intervals
                intervals = {
                    1.0: "Unison",
                    1.067: "Minor Second",
                    1.125: "Major Second",
                    1.2: "Minor Third",
                    1.25: "Major Third",
                    1.333: "Perfect Fourth",
                    1.4: "Augmented Fourth/Diminished Fifth",
                    1.5: "Perfect Fifth",
                    1.6: "Minor Sixth",
                    1.667: "Major Sixth",
                    1.8: "Minor Seventh",
                    1.875: "Major Seventh",
                    2.0: "Octave"
                }
                
                # Find closest interval
                closest_interval = min(intervals.keys(), key=lambda k: abs(k - ratio))
                interval_name = intervals[closest_interval]
                interval_error = abs(closest_interval - ratio)
                
                # Calculate consonance (0.0 to 1.0, higher is more consonant)
                if closest_interval in [1.0, 1.5, 2.0]:  # Perfect consonances
                    consonance = 1.0 - interval_error * 10
                elif closest_interval in [1.25, 1.333, 1.667]:  # Imperfect consonances
                    consonance = 0.8 - interval_error * 8
                else:  # Dissonances
                    consonance = 0.5 - interval_error * 5
                
                consonance = max(0.0, min(1.0, consonance))  # Clamp to [0, 1]
                
                resonance[name1][name2] = {
                    'ratio': ratio,
                    'closest_interval': closest_interval,
                    'interval_name': interval_name,
                    'consonance': consonance
                }
    
    return {
        'frequencies': frequencies,
        'resonance': resonance
    }

# Example usage
if __name__ == "__main__":
    # Generate Tree of Life structure
    tree_data = generate_tree_of_life()
    
    # Visualize
    fig = visualize_tree_of_life(tree_data, show_energy=True, show_grid=True)
    plt.savefig('tree_of_life.png', dpi=300, bbox_inches='tight')
    
    # Calculate metrics
    metrics = calculate_tree_of_life_metrics(tree_data)
    print("\nTree of Life Metrics:")
    
    print("\nSacred Proportions:")
    for key, value in metrics['proportions'].items():
        print(f"{key}: {value:.4f}")
    
    print("\nEnergy Metrics:")
    for key, value in metrics['energy_metrics'].items():
        print(f"{key}: {value:.4f}")
    
    print("\nPath Metrics:")
    for key, value in metrics['path_metrics'].items():
        print(f"{key}: {value:.4f}")
    
    # Generate and print resonance frequencies
    frequencies = generate_tree_of_life_resonance_frequencies()
    
    print("\nSephiroth Frequencies:")
    for name, data in frequencies['frequencies'].items():
        print(f"{name}: {data['primary']:.2f} Hz")
    
    # Generate 3D version
    tree_data_3d = generate_tree_of_life_3d()
    
    # Visualize 3D
    visualize_tree_of_life_3d(tree_data_3d)