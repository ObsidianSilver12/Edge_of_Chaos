"""
Vector Equilibrium Module

This module implements the vector equilibrium (cuboctahedron) sacred geometry pattern.
The vector equilibrium represents a unique geometric form where all vectors connecting
the center to the vertices are of equal length, and all vectors connecting adjacent 
vertices are also of equal length.

Key functions:
- Generate precise vector equilibrium geometry
- Calculate sacred ratios within the pattern
- Calculate energy distribution at vertices and edges
- Generate field embedding for the pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Any, Optional

def generate_vector_equilibrium(center: Tuple[float, float, float], 
                             radius: float) -> Dict[str, Any]:
    """
    Generate a vector equilibrium (cuboctahedron) with the given center and radius.
    
    Args:
        center: The (x, y, z) center of the vector equilibrium
        radius: The radius (distance from center to vertices)
        
    Returns:
        Dictionary containing the vector equilibrium geometry
    """
    # The vector equilibrium (cuboctahedron) has 12 vertices
    # These are positioned at the following coordinates (relative to center)
    # 6 vertices are at positions (±radius, 0, 0), (0, ±radius, 0), (0, 0, ±radius)
    # 8 vertices are at positions (±radius/√2, ±radius/√2, ±radius/√2)
    
    # Create the 12 vertices
    sqrt2 = np.sqrt(2)
    vertices = []
    
    # 6 vertices at the ends of the coordinate axes
    # (±radius, 0, 0)
    vertices.append((center[0] + radius, center[1], center[2]))
    vertices.append((center[0] - radius, center[1], center[2]))
    
    # (0, ±radius, 0)
    vertices.append((center[0], center[1] + radius, center[2]))
    vertices.append((center[0], center[1] - radius, center[2]))
    
    # (0, 0, ±radius)
    vertices.append((center[0], center[1], center[2] + radius))
    vertices.append((center[0], center[1], center[2] - radius))
    
    # Define the edges as pairs of vertex indices
    edges = [
        # Square in the xy-plane at z=0
        (0, 2), (2, 1), (1, 3), (3, 0),
        # Square in the xz-plane at y=0
        (0, 4), (4, 1), (1, 5), (5, 0),
        # Square in the yz-plane at x=0
        (2, 4), (4, 3), (3, 5), (5, 2)
    ]
    
    # Define faces (8 triangular faces and 6 square faces)
    triangular_faces = [
        (0, 2, 4), (1, 3, 5),  # Top and bottom triangles
        (0, 4, 3), (0, 2, 5),  # Side triangles
        (1, 5, 2), (1, 3, 4)   # Side triangles
    ]
    
    square_faces = [
        (0, 2, 1, 3),  # xy-plane at z=0
        (0, 4, 1, 5),  # xz-plane at y=0
        (2, 4, 3, 5)   # yz-plane at x=0
    ]
    
    # Calculate energy points (these are at vertices, edge midpoints, and face centers)
    energy_points = []
    
    # Vertices are the primary energy points
    energy_points.extend(vertices)
    
    # Edge midpoints are secondary energy points
    for edge in edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        midpoint = tuple((v1[i] + v2[i])/2 for i in range(3))
        energy_points.append(midpoint)
    
    # Face centers are tertiary energy points
    # Triangular face centers
    for face in triangular_faces:
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        center_point = tuple((v1[i] + v2[i] + v3[i])/3 for i in range(3))
        energy_points.append(center_point)
    
    # Square face centers
    for face in square_faces:
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        v4 = vertices[face[3]]
        center_point = tuple((v1[i] + v2[i] + v3[i] + v4[i])/4 for i in range(3))
        energy_points.append(center_point)
    
    return {
        'center': center,
        'radius': radius,
        'vertices': vertices,
        'edges': edges,
        'triangular_faces': triangular_faces,
        'square_faces': square_faces,
        'energy_points': energy_points
    }

def calculate_vector_equilibrium_sacred_ratios(ve_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the sacred ratios and proportions in the vector equilibrium.
    
    Args:
        ve_data: Dictionary containing vector equilibrium geometry data
        
    Returns:
        Dictionary of calculated sacred ratios
    """
    radius = ve_data['radius']
    vertices = ve_data['vertices']
    center = ve_data['center']
    
    # Calculate the edge length (distance between adjacent vertices)
    # In a vector equilibrium, this equals radius * sqrt(2)
    # Take the first edge and calculate its length
    edge = ve_data['edges'][0]
    v1 = vertices[edge[0]]
    v2 = vertices[edge[1]]
    edge_length = np.sqrt(sum((v1[i] - v2[i])**2 for i in range(3)))
    
    # Calculate the ratio of edge length to radius
    edge_radius_ratio = edge_length / radius  # Should be close to sqrt(2)
    
    # Calculate the volume of the vector equilibrium
    # Volume = (5/3) * sqrt(2) * radius^3
    volume = (5/3) * np.sqrt(2) * radius**3
    
    # Calculate the surface area of the vector equilibrium
    # Surface area = 4 * sqrt(3) * radius^2
    surface_area = 4 * np.sqrt(3) * radius**2
    
    # Calculate the dihedral angle (angle between adjacent faces)
    # In a vector equilibrium, the dihedral angle is approximately 144.7 degrees
    dihedral_angle = np.arccos(-1/3) * 180 / np.pi  # Convert to degrees
    
    # Calculate the ratio of the radius of the circumscribed sphere (which equals radius)
    # to the radius of the inscribed sphere
    inscribed_radius = radius * np.sqrt(2)/2
    sphere_radius_ratio = radius / inscribed_radius
    
    # Calculate relationship to golden ratio (phi)
    phi = (1 + np.sqrt(5)) / 2
    phi_approximation = edge_length / (radius * (phi - 1))
    
    return {
        'edge_length': edge_length,
        'edge_radius_ratio': edge_radius_ratio,
        'volume': volume,
        'surface_area': surface_area,
        'dihedral_angle': dihedral_angle,
        'sphere_radius_ratio': sphere_radius_ratio,
        'phi_approximation': phi_approximation
    }

def calculate_vector_equilibrium_energy_distribution(ve_data: Dict[str, Any], 
                                                 resolution: int = 50) -> Dict[str, Any]:
    """
    Calculate the energy distribution within and around the vector equilibrium.
    
    Args:
        ve_data: Dictionary containing vector equilibrium geometry data
        resolution: Resolution of the 3D grid for energy calculation
        
    Returns:
        Dictionary containing energy distribution data
    """
    center = ve_data['center']
    radius = ve_data['radius']
    vertices = ve_data['vertices']
    edges = ve_data['edges']
    energy_points = ve_data['energy_points']
    
    # Create a 3D grid for energy calculation
    x = np.linspace(center[0] - radius * 2, center[0] + radius * 2, resolution)
    y = np.linspace(center[1] - radius * 2, center[1] + radius * 2, resolution)
    z = np.linspace(center[2] - radius * 2, center[2] + radius * 2, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize energy field
    energy = np.zeros((resolution, resolution, resolution))
    
    # Calculate energy contribution from each vertex
    for vertex in vertices:
        dist = np.sqrt((X - vertex[0])**2 + (Y - vertex[1])**2 + (Z - vertex[2])**2)
        # Energy decreases with distance from the vertex
        vertex_energy = np.exp(-dist / (radius * 0.2))
        energy += vertex_energy
    
    # Calculate energy contribution from each edge
    for edge in edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Calculate the closest distance from each point to the line segment
        # This is a simplified approximation using sampling along the edge
        edge_energy = np.zeros_like(energy)
        for t in np.linspace(0, 1, 10):
            point = [v1[i] * (1-t) + v2[i] * t for i in range(3)]
            dist = np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2)
            edge_energy += np.exp(-dist / (radius * 0.2))
        
        energy += edge_energy / 20.0  # Reduced weight for edges
    
    # Calculate energy from special energy points
    for point in energy_points:
        dist = np.sqrt((X - point[0])**2 + (Y - point[1])**2 + (Z - point[2])**2)
        point_energy = np.exp(-dist / (radius * 0.15))
        energy += point_energy * 0.5  # Adjusting weight for energy points
    
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
        i, j, k = tuple(np.unravel_index(idx, energy.shape))
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

def embed_vector_equilibrium_in_field(field_array: np.ndarray, 
                                    center: Tuple[float, float, float],
                                    radius: float,
                                    strength: float = 1.0) -> np.ndarray:
    """
    Embed a vector equilibrium pattern into a 3D field array.
    
    Args:
        field_array: 3D numpy array representing the field
        center: Center coordinates (x, y, z)
        radius: Radius of the vector equilibrium
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
    
    # Generate vector equilibrium
    ve_data = generate_vector_equilibrium(center, radius)
    vertices = ve_data['vertices']
    edges = ve_data['edges']
    
    # Initialize energy field
    energy = np.zeros(field_shape)
    
    # Calculate energy from each vertex
    for vertex in vertices:
        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
        vertex_energy = np.exp(-dist / (radius * 0.2))
        energy += vertex_energy
    
    # Calculate energy from each edge
    for edge in edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        
        # Sample points along the edge
        for t in np.linspace(0, 1, 5):
            point = [v1[i] * (1-t) + v2[i] * t for i in range(3)]
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2 + (z - point[2])**2)
            energy += np.exp(-dist / (radius * 0.2)) * 0.5  # Half weight for edges
    
    # Normalize energy to [0, 1]
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Apply pattern to field with given strength
    modified_field = field_array * (1.0 + energy * strength)
    
    # Normalize field after modification
    if np.max(np.abs(modified_field)) > 0:
        modified_field = modified_field / np.max(np.abs(modified_field))
    
    return modified_field

def visualize_vector_equilibrium(ve_data: Dict[str, Any], 
                               show_energy: bool = False) -> plt.Figure:
    """
    Create a 3D visualization of the vector equilibrium structure.
    
    Args:
        ve_data: Dictionary containing vector equilibrium geometry data
        show_energy: Whether to show energy points
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    vertices = ve_data['vertices']
    edges = ve_data['edges']
    triangular_faces = ve_data['triangular_faces']
    square_faces = ve_data['square_faces']
    
    # Plot vertices
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]
    ax.scatter(x, y, z, color='blue', s=100, alpha=0.8)
    
    # Plot edges
    for edge in edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-', linewidth=2)
    
    # Plot triangular faces (semi-transparent)
    for face in triangular_faces:
        verts = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
        x = [v[0] for v in verts]
        y = [v[1] for v in verts]
        z = [v[2] for v in verts]
        
        # Add the first vertex again to close the polygon
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        
        ax.plot(x, y, z, 'r-', alpha=0.5)
        
        # Create a polygon (this is a bit hacky in matplotlib 3D)
        verts = list(zip(x, y, z))
        poly = Axes3D.art3d.Poly3DCollection([verts])
        poly.set_color('red')
        poly.set_alpha(0.2)
        ax.add_collection3d(poly)
    
    # Plot square faces (semi-transparent)
    for face in square_faces:
        verts = [vertices[face[0]], vertices[face[1]], 
                vertices[face[2]], vertices[face[3]]]
        x = [v[0] for v in verts]
        y = [v[1] for v in verts]
        z = [v[2] for v in verts]
        
        # Add the first vertex again to close the polygon
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        
        ax.plot(x, y, z, 'g-', alpha=0.5)
        
        # Create a polygon
        verts = list(zip(x, y, z))
        poly = Axes3D.art3d.Poly3DCollection([verts])
        poly.set_color('green')
        poly.set_alpha(0.2)
        ax.add_collection3d(poly)
    
    # Plot energy points if requested
    if show_energy and 'energy_points' in ve_data:
        energy_points = ve_data['energy_points']
        x = [p[0] for p in energy_points]
        y = [p[1] for p in energy_points]
        z = [p[2] for p in energy_points]
        ax.scatter(x, y, z, color='yellow', s=30, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Equilibrium (Cuboctahedron)')
    
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

def visualize_vector_equilibrium_energy(ve_data: Dict[str, Any], 
                                      energy_data: Dict[str, Any]) -> plt.Figure:
    """
    Visualize the energy distribution around the vector equilibrium structure.
    
    Args:
        ve_data: Dictionary containing vector equilibrium geometry data
        energy_data: Dictionary containing energy distribution data
        
    Returns:
        matplotlib Figure object for energy visualization
    """
    fig = plt.figure(figsize=(14, 5))
    
    # Create 3 subplots for energy slices in different planes
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Extract data
    energy = energy_data['energy']
    X = energy_data['grid_x']
    Y = energy_data['grid_y']
    Z = energy_data['grid_z']
    center = ve_data['center']
    
    # Find the indices closest to the center point in each dimension
    center_idx = [
        np.abs(X[:, 0, 0] - center[0]).argmin(),
        np.abs(Y[0, :, 0] - center[1]).argmin(),
        np.abs(Z[0, 0, :] - center[2]).argmin()
    ]
    
    # Create slices through the center in each plane
    slice_xy = energy[:, :, center_idx[2]]
    slice_xz = energy[:, center_idx[1], :]
    slice_yz = energy[center_idx[0], :, :]
    
    # Plot each slice
    im1 = ax1.imshow(slice_xy.T, cmap='viridis', 
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    origin='lower', aspect='equal')
    fig.colorbar(im1, ax=ax1, label='Energy')
    ax1.set_title('XY Plane (Z={:.2f})'.format(Z[0, 0, center_idx[2]]))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    im2 = ax2.imshow(slice_xz.T, cmap='viridis', 
                    extent=[X.min(), X.max(), Z.min(), Z.max()],
                    origin='lower', aspect='equal')
    fig.colorbar(im2, ax=ax2, label='Energy')
    ax2.set_title('XZ Plane (Y={:.2f})'.format(Y[0, center_idx[1], 0]))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    im3 = ax3.imshow(slice_yz.T, cmap='viridis', 
                    extent=[Y.min(), Y.max(), Z.min(), Z.max()],
                    origin='lower', aspect='equal')
    fig.colorbar(im3, ax=ax3, label='Energy')
    ax3.set_title('YZ Plane (X={:.2f})'.format(X[center_idx[0], 0, 0]))
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    
    plt.tight_layout()
    return fig

def get_base_glyph_elements(center: Tuple[float, float, float], radius: float) -> Dict[str, Any]:
    """
    Returns lines for Vector Equilibrium base glyph. Radius here is dist to vertex.
    """
    ve_data = generate_vector_equilibrium(center, radius)
    
    vertices_np_ve = np.array(ve_data['vertices']) # Renamed
    lines_data = []
    for edge_indices in ve_data['edges']:
        p1 = vertices_np_ve[edge_indices[0]].tolist()
        p2 = vertices_np_ve[edge_indices[1]].tolist()
        lines_data.append((p1, p2))
    
    min_coords_ve = np.min(vertices_np_ve, axis=0); max_coords_ve = np.max(vertices_np_ve, axis=0) # Renamed
    padding_ve = radius * 0.15 # Renamed

    return {
        'lines': lines_data,
        'projection_type': '3d',
        'bounding_box': {
            'xmin': float(min_coords_ve[0]-padding_ve), 'xmax': float(max_coords_ve[0]+padding_ve),
            'ymin': float(min_coords_ve[1]-padding_ve), 'ymax': float(max_coords_ve[1]+padding_ve),
            'zmin': float(min_coords_ve[2]-padding_ve), 'zmax': float(max_coords_ve[2]+padding_ve),
        }
    }


# Example usage
if __name__ == "__main__":
    # Create a vector equilibrium
    center = (0, 0, 0)
    radius = 1.0
    
    ve = generate_vector_equilibrium(center, radius)
    
    # Calculate sacred ratios
    ratios = calculate_vector_equilibrium_sacred_ratios(ve)
    print("Sacred Ratios:")
    for key, value in ratios.items():
        print(f"{key}: {value}")
    
    # Calculate energy distribution
    energy_data = calculate_vector_equilibrium_energy_distribution(ve)
    
    # Visualize structure
    fig1 = visualize_vector_equilibrium(ve, show_energy=True)
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.show()
    
    # Visualize energy
    fig2 = visualize_vector_equilibrium_energy(ve, energy_data)
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.show()