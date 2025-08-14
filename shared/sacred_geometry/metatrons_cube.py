"""
Metatron's Cube Pattern

This module implements the Metatron's Cube sacred geometry pattern.
Metatron's Cube is a complex sacred geometry pattern derived from the Flower of Life,
containing all five Platonic solids in geometric relationship.

This pattern serves as a map of the universe and connects all Platonic solids,
creating a template for multidimensional relationships in 3D space.

Author: Soul Development Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import logging
from scipy.spatial import ConvexHull

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='metatrons_cube.log'
)
logger = logging.getLogger('metatrons_cube')

class MetatronsCube:
    """
    Implementation of the Metatron's Cube sacred geometry pattern.
    
    Metatron's Cube contains all five Platonic solids (tetrahedron, hexahedron/cube,
    octahedron, dodecahedron, and icosahedron) within its structure and creates
    13 primary energy channels between its vertices.
    """
    
    def __init__(self, radius=1.0, resolution=64):
        """
        Initialize a new Metatron's Cube pattern.
        
        Args:
            radius (float): Radius of the circumscribing sphere
            resolution (int): Resolution of the generated pattern matrices
        """
        self.radius = radius
        self.resolution = resolution
        
        # Generate the key vertices of Metatron's Cube
        self.vertices = self._generate_vertices()
        
        # Generate the edges connecting vertices
        self.edges = self._generate_edges()
        
        # Generate Platonic solids within the cube
        self.platonic_solids = self._generate_platonic_solids()
        
        # Generate pattern matrices
        self.pattern_2d = None
        self.pattern_3d = None
        
        logger.info(f"Metatron's Cube initialized with radius {radius} and resolution {resolution}")
        logger.info(f"Generated {len(self.vertices)} vertices and {len(self.edges)} edges")
    
    def _generate_vertices(self):
        """
        Generate the vertices of Metatron's Cube.
        
        Returns:
            dict: Dictionary of vertex coordinates (13 vertices total)
        """
        # Metatron's Cube has 13 key vertices:
        # - Central point (origin)
        # - 12 vertices derived from the Flower of Life pattern
        
        vertices = {}
        
        # Central point (origin)
        vertices['center'] = np.array([0, 0, 0])
        
        # First layer: 6 vertices in hexagonal arrangement (2D plane)
        for i in range(6):
            angle = i * np.pi / 3  # 60 degrees apart
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = 0  # All in the same plane
            vertices[f'first_{i+1}'] = np.array([x, y, z])
        
        # Second layer: 6 vertices in hexagonal arrangement (alternating planes)
        for i in range(6):
            angle = (i * np.pi / 3) + (np.pi / 6)  # 60 degrees apart, rotated by 30 degrees
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = (i % 2) * self.radius  # Alternating z values
            vertices[f'second_{i+1}'] = np.array([x, y, z])
        
        return vertices
    
    def _generate_edges(self):
        """
        Generate the edges of Metatron's Cube.
        
        Returns:
            list: List of edge vertex pairs
        """
        edges = []
        vertex_ids = list(self.vertices.keys())
        
        # Connect center to all other vertices
        for vertex_id in vertex_ids:
            if vertex_id != 'center':
                edges.append(('center', vertex_id))
        
        # Connect first layer vertices sequentially
        first_layer = [vid for vid in vertex_ids if vid.startswith('first_')]
        for i in range(len(first_layer)):
            edges.append((first_layer[i], first_layer[(i+1) % len(first_layer)]))
        
        # Connect second layer vertices sequentially
        second_layer = [vid for vid in vertex_ids if vid.startswith('second_')]
        for i in range(len(second_layer)):
            edges.append((second_layer[i], second_layer[(i+1) % len(second_layer)]))
        
        # Connect first layer to second layer
        for i in range(len(first_layer)):
            edges.append((first_layer[i], second_layer[i]))
            edges.append((first_layer[i], second_layer[(i+1) % len(second_layer)]))
        
        return edges
    
    def _generate_platonic_solids(self):
        """
        Generate the five Platonic solids within Metatron's Cube.
        
        Returns:
            dict: Dictionary containing vertices and faces for each Platonic solid
        """
        platonic_solids = {}
        all_vertices = list(self.vertices.values())
        
        # 1. Tetrahedron (4 vertices, 4 faces)
        tetra_vertices = [
            self.vertices['center'],
            self.vertices['first_1'],
            self.vertices['first_3'],
            self.vertices['first_5']
        ]
        tetra_faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 3, 2)
        ]
        platonic_solids['tetrahedron'] = {
            'vertices': tetra_vertices,
            'faces': tetra_faces
        }
        
        # 2. Hexahedron/Cube (8 vertices, 6 faces)
        # Use vertices from first and second layers to create a cube
        hex_vertices = [
            self.vertices['first_1'],
            self.vertices['first_3'],
            self.vertices['first_5'],
            self.vertices['second_2'],
            self.vertices['second_4'],
            self.vertices['second_6'],
            self.vertices['center'],
            np.array([0, 0, self.radius])  # Top center point
        ]
        hex_faces = [
            (0, 1, 2), (0, 2, 5), (0, 5, 3), (0, 3, 1),  # Lower faces
            (6, 1, 2), (6, 2, 5), (6, 5, 3), (6, 3, 1),  # Upper faces
            (7, 1, 2), (7, 2, 5), (7, 5, 3), (7, 3, 1)   # Top faces
        ]
        platonic_solids['hexahedron'] = {
            'vertices': hex_vertices,
            'faces': hex_faces
        }
        
        # 3. Octahedron (6 vertices, 8 faces)
        # Use the center and symmetric points
        oct_vertices = [
            self.vertices['center'],
            self.vertices['first_1'],
            self.vertices['first_3'],
            self.vertices['first_5'],
            np.array([0, 0, self.radius]),
            np.array([0, 0, -self.radius])
        ]
        oct_faces = [
            (0, 1, 4), (0, 2, 4), (0, 3, 4), (0, 1, 5),
            (0, 2, 5), (0, 3, 5), (1, 2, 4), (2, 3, 4),
            (3, 1, 4), (1, 2, 5), (2, 3, 5), (3, 1, 5)
        ]
        platonic_solids['octahedron'] = {
            'vertices': oct_vertices,
            'faces': oct_faces
        }
        
        # 4. Dodecahedron (20 vertices, 12 faces)
        # This is a simplified approximation of the dodecahedron within Metatron's Cube
        # We'll create it using the golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Create the 20 vertices of a regular dodecahedron
        dodeca_vertices = []
        
        # Vertices are based on three orthogonal golden rectangles
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    dodeca_vertices.append(np.array([x, y, z]) * self.radius / np.sqrt(3))
        
        for i in range(8, 20):
            # Add the other 12 vertices derived from the golden ratio
            cycles = [
                [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
                [1/phi, 0, phi], [-1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, -phi],
                [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0]
            ]
            coords = cycles[i-8]
            dodeca_vertices.append(np.array(coords) * self.radius / np.sqrt(phi**2 + 1/phi**2 + 1))
        
        # Create the 12 pentagonal faces
        # This is a simplified version - actual face definition would be more complex
        platonic_solids['dodecahedron'] = {
            'vertices': dodeca_vertices,
            'faces': []  # We'll skip detailed face definition for simplicity
        }
        
        # 5. Icosahedron (12 vertices, 20 faces)
        # Create the 12 vertices of a regular icosahedron
        ico_vertices = []
        
        # Vertices are based on the golden ratio
        for x in [-1, 1]:
            for y in [-1, 1]:
                ico_vertices.append(np.array([0, x, y*phi]) * self.radius / np.sqrt(1 + phi**2))
                ico_vertices.append(np.array([x, y*phi, 0]) * self.radius / np.sqrt(1 + phi**2))
                ico_vertices.append(np.array([x*phi, 0, y]) * self.radius / np.sqrt(1 + phi**2))
        
        # Create the 20 triangular faces
        # This is a simplified version - actual face definition would be more complex
        platonic_solids['icosahedron'] = {
            'vertices': ico_vertices,
            'faces': []  # We'll skip detailed face definition for simplicity
        }
        
        logger.info(f"Generated {len(platonic_solids)} Platonic solids within Metatron's Cube")
        return platonic_solids
    
    def generate_2d_pattern(self):
        """
        Generate a 2D matrix representation of Metatron's Cube pattern.
        
        This creates a 2D projection of the 3D structure, showing the
        characteristic Star of David pattern with internal lines.
        
        Returns:
            ndarray: 2D matrix representation of Metatron's Cube
        """
        # Create the bounds of the pattern
        x = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        y = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Initialize pattern matrix
        self.pattern_2d = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        
        # Draw the edges
        line_width = max(1, int(self.resolution / 100))
        
        for v1_id, v2_id in self.edges:
            v1 = self.vertices[v1_id]
            v2 = self.vertices[v2_id]
            
            # Project 3D onto 2D (XY plane)
            start = (v1[0], v1[1])
            end = (v2[0], v2[1])
            
            # Convert to pixel coordinates
            start_px = (
                int((start[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((start[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            end_px = (
                int((end[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((end[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            
            # Draw line using Bresenham's algorithm
            self._draw_line(start_px, end_px, line_width)
        
        # Add vertex points
        for vertex_id, coords in self.vertices.items():
            # Project 3D onto 2D (XY plane)
            point = (coords[0], coords[1])
            
            # Convert to pixel coordinates
            point_px = (
                int((point[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((point[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            
            # Draw vertex point
            self._draw_point(point_px, radius=line_width*2)
        
        logger.info(f"2D Metatron's Cube pattern generated with shape {self.pattern_2d.shape}")
        return self.pattern_2d
    
    def _draw_line(self, start, end, width=1):
        """
        Draw a line on the 2D pattern matrix.
        
        Args:
            start (tuple): Start point (x, y) in pixel coordinates
            end (tuple): End point (x, y) in pixel coordinates
            width (int): Line width in pixels
        """
        x0, y0 = start
        x1, y1 = end
        
        # Ensure coordinates are within bounds
        x0 = max(0, min(x0, self.resolution-1))
        y0 = max(0, min(y0, self.resolution-1))
        x1 = max(0, min(x1, self.resolution-1))
        y1 = max(0, min(y1, self.resolution-1))
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Set the pixel at (x0, y0)
            for wx in range(-width//2, width//2 + 1):
                for wy in range(-width//2, width//2 + 1):
                    px = x0 + wx
                    py = y0 + wy
                    if 0 <= px < self.resolution and 0 <= py < self.resolution:
                        self.pattern_2d[py, px] = 1.0
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def _draw_point(self, point, radius=2, value=1.0):
        """
        Draw a point on the 2D pattern matrix.
        
        Args:
            point (tuple): Point (x, y) in pixel coordinates
            radius (int): Point radius in pixels
            value (float): Value to set at the point
        """
        x0, y0 = point
        
        for x in range(x0 - radius, x0 + radius + 1):
            for y in range(y0 - radius, y0 + radius + 1):
                if 0 <= x < self.resolution and 0 <= y < self.resolution:
                    if (x - x0)**2 + (y - y0)**2 <= radius**2:
                        self.pattern_2d[y, x] = value
    
    def generate_3d_pattern(self):
        """
        Generate a 3D matrix representation of Metatron's Cube.
        
        This creates a 3D array representing the edges and vertices of
        Metatron's Cube in 3D space.
        
        Returns:
            ndarray: 3D matrix representation of Metatron's Cube
        """
        # Create the bounds of the pattern
        x = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        y = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        z = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        
        # Initialize 3D pattern
        self.pattern_3d = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float64)
        
        # Draw the edges
        line_width = max(1, int(self.resolution / 100))
        
        for v1_id, v2_id in self.edges:
            v1 = self.vertices[v1_id]
            v2 = self.vertices[v2_id]
            
            # Convert to pixel coordinates
            start_px = (
                int((v1[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((v1[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((v1[2] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            end_px = (
                int((v2[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((v2[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((v2[2] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            
            # Draw 3D line
            self._draw_3d_line(start_px, end_px, width=line_width)
        
        # Add vertex points
        for vertex_id, coords in self.vertices.items():
            # Convert to pixel coordinates
            point_px = (
                int((coords[0] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((coords[1] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius)),
                int((coords[2] + self.radius*1.2) * (self.resolution-1) / (2.4*self.radius))
            )
            
            # Draw vertex point
            self._draw_3d_point(point_px, radius=line_width*2)
        
        logger.info(f"3D Metatron's Cube pattern generated with shape {self.pattern_3d.shape}")
        return self.pattern_3d
    
    def _draw_3d_line(self, start, end, width=1):
        """
        Draw a line in 3D space on the pattern matrix.
        
        Args:
            start (tuple): Start point (x, y, z) in pixel coordinates
            end (tuple): End point (x, y, z) in pixel coordinates
            width (int): Line width in pixels
        """
        x0, y0, z0 = start
        x1, y1, z1 = end
        
        # Ensure coordinates are within bounds
        x0 = max(0, min(x0, self.resolution-1))
        y0 = max(0, min(y0, self.resolution-1))
        z0 = max(0, min(z0, self.resolution-1))
        x1 = max(0, min(x1, self.resolution-1))
        y1 = max(0, min(y1, self.resolution-1))
        z1 = max(0, min(z1, self.resolution-1))
        
        # 3D Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        sz = 1 if z0 < z1 else -1
        
        # Driving axis is the one with the greatest change
        if dx >= dy and dx >= dz:
            err_1 = 2*dy - dx
            err_2 = 2*dz - dx
            
            for i in range(dx + 1):
                # Set the voxel at (x0, y0, z0)
                for wx in range(-width//2, width//2 + 1):
                    for wy in range(-width//2, width//2 + 1):
                        for wz in range(-width//2, width//2 + 1):
                            px = x0 + wx
                            py = y0 + wy
                            pz = z0 + wz
                            if (0 <= px < self.resolution and 
                                0 <= py < self.resolution and 
                                0 <= pz < self.resolution):
                                self.pattern_3d[px, py, pz] = 1.0
                
                if x0 == x1 and y0 == y1 and z0 == z1:
                    break
                
                if err_1 > 0:
                    y0 += sy
                    err_1 -= 2*dx
                
                if err_2 > 0:
                    z0 += sz
                    err_2 -= 2*dx
                
                err_1 += 2*dy
                err_2 += 2*dz
                x0 += sx
                
        elif dy >= dx and dy >= dz:
            err_1 = 2*dx - dy
            err_2 = 2*dz - dy
            
            for i in range(dy + 1):
                # Set the voxel at (x0, y0, z0)
                for wx in range(-width//2, width//2 + 1):
                    for wy in range(-width//2, width//2 + 1):
                        for wz in range(-width//2, width//2 + 1):
                            px = x0 + wx
                            py = y0 + wy
                            pz = z0 + wz
                            if (0 <= px < self.resolution and 
                                0 <= py < self.resolution and 
                                0 <= pz < self.resolution):
                                self.pattern_3d[px, py, pz] = 1.0
                
                if x0 == x1 and y0 == y1 and z0 == z1:
                    break
                
                if err_1 > 0:
                    x0 += sx
                    err_1 -= 2*dy
                
                if err_2 > 0:
                    z0 += sz
                    err_2 -= 2*dy
                
                err_1 += 2*dx
                err_2 += 2*dz
                y0 += sy
                
        else:  # dz >= dx and dz >= dy
            err_1 = 2*dy - dz
            err_2 = 2*dx - dz
            
            for i in range(dz + 1):
                # Set the voxel at (x0, y0, z0)
                for wx in range(-width//2, width//2 + 1):
                    for wy in range(-width//2, width//2 + 1):
                        for wz in range(-width//2, width//2 + 1):
                            px = x0 + wx
                            py = y0 + wy
                            pz = z0 + wz
                            if (0 <= px < self.resolution and 
                                0 <= py < self.resolution and 
                                0 <= pz < self.resolution):
                                self.pattern_3d[px, py, pz] = 1.0
                
                if x0 == x1 and y0 == y1 and z0 == z1:
                    break
                
                if err_1 > 0:
                    y0 += sy
                    err_1 -= 2*dz
                
                if err_2 > 0:
                    x0 += sx
                    err_2 -= 2*dz
                
                err_1 += 2*dy
                err_2 += 2*dx
                z0 += sz
    
    def _draw_3d_point(self, point, radius=2, value=1.0):
        """
        Draw a 3D point (sphere) on the pattern matrix.
        
        Args:
            point (tuple): Point (x, y, z) in pixel coordinates
            radius (int): Point radius in pixels
            value (float): Value to set at the point
        """
        x0, y0, z0 = point
        
        for x in range(x0 - radius, x0 + radius + 1):
            for y in range(y0 - radius, y0 + radius + 1):
                for z in range(z0 - radius, z0 + radius + 1):
                    if (0 <= x < self.resolution and 
                        0 <= y < self.resolution and 
                        0 <= z < self.resolution):
                        if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 <= radius**2:
                            self.pattern_3d[x, y, z] = value
    
    def get_2d_pattern(self):
        """
        Get the 2D pattern matrix.
        
        Returns:
            ndarray: 2D matrix representation of Metatron's Cube
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
        return self.pattern_2d
    
    def get_3d_pattern(self):
        """
        Get the 3D pattern matrix.
        
        Returns:
            ndarray: 3D matrix representation of Metatron's Cube
        """
        if self.pattern_3d is None:
            self.generate_3d_pattern()
        return self.pattern_3d
    
    def visualize_2d(self, show_vertices=True, show_edges=True, 
                    show=True, save_path=None):
        """
        Visualize the 2D projection of Metatron's Cube.
        
        Args:
            show_vertices (bool): Whether to highlight vertices
            show_edges (bool): Whether to highlight edges
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Generate pattern if not already done
            if self.pattern_2d is None:
                self.generate_2d_pattern()
                
            # Create the figure
            plt.figure(figsize=(10, 8))
            
            # Show the pattern
            plt.imshow(self.pattern_2d, cmap='viridis', origin='lower', 
                      extent=[-self.radius*1.2, self.radius*1.2, -self.radius*1.2, self.radius*1.2])
            
            if show_edges:
                # Plot the edges
                for v1_id, v2_id in self.edges:
                    v1 = self.vertices[v1_id]
                    v2 = self.vertices[v2_id]
                    plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'w-', alpha=0.7, linewidth=1)
            
            if show_vertices:
                # Plot the vertices
                for vertex_id, coords in self.vertices.items():
                    plt.plot(coords[0], coords[1], 'ro' if vertex_id == 'center' else 'wo', 
                           markersize=8 if vertex_id == 'center' else 5)
            
            plt.title("Metatron's Cube - 2D Projection")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"2D visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in 2D visualization: {str(e)}")
            return False
    
    def visualize_3d(self, show_vertices=True, show_edges=True, 
                    show_platonic=None, show=True, save_path=None):
        """
        Visualize the 3D structure of Metatron's Cube.
        
        Args:
            show_vertices (bool): Whether to highlight vertices
            show_edges (bool): Whether to highlight edges
            show_platonic (str): Name of Platonic solid to highlight (None for none)
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Create the figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if show_edges:
                # Plot the edges
                for v1_id, v2_id in self.edges:
                    v1 = self.vertices[v1_id]
                    v2 = self.vertices[v2_id]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                          'w-', alpha=0.7, linewidth=1)
            
            if show_vertices:
                # Plot the vertices
                vertex_xs = []
                vertex_ys = []
                vertex_zs = []
                colors = []
                
                for vertex_id, coords in self.vertices.items():
                    vertex_xs.append(coords[0])
                    vertex_ys.append(coords[1])
                    vertex_zs.append(coords[2])
                    colors.append('red' if vertex_id == 'center' else 'white')
                
                ax.scatter(vertex_xs, vertex_ys, vertex_zs, c=colors, s=50)
            
            # Highlight a specific Platonic solid if requested
            if show_platonic and show_platonic in self.platonic_solids:
                solid = self.platonic_solids[show_platonic]
                
                # Different colors for different solids
                color_map = {
                    'tetrahedron': 'red',
                    'hexahedron': 'blue',
                    'octahedron': 'green',
                    'dodecahedron': 'purple',
                    'icosahedron': 'orange'
                }
                color = color_map.get(show_platonic, 'cyan')
                
                # Plot vertices
                vertices = solid['vertices']
                vertex_xs = [v[0] for v in vertices]
                vertex_ys = [v[1] for v in vertices]
                vertex_zs = [v[2] for v in vertices]
                
                ax.scatter(vertex_xs, vertex_ys, vertex_zs, c=color, s=100, alpha=0.8)
                
                # Plot faces if available
                if 'faces' in solid and solid['faces']:
                    for face in solid['faces']:
                        face_vertices = [vertices[i] for i in face]
                        face_vertices.append(face_vertices[0])  # Close the loop
                        xs = [v[0] for v in face_vertices]
                        ys = [v[1] for v in face_vertices]
                        zs = [v[2] for v in face_vertices]
                        
                        ax.plot(xs, ys, zs, color=color, alpha=0.5, linewidth=2)
                        
                        # Add a transparent face
                        if len(face) == 3:  # Triangular face
                            tri = [[vertices[i] for i in face]]
                            ax.add_collection3d(plt.art3d.Poly3DCollection(
                                tri, alpha=0.2, color=color))
            
            # Set axis labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Metatron's Cube - 3D Structure")
            
            # Set equal aspect ratio
            max_range = max([
                max(ax.get_xlim()) - min(ax.get_xlim()),
                max(ax.get_ylim()) - min(ax.get_ylim()),
                max(ax.get_zlim()) - min(ax.get_zlim())
            ])
            
            mid_x = np.mean(ax.get_xlim())
            mid_y = np.mean(ax.get_ylim())
            mid_z = np.mean(ax.get_zlim())
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"3D visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in 3D visualization: {str(e)}")
            return False
    
    def visualize_platonic_solid(self, solid_name, show=True, save_path=None):
        """
        Visualize a specific Platonic solid from Metatron's Cube.
        
        Args:
            solid_name (str): Name of the Platonic solid to visualize
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            if solid_name not in self.platonic_solids:
                logger.error(f"Unknown Platonic solid: {solid_name}")
                return False
                
            # Create the figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get the solid data
            solid = self.platonic_solids[solid_name]
            vertices = solid['vertices']
            
            # Plot vertices
            vertex_xs = [v[0] for v in vertices]
            vertex_ys = [v[1] for v in vertices]
            vertex_zs = [v[2] for v in vertices]
            
            ax.scatter(vertex_xs, vertex_ys, vertex_zs, c='blue', s=100, alpha=0.8)
            
            # Plot edges and faces if available
            if 'faces' in solid and solid['faces']:
                for face in solid['faces']:
                    face_vertices = [vertices[i] for i in face]
                    face_vertices.append(face_vertices[0])  # Close the loop
                    xs = [v[0] for v in face_vertices]
                    ys = [v[1] for v in face_vertices]
                    zs = [v[2] for v in face_vertices]
                    
                    ax.plot(xs, ys, zs, 'b-', alpha=0.7, linewidth=2)
                    
                    # Add a transparent face
                    if len(face) == 3:  # Triangular face
                        tri = [[vertices[i] for i in face]]  # Create triangle vertices
                        ax.add_collection3d(art3d.Poly3DCollection(
                            tri, alpha=0.2, color='cyan'))

            
            # Set axis labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{solid_name.capitalize()} from Metatron's Cube")
            
            # Set equal aspect ratio
            max_range = max([
                max(ax.get_xlim()) - min(ax.get_xlim()),
                max(ax.get_ylim()) - min(ax.get_ylim()),
                max(ax.get_zlim()) - min(ax.get_zlim())
            ])
            
            mid_x = np.mean(ax.get_xlim())
            mid_y = np.mean(ax.get_ylim())
            mid_z = np.mean(ax.get_zlim())
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Platonic solid visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in Platonic solid visualization: {str(e)}")
            return False
    
    def get_platonic_solid(self, solid_name):
        """
        Get the vertices and faces of a specific Platonic solid.
        
        Args:
            solid_name (str): Name of the Platonic solid
            
        Returns:
            dict: Dictionary with vertices and faces, or None if not found
        """
        if solid_name in self.platonic_solids:
            return self.platonic_solids[solid_name]
        else:
            logger.warning(f"Unknown Platonic solid: {solid_name}")
            return None
    
    def get_energy_channels(self):
        """
        Get the 13 primary energy channels of Metatron's Cube.
        
        Returns:
            list: List of edges representing energy channels
        """
        # In Metatron's Cube, the edges connecting the center to other vertices
        # and the edges forming the two hexagons are considered primary energy channels
        primary_channels = []
        
        # Center to all other vertices
        center_edges = [edge for edge in self.edges if 'center' in edge]
        primary_channels.extend(center_edges)
        
        # First layer hexagon
        first_layer = [vid for vid in self.vertices.keys() if vid.startswith('first_')]
        for i in range(len(first_layer)):
            primary_channels.append((first_layer[i], first_layer[(i+1) % len(first_layer)]))
        
        # Second layer hexagon
        second_layer = [vid for vid in self.vertices.keys() if vid.startswith('second_')]
        for i in range(len(second_layer)):
            primary_channels.append((second_layer[i], second_layer[(i+1) % len(second_layer)]))
        
        return primary_channels
    
    def __str__(self):
        """String representation of the Metatron's Cube pattern."""
        return (f"Metatron's Cube Pattern\n"
                f"Radius: {self.radius}\n"
                f"Number of Vertices: {len(self.vertices)}\n"
                f"Number of Edges: {len(self.edges)}\n"
                f"Number of Platonic Solids: {len(self.platonic_solids)}\n"
                f"Resolution: {self.resolution}x{self.resolution}")

    def get_base_glyph_elements(self) -> Dict[str, Any]:
        """
        Returns lines for a 2D projection of Metatron's Cube.
        """
        if not hasattr(self, 'vertices') or not self.vertices: self.vertices = self._generate_vertices()
        if not hasattr(self, 'edges') or not self.edges: self.edges = self._generate_edges()

        lines_data = []
        all_x_mc = []; all_y_mc = [] # Renamed

        for v1_id, v2_id in self.edges:
            v1_coords = self.vertices[v1_id]
            v2_coords = self.vertices[v2_id]
            p1 = (v1_coords[0], v1_coords[1]) # 2D Projection
            p2 = (v2_coords[0], v2_coords[1]) # 2D Projection
            lines_data.append((p1, p2))
            all_x_mc.extend([p1[0], p2[0]])
            all_y_mc.extend([p1[1], p2[1]])
        
        padding_mc = self.radius * 0.2 if hasattr(self, 'radius') else 0.2 # Renamed

        bbox_mc = {'xmin':-1.2*self.radius, 'xmax':1.2*self.radius, 'ymin':-1.2*self.radius, 'ymax':1.2*self.radius}
        if all_x_mc and all_y_mc:
            bbox_mc = {
                'xmin': float(min(all_x_mc) - padding_mc), 'xmax': float(max(all_x_mc) + padding_mc),
                'ymin': float(min(all_y_mc) - padding_mc), 'ymax': float(max(all_y_mc) + padding_mc),
            }

        return { 'lines': lines_data, 'projection_type': '2d', 'bounding_box': bbox_mc }

if __name__ == "__main__":
    # Example usage
    cube = MetatronsCube(radius=1.0, resolution=256)
    print(cube)
    
    # Visualize the 2D projection
    cube.visualize_2d(save_path="metatrons_cube_2d.png")
    
    # Visualize the 3D structure
    cube.visualize_3d(save_path="metatrons_cube_3d.png")
    
    # Visualize specific Platonic solids
    for solid in ['tetrahedron', 'hexahedron', 'octahedron']:
        cube.visualize_platonic_solid(solid, save_path=f"{solid}_from_metatrons_cube.png")
    
    # Get pattern matrices for use in field systems
    pattern_2d = cube.get_2d_pattern()
    pattern_3d = cube.get_3d_pattern()