"""
Merkaba Pattern

This module implements the Merkaba sacred geometry pattern.
The Merkaba consists of two counter-rotating tetrahedra, creating a three-dimensional
Star of David pattern that forms a vehicle of light for consciousness.

The Merkaba (also spelled Merkabah) represents the energy field that surrounds
the body and can be used for interdimensional travel and protection during
spiritual transitions.

Author: Soul Development Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='merkaba.log'
)
logger = logging.getLogger('merkaba')

class Merkaba:
    """
    Implementation of the Merkaba sacred geometry pattern.
    
    The Merkaba consists of two interlocked tetrahedra, one pointing upward
    (representing spiritual energy) and one pointing downward (representing
    physical energy). When counter-rotating, these tetrahedra create a stable
    energy field for interdimensional travel and spiritual protection.
    """
    
    def __init__(self, radius=1.0, resolution=64, rotation_angle=0.0, phi_ratio=1.618):
        """
        Initialize a new Merkaba pattern.
        
        Args:
            radius (float): Radius of the circumscribing sphere
            resolution (int): Resolution of the generated pattern matrices
            rotation_angle (float): Initial rotation angle between tetrahedra (radians)
            phi_ratio (float): Golden ratio factor for harmonic resonance (default: 1.618)
        """
        self.radius = radius
        self.resolution = resolution
        self.rotation_angle = rotation_angle
        self.phi_ratio = phi_ratio
        
        # Generate tetrahedron vertices
        self.upward_vertices = self._generate_tetrahedron_vertices(up=True)
        self.downward_vertices = self._generate_tetrahedron_vertices(up=False)
        
        # Generate tetrahedron edges
        self.upward_edges = self._generate_tetrahedron_edges(self.upward_vertices)
        self.downward_edges = self._generate_tetrahedron_edges(self.downward_vertices)
        
        # Generate tetrahedron faces
        self.upward_faces = self._generate_tetrahedron_faces(self.upward_vertices)
        self.downward_faces = self._generate_tetrahedron_faces(self.downward_vertices)
        
        # Generate pattern matrices
        self.pattern_3d = None
        
        logger.info(f"Merkaba initialized with radius {radius} and resolution {resolution}")
        logger.info(f"Rotation angle: {rotation_angle}, Phi ratio: {phi_ratio}")
    
    def _generate_tetrahedron_vertices(self, up=True):
        """
        Generate the vertices of a tetrahedron.
        
        Args:
            up (bool): True for upward-pointing, False for downward-pointing
            
        Returns:
            ndarray: Array of vertex coordinates (4 vertices, 3 coordinates each)
        """
        # Base tetrahedron vertices (centered at origin)
        vertices = np.array([
            [0, 0, self.radius],             # Apex
            [self.radius * np.sin(0), self.radius * np.cos(0), -self.radius/3],  # Base vertex 1
            [self.radius * np.sin(2*np.pi/3), self.radius * np.cos(2*np.pi/3), -self.radius/3],  # Base vertex 2
            [self.radius * np.sin(4*np.pi/3), self.radius * np.cos(4*np.pi/3), -self.radius/3]   # Base vertex 3
        ])
        
        # Invert for downward tetrahedron
        if not up:
            vertices[:, 2] = -vertices[:, 2]  # Invert z-coordinates
            
            # Apply rotation around z-axis
            rotation_matrix = np.array([
                [np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation to each vertex
            for i in range(vertices.shape[0]):
                vertices[i] = rotation_matrix @ vertices[i]
        
        return vertices
    
    def _generate_tetrahedron_edges(self, vertices):
        """
        Generate the edges of a tetrahedron.
        
        Args:
            vertices (ndarray): Array of vertex coordinates
            
        Returns:
            list: List of edge vertex pairs (6 edges total)
        """
        # Each tetrahedron has 6 edges
        edges = [
            (0, 1),  # Apex to base vertex 1
            (0, 2),  # Apex to base vertex 2
            (0, 3),  # Apex to base vertex 3
            (1, 2),  # Base edge 1
            (2, 3),  # Base edge 2
            (3, 1)   # Base edge 3
        ]
        
        return edges
    
    def _generate_tetrahedron_faces(self, vertices):
        """
        Generate the faces of a tetrahedron.
        
        Args:
            vertices (ndarray): Array of vertex coordinates
            
        Returns:
            list: List of face vertex triplets (4 faces total)
        """
        # Each tetrahedron has 4 triangular faces
        faces = [
            (0, 1, 2),  # Face 1 (apex, base1, base2)
            (0, 2, 3),  # Face 2 (apex, base2, base3)
            (0, 3, 1),  # Face 3 (apex, base3, base1)
            (1, 3, 2)   # Face 4 (base face: base1, base3, base2)
        ]
        
        return faces
    
    def generate_3d_pattern(self):
        """
        Generate a 3D matrix representation of the Merkaba pattern.
        
        This creates a 3D array where higher values represent the
        tetrahedra, with the highest values at intersection regions.
        
        Returns:
            ndarray: 3D matrix representation of the Merkaba
        """
        # Create coordinate grids
        x = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        y = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        z = np.linspace(-self.radius*1.2, self.radius*1.2, self.resolution)
        
        # Create grid meshes for vectorized operations
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize 3D pattern
        self.pattern_3d = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float64)
        
        # Add upward tetrahedron
        for face in self.upward_faces:
            v1, v2, v3 = self.upward_vertices[face[0]], self.upward_vertices[face[1]], self.upward_vertices[face[2]]
            self._add_tetrahedron_face_to_pattern(v1, v2, v3, X, Y, Z, value=1.0)
        
        # Add downward tetrahedron
        for face in self.downward_faces:
            v1, v2, v3 = self.downward_vertices[face[0]], self.downward_vertices[face[1]], self.downward_vertices[face[2]]
            self._add_tetrahedron_face_to_pattern(v1, v2, v3, X, Y, Z, value=1.0)
        
        logger.info(f"3D Merkaba pattern generated with shape {self.pattern_3d.shape}")
        return self.pattern_3d
    
    def _add_tetrahedron_face_to_pattern(self, v1, v2, v3, X, Y, Z, value=1.0, thickness=0.1):
        """
        Add a triangular face to the 3D pattern.
        
        Args:
            v1, v2, v3 (ndarray): Vertices of the triangle
            X, Y, Z (ndarray): Coordinate grids
            value (float): Value to add to the pattern
            thickness (float): Thickness of the face
            
        Returns:
            None (modifies self.pattern_3d in place)
        """
        # Calculate face normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate the plane equation: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, v1)
        
        # Calculate distance from each point to the plane
        distance = abs(a*X + b*Y + c*Z + d) / np.sqrt(a**2 + b**2 + c**2)
        
        # Create a binary mask for points within the triangular face
        # This is a simplified approach that works for the Merkaba visualization
        # For more accurate triangle rendering, additional checks would be needed
        in_face = distance < thickness
        
        # Add the value to the pattern at face locations
        self.pattern_3d = np.where(in_face, self.pattern_3d + value, self.pattern_3d)
    
    def get_3d_pattern(self):
        """
        Get the 3D pattern matrix.
        
        Returns:
            ndarray: 3D matrix representation of the Merkaba
        """
        if self.pattern_3d is None:
            self.generate_3d_pattern()
        return self.pattern_3d
    
    def rotate_merkaba(self, angle_step):
        """
        Rotate the Merkaba pattern.
        
        The counter-rotation of the two tetrahedra is what activates
        the Merkaba energy field.
        
        Args:
            angle_step (float): Angle to rotate (radians)
            
        Returns:
            None (modifies the Merkaba in place)
        """
        # Update rotation angle
        self.rotation_angle += angle_step
        
        # Regenerate the downward tetrahedron with the new rotation
        self.downward_vertices = self._generate_tetrahedron_vertices(up=False)
        self.downward_edges = self._generate_tetrahedron_edges(self.downward_vertices)
        self.downward_faces = self._generate_tetrahedron_faces(self.downward_vertices)
        
        # Clear the 3D pattern to force regeneration
        self.pattern_3d = None
        
        logger.info(f"Merkaba rotated to angle {self.rotation_angle}")
    
    def activate_merkaba(self, num_steps=12, phi_modulation=True):
        """
        Activate the Merkaba through a sequence of counter-rotations.
        
        This simulates the activation process that creates the energy
        field for interdimensional travel and spiritual protection.
        
        Args:
            num_steps (int): Number of rotation steps
            phi_modulation (bool): Whether to modulate rotation by phi ratio
            
        Returns:
            list: List of 3D patterns at each step of activation
        """
        activation_patterns = []
        
        # Determine rotation step angle
        full_circle = 2 * np.pi
        base_step = full_circle / num_steps
        
        for i in range(num_steps):
            # Apply phi ratio modulation if requested
            if phi_modulation:
                # Modulate rotation speed using golden ratio
                phi_factor = 1.0 + 0.2 * np.sin(i * 2 * np.pi / num_steps * self.phi_ratio)
                angle_step = base_step * phi_factor
            else:
                angle_step = base_step
            
            # Rotate the Merkaba
            self.rotate_merkaba(angle_step)
            
            # Generate the pattern at this rotation
            pattern = self.generate_3d_pattern()
            
            # Store a copy of the pattern
            activation_patterns.append(pattern.copy())
            
            logger.info(f"Activation step {i+1}/{num_steps} completed")
        
        return activation_patterns
    
    def visualize_merkaba_3d(self, show_vertices=True, show_edges=True, 
                           show_faces=False, show=True, save_path=None):
        """
        Create a 3D visualization of the Merkaba geometry.
        
        Args:
            show_vertices (bool): Whether to show vertices
            show_edges (bool): Whether to show edges
            show_faces (bool): Whether to show faces
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Drawing upward tetrahedron
            if show_vertices:
                ax.scatter(self.upward_vertices[:, 0], self.upward_vertices[:, 1], 
                         self.upward_vertices[:, 2], color='blue', s=100, label='Upward Vertices')
            
            if show_edges:
                for edge in self.upward_edges:
                    v1, v2 = self.upward_vertices[edge[0]], self.upward_vertices[edge[1]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'b-', linewidth=2)
            
            if show_faces:
                for face in self.upward_faces:
                    v1, v2, v3 = self.upward_vertices[face[0]], self.upward_vertices[face[1]], self.upward_vertices[face[2]]
                    verts = [list(v1), list(v2), list(v3)]
                    ax.add_collection3d(plt.art3d.Poly3DCollection([verts], 
                                                               alpha=0.3, color='blue'))
            
            # Drawing downward tetrahedron
            if show_vertices:
                ax.scatter(self.downward_vertices[:, 0], self.downward_vertices[:, 1], 
                         self.downward_vertices[:, 2], color='red', s=100, label='Downward Vertices')
            
            if show_edges:
                for edge in self.downward_edges:
                    v1, v2 = self.downward_vertices[edge[0]], self.downward_vertices[edge[1]]
                    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'r-', linewidth=2)
            
            if show_faces:
                for face in self.downward_faces:
                    v1, v2, v3 = self.downward_vertices[face[0]], self.downward_vertices[face[1]], self.downward_vertices[face[2]]
                    verts = [list(v1), list(v2), list(v3)]
                    ax.add_collection3d(plt.art3d.Poly3DCollection([verts], 
                                                               alpha=0.3, color='red'))
            
            # Setting equal aspect ratio
            vertices = np.vstack((self.upward_vertices, self.downward_vertices))
            X = vertices[:, 0]
            Y = vertices[:, 1]
            Z = vertices[:, 2]
            
            max_range = np.array([
                X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()
            ]).max() / 2.0
            
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Merkaba Geometry (Rotation: {self.rotation_angle:.2f} rad)")
            
            # Add legend if showing vertices
            if show_vertices:
                ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Merkaba visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in Merkaba visualization: {str(e)}")
            return False
    
    def visualize_3d_pattern(self, threshold=1.0, opacity=0.4, 
                           show=True, save_path=None):
        """
        Visualize the 3D pattern of the Merkaba.
        
        Args:
            threshold (float): Value threshold for visualization
            opacity (float): Opacity of the volume rendering
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            if self.pattern_3d is None:
                self.generate_3d_pattern()
                
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get the data
            x, y, z = np.indices(self.pattern_3d.shape)
            
            # Create a mask for thresholding
            mask = self.pattern_3d > threshold
            
            # Plot the voxels
            colors = np.empty(self.pattern_3d.shape + (4,))
            
            # Color mapping: blue for upward, red for downward, purple for intersection
            colors[..., 0] = 1.0  # Red channel
            colors[..., 1] = 0.0  # Green channel
            colors[..., 2] = 1.0  # Blue channel
            colors[..., 3] = opacity * (self.pattern_3d > threshold)  # Alpha channel
            
            ax.voxels(mask, facecolors=colors)
            
            # Add tetrahedron edges for better visualization
            for edge in self.upward_edges:
                v1, v2 = self.upward_vertices[edge[0]], self.upward_vertices[edge[1]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'b-', linewidth=1)
                
            for edge in self.downward_edges:
                v1, v2 = self.downward_vertices[edge[0]], self.downward_vertices[edge[1]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'r-', linewidth=1)
            
            # Setting equal aspect ratio
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Merkaba 3D Pattern (Rotation: {self.rotation_angle:.2f} rad)")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"3D pattern visualization saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in 3D pattern visualization: {str(e)}")
            return False
    
    def get_merkaba_metrics(self):
        """
        Get metrics about the Merkaba pattern.
        
        Returns:
            dict: Dictionary of merkaba metrics
        """
        # Calculate the volume of a tetrahedron
        def tetrahedron_volume(vertices):
            v1, v2, v3, v4 = vertices
            return abs(np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4))) / 6
        
        # Calculate activation energy based on rotation angle
        # Maximum at phi * pi radians
        rotation_phase = (self.rotation_angle % (2 * np.pi)) / (2 * np.pi)
        activation_energy = np.sin(rotation_phase * np.pi * self.phi_ratio) ** 2
        
        # Calculate intersection volume (approximation)
        if self.pattern_3d is not None:
            intersection_volume = np.sum(self.pattern_3d > 1.5) / np.sum(self.pattern_3d > 0.5)
        else:
            intersection_volume = 0.0
        
        # Calculate symmetry metric
        upward_volume = tetrahedron_volume(self.upward_vertices)
        downward_volume = tetrahedron_volume(self.downward_vertices)
        volume_ratio = min(upward_volume, downward_volume) / max(upward_volume, downward_volume)
        
        # Calculate harmonic resonance based on phi ratio
        harmonic_resonance = np.cos(rotation_phase * 2 * np.pi * self.phi_ratio) ** 2
        
        # Combine metrics
        metrics = {
            'radius': self.radius,
            'rotation_angle': self.rotation_angle,
            'phi_ratio': self.phi_ratio,
            'upward_volume': upward_volume,
            'downward_volume': downward_volume,
            'volume_ratio': volume_ratio,
            'activation_energy': activation_energy,
            'intersection_volume': intersection_volume,
            'harmonic_resonance': harmonic_resonance,
            'dimensional_stability': activation_energy * harmonic_resonance
        }
        
        return metrics
    
    def __str__(self):
        """String representation of the Merkaba pattern."""
        metrics = self.get_merkaba_metrics()
        
        return (f"Merkaba Pattern\n"
                f"Radius: {self.radius}\n"
                f"Rotation Angle: {self.rotation_angle:.4f} rad\n"
                f"Phi Ratio: {self.phi_ratio}\n"
                f"Activation Energy: {metrics['activation_energy']:.4f}\n"
                f"Harmonic Resonance: {metrics['harmonic_resonance']:.4f}\n"
                f"Dimensional Stability: {metrics['dimensional_stability']:.4f}\n"
                f"Resolution: {self.resolution}x{self.resolution}x{self.resolution}")

    def get_base_glyph_elements(self) -> Dict[str, Any]:
        """
        Returns 3D lines for Merkaba base glyph.
        """
        if not hasattr(self, 'upward_vertices'): self.upward_vertices = self._generate_tetrahedron_vertices(up=True)
        if not hasattr(self, 'downward_vertices'): self.downward_vertices = self._generate_tetrahedron_vertices(up=False)
        if not hasattr(self, 'upward_edges'): self.upward_edges = self._generate_tetrahedron_edges(self.upward_vertices)
        if not hasattr(self, 'downward_edges'): self.downward_edges = self._generate_tetrahedron_edges(self.downward_vertices)

        lines_data = []
        all_verts_list_mk = [] # Renamed

        up_verts_mk_np = np.array(self.upward_vertices) # Renamed
        all_verts_list_mk.extend(up_verts_mk_np.tolist())
        for v_idx1, v_idx2 in self.upward_edges:
            lines_data.append((up_verts_mk_np[v_idx1].tolist(), up_verts_mk_np[v_idx2].tolist()))

        down_verts_mk_np = np.array(self.downward_vertices) # Renamed
        all_verts_list_mk.extend(down_verts_mk_np.tolist())
        for v_idx1, v_idx2 in self.downward_edges:
            lines_data.append((down_verts_mk_np[v_idx1].tolist(), down_verts_mk_np[v_idx2].tolist()))
        
        overall_verts_mk_np = np.array(all_verts_list_mk) # Renamed
        min_coords_mk = np.min(overall_verts_mk_np, axis=0); max_coords_mk = np.max(overall_verts_mk_np, axis=0) # Renamed
        padding_mk = self.radius * 0.15 # Renamed

        return {
            'lines': lines_data,
            'projection_type': '3d',
            'bounding_box': {
                'xmin': float(min_coords_mk[0]-padding_mk), 'xmax': float(max_coords_mk[0]+padding_mk),
                'ymin': float(min_coords_mk[1]-padding_mk), 'ymax': float(max_coords_mk[1]+padding_mk),
                'zmin': float(min_coords_mk[2]-padding_mk), 'zmax': float(max_coords_mk[2]+padding_mk),
            }
        }
if __name__ == "__main__":
    # Example usage
    merkaba = Merkaba(radius=1.0, resolution=64, phi_ratio=1.618)
    print(merkaba)
    
    # Generate 3D pattern
    pattern_3d = merkaba.get_3d_pattern()
    
    # Visualize
    merkaba.visualize_merkaba_3d(save_path="merkaba_geometry.png")
    merkaba.visualize_3d_pattern(save_path="merkaba_pattern.png")
    
    # Activate
    activation_patterns = merkaba.activate_merkaba(num_steps=12, phi_modulation=True)
    print(f"Merkaba activated through {len(activation_patterns)} steps")
    
    # Get metrics after activation
    metrics = merkaba.get_merkaba_metrics()
    print(f"Activation Energy: {metrics['activation_energy']:.4f}")
    print(f"Dimensional Stability: {metrics['dimensional_stability']:.4f}")