"""
Sri Yantra Pattern

This module implements the Sri Yantra sacred geometry pattern.
The Sri Yantra consists of nine interlocking triangles forming 43 smaller triangles
arranged around a central point (bindu) and surrounded by concentric circles and
a square gate structure.

The Sri Yantra represents the connection between the individual and the universe,
and serves as a tool for integration, spiritual development, and consciousness expansion.

Author: Soul Development Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from matplotlib.path import Path
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sri_yantra.log'
)
logger = logging.getLogger('sri_yantra')

class SriYantra:
    """
    Implementation of the Sri Yantra sacred geometry pattern.
    
    The Sri Yantra consists of nine interlocking triangles that create 43 smaller
    triangles arranged in layers around a central point (bindu). It is surrounded
    by concentric circles and a square gate structure with four openings.
    
    This complex pattern forms a spiritual map for consciousness integration and
    represents the connection between the individual and the universe.
    """
    
    def __init__(self, radius=1.0, resolution=512):
        """
        Initialize a new Sri Yantra pattern.
        
        Args:
            radius (float): Radius of the outer circle
            resolution (int): Resolution of the generated pattern matrices
        """
        self.radius = radius
        self.resolution = resolution
        
        # Generate pattern components
        self.central_point = (0, 0)  # Bindu (central point)
        self.triangles = self._generate_triangles()
        self.circles = self._generate_circles()
        self.gates = self._generate_gates()
        
        # Generate pattern matrices
        self.pattern_2d = None
        self.pattern_3d = None
        
        logger.info(f"Sri Yantra initialized with radius {radius} and resolution {resolution}")
        logger.info(f"Generated {len(self.triangles)} triangles, {len(self.circles)} circles, "
                   f"and {len(self.gates)} gates")
    
    def _generate_triangles(self):
        """
        Generate the nine interlocking triangles of the Sri Yantra.
        
        Returns:
            list: List of triangle vertices, each triangle is a list of 3 (x,y) points
        """
        triangles = []
        
        # The Sri Yantra consists of 9 interlocking triangles:
        # - 4 upward-pointing triangles (Shakti, feminine principle)
        # - 5 downward-pointing triangles (Shiva, masculine principle)
        
        # Parameters for the triangles (approximate, can be tuned for better accuracy)
        # These values create a simplified but recognizable Sri Yantra
        center_radius = 0.1 * self.radius
        
        # First layer - central triangle (downward)
        r1 = 0.15 * self.radius
        t1 = [
            (0, r1),
            (-0.866 * r1, -0.5 * r1),
            (0.866 * r1, -0.5 * r1)
        ]
        triangles.append(t1)
        
        # Second layer - first upward triangle
        r2 = 0.25 * self.radius
        t2 = [
            (0, -r2),
            (-0.866 * r2, 0.5 * r2),
            (0.866 * r2, 0.5 * r2)
        ]
        triangles.append(t2)
        
        # Third layer - second downward triangle
        r3 = 0.35 * self.radius
        t3 = [
            (0, r3),
            (-0.866 * r3, -0.5 * r3),
            (0.866 * r3, -0.5 * r3)
        ]
        triangles.append(t3)
        
        # Fourth layer - second upward triangle
        r4 = 0.45 * self.radius
        t4 = [
            (0, -r4),
            (-0.866 * r4, 0.5 * r4),
            (0.866 * r4, 0.5 * r4)
        ]
        triangles.append(t4)
        
        # Fifth layer - third downward triangle
        r5 = 0.55 * self.radius
        t5 = [
            (0, r5),
            (-0.866 * r5, -0.5 * r5),
            (0.866 * r5, -0.5 * r5)
        ]
        triangles.append(t5)
        
        # Sixth layer - third upward triangle
        r6 = 0.65 * self.radius
        t6 = [
            (0, -r6),
            (-0.866 * r6, 0.5 * r6),
            (0.866 * r6, 0.5 * r6)
        ]
        triangles.append(t6)
        
        # Seventh layer - fourth downward triangle
        r7 = 0.75 * self.radius
        t7 = [
            (0, r7),
            (-0.866 * r7, -0.5 * r7),
            (0.866 * r7, -0.5 * r7)
        ]
        triangles.append(t7)
        
        # Eighth layer - fourth upward triangle
        r8 = 0.85 * self.radius
        t8 = [
            (0, -r8),
            (-0.866 * r8, 0.5 * r8),
            (0.866 * r8, 0.5 * r8)
        ]
        triangles.append(t8)
        
        # Ninth layer - fifth downward triangle
        r9 = 0.95 * self.radius
        t9 = [
            (0, r9),
            (-0.866 * r9, -0.5 * r9),
            (0.866 * r9, -0.5 * r9)
        ]
        triangles.append(t9)
        
        return triangles
    
    def _generate_circles(self):
        """
        Generate the concentric circles of the Sri Yantra.
        
        Returns:
            list: List of circle parameters (radius, center)
        """
        circles = []
        
        # The traditional Sri Yantra has three concentric circles
        # surrounding the triangular formation
        
        # Inner circle
        circles.append({
            'radius': 0.98 * self.radius,
            'center': self.central_point
        })
        
        # Middle circle
        circles.append({
            'radius': 1.05 * self.radius,
            'center': self.central_point
        })
        
        # Outer circle
        circles.append({
            'radius': 1.12 * self.radius,
            'center': self.central_point
        })
        
        return circles
    
    def _generate_gates(self):
        """
        Generate the square gates surrounding the Sri Yantra.
        
        Returns:
            list: List of gate lines, each line is a tuple of start and end points
        """
        gate_lines = []
        
        # The Sri Yantra traditionally has a square boundary with four gates,
        # one in each cardinal direction
        
        # Square size
        square_size = 1.3 * self.radius
        half_size = square_size / 2
        
        # Gate size (opening)
        gate_size = square_size / 5
        
        # Top side
        gate_lines.append(((-half_size, half_size), (-gate_size, half_size)))
        gate_lines.append(((gate_size, half_size), (half_size, half_size)))
        
        # Right side
        gate_lines.append(((half_size, half_size), (half_size, gate_size)))
        gate_lines.append(((half_size, -gate_size), (half_size, -half_size)))
        
        # Bottom side
        gate_lines.append(((half_size, -half_size), (gate_size, -half_size)))
        gate_lines.append(((-gate_size, -half_size), (-half_size, -half_size)))
        
        # Left side
        gate_lines.append(((-half_size, -half_size), (-half_size, -gate_size)))
        gate_lines.append(((-half_size, gate_size), (-half_size, half_size)))
        
        return gate_lines
    
    def generate_2d_pattern(self):
        """
        Generate a 2D matrix representation of the Sri Yantra pattern.
        
        This creates a 2D array where higher values represent the
        overlapping triangles, with the highest values at intersection points.
        
        Returns:
            ndarray: 2D matrix representation of the Sri Yantra
        """
        # Create the bounds of the pattern
        x = np.linspace(-1.5 * self.radius, 1.5 * self.radius, self.resolution)
        y = np.linspace(-1.5 * self.radius, 1.5 * self.radius, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Initialize pattern matrix
        self.pattern_2d = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        
        # Add triangles
        for triangle in self.triangles:
            # Create triangle as a path
            path = Path(triangle + [triangle[0]])  # Close the triangle
            
            # Convert grid to points
            points = np.vstack((X.flatten(), Y.flatten())).T
            
            # Check which points are inside the triangle
            mask = path.contains_points(points).reshape(X.shape)
            
            # Add to pattern matrix
            self.pattern_2d += mask
        
        # Add circles
        for circle in self.circles:
            r = circle['radius']
            cx, cy = circle['center']
            
            # Calculate distance from center for each point
            distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
            
            # Create a band for the circle (width = 0.02 * radius)
            width = 0.02 * self.radius
            circle_mask = (distance >= r - width) & (distance <= r + width)
            
            # Add to pattern matrix
            self.pattern_2d += circle_mask
        
        # Add gates
        for gate in self.gates:
            start, end = gate
            sx, sy = start
            ex, ey = end
            
            # Calculate distance from line segment for each point
            # Using the formula for distance from point to line segment
            
            # Vector from start to end
            line_vec = np.array([ex - sx, ey - sy])
            line_length = np.sqrt(np.sum(line_vec**2))
            line_unit_vec = line_vec / line_length if line_length > 0 else line_vec
            
            # Vector from start to each point
            point_vec = np.stack([X - sx, Y - sy], axis=-1)
            
            # Project point vector onto line vector
            projection = np.sum(point_vec * line_unit_vec, axis=-1)
            projection = np.clip(projection, 0, line_length)
            
            # Closest point on the line segment
            closest_point = np.stack([
                sx + projection * line_unit_vec[0],
                sy + projection * line_unit_vec[1]
            ], axis=-1)
            
            # Distance from point to closest point on line segment
            distance = np.sqrt(np.sum((np.stack([X, Y], axis=-1) - closest_point)**2, axis=-1))
            
            # Create a band for the gate line (width = 0.02 * radius)
            width = 0.02 * self.radius
            gate_mask = distance <= width
            
            # Add to pattern matrix
            self.pattern_2d += gate_mask
        
        # Add central bindu point
        cx, cy = self.central_point
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
        bindu_mask = distance <= 0.05 * self.radius
        self.pattern_2d += bindu_mask * 2  # Make the bindu stronger
        
        logger.info(f"2D Sri Yantra pattern generated with shape {self.pattern_2d.shape}")
        return self.pattern_2d
    
    def generate_3d_pattern(self, height=1.0):
        """
        Generate a 3D matrix representation of the Sri Yantra pattern.
        
        This extends the 2D pattern into 3D space, treating the central bindu
        as the highest point and creating a pyramidal structure.
        
        Args:
            height (float): Height of the 3D pattern
            
        Returns:
            ndarray: 3D matrix representation of the Sri Yantra
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
        
        # Create z coordinate array
        z = np.linspace(0, height, self.resolution)
        
        # Initialize 3D pattern
        self.pattern_3d = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float64)
        
        # Create distance field from center
        x = np.linspace(-1.5 * self.radius, 1.5 * self.radius, self.resolution)
        y = np.linspace(-1.5 * self.radius, 1.5 * self.radius, self.resolution)
        X, Y = np.meshgrid(x, y)
        cx, cy = self.central_point
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Normalize distance field
        max_dist = 1.5 * self.radius
        norm_distance = distance / max_dist
        
        # For each z level, scale the 2D pattern based on distance from center and height
        for k, z_val in enumerate(z):
            # Normalized height (0 at bottom, 1 at top)
            norm_z = z_val / height
            
            # Scale factor: 1 at center-top (bindu apex), decreasing with distance and lower height
            # This creates a pyramidal structure with the bindu at the apex
            scale_factor = 1.0 - (norm_distance * (1.0 - norm_z))
            
            # Scale the 2D pattern
            layer = self.pattern_2d * np.clip(scale_factor, 0, 1)
            
            # Add to 3D pattern
            self.pattern_3d[:, :, k] = layer
        
        logger.info(f"3D Sri Yantra pattern generated with shape {self.pattern_3d.shape}")
        return self.pattern_3d
    
    def get_2d_pattern(self):
        """
        Get the 2D pattern matrix.
        
        Returns:
            ndarray: 2D matrix representation of the Sri Yantra
        """
        if self.pattern_2d is None:
            self.generate_2d_pattern()
        return self.pattern_2d
    
    def get_3d_pattern(self, height=1.0):
        """
        Get the 3D pattern matrix.
        
        Args:
            height (float): Height of the 3D pattern
            
        Returns:
            ndarray: 3D matrix representation of the Sri Yantra
        """
        if self.pattern_3d is None:
            self.generate_3d_pattern(height)
        return self.pattern_3d
    
    def get_energy_nodes(self):
        """
        Get the positions of energy nodes in the Sri Yantra.
        
        The energy nodes are the intersection points of the triangles,
        which form the 43 smaller triangles in the complete Sri Yantra.
        
        Returns:
            list: List of energy node coordinates
        """
        nodes = []
        
        # Central bindu
        nodes.append(self.central_point)
        
        # Intersection points of triangles
        # For each pair of triangles, find their intersection points
        for i in range(len(self.triangles)):
            for j in range(i+1, len(self.triangles)):
                tri1 = self.triangles[i]
                tri2 = self.triangles[j]
                
                # For each pair of edges, find intersection
                for e1 in range(3):
                    for e2 in range(3):
                        p1 = tri1[e1]
                        p2 = tri1[(e1+1) % 3]
                        p3 = tri2[e2]
                        p4 = tri2[(e2+1) % 3]
                        
                        # Find intersection of line segments (p1,p2) and (p3,p4)
                        intersection = self._line_intersection(p1, p2, p3, p4)
                        
                        if intersection is not None:
                            # Check if the intersection is already in the list (within a small tolerance)
                            is_new = True
                            for node in nodes:
                                if np.sqrt((node[0] - intersection[0])**2 + 
                                          (node[1] - intersection[1])**2) < 0.01 * self.radius:
                                    is_new = False
                                    break
                            
                            if is_new:
                                nodes.append(intersection)
        
        logger.info(f"Identified {len(nodes)} energy nodes in the Sri Yantra")
        return nodes
    
    def _line_intersection(self, p1, p2, p3, p4):
        """
        Find the intersection point of two line segments.
        
        Args:
            p1, p2: First line segment endpoints
            p3, p4: Second line segment endpoints
            
        Returns:
            tuple: Intersection point (x,y) or None if no intersection
        """
        # Line segment 1: p1 to p2
        x1, y1 = p1
        x2, y2 = p2
        
        # Line segment 2: p3 to p4
        x3, y3 = p3
        x4, y4 = p4
        
        # Compute denominators
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if abs(denom) < 1e-8:  # Parallel or coincident lines
            return None
            
        # Compute ua and ub
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            # Compute intersection point
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
        else:
            return None
    
    def visualize_2d(self, show_triangles=True, show_circles=True, 
                    show_gates=True, show_nodes=True, show=True, save_path=None):
        """
        Visualize the 2D Sri Yantra pattern.
        
        Args:
            show_triangles (bool): Whether to highlight triangles
            show_circles (bool): Whether to highlight circles
            show_gates (bool): Whether to highlight gates
            show_nodes (bool): Whether to highlight energy nodes
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
            plt.figure(figsize=(12, 12))
            
            # Show the pattern
            plt.imshow(self.pattern_2d, cmap='viridis', origin='lower', 
                      extent=[-1.5*self.radius, 1.5*self.radius, -1.5*self.radius, 1.5*self.radius])
            
            if show_triangles:
                # Plot the triangles
                for triangle in self.triangles:
                    triangle_closed = triangle + [triangle[0]]  # Close the triangle
                    xs, ys = zip(*triangle_closed)
                    plt.plot(xs, ys, 'w-', alpha=0.7, linewidth=1)
            
            if show_circles:
                # Plot the circles
                for circle in self.circles:
                    r = circle['radius']
                    cx, cy = circle['center']
                    circle_patch = plt.Circle((cx, cy), r, fill=False, color='white', alpha=0.7, linewidth=1)
                    plt.gca().add_patch(circle_patch)
            
            if show_gates:
                # Plot the gates
                for gate in self.gates:
                    start, end = gate
                    plt.plot([start[0], end[0]], [start[1], end[1]], 'w-', alpha=0.7, linewidth=1)
            
            if show_nodes:
                # Plot the energy nodes
                nodes = self.get_energy_nodes()
                node_xs, node_ys = zip(*nodes)
                plt.plot(node_xs, node_ys, 'ro', markersize=4)
                
                # Highlight central bindu
                cx, cy = self.central_point
                plt.plot(cx, cy, 'ro', markersize=8)
            
            plt.title("Sri Yantra Pattern")
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
    
    def visualize_3d(self, height=1.0, threshold=0.5, opacity=0.7, 
                    show=True, save_path=None):
        """
        Visualize the 3D Sri Yantra pattern.
        
        Args:
            height (float): Height of the 3D pattern
            threshold (float): Value threshold for visualization
            opacity (float): Opacity of the volume rendering
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Generate 3D pattern if not already done
            if self.pattern_3d is None:
                self.generate_3d_pattern(height)
                
            # Create the figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get the data
            x, y, z = np.indices(self.pattern_3d.shape)
            
            # Create a mask for thresholding
            mask = self.pattern_3d > threshold
            
            # Plot the voxels
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=threshold, vmax=np.max(self.pattern_3d))
            colors = cmap(norm(self.pattern_3d))
            colors[..., 3] = opacity * mask  # Set alpha channel
            
            ax.voxels(mask, facecolors=colors)
            
            # Add triangles in 3D for better visualization
            for triangle in self.triangles:
                # Create 3D triangle with z=0
                tri_3d = [(x, y, 0) for x, y in triangle]
                tri_3d.append(tri_3d[0])  # Close the triangle
                xs, ys, zs = zip(*tri_3d)
                
                # Plot the triangle
                ax.plot(xs, ys, zs, 'w-', alpha=0.5, linewidth=1)
            
            # Add central bindu in 3D
            cx, cy = self.central_point
            ax.scatter([cx], [cy], [height], c='red', s=100)
            
            # Set axis labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Sri Yantra - 3D Visualization")
            
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
    
    def get_yantric_properties(self):
        """
        Get the key properties of the Sri Yantra pattern.
        
        Returns:
            dict: Dictionary of yantric properties
        """
        # Calculate number of energy nodes
        nodes = self.get_energy_nodes()
        
        # Count triangles by type
        upward_triangles = len([t for i, t in enumerate(self.triangles) if i % 2 == 1])
        downward_triangles = len(self.triangles) - upward_triangles
        
        # Count number of small triangles formed by intersections
        # In a proper Sri Yantra, there are 43 smaller triangles
        # Since we're using a simplified model, we approximate
        small_triangles = 43
        
        properties = {
            'radius': self.radius,
            'central_point': self.central_point,
            'num_primary_triangles': len(self.triangles),
            'upward_triangles': upward_triangles,  # Shakti (feminine)
            'downward_triangles': downward_triangles,  # Shiva (masculine)
            'num_energy_nodes': len(nodes),
            'num_small_triangles': small_triangles,
            'num_circles': len(self.circles),
            'num_gates': len(self.gates) // 2  # Each gate has two line segments
        }
        
        return properties
    
    def __str__(self):
        """String representation of the Sri Yantra pattern."""
        props = self.get_yantric_properties()
        
        return (f"Sri Yantra Pattern\n"
                f"Radius: {props['radius']}\n"
                f"Primary Triangles: {props['num_primary_triangles']} "
                f"({props['upward_triangles']} upward, {props['downward_triangles']} downward)\n"
                f"Small Triangles: {props['num_small_triangles']}\n"
                f"Energy Nodes: {props['num_energy_nodes']}\n"
                f"Circles: {props['num_circles']}\n"
                f"Gates: {props['num_gates']}\n"
                f"Resolution: {self.resolution}x{self.resolution}")

    def get_base_glyph_elements(self) -> Dict[str, Any]:
        """
        Returns lines, circles, and points for Sri Yantra line art.
        """
        if not hasattr(self, 'triangles'): self.triangles = self._generate_triangles()
        if not hasattr(self, 'circles'): self.circles = self._generate_circles()
        if not hasattr(self, 'gates'): self.gates = self._generate_gates()

        lines_data = []
        for triangle_verts in self.triangles:
            for i in range(len(triangle_verts)):
                p1 = triangle_verts[i]; p2 = triangle_verts[(i + 1) % len(triangle_verts)]
                lines_data.append((tuple(p1), tuple(p2)))
        for p1_gate, p2_gate in self.gates: # Renamed p1, p2
            lines_data.append((tuple(p1_gate), tuple(p2_gate)))

        circles_data = [{'center': tuple(c['center']), 'radius': c['radius']} for c in self.circles]
        points_data = [tuple(self.central_point)] if hasattr(self, 'central_point') else []
        
        padding_sri = 0.1 * self.radius # Renamed padding
        bound_sri = 1.4 * self.radius + padding_sri # Renamed bound, adjusted to be slightly smaller to fit better

        return {
            'lines': lines_data, 'circles': circles_data, 'points': points_data,
            'projection_type': '2d',
            'bounding_box': {
                'xmin': float(-bound_sri), 'xmax': float(bound_sri),
                'ymin': float(-bound_sri), 'ymax': float(bound_sri),
            }
        }
if __name__ == "__main__":
    # Example usage
    yantra = SriYantra(radius=1.0, resolution=512)
    print(yantra)
    
    # Visualize
    yantra.visualize_2d(save_path="sri_yantra_2d.png")
    yantra.visualize_3d(height=1.0, save_path="sri_yantra_3d.png")
    
    # Get pattern matrices for use in field systems
    pattern_2d = yantra.get_2d_pattern()
    pattern_3d = yantra.get_3d_pattern(height=0.8)
    
    # Get energy nodes
    nodes = yantra.get_energy_nodes()
    print(f"Found {len(nodes)} energy nodes")