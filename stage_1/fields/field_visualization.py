"""
Field System Visualization Module

Provides comprehensive 3D and 2D visualization capabilities for the
field system components including VoidField, SephirothField, 
and overall field interactions.
"""

import logging
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import plasma
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

# Constants (import from project constants if needed)
PI = 3.14159265358979323846

logger = logging.getLogger(__name__)

class FieldVisualizer:
    """Creates visualizations for the field system and its components."""
    
    def __init__(self, output_dir: str = "output/visualizations/fields"):
        """Initialize the field visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise
            
        # Define colormaps for different field properties
        self.energy_cmap = LinearSegmentedColormap.from_list(
            "energy_cmap", ["#0074D9", "#7FDBFF", "#FFDC00", "#FF4136"]
        )  # Blue -> Light Blue -> Yellow -> Red (Low to High)
        
        self.frequency_cmap = LinearSegmentedColormap.from_list(
            "frequency_cmap", ["#B10DC9", "#F012BE", "#FF4136", "#FF851B", "#FFDC00", "#2ECC40", "#0074D9"]
        )  # Purple -> Pink -> Red -> Orange -> Yellow -> Green -> Blue
        
        self.coherence_cmap = LinearSegmentedColormap.from_list(
            "coherence_cmap", ["#111111", "#AAAAAA", "#FF851B", "#FFDC00", "#FFFFFF"]
        )  # Black -> Grey -> Orange -> Yellow -> White
        
        self.phi_resonance_cmap = LinearSegmentedColormap.from_list(
            "phi_resonance_cmap", ["#001f3f", "#0074D9", "#7FDBFF", "#39CCCC", "#3D9970", "#2ECC40", "#01FF70"]
        )  # Dark Blue -> Blue -> Light Blue -> Teal -> Green -> Bright Green

        logger.info(f"FieldVisualizer initialized")
    
    def _save_or_show(self, fig, filename: str, show: bool, save: bool):
        """Helper to save and/or show a matplotlib figure."""
        try:
            if save:
                full_path = os.path.join(self.output_dir, filename)
                fig.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {full_path}")
            if show:
                # Check if running in interactive environment before showing
                if hasattr(plt, 'isinteractive') and plt.isinteractive():
                    plt.show()
                else:
                    logger.info("Non-interactive environment detected, plot not shown automatically.")
            if not show:  # Close if not shown to free memory
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error saving/showing plot {filename}: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
    
    def visualize_void_field_slice(self, void_field, property_name: str = 'energy', 
                            slice_axis: int = 2, slice_idx: Optional[int] = None,
                            show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize a 2D slice of the void field grid.
        
        Args:
            void_field: VoidField instance
            property_name: Property to visualize ('energy', 'frequency', 'coherence', etc.)
            slice_axis: Axis to slice along (0=x, 1=y, 2=z)
            slice_idx: Index for the slice (if None, uses the middle)
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(10, 8))
        
        try:
            # Get the grid data
            if hasattr(void_field, 'grid') and hasattr(void_field.grid, property_name):
                grid_data = getattr(void_field.grid, property_name)
            else:
                logger.error(f"Property {property_name} not found in void_field.grid")
                return None
                
            # Determine slice index if not provided
            if slice_idx is None:
                slice_idx = grid_data.shape[slice_axis] // 2
                
            # Select the appropriate colormap
            cmap = getattr(self, f"{property_name}_cmap", plt.colormaps.get('viridis'))
            
            # Create the slice view
            slices = [slice(None)] * 3
            slices[slice_axis] = slice_idx
            slice_data = grid_data[tuple(slices)]
            
            # Apply Gaussian smoothing for better visualization
            smoothed_data = gaussian_filter(slice_data, sigma=1.0)
            
            # Create heat map
            ax = fig.add_subplot(111)
            im = ax.imshow(smoothed_data.T, origin='lower', cmap=cmap, interpolation='bilinear')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(property_name.capitalize())
            
            # Determine axis labels
            axis_labels = ['X', 'Y', 'Z']
            visible_axes = [i for i in range(3) if i != slice_axis]
            ax.set_xlabel(f"{axis_labels[visible_axes[0]]} Position")
            ax.set_ylabel(f"{axis_labels[visible_axes[1]]} Position")
            
            # Set title
            ax.set_title(f"Void Field {property_name.capitalize()} (Slice at {axis_labels[slice_axis]}={slice_idx})")
            
            self._save_or_show(fig, f"void_field_{property_name}_slice_{axis_labels[slice_axis]}{slice_idx}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating void field slice visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_edge_of_chaos(self, void_field, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the Edge of Chaos regions in the void field.
        
        Args:
            void_field: VoidField instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(12, 10))
        
        try:
            # Get the Edge of Chaos calculation result
            if hasattr(void_field, 'calculate_edge_of_chaos'):
                edge_values = void_field.calculate_edge_of_chaos()
            else:
                logger.error("VoidField instance does not have calculate_edge_of_chaos method")
                return None
                
            # Create a grid of subplots for different slices
            axes = []
            n_slices = min(4, edge_values.shape[2])  # Show up to 4 slices
            slice_indices = np.linspace(0, edge_values.shape[2]-1, n_slices, dtype=int)
            
            for i, slice_idx in enumerate(slice_indices):
                ax = fig.add_subplot(2, 2, i+1)
                axes.append(ax)
                
                # Get a slice and apply smoothing
                slice_data = edge_values[:, :, slice_idx]
                smoothed_data = gaussian_filter(slice_data, sigma=1.0)
                
                # Create heat map
                im = ax.imshow(smoothed_data.T, origin='lower', cmap='plasma', interpolation='bilinear')
                ax.set_title(f"Z Slice {slice_idx}")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Edge of Chaos Value")
            
            fig.suptitle("Edge of Chaos Distribution", fontsize=16)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            
            self._save_or_show(fig, "void_field_edge_of_chaos.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating Edge of Chaos visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_sephiroth_field(self, sephiroth_field, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize a Sephiroth field's pattern influence.
        
        Args:
            sephiroth_field: SephirothField instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(12, 10))
        
        try:
            # Get field properties
            name = getattr(sephiroth_field, 'name', 'Unknown')
            base_frequency = getattr(sephiroth_field, 'base_frequency', 0.0)
            sephiroth_color = getattr(sephiroth_field, 'color', [0.5, 0.5, 0.5])
            pattern_type = getattr(sephiroth_field, 'pattern_type', 'Unknown')
            pattern_influence = getattr(sephiroth_field, 'pattern_influence', None)
            
            # Convert color to hex for visualization
            color_hex = f"#{int(sephiroth_color[0]*255):02x}{int(sephiroth_color[1]*255):02x}{int(sephiroth_color[2]*255):02x}"
            
            # Create a 3D visualization of the pattern influence
            if pattern_influence is not None and hasattr(pattern_influence, 'shape'):
                ax = fig.add_subplot(111, projection='3d')
                
                # Downsample if grid is too large
                x_step = max(1, pattern_influence.shape[0] // 20)
                y_step = max(1, pattern_influence.shape[1] // 20)
                z_step = max(1, pattern_influence.shape[2] // 20)
                
                # Create grid points
                x, y, z = np.meshgrid(
                    np.arange(0, pattern_influence.shape[0], x_step),
                    np.arange(0, pattern_influence.shape[1], y_step),
                    np.arange(0, pattern_influence.shape[2], z_step)
                )
                
                # Get values at grid points
                values = pattern_influence[::x_step, ::y_step, ::z_step]
                
                # Normalize for scaling points
                norm_values = values / np.max(values) if np.max(values) > 0 else values
                
                # Plot points with size and color based on pattern influence
                scatter = ax.scatter(
                    x, y, z, 
                    c=values.flatten(), 
                    cmap=plt.cm.plasma,
                    alpha=0.6
                )
                
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label("Pattern Influence Strength")
                
                # Set labels and title
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Sephiroth: {name}\nPattern: {pattern_type}, Frequency: {base_frequency:.2f} Hz")
                
                # Add color indicator
                color_label = fig.text(
                    0.85, 0.90, 
                    f"Sephiroth Color", 
                    fontsize=10,
                    ha='center',
                    bbox=dict(facecolor=color_hex, alpha=0.8, pad=4)
                )
            else:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No pattern influence data available", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f"Sephiroth: {name}")
            
            self._save_or_show(fig, f"sephiroth_field_{name.lower()}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating Sephiroth field visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_sephiroth_tree(self, field_controller, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the Tree of Life structure with all Sephiroth.
        
        Args:
            field_controller: FieldController instance containing Sephiroth fields
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(12, 14))
        
        try:
            # Extract Sephiroth fields from the controller
            sephiroth_fields = []
            if hasattr(field_controller, 'sephiroth_fields'):
                sephiroth_fields = field_controller.sephiroth_fields
            elif hasattr(field_controller, 'fields'):
                sephiroth_fields = [f for f in field_controller.fields.values() 
                                   if hasattr(f, 'is_sephiroth') and f.is_sephiroth]
            
            if not sephiroth_fields:
                logger.error("No Sephiroth fields found in field_controller")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No Sephiroth fields available", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self._save_or_show(fig, "sephiroth_tree_empty.png", show, save)
                return fig
            
            # Tree of Life 2D positions (traditional layout)
            tree_positions = {
                "Kether": (5, 9.5),
                "Chokmah": (3, 8),
                "Binah": (7, 8),
                "Chesed": (3, 6),
                "Geburah": (7, 6),
                "Tiphareth": (5, 5),
                "Netzach": (3, 4),
                "Hod": (7, 4),
                "Yesod": (5, 2.5),
                "Malkuth": (5, 1)
            }
            
            # Tree of Life connections (paths)
            tree_paths = [
                ("Kether", "Chokmah"),
                ("Kether", "Binah"),
                ("Chokmah", "Binah"),
                ("Chokmah", "Tiphareth"),
                ("Chokmah", "Chesed"),
                ("Binah", "Tiphareth"),
                ("Binah", "Geburah"),
                ("Chesed", "Geburah"),
                ("Chesed", "Tiphareth"),
                ("Chesed", "Netzach"),
                ("Geburah", "Tiphareth"),
                ("Geburah", "Hod"),
                ("Tiphareth", "Netzach"),
                ("Tiphareth", "Hod"),
                ("Tiphareth", "Yesod"),
                ("Netzach", "Hod"),
                ("Netzach", "Yesod"),
                ("Netzach", "Malkuth"),
                ("Hod", "Yesod"),
                ("Hod", "Malkuth"),
                ("Yesod", "Malkuth")
            ]
            
            ax = fig.add_subplot(111)
            
            # Draw paths (connections)
            for path in tree_paths:
                if path[0] in tree_positions and path[1] in tree_positions:
                    x = [tree_positions[path[0]][0], tree_positions[path[1]][0]]
                    y = [tree_positions[path[0]][1], tree_positions[path[1]][1]]
                    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.7)
            
            # Extract field data for visualization
            sephiroth_data = {}
            for field in sephiroth_fields:
                name = getattr(field, 'name', None)
                if name in tree_positions:
                    sephiroth_data[name] = {
                        'color': getattr(field, 'color', [0.5, 0.5, 0.5]),
                        'strength': getattr(field, 'field_strength', 0.5),
                        'frequency': getattr(field, 'base_frequency', 0.0),
                        'element': getattr(field, 'element', 'unknown')
                    }
            
            # Draw Sephiroth nodes
            for name, position in tree_positions.items():
                if name in sephiroth_data:
                    data = sephiroth_data[name]
                    color = data['color']
                    color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                    size = 1000 * data['strength'] + 500
                    
                    ax.scatter(position[0], position[1], s=size, c=color_hex, 
                              alpha=0.85, edgecolors='white', linewidth=1.5)
                    
                    # Add label with name and frequency
                    ax.text(position[0], position[1], f"{name}\n{data['frequency']:.1f}Hz",
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Customize plot
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            fig.suptitle("Tree of Life - Sephiroth Fields", fontsize=18, y=0.98)
            
            self._save_or_show(fig, "sephiroth_tree.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating Sephiroth Tree visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_soul_field_interaction(self, soul, field, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the interaction between a soul and a field.
        
        Args:
            soul: SoulSpark instance
            field: Field instance (VoidField or SephirothField)
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(10, 8))
        
        try:
            # Get soul properties
            soul_name = getattr(soul, 'name', 'Unknown Soul')
            soul_position = getattr(soul, 'position', [0, 0, 0])
            soul_resonance = getattr(soul, 'resonance', 0.5)
            soul_stability = getattr(soul, 'stability', 0.5)
            soul_coherence = getattr(soul, 'coherence', 0.5)
            soul_energy = getattr(soul, 'energy', 0.5)
            
            # Get field properties
            field_name = getattr(field, 'name', 'Unknown Field')
            field_type = type(field).__name__
            
            # Create 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot soul position
            soul_size = 100 + 200 * soul_energy
            ax.scatter(
                soul_position[0], soul_position[1], soul_position[2],
                s=soul_size, c='blue', alpha=0.7, label='Soul'
            )
            
            # Calculate radius for influence sphere
            influence_radius = 0.5 + soul_resonance * 2.0
            
            # Create sphere surface points
            phi = np.linspace(0, 2*PI, 20)
            theta = np.linspace(0, PI, 20)
            x = influence_radius * np.outer(np.cos(phi), np.sin(theta)) + soul_position[0]
            y = influence_radius * np.outer(np.sin(phi), np.sin(theta)) + soul_position[1]
            z = influence_radius * np.outer(np.ones_like(phi), np.cos(theta)) + soul_position[2]
            
            # Plot influence sphere as a wireframe
            ax.plot_wireframe(x, y, z, color='cyan', alpha=0.3, linewidth=0.6)
            
            # Plot field influence at soul position
            if hasattr(field, 'grid') and hasattr(field.grid, 'energy'):
                # Sample field properties at soul position
                pos_x, pos_y, pos_z = [int(p) % s for p, s in zip(soul_position, field.grid.energy.shape)]
                field_energy = field.grid.energy[pos_x, pos_y, pos_z]
                field_freq = field.grid.frequency[pos_x, pos_y, pos_z] if hasattr(field.grid, 'frequency') else 0
                
                # Draw field influence vectors
                arrow_length = 1.5 * field_energy
                ax.quiver(
                    soul_position[0], soul_position[1], soul_position[2],
                    arrow_length, 0, 0, 
                    color='red', alpha=0.6, arrow_length_ratio=0.3
                )
                ax.quiver(
                    soul_position[0], soul_position[1], soul_position[2],
                    0, arrow_length, 0, 
                    color='green', alpha=0.6, arrow_length_ratio=0.3
                )
                ax.quiver(
                    soul_position[0], soul_position[1], soul_position[2],
                    0, 0, arrow_length, 
                    color='blue', alpha=0.6, arrow_length_ratio=0.3
                )
                
                # Add field info text
                field_info = (
                    f"Field Energy: {field_energy:.2f}\n"
                    f"Field Frequency: {field_freq:.2f}"
                )
            else:
                field_info = "No field grid data available"
            
            # Set labels and title
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            
            title = (
                f"Soul-Field Interaction\n"
                f"Soul: {soul_name} (R:{soul_resonance:.2f}, S:{soul_stability:.2f}, C:{soul_coherence:.2f})\n"
                f"Field: {field_name} ({field_type})"
            )
            ax.set_title(title)
            
            # Add info text to figure
            fig.text(0.02, 0.02, field_info, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Equal aspect ratio for 3D plot
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            y_range = abs(y_limits[1] - y_limits[0])
            z_range = abs(z_limits[1] - z_limits[0])
            max_range = max(x_range, y_range, z_range)
            
            mid_x = np.mean(x_limits)
            mid_y = np.mean(y_limits)
            mid_z = np.mean(z_limits)
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            self._save_or_show(fig, f"soul_field_interaction_{soul_name}_{field_name}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating soul-field interaction visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
    
    def visualize_field_frequency_spectrum(self, field, position: List[int] = None, 
                                   show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Visualize the frequency spectrum at a specific position in the field.
        
        Args:
            field: Field instance
            position: [x, y, z] coordinates (defaults to field center)
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(10, 6))
        
        try:
            field_name = getattr(field, 'name', 'Unknown Field')
            
            # Use field center if position not specified
            if position is None:
                if hasattr(field, 'grid') and hasattr(field.grid, 'energy'):
                    grid_shape = field.grid.energy.shape
                    position = [s // 2 for s in grid_shape]
                else:
                    position = [0, 0, 0]
            
            # Get frequency data
            has_harmonics = False
            if hasattr(field, 'get_frequency_spectrum'):
                frequencies, amplitudes = field.get_frequency_spectrum(position)
                has_harmonics = True
            elif hasattr(field, 'grid') and hasattr(field.grid, 'frequency'):
                # If no spectrum method, use base frequency at position
                pos_x, pos_y, pos_z = [min(p, s-1) for p, s in zip(position, field.grid.frequency.shape)]
                base_freq = field.grid.frequency[pos_x, pos_y, pos_z]
                
                # Generate simple harmonics
                frequencies = np.array([base_freq * i for i in range(1, 11)])  # First 10 harmonics
                amplitudes = np.array([1.0 / i for i in range(1, 11)])  # Amplitude decreases with harmonic number
            else:
                # No frequency data available
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No frequency data available",
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f"Frequency Spectrum - {field_name}")
                self._save_or_show(fig, f"field_frequency_spectrum_{field_name}.png", show, save)
                return fig
            
            # Create plot
            ax = fig.add_subplot(111)
            
            # Plot frequency spectrum
            ax.stem(frequencies, amplitudes, linefmt='b-', markerfmt='bo', basefmt='r-')
            
            if has_harmonics:
                # Highlight phi-related frequencies if available
                phi = (1 + np.sqrt(5)) / 2
                phi_freqs = [f for f in frequencies if any(abs(f / frequencies[0] - phi**i) < 0.1 for i in range(-3, 4))]
                phi_amps = [amplitudes[np.where(frequencies == f)[0][0]] for f in phi_freqs]
                
                if phi_freqs:
                    ax.scatter(phi_freqs, phi_amps, color='gold', s=100, alpha=0.7, zorder=3, 
                              label='Phi-related harmonics')
            
            # Plot formatting
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f"Frequency Spectrum at Position {position}\nField: {field_name}")
            
            if has_harmonics:
                ax.legend()
                
            # Add grid and make plot pretty
            ax.grid(True, alpha=0.3)
            
            self._save_or_show(fig, f"field_frequency_spectrum_{field_name}.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating frequency spectrum visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None
                
    def create_field_dashboard(self, field_controller, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """Create a comprehensive dashboard view of the entire field system.
        
        Args:
            field_controller: FieldController instance
            show: Whether to display the plot
            save: Whether to save the plot to file
            
        Returns:
            Matplotlib figure or None on error
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)
        
        try:
            # --- 1. Void Field Energy (Top Left) ---
            ax1 = fig.add_subplot(gs[0, 0])
            void_field = None
            if hasattr(field_controller, 'void_field'):
                void_field = field_controller.void_field
            elif hasattr(field_controller, 'fields') and 'void' in field_controller.fields:
                void_field = field_controller.fields['void']
                
            if void_field and hasattr(void_field, 'grid') and hasattr(void_field.grid, 'energy'):
                # Get middle slice of energy grid
                z_slice = void_field.grid.energy.shape[2] // 2
                energy_slice = void_field.grid.energy[:, :, z_slice]
                # Apply smoothing
                energy_slice = gaussian_filter(energy_slice, sigma=1.0)
                # Plot
                im1 = ax1.imshow(energy_slice.T, origin='lower', cmap=self.energy_cmap, interpolation='bilinear')
                fig.colorbar(im1, ax=ax1, label='Energy')
                ax1.set_title("Void Field - Energy Distribution")
                ax1.set_xlabel("X Position")
                ax1.set_ylabel("Y Position")
            else:
                ax1.text(0.5, 0.5, "No Void Field data", ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title("Void Field - Energy")
            
            # --- 2. Void Field Edge of Chaos (Top Middle) ---
            ax2 = fig.add_subplot(gs[0, 1])
            if void_field and hasattr(void_field, 'get_edge_of_chaos_grid'): 
                try:
                    edge_values = void_field.get_edge_of_chaos_grid(low_res=True) # MODIFIED - Get grid (low_res recommended)
                    # Get middle slice (adjust if low_res shape differs)
                    z_slice = edge_values.shape[2] // 2
                    edge_slice = edge_values[:, :, z_slice]
                    # Apply smoothing
                    edge_slice = gaussian_filter(edge_slice, sigma=1.0)
                    # Plot (rest of plotting is likely OK)
                    im2 = ax2.imshow(edge_slice.T, origin='lower', cmap='plasma', interpolation='bilinear')
                    fig.colorbar(im2, ax=ax2, label='Edge Value')
                    ax2.set_title("Void Field - Edge of Chaos")
                    ax2.set_xlabel("X Position")
                    ax2.set_ylabel("Y Position")
                except Exception as e:
                    logger.error(f"Error calculating Edge of Chaos: {e}")
                    ax2.text(0.5, 0.5, "Edge of Chaos calculation failed", ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title("Void Field - Edge of Chaos")
            else:
                ax2.text(0.5, 0.5, "No Edge of Chaos data", ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("Void Field - Edge of Chaos")
            
            # --- 3. Sephiroth Tree (Top Right spanning down) ---
            ax3 = fig.add_subplot(gs[0:2, 2])
            
            # Define Tree of Life positions
            tree_positions = {
                "Kether": (5, 9.5),
                "Chokmah": (3, 8),
                "Binah": (7, 8),
                "Chesed": (3, 6),
                "Geburah": (7, 6),
                "Tiphareth": (5, 5),
                "Netzach": (3, 4),
                "Hod": (7, 4),
                "Yesod": (5, 2.5),
                "Malkuth": (5, 1)
            }
            
            # Define paths
            tree_paths = [
                ("Kether", "Chokmah"), ("Kether", "Binah"), ("Chokmah", "Binah"),
                ("Chokmah", "Tiphareth"), ("Chokmah", "Chesed"), ("Binah", "Tiphareth"),
                ("Binah", "Geburah"), ("Chesed", "Geburah"), ("Chesed", "Tiphareth"),
                ("Chesed", "Netzach"), ("Geburah", "Tiphareth"), ("Geburah", "Hod"),
                ("Tiphareth", "Netzach"), ("Tiphareth", "Hod"), ("Tiphareth", "Yesod"),
                ("Netzach", "Hod"), ("Netzach", "Yesod"), ("Netzach", "Malkuth"),
                ("Hod", "Yesod"), ("Hod", "Malkuth"), ("Yesod", "Malkuth")
            ]
            
            # Extract Sephiroth fields data
            sephiroth_fields = []
            if hasattr(field_controller, 'sephiroth_fields'):
                sephiroth_fields = field_controller.sephiroth_fields
            elif hasattr(field_controller, 'fields'):
                sephiroth_fields = [f for f in field_controller.fields.values() 
                                   if hasattr(f, 'is_sephiroth') and f.is_sephiroth]
            
            # Draw paths
            for path in tree_paths:
                if path[0] in tree_positions and path[1] in tree_positions:
                    x = [tree_positions[path[0]][0], tree_positions[path[1]][0]]
                    y = [tree_positions[path[0]][1], tree_positions[path[1]][1]]
                    ax3.plot(x, y, 'k-', linewidth=1, alpha=0.5)
            
            # Extract field data
            sephiroth_data = {}
            for field in sephiroth_fields:
                name = getattr(field, 'name', None)
                if name in tree_positions:
                    color = getattr(field, 'color', [0.5, 0.5, 0.5])
                    # Convert color to hex
                    color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                    strength = getattr(field, 'field_strength', 0.5)
                    sephiroth_data[name] = {
                        'color': color_hex,
                        'strength': strength
                    }
            
            # Draw Sephiroth nodes
            for name, position in tree_positions.items():
                if name in sephiroth_data:
                    data = sephiroth_data[name]
                    ax3.scatter(position[0], position[1], s=700*data['strength']+300, 
                               c=data['color'], alpha=0.7, edgecolors='white', linewidth=1)
                else:
                    ax3.scatter(position[0], position[1], s=200, c='gray', alpha=0.3)
                
                ax3.text(position[0], position[1], name, ha='center', va='center', fontsize=8)
            
            ax3.set_xlim(0, 10)
            ax3.set_ylim(0, 10.5)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title("Tree of Life - Sephiroth Fields")
            
            # --- 4. Field Metrics (Middle Left) ---
            ax4 = fig.add_subplot(gs[1, 0])
            
            # Extract metrics from different fields
            field_metrics = {}
            if void_field:
                field_metrics['Void Energy'] = np.mean(void_field.grid.energy) if hasattr(void_field, 'grid') and hasattr(void_field.grid, 'energy') else 0
                field_metrics['Void Coherence'] = np.mean(void_field.grid.coherence) if hasattr(void_field, 'grid') and hasattr(void_field.grid, 'coherence') else 0
            
            # Add metrics from Sephiroth fields
            sephiroth_energy = [np.mean(f.grid.energy) if hasattr(f, 'grid') and hasattr(f.grid, 'energy') else 0 for f in sephiroth_fields]
            field_metrics['Sephiroth Energy Avg'] = np.mean(sephiroth_energy) if sephiroth_energy else 0
            
            # Plot metrics as horizontal bars
            metrics_names = list(field_metrics.keys())
            metrics_values = list(field_metrics.values())
            y_pos = np.arange(len(metrics_names))
            bars = ax4.barh(y_pos, metrics_values, align='center', color='skyblue', alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(metrics_names)
            ax4.set_xlim(0, 1)
            ax4.set_title("Field System Metrics")
            ax4.set_xlabel("Value")
            
            # Add text labels
            for bar in bars:
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                        va='center', fontsize=8)
            
            # --- 5. Phi Resonance Distribution (Middle Middle) ---
            ax5 = fig.add_subplot(gs[1, 1])
            
            # Create a simple phi resonance distribution visualization
            # This would normally use actual data from the fields
            phi = (1 + np.sqrt(5)) / 2
            x = np.linspace(0, 10, 1000)
            y = np.exp(-0.5 * ((x - phi) / 0.2)**2)  # Gaussian centered at phi
            
            # Add more peaks at phi-related frequencies
            for i in range(1, 4):
                y += 0.7 * np.exp(-0.5 * ((x - phi * i) / (0.2 * i))**2) / i
                y += 0.5 * np.exp(-0.5 * ((x - i / phi) / (0.15 * i))**2) / i
            
            ax5.plot(x, y, 'b-', linewidth=2)
            ax5.fill_between(x, 0, y, color='blue', alpha=0.2)
            
            # Mark phi value
            ax5.axvline(x=phi, color='gold', linestyle='--', alpha=0.7)
            ax5.text(phi+0.1, 0.9*max(y), f'φ = {phi:.3f}', color='darkgoldenrod')
            
            # Format plot
            ax5.set_xlabel('Frequency Ratio')
            ax5.set_ylabel('Resonance Strength')
            ax5.set_title('Phi Resonance Distribution')
            ax5.grid(True, alpha=0.3)
            
            # --- 6. System Status (Bottom row) ---
            ax6 = fig.add_subplot(gs[2, :])
            
            # Create system status visualization
            status_metrics = {
                'Void Field Stability': 0.85,
                'Sephiroth Alignment': 0.92,
                'Harmonic Integrity': 0.78,
                'Pattern Coherence': 0.81,
                'Edge of Chaos Balance': 0.73,
                'System Integration': 0.88
            }
            
            # These would normally be calculated from actual field data
            # Replace with real calculations when integrating
            
            status_names = list(status_metrics.keys())
            status_values = list(status_metrics.values())
            x_pos = np.arange(len(status_names))
            
            # Create stacked bars (green for value, red for remaining)
            bars1 = ax6.bar(x_pos, status_values, 0.7, color='forestgreen', alpha=0.7)
            bars2 = ax6.bar(x_pos, [1-v for v in status_values], 0.7, 
                           bottom=status_values, color='firebrick', alpha=0.4)
            
            # Add labels
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(status_names, rotation=20, ha='right')
            ax6.set_ylim(0, 1)
            ax6.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax6.set_title('Field System Status')
            
            # Add percentage text
            for bar in bars1:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2, height/2, f'{height:.0%}', 
                        ha='center', va='center', color='white', fontweight='bold')
            
            # Main title
            fig.suptitle("Field System Dashboard", fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            self._save_or_show(fig, "field_system_dashboard.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating field dashboard: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
            return None