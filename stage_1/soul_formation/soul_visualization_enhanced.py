# --- START OF FILE soul_visualization_enhanced.py ---

"""
Enhanced Soul Visualization (Refactored)

Provides detailed visualization capabilities for fully formed souls,
creating rich visual representations based on current SoulSpark attributes.
"""

import logging
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple, List

# --- Matplotlib/Plotly Imports ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go
    import plotly.express as px
    import seaborn as sns  # Keep for potential heatmap/pairplot use later
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_ENABLED = True
except ImportError as e:
    logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.warning(
    f"Visualization libraries (matplotlib, plotly, seaborn) not found: {e}. Visualizer disabled.")
    VISUALIZATION_ENABLED = False
    # Define dummy classes/functions if needed to prevent downstream errors
    class plt: pass
    class go: pass
    class px: pass
    class sns: pass
    class Axes3D: pass
    class Poly3DCollection: pass
    def LinearSegmentedColormap(): pass


# --- Other Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from constants.constants import *  # Import constants
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    # Basic log if plt failed too
    if not VISUALIZATION_ENABLED: logging.basicConfig(level=logging.INFO)
    logging.critical(
    f"CRITICAL ERROR: Failed to import SoulSpark or constants: {e}. Visualizer cannot function.")
    DEPENDENCIES_AVAILABLE = False
    # Define SoulSpark placeholder if needed
    class SoulSpark: pass


logger = logging.getLogger(__name__)


class EnhancedSoulVisualizer:
    """ Creates detailed visualizations for a SoulSpark object. """

    def __init__(
    self,
    soul: SoulSpark,
     output_dir: str = "output/visualizations"):
        if not VISUALIZATION_ENABLED or not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(
                "Visualization dependencies or SoulSpark/constants are missing. Visualizer cannot be initialized.")
        if not isinstance(soul, SoulSpark):
            raise TypeError("Input 'soul' must be a SoulSpark instance.")

        self.soul = soul
        # Ensure output dir is specific to the soul ID
        self.output_dir = os.path.join(
    output_dir,
    f"soul_{
        getattr(
            soul,
            'spark_id',
             'unknown')}")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(
    f"Failed to create output directory {
        self.output_dir}: {e}")
            raise  # Fail if dir cannot be made

        logger.info(
    f"EnhancedSoulVisualizer initialized for Soul ID: {
        getattr(
            soul,
            'spark_id',
             'unknown')}")
        # Define colormaps
        self.stability_cmap = LinearSegmentedColormap.from_list(
    "stability_cmap", [
        "#FF4136", "#FFDC00", "#7FDBFF", "#0074D9"])  # Red-Yellow-LightBlue-Blue
        self.coherence_cmap = LinearSegmentedColormap.from_list(
    "coherence_cmap", [
        "#AAAAAA", "#FF851B", "#FFDC00", "#FFFFFF"])  # Grey-Orange-Yellow-White
        # Teal-Green-Yellow-Red (Low to High?) -> Let's reverse:
        # Red-Yellow-Green-Teal (Low to High)
        self.energy_cmap = LinearSegmentedColormap.from_list(
            "energy_cmap", ["#3D9970", "#2ECC40", "#FFDC00", "#FF4136"])
        self.energy_cmap = LinearSegmentedColormap.from_list(
            "energy_cmap", ["#FF4136", "#FFDC00", "#2ECC40", "#3D9970"])
        self.aspect_cmap = plt.get_cmap('viridis')

    def _save_or_show(self, fig, filename: str, show: bool, save: bool):
        """ Helper to save and/or show a matplotlib figure. """
        if not VISUALIZATION_ENABLED: return
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
                    logger.info(
                        "Non-interactive environment detected, plot not shown automatically.")
                    # Consider plt.draw() and plt.pause(0.001) if needed in
                    # some backends
            if not show:  # Close if not shown to free memory
                plt.close(fig)
        except Exception as e:
            logger.error(
    f"Error saving/showing plot {filename}: {e}",
     exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)

    def _get_soul_color_rgb(self) -> List[float]:
        """ Safely gets the soul's primary color as an RGB list. """
        color_name = getattr(
    self.soul,
    'soul_color',
     'grey')  # Default to grey
        if not isinstance(color_name, str): color_name = 'grey'

        # Reuse the color conversion logic from SephirothField (or move to a
        # utility)
        color_map = {
            'white': [1.0, 1.0, 1.0], 'grey': [0.5, 0.5, 0.5], 'black': [0.05, 0.05, 0.05],
            'blue': [0.1, 0.2, 0.9], 'red': [0.9, 0.1, 0.1], 'yellow': [0.9, 0.9, 0.1],
            'gold': [1.0, 0.84, 0.0], 'green': [0.1, 0.9, 0.2], 'orange': [1.0, 0.65, 0.0],
            'purple': [0.5, 0.1, 0.9], 'violet': [0.58, 0.0, 0.83], 'lavender': [0.9, 0.9, 1.0],
            'earth_tones': [0.6, 0.4, 0.2], 'brown': [0.6, 0.3, 0.05], 'russet': [0.5, 0.2, 0.1],
            'olive': [0.5, 0.5, 0.0], 'citrine': [0.9, 0.8, 0.2], 'silver': [0.75, 0.75, 0.75],
            'magenta': [1.0, 0.0, 1.0], 'indigo': [0.29, 0.0, 0.51], 'sky_blue': [0.53, 0.81, 0.92]
            # Add more colors as needed from constants/sephiroth_data
        }
        # Handle multi-colors like 'white/clear' -> 'white'
        if '/' in color_name: color_name = color_name.split('/')[0]
        return color_map.get(
    color_name.lower(), [
        0.5, 0.5, 0.5])  # Default grey

    # --- START OF REPLACEMENT visualize_core_structure METHOD ---
    def visualize_core_structure(self, show: bool = False, save: bool = True) -> Optional[plt.Figure]:
        """ Visualize the 3D core structure using current attributes. """
        if not VISUALIZATION_ENABLED: return None
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        try:
            # --- Get Soul Attributes ---
            name = getattr(self.soul, 'name', self.soul.spark_id[:8])
            stability = getattr(self.soul, 'stability', 0.0)
            coherence = getattr(self.soul, 'coherence', 0.0)
            energy = getattr(self.soul, 'energy', 0.0)
            resonance = getattr(self.soul, 'resonance', 0.0)
            position = [float(p) for p in getattr(self.soul, 'position', [0.0, 0.0, 0.0])] # Ensure float
            viz_radius = 0.3 + energy * 0.7 # Base radius

            # --- Create Sphere (Basic Representation) ---
            u, v = np.linspace(0, 2 * PI, 40), np.linspace(0, PI, 20)
            x = viz_radius * np.outer(np.cos(u), np.sin(v)) + position[0]
            y = viz_radius * np.outer(np.sin(u), np.sin(v)) + position[1]
            z = viz_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + position[2]

            # --- Color/Texture based on Soul State ---
            base_rgb = self._get_soul_color_rgb() # Shape (3,)
            stability_colors = self.stability_cmap(stability * np.ones_like(x))[:, :, :3] # Shape (40, 20, 3)
            alpha_channel = 0.3 + 0.6 * coherence # Scalar

            texture_freq = 1.0 + resonance * 6.0
            texture_pattern = 0.8 + 0.2 * np.sin(texture_freq * u[:, None]) * np.cos(texture_freq * v[None, :]) # Shape (40, 20)

            # --- Corrected Calculation for final_rgb ---
            # Multiply texture_pattern (broadcasted) with base_rgb
            base_colored_texture = texture_pattern[..., np.newaxis] * base_rgb # Shape (40, 20, 3)

            # Blend with stability color
            final_rgb = (base_colored_texture * 0.6 + stability_colors * 0.4)
            final_rgb = np.clip(final_rgb, 0.0, 1.0) # Shape is (40, 20, 3)
            # --- End Corrected Calculation ---

            # Assign to face_colors
            face_colors = np.zeros((*x.shape, 4)) # RGBA
            face_colors[..., :3] = final_rgb
            face_colors[..., 3] = alpha_channel

            # Plot the surface
            ax.plot_surface(x, y, z, facecolors=face_colors, rstride=1, cstride=1, linewidth=0, antialiased=True)

            # --- Add Connection Lines (Improved) ---
            # Creator Connection
            conn_strength = getattr(self.soul, 'creator_connection_strength', 0.0)
            if conn_strength > 0.1:
                line_len = 0.5 + 1.5 * conn_strength
                line_end_z = position[2] + viz_radius + line_len
                ax.plot([position[0], position[0]], [position[1], position[1]], [position[2] + viz_radius, line_end_z],
                         color='gold', linestyle='--', linewidth=1.5 + conn_strength * 2, alpha=0.8, label=f'Creator ({conn_strength:.2f})')
                ax.scatter([position[0]], [position[1]], [line_end_z], color='gold', s=30 + conn_strength*100, marker='*')

            # Life Cord
            cord_integrity = getattr(self.soul, 'cord_integrity', 0.0)
            if cord_integrity > 0.1:
                line_len = 0.5 + 1.5 * cord_integrity
                line_end_z = position[2] - viz_radius - line_len
                line_col = self._get_soul_color_rgb() # Use soul color blend for cord?
                line_col_earth = [0.6, 0.4, 0.2] # Earthy color
                line_col_blend = np.clip([c1*0.7+c2*0.3 for c1, c2 in zip(line_col, line_col_earth)], 0, 1)
                ax.plot([position[0], position[0]], [position[1], position[1]], [position[2] - viz_radius, line_end_z],
                         color=tuple(line_col_blend), linestyle='-', linewidth=1.5 + cord_integrity*2, alpha=0.8, label=f'Life Cord ({cord_integrity:.2f})')
                ax.scatter([position[0]], [position[1]], [line_end_z], color=tuple(line_col_blend), s=25+cord_integrity*80, marker='v')

            # --- Plot Setup ---
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            title = f"Soul Core: {name}\nS:{stability:.2f} C:{coherence:.2f} E:{energy:.2f} R:{resonance:.2f}"
            ax.set_title(title)
            if conn_strength > 0.1 or cord_integrity > 0.1: ax.legend(loc='best')

            # --- Auto-scaling ---
            all_points_list = [list(position)]
            if conn_strength > 0.1: all_points_list.append([position[0], position[1], position[2] + viz_radius + 1.5 * conn_strength])
            if cord_integrity > 0.1: all_points_list.append([position[0], position[1], position[2] - viz_radius - 1.5 * cord_integrity])
            # Ensure all points are added before converting to numpy array
            if len(all_points_list) > 1:
                all_points = np.array(all_points_list) # Convert list of lists/arrays
                if all_points.ndim == 1: # Handle case where only center point exists
                     all_points = all_points.reshape(1, -1) # Reshape to 2D array

                if all_points.size > 0:
                    min_coords = np.min(all_points, axis=0) - viz_radius
                    max_coords = np.max(all_points, axis=0) + viz_radius
                    means = (min_coords + max_coords) / 2.0
                    ranges = max_coords - min_coords
                    max_range = max(max(ranges), viz_radius * 3.0)
                    ax.set_xlim(means[0] - max_range / 2, means[0] + max_range / 2)
                    ax.set_ylim(means[1] - max_range / 2, means[1] + max_range / 2)
                    ax.set_zlim(means[2] - max_range / 2, means[2] + max_range / 2)
                else: # Fallback if no points generated (shouldn't happen if position exists)
                     ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            else: # Only center position
                 ax.set_xlim(position[0]-1, position[0]+1)
                 ax.set_ylim(position[1]-1, position[1]+1)
                 ax.set_zlim(position[2]-1, position[2]+1)

            self._save_or_show(fig, "soul_core_structure_enhanced.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating core structure visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None
    # --- END OF REPLACEMENT visualize_core_structure METHOD ---
    
    def visualize_aspects_map(self,
    show: bool = True,
     save: bool = True) -> Optional[plt.Figure]:
        """ Visualize the acquired aspects and their strengths via radar chart. """
        if not VISUALIZATION_ENABLED: return None
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
        try:
            aspects_data = getattr(self.soul, 'aspects', {})
            if not aspects_data:
                ax.text(0, 0, "No aspects acquired", ha='center', va='center')
                ax.set_title("Soul Aspects Map (Empty)"); ax.set_xticks(
                    []); ax.set_yticks([])
                self._save_or_show(
    fig, "soul_aspects_map_empty.png", show, save)
                return fig

            # Limit displayed aspects if too many (e.g., top 12)
            num_to_show = 12
            sorted_aspects = sorted(
    aspects_data.items(), key=lambda item: item[1].get(
        'strength', 0.0), reverse=True)
            labels = [name for name, data in sorted_aspects[:num_to_show]]
            strengths = [data.get('strength', 0.0)
                                  for name, data in sorted_aspects[:num_to_show]]
            num_vars = len(labels)

            if num_vars == 0:  # Handle case where no aspects have strength > 0
                 ax.text(
    0,
    0,
    "No aspects with strength > 0",
    ha='center',
     va='center')
                 ax.set_title("Soul Aspects Map (Zero Strength)"); ax.set_xticks(
                     []); ax.set_yticks([])
                 self._save_or_show(
    fig, "soul_aspects_map_empty.png", show, save)
                 return fig

            angles = np.linspace(
    0, 2 * np.pi, num_vars, endpoint=False).tolist()
            strengths = np.clip(strengths, 0, 1).tolist()
            strengths += strengths[:1]; angles += angles[:1]  # Complete loop

            ax.plot(
    angles,
    strengths,
    linewidth=2,
    linestyle='solid',
    color='magenta',
     marker='o')
            ax.fill(angles, strengths, 'magenta', alpha=0.4)

            # Adjust font size
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_ylim(0, 1)
            ax.set_title(
                f'Soul Aspects (Top {num_vars}): {getattr(self.soul, "name", self.soul.spark_id[:8])}', size=16, y=1.1)

            self._save_or_show(
    fig, "soul_aspects_map_enhanced.png", show, save)
            return fig
        except Exception as e:
            logger.error(
    f"Error generating aspects map visualization: {e}",
     exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    def visualize_frequency_signature(
        self, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
        """ Visualize the frequency signature (uses SoulSpark method). """
        if not VISUALIZATION_ENABLED: return None
        save_path = os.path.join(
    self.output_dir,
     "soul_frequency_signature_enhanced.png") if save else None
        try:
            # Call the method on the soul object itself
            fig = self.soul.visualize_energy_signature(
    show=False, save_path=None)  # Generate fig without showing/saving yet
            if fig is None: return None  # Generation failed within soul object
            self._save_or_show(
    fig, "soul_frequency_signature_enhanced.png", show, save)
            return fig
        except AttributeError:
            logger.error(
                "SoulSpark object missing 'visualize_energy_signature' method.")
            return None
        except Exception as e:
            logger.error(
    f"Error calling/saving frequency signature visualization: {e}",
     exc_info=True)
            return None

    def visualize_life_cord(self, show: bool = True,
                            save: bool = True) -> Optional[plt.Figure]:
        """ Visualize the detailed life cord structure (2D Representation). """
        if not VISUALIZATION_ENABLED: return None
        fig, ax = plt.subplots(figsize=(12, 6))  # Wider for legend

        try:
            life_cord_data = getattr(self.soul, 'life_cord', {})
            cord_integrity = getattr(self.soul, 'cord_integrity', 0.0)
            if not life_cord_data or not getattr(
    self.soul, 'cord_formation_complete', False):
                ax.text(
    0.5,
    0.5,
    "Life Cord Not Formed",
    ha='center',
    va='center',
     transform=ax.transAxes)
                ax.set_title("Life Cord Structure (Not Formed)"); ax.set_xticks(
                    []); ax.set_yticks([])
                self._save_or_show(
    fig, "soul_life_cord_absent.png", show, save)
                return fig

            # --- Extract Data ---
            soul_anchor_pos = life_cord_data.get(
    'soul_anchor', {}).get(
        'position', [
            0, 1])  # Use 2D placeholder
            earth_anchor_pos = life_cord_data.get(
    'earth_anchor', {}).get(
        'position', [
            0, 0])  # Use 2D placeholder
            primary_freq = life_cord_data.get('primary_frequency', 0.0)
            bandwidth = life_cord_data.get('bandwidth', 0.0)
            stability = life_cord_data.get('stability', 0.0)
            elasticity = life_cord_data.get('elasticity', 0.0)
            num_channels = life_cord_data.get('channel_count', 1)
            harmonic_nodes = life_cord_data.get('harmonic_nodes', [])
            secondary_channels = life_cord_data.get('secondary_channels', [])
            earth_conn = life_cord_data.get('earth_connection', 0.0)

            # --- Plotting (2D Representation) ---
            ax.set_title(
    f"Life Cord: {
        getattr(
            self.soul,
            'name',
            self.soul.spark_id[
                :8])}\nIntegrity: {
                    cord_integrity:.3f}, EarthConn: {
                        earth_conn:.3f}, BW: {
                            bandwidth:.1f} Hz")
            ax.set_ylim(-0.6, 1.6)  # Adjust Y-lim for labels
            ax.set_xlim(-0.1, 1.1)
            ax.set_yticks([])
            ax.set_xlabel("Connection Path (Earth Anchor -> Soul Anchor)")

            # Draw Anchors
            ax.plot(1, 0.5, 'o', markersize=18, color='purple',
                    alpha=0.7, label=f'Soul (Stab: {stability:.2f})')
            ax.plot(0, 0.5, 'o', markersize=18, color='saddlebrown',
                    alpha=0.7, label=f'Earth (Elast: {elasticity:.2f})')

            # Draw Primary Channel
            line_width = 2 + cord_integrity * 6  # Thicker line based on integrity
            ax.plot([0,
    1],
    [0.5,
    0.5],
    color='gold',
    linewidth=line_width,
    solid_capstyle='round',
    alpha=0.7,
     label=f'Primary ({primary_freq:.0f} Hz)')

            # Draw Secondary Channels
            y_offset_base = 0.20
            y_increment = 0.10
            channel_colors = {
    'emotional': '#6495ED',
    'mental': '#90EE90',
     'spiritual': '#E6E6FA'}
            for i, chan in enumerate(secondary_channels):
                y_pos = 0.5 + (y_offset_base + i * y_increment) * \
                               (1 if i % 2 == 0 else -1)
                chan_bw = chan.get(
    'bandwidth', 10.0); chan_res = chan.get(
        'interference_resistance', 0.5)
                chan_lw = 1 + (chan_bw / 50.0) * 3  # Width based on bandwidth
                chan_alpha = 0.4 + chan_res * 0.4  # Alpha based on resistance
                chan_col = channel_colors.get(chan.get('type'), 'grey')
                ax.plot([0,
    1],
    [y_pos,
    y_pos],
    color=chan_col,
    linewidth=chan_lw,
    alpha=chan_alpha,
    label=f"{chan.get('type',
     'Sec')[:3]}. (Res:{chan_res:.2f})")

            # Draw Harmonic Nodes
            if harmonic_nodes:
                node_x = [n.get('position', 0.5) for n in harmonic_nodes]
                node_y = [0.5] * len(harmonic_nodes)  # On primary channel
                node_size = [30 +
    n.get('amplitude', 0.5) *
     120 for n in harmonic_nodes]
                node_freqs = [n.get('frequency', 0.0) for n in harmonic_nodes]
                cmap_nodes = plt.get_cmap('plasma')
                norm_nodes = mcolors.LogNorm(
    vmin=max(
        1, min(
            f for f in node_freqs if f > 0)), vmax=max(
                f for f in node_freqs if f > 0)) if any(
                    f > 0 for f in node_freqs) else mcolors.Normalize(
                        0, 1)
                node_colors = cmap_nodes(norm_nodes(node_freqs)) if any(
                    f > 0 for f in node_freqs) else ['grey'] * len(node_freqs)
                scatter = ax.scatter(
    node_x,
    node_y,
    s=node_size,
    c=node_colors,
    alpha=0.9,
    zorder=10,
    edgecolors='black',
     linewidth=0.5)
                # Add colorbar if nodes exist and frequencies vary
                if len(set(nf for nf in node_freqs if nf > 0)) > 1:
                     cbar = fig.colorbar(
    scatter,
    ax=ax,
    label='Node Resonance Freq (Hz)',
    pad=0.02,
     aspect=30)

            ax.legend(
    loc='center left',
    bbox_to_anchor=(
        1.02,
        0.5),
         fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout

            self._save_or_show(fig, "soul_life_cord_enhanced.png", show, save)
            return fig
        except Exception as e:
            logger.error(f"Error generating life cord visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    def visualize_identity(self, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
         """ Visualize key identity attributes in a panel. """
         if not VISUALIZATION_ENABLED: return None
         fig = plt.figure(figsize=(10, 12)) # Taller figure
         gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3) # 3 rows

         try:
            name = getattr(self.soul, 'name', 'N/A')
            soul_color_str = getattr(self.soul, 'soul_color', 'grey')
            soul_color_rgb = self._get_soul_color_rgb()
            voice_freq = getattr(self.soul, 'voice_frequency', 0.0)
            crystallization = getattr(self.soul, 'crystallization_level', 0.0)
            seph_aspect = getattr(self.soul, 'sephiroth_aspect', 'N/A')
            elem_affinity = getattr(self.soul, 'elemental_affinity', 'N/A')
            plat_symbol = getattr(self.soul, 'platonic_symbol', 'N/A')
            attr_coherence = getattr(self.soul, 'attribute_coherence', 0.0)
            name_resonance = getattr(self.soul, 'name_resonance', 0.0)
            gematria = getattr(self.soul, 'gematria_value', 0)

            # --- 1. Name & Color Swatch (Top Left) ---
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title("Name & Color")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_facecolor(mcolors.to_hex(soul_color_rgb) if soul_color_rgb else 'lightgrey')
            id_text = f"Name: {name}\n" \
                      f"(Gematria: {gematria})\n" \
                      f"Soul Color: {soul_color_str.capitalize()}\n" \
                      f"Name Res: {name_resonance:.3f}"
            ax1.text(0.5, 0.5, id_text, ha='center', va='center', fontsize=11, wrap=True,
                     bbox=dict(facecolor='white', alpha=0.85, pad=0.8))

            # --- 2. Voice Frequency Waveform (Top Right) ---
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_title(f"Voice Signature ({voice_freq:.1f} Hz)")
            if voice_freq > FLOAT_EPSILON:
                 t = np.linspace(0, 5 / voice_freq, 500) # Show 5 cycles
                 wave = np.sin(2 * PI * voice_freq * t) * 0.8 # Base wave
                 # Add harmonic complexity based on attribute coherence
                 num_harmonics = int(2 + attr_coherence * 5)
                 for i in range(2, num_harmonics + 1):
                      wave += (0.8 / (i**1.2)) * np.sin(2 * PI * voice_freq * i * t + PI*i/3) # Add harmonics w/ phase
                 ax2.plot(t * 1000, wave, color=mcolors.to_hex(soul_color_rgb) if soul_color_rgb else 'blue')
                 ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Amplitude"); ax2.set_ylim(-1, 1)
            else: ax2.text(0.5, 0.5, "No Voice Freq", ha='center', va='center', transform=ax2.transAxes)

            # --- 3. Affinities Text (Middle Left) ---
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.set_title("Affinities")
            ax3.set_xticks([]); ax3.set_yticks([])
            aff_text = f"Sephiroth: {seph_aspect.capitalize()}\n" \
                       f"Element: {elem_affinity.capitalize()}\n" \
                       f"Platonic: {plat_symbol.capitalize()}"
            ax3.text(0.5, 0.5, aff_text, ha='center', va='center', fontsize=11,
                     bbox=dict(facecolor='white', alpha=0.85, pad=0.8))

            # --- 4. Emotional Resonance (Middle Right) ---
            ax4 = fig.add_subplot(gs[1, 1])
            emo_res = getattr(self.soul, 'emotional_resonance', {})
            if emo_res:
                 labels = list(emo_res.keys())
                 values = list(emo_res.values())
                 colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(labels)))
                 bars = ax4.barh(labels, values, color=colors, alpha=0.7)
                 ax4.set_xlim(0, 1); ax4.set_xlabel("Resonance Level")
                 ax4.set_title("Emotional Resonance")
                 for bar in bars: ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')
            else:
                 ax4.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax4.transAxes); ax4.set_title("Emotional Resonance")

            # --- 5. Crystallization Gauge (Bottom Left) ---
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.set_title(f"Crystallization Level: {crystallization:.3f}")
            ax5.barh([0], [crystallization], height=0.4, color='gold', alpha=0.8)
            ax5.barh([0], [1.0-crystallization], left=crystallization, height=0.4, color='grey', alpha=0.3)
            ax5.set_xlim(0, 1); ax5.set_yticks([]); ax5.set_xticks([0, 0.5, 1.0])
            ax5.text(crystallization / 2, 0, f"{crystallization*100:.1f}%", ha='center', va='center', color='black', fontweight='bold')

            # --- 6. Attribute Coherence Gauge (Bottom Right) ---
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.set_title(f"Attribute Coherence: {attr_coherence:.3f}")
            ax6.barh([0], [attr_coherence], height=0.4, color='cyan', alpha=0.8)
            ax6.barh([0], [1.0-attr_coherence], left=attr_coherence, height=0.4, color='grey', alpha=0.3)
            ax6.set_xlim(0, 1); ax6.set_yticks([]); ax6.set_xticks([0, 0.5, 1.0])
            ax6.text(attr_coherence / 2, 0, f"{attr_coherence*100:.1f}%", ha='center', va='center', color='black', fontweight='bold')


            fig.suptitle(f'Identity Profile: {name}', fontsize=16)
            self._save_or_show(fig, "soul_identity_enhanced.png", show, save)
            return fig
         except Exception as e:
            logger.error(f"Error generating identity visualization: {e}", exc_info=True)
            if 'fig' in locals() and fig:
                plt.close(fig)
         return None

    def create_soul_dashboard(self, show: bool = True, save: bool = True) -> Optional[plt.Figure]:
         """ Create a comprehensive dashboard view (Refined Layout). """
         if not VISUALIZATION_ENABLED: return None
         fig = plt.figure(figsize=(14, 15)) # Adjusted size
         gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1], hspace=0.5, wspace=0.3) # 4 rows

         try:
            name = getattr(self.soul, 'name', self.soul.spark_id[:8])
            is_incarnated = getattr(self.soul, 'incarnated', False)
            status = "Incarnated" if is_incarnated else "Forming"

            # --- 1. Core Metrics Radar (Top Left) ---
            ax1 = fig.add_subplot(gs[0, 0], polar=True)
            core_metrics = { "Stab": self.soul.stability, "Res": self.soul.resonance, "Coh": self.soul.coherence, "Align": self.soul.creator_alignment, "Energy": self.soul.energy, }
            labels = list(core_metrics.keys()); values = list(core_metrics.values())
            values = np.clip(values, 0, 1).tolist(); angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]; angles += angles[:1] # Complete loop
            ax1.plot(angles, values, linewidth=1.5, linestyle='solid', color='blue', marker='o', markersize=4)
            ax1.fill(angles, values, 'blue', alpha=0.3)
            ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(labels, fontsize=9); ax1.set_yticks(np.arange(0, 1.1, 0.25)); ax1.set_ylim(0, 1.05)
            ax1.tick_params(axis='y', labelsize=8)
            ax1.set_title("Core State", fontsize=11, pad=15)

            # --- 2. Identity/Harmony Radar (Top Right) ---
            ax2 = fig.add_subplot(gs[0, 1], polar=True)
            id_metrics = { "Cryst": self.soul.crystallization_level, "AttrCoh": self.soul.attribute_coherence, "NameRes": self.soul.name_resonance, "Response": self.soul.response_level, "Harmony": self.soul.harmony, }
            labels = list(id_metrics.keys()); values = list(id_metrics.values())
            values = np.clip(values, 0, 1).tolist(); angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]; angles += angles[:1]
            ax2.plot(angles, values, linewidth=1.5, linestyle='solid', color='gold', marker='o', markersize=4)
            ax2.fill(angles, values, 'gold', alpha=0.3)
            ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(labels, fontsize=9); ax2.set_yticks(np.arange(0, 1.1, 0.25)); ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis='y', labelsize=8)
            ax2.set_title("Identity & Harmony", fontsize=11, pad=15)

            # --- 3. Earth/Incarnation Radar (Middle Left) ---
            ax3 = fig.add_subplot(gs[1, 0], polar=True)
            earth_metrics = { "EarthRes": self.soul.earth_resonance, "GaiaConn": self.soul.gaia_connection, "Planetary": self.soul.planetary_resonance, "ElemAlign": self.soul.elemental_alignment, "CycleSync": self.soul.cycle_synchronization, }
            labels = list(earth_metrics.keys()); values = list(earth_metrics.values())
            values = np.clip(values, 0, 1).tolist(); angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]; angles += angles[:1]
            ax3.plot(angles, values, linewidth=1.5, linestyle='solid', color='green', marker='o', markersize=4)
            ax3.fill(angles, values, 'green', alpha=0.3)
            ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(labels, fontsize=9); ax3.set_yticks(np.arange(0, 1.1, 0.25)); ax3.set_ylim(0, 1.05)
            ax3.tick_params(axis='y', labelsize=8)
            ax3.set_title("Earth Attunement", fontsize=11, pad=15)

            # --- 4. Life Cord/Incarnation Radar (Middle Right) ---
            ax4 = fig.add_subplot(gs[1, 1], polar=True)
            incarn_metrics = { "CordInt": self.soul.cord_integrity, "FieldInt": self.soul.field_integration, "PhysInt": self.soul.physical_integration, "MemRet": self.soul.memory_retention, "HbEntrain": self.soul.heartbeat_entrainment, }
            labels = list(incarn_metrics.keys()); values = list(incarn_metrics.values())
            values = np.clip(values, 0, 1).tolist(); angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]; angles += angles[:1]
            ax4.plot(angles, values, linewidth=1.5, linestyle='solid', color='saddlebrown', marker='o', markersize=4)
            ax4.fill(angles, values, 'saddlebrown', alpha=0.3)
            ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(labels, fontsize=9); ax4.set_yticks(np.arange(0, 1.1, 0.25)); ax4.set_ylim(0, 1.05)
            ax4.tick_params(axis='y', labelsize=8)
            ax4.set_title("Incarnation Readiness", fontsize=11, pad=15)

            # --- 5. Aspect Strength Bar Chart (Bottom Left) ---
            ax5 = fig.add_subplot(gs[2, 0])
            aspects_data = getattr(self.soul, 'aspects', {})
            if aspects_data:
                sorted_aspects = sorted(aspects_data.items(), key=lambda item: item[1].get('strength', 0.0), reverse=True)
                aspect_names = [name.replace('_', ' ').title() for name, data in sorted_aspects[:10]] # Top 10, formatted
                aspect_strengths = [data.get('strength', 0.0) for name, data in sorted_aspects[:10]]
                colors = self.aspect_cmap(np.linspace(0.1, 0.9, len(aspect_names)))
                bars = ax5.barh(aspect_names, aspect_strengths, color=colors, alpha=0.8)
                ax5.set_xlim(0, 1.05); ax5.set_xlabel("Strength", fontsize=9)
                ax5.tick_params(axis='y', labelsize=8); ax5.tick_params(axis='x', labelsize=8)
                ax5.set_title(f"Top {len(aspect_names)} Aspects", fontsize=11)
                ax5.invert_yaxis() # Show strongest at top
                # Add text labels
                for bar in bars: ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center', fontsize=7)
            else: ax5.text(0.5, 0.5, "No Aspects", ha='center', va='center', transform=ax5.transAxes); ax5.set_title("Aspects", fontsize=11)

            # --- 6. Key Identity Info (Bottom Right) ---
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.set_title("Identity Summary", fontsize=11)
            ax6.set_xticks([]); ax6.set_yticks([])
            ax6.set_facecolor(mcolors.to_hex(self._get_soul_color_rgb()) if self._get_soul_color_rgb() else 'lightgrey')
            id_text = ( f"Name: {getattr(self.soul, 'name', 'N/A')}\n"
                        f"Color: {getattr(self.soul, 'soul_color', 'N/A').capitalize()}\n"
                        f"Voice: {getattr(self.soul, 'voice_frequency', 0.0):.1f} Hz\n"
                        f"Sephirah: {getattr(self.soul, 'sephiroth_aspect', 'N/A').capitalize()}\n"
                        f"Element: {getattr(self.soul, 'elemental_affinity', 'N/A').capitalize()}\n"
                        f"Symbol: {getattr(self.soul, 'platonic_symbol', 'N/A').capitalize()}\n"
                        f"Crystallized: {getattr(self.soul, 'identity_crystallized', False)}" )
            ax6.text(0.5, 0.5, id_text, ha='center', va='center', fontsize=10, linespacing=1.5,
                     bbox=dict(facecolor='white', alpha=0.88, pad=0.7))

            # --- 7. Stages Completed (Bottom Row spanning 2 cols) ---
            ax7 = fig.add_subplot(gs[3, :]) # Span both columns
            stages = [ 'Guff', 'Journey', 'Entangle', 'Harmonic', 'Cord', 'Earth', 'Identity', 'Incarnate' ]
            flags = [ getattr(self.soul, flag, False) for flag in [
                        'guff_strengthened', 'sephiroth_journey_complete', 'creator_channel_id', # Check channel ID existence for Entangle
                        'harmonically_strengthened', 'cord_formation_complete', 'earth_harmonized',
                        'identity_crystallized', 'incarnated' ] ]
            readiness = [ getattr(self.soul, flag, False) for flag in [
                        'ready_for_guff', 'ready_for_journey', 'ready_for_entanglement', 'ready_for_strengthening',
                        'ready_for_life_cord', 'ready_for_earth', 'ready_for_identity', 'ready_for_birth' ] ]

            bar_width = 0.35
            x_pos = np.arange(len(stages))
            rects1 = ax7.bar(x_pos - bar_width/2, [1 if f else 0 for f in flags], bar_width, label='Completed', color='forestgreen', alpha=0.8)
            rects2 = ax7.bar(x_pos + bar_width/2, [1 if r else 0 for r in readiness], bar_width, label='Ready For Next', color='skyblue', alpha=0.6)

            ax7.set_ylabel("Status (1=Yes)", fontsize=9)
            ax7.set_title("Formation Stage Progress", fontsize=11)
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(stages, rotation=30, ha='right', fontsize=9)
            ax7.set_yticks([0, 1]); ax7.set_yticklabels(['No', 'Yes'], fontsize=8)
            ax7.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2) # Legend below

            fig.suptitle(f"Soul Dashboard: {name} (Status: {status})", fontsize=18, y=0.99)
            self._save_or_show(fig, "soul_dashboard_enhanced.png", show, save)
            return fig
         except Exception as e:
            logger.error(f"Error generating soul dashboard: {e}", exc_info=True)
            if 'fig' in locals() and fig: plt.close(fig)
            return None

# --- END OF FILE soul_visualization_enhanced.py ---



