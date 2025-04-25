# --- START OF FILE field_visualization.py ---

"""
Field Visualization Module

Provides functions to visualize the spatial layout and basic properties
of fields within the Soul Development Framework, particularly the Tree of Life structure.

Author: Soul Development Framework Team
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional, List

# Use try-except for plotting library
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
    import numpy as np # For calculations if needed
    MATPLOTLIB_AVAILABLE = True
    # Define Figure type hint based on actual import
    Figure = plt.Figure
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Define placeholder for type hint if matplotlib is missing
    plt = None
    Axes3D = None
    np = None
    Figure = Any # Placeholder type hint
    logging.basicConfig(level=logging.INFO) # Basic logging if constants failed elsewhere

# Import necessary classes for type hinting and access
# Use absolute imports assuming standard structure relative to project root 'src'
try:
    from src.stage_1.fields.field_registry import FieldRegistry
    from src.stage_1.fields.void_field import VoidField
    from src.stage_1.fields.sephiroth_field import SephirothField
    from src.stage_1.fields.daath_field import DaathField # Import Daath specifically
except ImportError as e:
     # This indicates a problem with the project structure or PATH setup
     logging.critical(f"CRITICAL: Failed to import core field classes in visualization: {e}")
     # Define dummy classes for type hinting if essential imports fail? Risky.
     # Best to ensure imports work correctly from the calling context.
     FieldRegistry = Any
     VoidField = Any
     SephirothField = Any
     DaathField = Any

# Configure logging (ensure it's configured by the main entry point ideally)
logger = logging.getLogger(__name__)


def visualize_field_layout_3d(registry: FieldRegistry,
                              highlight_sephiroth: Optional[List[str]] = None,
                              show_connections: bool = True,
                              show_labels: bool = True,
                              show_void_bounds: bool = False,
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> Optional[Figure]:
    """
    Creates a simplified 3D visualization of the registered field layout,
    focusing on Sephiroth positions within the Void.

    Args:
        registry: The initialized FieldRegistry instance containing the fields.
        highlight_sephiroth: Optional list of lowercase Sephirah names to highlight.
        show_connections: Whether to draw lines for registered connections between plotted fields.
        show_labels: Whether to label the Sephiroth nodes.
        show_void_bounds: Whether to draw the bounding box of the Void field.
        save_path: Optional path to save the visualization image.
        show_plot: Whether to display the plot window interactively.

    Returns:
        Optional[plt.Figure]: Matplotlib Figure object if successful and matplotlib is available, None otherwise.

    Raises:
        RuntimeError: If registry is invalid, visualization fails critically, or matplotlib unavailable.
        ValueError: If essential data (like Void field or positions) is missing/invalid.
        TypeError: If registry is not a FieldRegistry instance.
    """
    if not MATPLOTLIB_AVAILABLE:
        # Raise RuntimeError instead of just logging, as visualization is impossible
        raise RuntimeError("Matplotlib is required for visualization but not installed.")
    if not isinstance(registry, FieldRegistry):
        raise TypeError("Invalid registry object provided (must be FieldRegistry instance).")
    if not registry.initialized or registry.void_field_id is None:
        raise RuntimeError("Registry is not initialized or Void field is missing.")

    logger.info("Generating Tree of Life field layout visualization...")

    fig = None # Initialize fig to None
    try:
        fig = plt.figure(figsize=(13, 15)) # Adjusted size
        ax = fig.add_subplot(111, projection='3d')

        plotted_nodes = {} # Store {field_id: {'name': name_lower, 'display_name': display, 'pos': pos, 'color': c, 'size': s}}

        # --- Get Void Field for Bounds and Contained Fields ---
        try:
            void_field = registry.get_field(registry.void_field_id)
            if not isinstance(void_field, VoidField):
                 raise ValueError("Object registered as Void field is not a VoidField instance.")
            if not hasattr(void_field, 'contained_fields') or not isinstance(void_field.contained_fields, dict):
                 raise ValueError("Void field lacks a valid 'contained_fields' dictionary.")
            if not isinstance(void_field.dimensions, tuple) or len(void_field.dimensions) != 3 or not all(isinstance(d,(int,float)) and d > 0 for d in void_field.dimensions):
                 raise ValueError(f"Void field has invalid dimensions: {void_field.dimensions}")
            void_dims = void_field.dimensions
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
             # Catch specific errors related to accessing Void data
             raise ValueError(f"Cannot visualize layout: Error accessing Void field data: {e}") from e

        # --- Identify and Prepare Sephiroth Nodes ---
        for field_id, contained_data in void_field.contained_fields.items():
            if field_id not in registry.fields:
                 logger.warning(f"Contained field {field_id} not found in registry, skipping.")
                 continue

            field = registry.fields[field_id]
            field_name_lower = None
            # Prefer display name from field object if available
            display_name = getattr(field, 'name', f"Field_{field_id[:4]}")

            # Determine the sephirah_name (lowercase) for identification
            if isinstance(field, SephirothField) and hasattr(field, 'sephiroth_name') and isinstance(field.sephiroth_name, str):
                 field_name_lower = field.sephiroth_name.lower()
            elif isinstance(field, DaathField): # Explicit check for Daath
                 field_name_lower = "daath"

            if field_name_lower: # Only process Sephiroth/Daath fields found in Void
                pos = contained_data.get('position')
                # Use field attributes, fallback to defaults/constants if missing
                color_attr = getattr(field, 'primary_color', 'grey') # Default color grey

                # Handle complex color strings (like "yellow/gold") or list/tuple colors
                if isinstance(color_attr, str) and '/' in color_attr: color = color_attr.split('/')[0].lower()
                elif isinstance(color_attr, (list, tuple)) and color_attr: color = str(color_attr[0]).lower() # Take first if list/tuple
                elif isinstance(color_attr, str): color = color_attr.lower()
                else: color = 'grey'

                # Validate color - check if it's a recognized matplotlib color name
                is_valid_color = False
                try:
                    plt.cm.get_cmap(color) # Check cmap first
                    is_valid_color = True
                except ValueError:
                    try:
                        plt.colors.to_rgba(color) # Check color name/hex
                        is_valid_color = True
                    except ValueError:
                         logger.warning(f"Invalid color '{color}' for {field_name_lower}. Using gray.")
                         color = 'gray'
                         is_valid_color = True # Gray is valid

                # Size based on stability (more stable = larger?) or fixed size
                base_size = 100
                size_factor = getattr(field, 'stability', 0.5) # Use stability
                if not isinstance(size_factor, (int, float)): size_factor = 0.5 # Default stability
                node_size = base_size + min(1.0, max(0.0, size_factor)) * 150

                # Validate position data before adding
                if isinstance(pos, tuple) and len(pos) == 3 and all(isinstance(c, (int,float)) and np.isfinite(c) for c in pos):
                     plotted_nodes[field_id] = {
                          'name': field_name_lower, # Store lowercase name for lookup
                          'display_name': display_name, # Store original display name
                          'pos': pos,
                          'color': color,
                          'size': node_size
                     }
                else:
                     logger.warning(f"Field {field_id} ({display_name}) has invalid position data in Void: {pos}. Skipping.")


        # --- Plot Nodes ---
        if not plotted_nodes:
             logger.warning("No valid Sephiroth/Daath nodes found in Void's contained_fields to visualize.")
             # Setup empty plot axes if desired
             ax.set_xlim(0, void_dims[0]); ax.set_ylim(0, void_dims[1]); ax.set_zlim(0, void_dims[2])
             ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
             ax.set_title("Tree of Life Field Layout (No Fields Plotted)")
             plt.tight_layout()
             if show_plot: plt.show()
             elif fig: plt.close(fig)
             return fig # Return figure with empty axes

        highlight_list = [name.lower() for name in highlight_sephiroth] if highlight_sephiroth else []

        for field_id, data in plotted_nodes.items():
            pos = data['pos']
            name_lower = data['name']
            # Use display name for label, maybe just the first part
            display_name_short = data['display_name'].split('-')[0].strip().capitalize()
            is_highlighted = name_lower in highlight_list
            node_color = data['color']
            node_size = data['size'] * (1.5 if is_highlighted else 1.0)
            edge_color = 'black' # Black edge for all for clarity
            edge_width = 1.5 if is_highlighted else 0.7
            alpha = 0.9 if is_highlighted else 0.75

            ax.scatter(pos[0], pos[1], pos[2], c=node_color, s=node_size,
                   label=display_name_short if show_labels else None,
                   alpha=alpha, depthshade=True, edgecolor=edge_color, linewidth=edge_width)
            if show_labels:
                # Offset label slightly above the node
                ax.text(pos[0], pos[1], pos[2] + node_size*0.009, f" {display_name_short}", # Adjusted offset factor
                        size=8, zorder=10, color='black', ha='center', va='bottom')

        # --- Plot Connections ---
        if show_connections:
            logger.debug("Plotting connections between visualized fields...")
            plotted_connections = set()
            for source_id, targets in registry.field_connections.items():
                 if source_id not in plotted_nodes: continue

                 for target_id in targets.keys():
                      if target_id not in plotted_nodes: continue

                      pair = tuple(sorted((source_id, target_id)))
                      if pair in plotted_connections: continue
                      plotted_connections.add(pair)

                      pos1 = plotted_nodes[source_id]['pos']
                      pos2 = plotted_nodes[target_id]['pos']
                      ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                              color='dimgray', linestyle=':', linewidth=1.2, alpha=0.6)

        # --- Plot Void Bounds (Optional) ---
        if show_void_bounds:
             x0, y0, z0 = 0, 0, 0
             x1, y1, z1 = void_dims
             verts = [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0), (x0,y0,z1),(x0,y1,z1),(x1,y1,z1),(x1,y0,z1)]
             edges = [[verts[i], verts[j]] for i,j in [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]]
             for edge in edges:
                  xs, ys, zs = zip(*edge)
                  ax.plot(xs, ys, zs, color='lightblue', linestyle='--', linewidth=0.8, alpha=0.4)

        # --- Setup Axes ---
        ax.set_xlabel("X Dimension (Void)")
        ax.set_ylabel("Y Dimension (Void)")
        ax.set_zlabel("Z Dimension (Void)")
        ax.set_title("Tree of Life Field Layout in Void")

        # Set limits based on void dimensions, add padding
        padding_x = void_dims[0] * 0.05
        padding_y = void_dims[1] * 0.05
        padding_z = void_dims[2] * 0.05
        ax.set_xlim(-padding_x, void_dims[0] + padding_x)
        ax.set_ylim(-padding_y, void_dims[1] + padding_y)
        ax.set_zlim(-padding_z, void_dims[2] + padding_z)
        # Use equal aspect visually for better representation
        ax.set_box_aspect([1,1,1])

        # Improve view angle and background color
        ax.view_init(elev=25., azim=-55)
        ax.set_facecolor('#EAEAF2') # Lighter grey background
        fig.patch.set_facecolor('white') # Figure background

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        if save_path:
            try:
                output_dir = os.path.dirname(save_path)
                if output_dir: os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
                logger.info(f"Tree of Life layout visualization saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save layout visualization to {save_path}: {e}")
        if show_plot:
            plt.show()
        elif fig: # Only close if not showing, and fig exists
            plt.close(fig)

        return fig

    except Exception as e:
        logger.error(f"Error during Tree of Life layout visualization: {str(e)}", exc_info=True)
        if fig: plt.close(fig) # Attempt to close figure on error
        raise RuntimeError(f"Tree of Life layout visualization failed: {e}") from e

# --- END OF FILE field_visualization.py ---