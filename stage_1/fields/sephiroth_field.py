# --- START OF FILE src/stage_1/fields/sephiroth_field.py ---

"""
Sephiroth Field Module

Implements the Sephiroth Field 'influencer'. It doesn't own a grid but modifies
the underlying VoidField grid within its defined location and radius based on
its specific aspects, including geometry and platonic affinities.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Assuming field_base and void_field are in the same directory
from .field_base import FieldBase
from .void_field import VoidField
# Assuming aspect dictionary is accessible
from stage_1.soul_formation.sephiroth_aspect_dictionary import aspect_dictionary
# Assuming constants are accessible
from constants.constants import *

# --- Dependency Imports for Geometry/Platonics (Example) ---
# These imports depend on the refactored functions existing
try:
    from geometry.tree_of_life import get_tree_of_life_influence # Example
    from platonics.hexahedron import get_hexahedron_influence   # Example
    # Import other necessary influence functions...
    GEOM_PLAT_FUNCS_AVAILABLE = True
    # Map names to functions (adjust keys to match sephiroth_data)
    GEOM_PLAT_FUNCTION_MAP = {
        'tree_of_life': get_tree_of_life_influence,
        'hexahedron': get_hexahedron_influence,
        # Add other mappings...
    }
except ImportError as e:
    logging.warning(f"Could not import geometry/platonic influence functions: {e}. Influence disabled.")
    GEOM_PLAT_FUNCS_AVAILABLE = False
    GEOM_PLAT_FUNCTION_MAP = {}


logger = logging.getLogger(__name__)

class SephirothField:
    """
    Represents the influence of a specific Sephirah within the VoidField.
    Modifies the Void grid based on its unique aspects, frequency, color, geometry etc.
    """

    def __init__(self, sephirah_name: str, location: Tuple[int, int, int], radius: float):
        """ Initialize the Sephirah Field influencer. (Validation as before) """
        self.sephirah_name: str = sephirah_name.lower()
        if not isinstance(location, tuple) or len(location) != 3 or not all(isinstance(i, int) for i in location):
            raise ValueError("Location must be a tuple of 3 integers.")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")

        self.location: Tuple[int, int, int] = location
        self.radius: float = radius
        self.radius_sq: float = radius * radius

        if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
        self.aspect_data: Dict[str, Any] = aspect_dictionary.get_aspects(self.sephirah_name)
        if not self.aspect_data: raise ValueError(f"No aspect data for '{self.sephirah_name}'")

        # Extract key properties (as before)
        self.target_frequency: float = self.aspect_data.get('base_frequency', SOUL_SPARK_DEFAULT_FREQ)
        self.target_energy: float = self.aspect_data.get('energy_level', VOID_BASE_ENERGY)
        # Use modifiers to calculate target coherence/order relative to Void baseline
        self.target_coherence: float = VOID_BASE_COHERENCE * self.aspect_data.get('coherence_modifier', 1.0)
        self.target_order: float = (VOID_CHAOS_ORDER_BALANCE + self.aspect_data.get('chaos_order_bias', 0.0)) # Adjust target order bias
        self.target_coherence = max(0.0, min(1.0, self.target_coherence))
        self.target_order = max(0.0, min(1.0, self.target_order))

        self.primary_color_str: str = self.aspect_data.get('primary_color', 'white')
        self.target_color: np.ndarray = self._get_target_color_vector(self.primary_color_str) # Defined as before
        self.geometric_pattern: Optional[str] = self.aspect_data.get('geometric_correspondence')
        self.platonic_affinity: Optional[str] = self.aspect_data.get('platonic_affinity')

        logger.info(f"SephirothField influencer '{self.sephirah_name.capitalize()}' initialized at {self.location} with radius {self.radius:.1f}.")

    def _get_target_color_vector(self, color_name: str) -> np.ndarray:
        """ Converts color name (if known) to an RGB vector. (Implementation as before) """
        color_map = { # Expand this map
            'white': [1.0, 1.0, 1.0], 'grey': [0.5, 0.5, 0.5], 'black': [0.05, 0.05, 0.05],
            'blue': [0.1, 0.2, 0.9], 'red': [0.9, 0.1, 0.1], 'yellow': [0.9, 0.9, 0.1],
            'gold': [1.0, 0.84, 0.0], 'green': [0.1, 0.9, 0.2], 'orange': [1.0, 0.65, 0.0],
            'purple': [0.5, 0.1, 0.9], 'violet': [0.58, 0.0, 0.83], 'lavender': [0.9, 0.9, 1.0],
            'earth_tones': [0.6, 0.4, 0.2] # Average
        }
        return np.array(color_map.get(color_name.lower(), [0.7, 0.7, 0.7]), dtype=np.float32)

    def apply_sephiroth_influence(self, void_field: VoidField) -> None:
        """ Applies this Sephirah's influence onto the provided VoidField grid. """
        if not isinstance(void_field, VoidField): raise TypeError("void_field must be VoidField.")
        if void_field.energy is None: raise AttributeError("VoidField grids not initialized.")

        logger.debug(f"Applying influence of '{self.sephirah_name}' onto VoidField...")
        center_x, center_y, center_z = self.location
        grid_shape = void_field.get_grid_shape()

        # --- Calculate Bounding Box and Mask (as before) ---
        min_x = max(0, center_x - int(self.radius)); max_x = min(grid_shape[0], center_x + int(self.radius) + 1)
        min_y = max(0, center_y - int(self.radius)); max_y = min(grid_shape[1], center_y + int(self.radius) + 1)
        min_z = max(0, center_z - int(self.radius)); max_z = min(grid_shape[2], center_z + int(self.radius) + 1)

        # Check for invalid box dimensions
        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
             logger.warning(f"Skipping influence for '{self.sephirah_name}': bounding box has zero volume.")
             return

        x_idx, y_idx, z_idx = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z), indexing='ij')
        dist_sq = (x_idx - center_x)**2 + (y_idx - center_y)**2 + (z_idx - center_z)**2
        within_radius_mask = dist_sq <= self.radius_sq
        if not np.any(within_radius_mask): return # Skip if no cells are within radius

        influence_strength = np.maximum(0.0, 1.0 - np.sqrt(dist_sq / max(FLOAT_EPSILON, self.radius_sq)))
        influence_strength[~within_radius_mask] = 0.0

        # --- Get Void Grid Slices ---
        box_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
        # Use the boolean mask 'within_radius_mask' to select elements *within the box*
        influence_slice_mask = influence_strength > FLOAT_EPSILON # Mask relative to the box slice

        # --- Blend Properties ---
        influence_vals = influence_strength[influence_slice_mask] # Get valid influence values
        inv_influence_vals = 1.0 - influence_vals

        # Blend Energy
        current_energy_slice = void_field.energy[box_slice][influence_slice_mask]
        void_field.energy[box_slice][influence_slice_mask] = current_energy_slice * inv_influence_vals + self.target_energy * influence_vals

        # Blend Frequency
        current_freq_slice = void_field.frequency[box_slice][influence_slice_mask]
        void_field.frequency[box_slice][influence_slice_mask] = current_freq_slice * inv_influence_vals + self.target_frequency * influence_vals

        # Blend Coherence
        current_coh_slice = void_field.coherence[box_slice][influence_slice_mask]
        void_field.coherence[box_slice][influence_slice_mask] = current_coh_slice * inv_influence_vals + self.target_coherence * influence_vals

        # Blend Order (Chaos will be recalculated later if needed or derived)
        current_order_slice = void_field.order[box_slice][influence_slice_mask]
        void_field.order[box_slice][influence_slice_mask] = current_order_slice * inv_influence_vals + self.target_order * influence_vals

        # Blend Color (Requires broadcasting influence)
        current_color_slice = void_field.color[box_slice][influence_slice_mask]
        influence_broadcast = influence_vals[:, np.newaxis] # Shape (N, 1)
        inv_influence_broadcast = 1.0 - influence_broadcast
        void_field.color[box_slice][influence_slice_mask] = current_color_slice * inv_influence_broadcast + self.target_color * influence_broadcast

        # --- Apply Geometric/Platonic Influence to pattern_influence grid ---
        geom_influence_applied = False
        plat_influence_applied = False
        sephirah_pattern_strength = 0.5 # Base strength of this Sephirah's pattern imprint

        if GEOM_PLAT_FUNCS_AVAILABLE:
            # Geometry
            if self.geometric_pattern and self.geometric_pattern in GEOM_PLAT_FUNCTION_MAP:
                try:
                    geom_func = GEOM_PLAT_FUNCTION_MAP[self.geometric_pattern]
                    # Generate influence grid for the *entire box* then mask
                    box_shape = (max_x-min_x, max_y-min_y, max_z-min_z)
                    geom_grid = geom_func(shape=box_shape, strength=sephirah_pattern_strength)
                    if geom_grid is not None and geom_grid.shape == box_shape:
                        # Add influence where Sephirah has influence
                        void_field.pattern_influence[box_slice][influence_slice_mask] += geom_grid[influence_slice_mask] * influence_vals
                        geom_influence_applied = True
                    else: logger.warning(f"Invalid grid from geometry function {self.geometric_pattern}")
                except Exception as e: logger.error(f"Error applying geometry {self.geometric_pattern}: {e}")

            # Platonics
            if self.platonic_affinity and self.platonic_affinity in GEOM_PLAT_FUNCTION_MAP:
                 try:
                    plat_func = GEOM_PLAT_FUNCTION_MAP[self.platonic_affinity]
                    box_shape = (max_x-min_x, max_y-min_y, max_z-min_z)
                    plat_grid = plat_func(shape=box_shape, strength=sephirah_pattern_strength * 0.8) # Slightly less strength?
                    if plat_grid is not None and plat_grid.shape == box_shape:
                        void_field.pattern_influence[box_slice][influence_slice_mask] += plat_grid[influence_slice_mask] * influence_vals
                        plat_influence_applied = True
                    else: logger.warning(f"Invalid grid from platonic function {self.platonic_affinity}")
                 except Exception as e: logger.error(f"Error applying platonic {self.platonic_affinity}: {e}")

            # Clamp pattern influence grid after additions
            void_field.pattern_influence[box_slice] = np.clip(void_field.pattern_influence[box_slice], 0.0, 1.0) # Clamp whole slice affected

        debug_geom = f" GeomApplied={geom_influence_applied}" if GEOM_PLAT_FUNCS_AVAILABLE else ""
        debug_plat = f" PlatApplied={plat_influence_applied}" if GEOM_PLAT_FUNCS_AVAILABLE else ""
        logger.debug(f"Influence of '{self.sephirah_name}' applied.{debug_geom}{debug_plat}")


# --- END OF FILE src/stage_1/fields/sephiroth_field.py ---