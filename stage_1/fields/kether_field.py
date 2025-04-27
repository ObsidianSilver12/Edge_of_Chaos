# --- START OF FILE src/stage_1/fields/kether_field.py ---

""" Kether Field Module (Ensure constants are correct) """

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Use relative imports assuming same directory structure
from .sephiroth_field import SephirothField
from .void_field import VoidField
from constants.constants import * # Import all constants

logger = logging.getLogger(__name__)

class KetherField(SephirothField):
    """ Represents the Kether Field influence, including the Guff (Hall of Souls). """

    def __init__(self, location: Tuple[int, int, int], radius: float):
        """ Initialize Kether Field. """
        super().__init__(sephirah_name='kether', location=location, radius=radius) # Calls parent __init__ which loads aspects

        # --- Define Guff Region ---
        self.guff_radius: float = self.radius * GUFF_RADIUS_FACTOR # Use constant
        self.guff_radius_sq: float = self.guff_radius * self.guff_radius
        # Correct calculation for offsets
        self.guff_min_offset: int = -int(np.ceil(self.guff_radius))
        self.guff_max_offset: int = int(np.ceil(self.guff_radius)) # No +1 needed if using < max in loop/slice

        logger.info(f"KetherField initialized. Guff Radius: {self.guff_radius:.2f} (within Kether Radius: {self.radius:.1f})")

    def apply_sephiroth_influence(self, void_field: VoidField) -> None:
        """ Applies Kether influence and then modifies the Guff region. """
        # 1. Apply base Kether influence using the parent method
        super().apply_sephiroth_influence(void_field)
        logger.debug("Base Kether influence applied. Now applying Guff modifications...")

        # 2. Apply Guff-specific modifications within its radius
        center_x, center_y, center_z = self.location
        grid_shape = void_field.get_grid_shape()

        # --- Calculate Guff Bounding Box & Mask (as before) ---
        guff_min_x = max(0, center_x + self.guff_min_offset)
        guff_max_x = min(grid_shape[0], center_x + self.guff_max_offset + 1) # +1 for slice upper bound
        guff_min_y = max(0, center_y + self.guff_min_offset)
        guff_max_y = min(grid_shape[1], center_y + self.guff_max_offset + 1)
        guff_min_z = max(0, center_z + self.guff_min_offset)
        guff_max_z = min(grid_shape[2], center_z + self.guff_max_offset + 1)

        if guff_min_x >= guff_max_x or guff_min_y >= guff_max_y or guff_min_z >= guff_max_z:
             logger.warning("Guff bounding box has zero volume. Skipping Guff modifications.")
             return

        gx_idx, gy_idx, gz_idx = np.meshgrid(np.arange(guff_min_x, guff_max_x), np.arange(guff_min_y, guff_max_y), np.arange(guff_min_z, guff_max_z), indexing='ij')
        guff_dist_sq = (gx_idx - center_x)**2 + (gy_idx - center_y)**2 + (gz_idx - center_z)**2
        guff_mask = guff_dist_sq <= self.guff_radius_sq

        if not np.any(guff_mask): return # Skip if no cells inside guff radius

        guff_slice = (slice(guff_min_x, guff_max_x), slice(guff_min_y, guff_max_y), slice(guff_min_z, guff_max_z))

        # --- Guff Modifications (Apply where guff_mask is True) ---
        # Energy Boost (Multiply energy ONLY within the mask)
        void_field.energy[guff_slice][guff_mask] *= GUFF_ENERGY_MULTIPLIER # Use constant

        # Coherence/Stability Push (Push towards target ONLY within the mask)
        current_coh = void_field.coherence[guff_slice][guff_mask]
        void_field.coherence[guff_slice][guff_mask] = current_coh + (GUFF_COHERENCE_TARGET - current_coh) * 0.8 # Strong push
        current_order = void_field.order[guff_slice][guff_mask]
        void_field.order[guff_slice][guff_mask] = current_order + (GUFF_STABILITY_TARGET - current_order) * 0.8 # Strong push

        # Frequency - Set to Kether frequency (override base blend)
        void_field.frequency[guff_slice][guff_mask] = self.target_frequency # Use Kether's target

        # Pattern Influence - Increase density/strength
        void_field.pattern_influence[guff_slice][guff_mask] += 0.5 # Add strong base pattern influence
        # Optionally add Guff-specific pattern here if defined

        # --- Clamping ---
        void_field.energy[guff_slice][guff_mask] = np.clip(void_field.energy[guff_slice][guff_mask], 0.0, 1.0)
        void_field.coherence[guff_slice][guff_mask] = np.clip(void_field.coherence[guff_slice][guff_mask], 0.0, 1.0)
        void_field.order[guff_slice][guff_mask] = np.clip(void_field.order[guff_slice][guff_mask], 0.0, 1.0)
        void_field.pattern_influence[guff_slice][guff_mask] = np.clip(void_field.pattern_influence[guff_slice][guff_mask], 0.0, 1.0)
        # Recalculate chaos for affected cells
        void_field.chaos[guff_slice][guff_mask] = 1.0 - void_field.order[guff_slice][guff_mask]

        logger.debug(f"Guff modifications applied within radius {self.guff_radius:.2f}.")

    # is_in_guff and __str__/__repr__ methods remain the same as previous definition.
    def is_in_guff(self, coords: Tuple[int, int, int]) -> bool:
        """Check if coordinates are within the Guff region."""
        if not self._validate_coordinates(coords): return False
        dist_sq = sum((coords[i] - self.location[i])**2 for i in range(3))
        return dist_sq <= self.guff_radius_sq

    def _validate_coordinates(self, coordinates: Tuple[int, int, int]) -> bool:
        """ Inherited validation - checks against full Kether radius, not just Guff """
        # This check is a bit ambiguous here. Does it check Kether bounds or grid bounds?
        # Let's assume it checks grid bounds, which is safer.
        # If the location itself is near the edge, the radius might extend beyond.
        # The apply_influence handles boundary conditions correctly via slicing.
        grid_size = GRID_SIZE # Assuming access to global constant or passed grid size
        return all(0 <= coordinates[i] < grid_size[i] for i in range(3))

    def __str__(self) -> str:
        return f"KetherField(Name: {self.sephirah_name.capitalize()}, Loc: {self.location}, Rad: {self.radius:.1f}, GuffRad: {self.guff_radius:.1f})"

    def __repr__(self) -> str:
        return f"<KetherField name='{self.sephirah_name}' location={self.location} radius={self.radius} guff_radius={self.guff_radius}>"


# --- END OF FILE src/stage_1/fields/kether_field.py ---