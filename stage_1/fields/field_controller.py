# --- START OF FILE field_controller.py ---

"""
Field Controller Module

Manages the VoidField and SephirothField influencers, orchestrates field updates,
and handles the conceptual location and interaction of souls within the fields.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from stage_1.fields.field_base import FieldBase
from stage_1.fields.void_field import VoidField
from stage_1.fields.sephiroth_field import SephirothField
from stage_1.fields.kether_field import KetherField
from stage_1.soul_formation.soul_spark import SoulSpark
from stage_1.soul_formation.sephiroth_aspect_dictionary import aspect_dictionary
from constants.constants import *

logger = logging.getLogger(__name__)

class FieldController:
    """
    Manages the field environment, including the Void and Sephiroth influencers.
    Handles soul positioning and field interactions conceptually.
    """

    def __init__(self, grid_size: Tuple[int, int, int] = GRID_SIZE):
        """
        Initialize the FieldController, creating the Void and Sephiroth influencers.

        Args:
            grid_size (Tuple[int, int, int]): Dimensions for the underlying Void grid.
        """
        if aspect_dictionary is None:
             raise RuntimeError("Aspect Dictionary is not available. FieldController cannot initialize.")

        self.grid_size: Tuple[int, int, int] = grid_size
        logger.info(f"Initializing FieldController with grid size {self.grid_size}...")

        # 1. Create the Void Field
        try:
            self.void_field: VoidField = VoidField(grid_size=self.grid_size)
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize VoidField: {e}", exc_info=True)
            raise RuntimeError("VoidField initialization failed.") from e

        # 2. Define Sephiroth Locations and Radii
        # Example: Arrange Sephiroth based on Tree of Life layout within the grid
        # Scaling the 9x16 ToL layout to the grid size
        center_x, center_y, center_z = grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2
        scale_x = grid_size[0] / 9.0
        scale_y = grid_size[1] / 16.0 # Use Y for height in ToL layout mapping
        default_radius = SEPHIROTH_DEFAULT_RADIUS * (min(grid_size) / 64.0) # Scale radius with grid size

        # Re-fetch positions from constants (used in ToL viz, adapt here)
        # These need translation/scaling to fit grid_size
        tol_base_positions = {
            'kether': (4.5, 1), 'chokmah': (9, 3), 'binah': (0, 3), 'chesed': (9, 7.5),
            'geburah': (0, 7.5), 'tiphareth': (4.5, 9), 'netzach': (9, 12), 'hod': (0, 12),
            'yesod': (4.5, 14), 'malkuth': (4.5, 16), 'daath': (4.5, 4.75)
        }

        self.sephiroth_locations: Dict[str, Tuple[int, int, int]] = {}
        self.sephiroth_radii: Dict[str, float] = {}
        for name, (tol_x, tol_y) in tol_base_positions.items():
            grid_x = int(tol_x * scale_x)
            grid_y = int(tol_y * scale_y) # Map ToL height to grid Y
            # Add Z variation (e.g., pillars slightly offset)
            grid_z = center_z
            if name in ['chokmah', 'chesed', 'netzach']: grid_z += int(grid_size[2] * 0.1) # Right pillar forward
            if name in ['binah', 'geburah', 'hod']: grid_z -= int(grid_size[2] * 0.1) # Left pillar back
            # Clamp coordinates to grid boundaries
            grid_x = max(0, min(grid_size[0] - 1, grid_x))
            grid_y = max(0, min(grid_size[1] - 1, grid_y))
            grid_z = max(0, min(grid_size[2] - 1, grid_z))
            self.sephiroth_locations[name] = (grid_x, grid_y, grid_z)
            self.sephiroth_radii[name] = default_radius # Can customize radius per Sephirah later

        # 3. Create Sephiroth Influencer Instances
        self.sephiroth_influencers: Dict[str, SephirothField] = {}
        self.kether_field: Optional[KetherField] = None # Specific reference to Kether
        for name in aspect_dictionary.sephiroth_names:
            loc = self.sephiroth_locations.get(name)
            rad = self.sephiroth_radii.get(name)
            if loc is None or rad is None:
                logger.error(f"Missing location or radius for Sephirah '{name}'. Skipping influencer creation.")
                continue
            try:
                if name == 'kether':
                    kether_instance = KetherField(location=loc, radius=rad)
                    self.sephiroth_influencers[name] = kether_instance
                    self.kether_field = kether_instance
                else:
                    seph_instance = SephirothField(sephirah_name=name, location=loc, radius=rad)
                    self.sephiroth_influencers[name] = seph_instance
            except Exception as e:
                logger.error(f"Failed to create influencer for '{name}': {e}", exc_info=True)
                # Continue trying to create others

        if len(self.sephiroth_influencers) != 11:
             logger.warning(f"Expected 11 Sephiroth influencers, but created {len(self.sephiroth_influencers)}.")
        if not self.kether_field:
             raise RuntimeError("KetherField instance could not be created.")

        # 4. Apply Initial Influences
        self.apply_initial_influences()
        logger.info("FieldController initialized and initial Sephiroth influences applied.")

    def apply_initial_influences(self) -> None:
        """Applies the influence of all Sephiroth onto the Void grid."""
        logger.info("Applying initial Sephiroth influences to Void grid...")
        for name, influencer in self.sephiroth_influencers.items():
            try:
                influencer.apply_sephiroth_influence(self.void_field)
            except Exception as e:
                logger.error(f"Error applying initial influence for '{name}': {e}", exc_info=True)
        logger.info("Initial influences applied.")

    def update_fields(self, delta_time: float) -> None:
        """
        Updates the Void field dynamics and reapplies Sephiroth influences.

        Args:
            delta_time (float): Time step duration.
        """
        # 1. Update Void dynamics
        try:
            self.void_field.update_step(delta_time)
        except Exception as e:
            logger.error(f"Error updating VoidField dynamics: {e}", exc_info=True)
            # Decide whether to continue or raise based on severity

        # 2. Reapply Sephiroth influences (reinforce their presence)
        # Optimization: Could be done less frequently than Void update?
        for name, influencer in self.sephiroth_influencers.items():
            try:
                influencer.apply_sephiroth_influence(self.void_field)
            except Exception as e:
                logger.error(f"Error reapplying influence for '{name}' during update: {e}", exc_info=True)

    def get_field(self, field_key: str) -> Optional[FieldBase]:
        """Gets a specific field instance (Void or Sephiroth influencer)."""
        field_key_lower = field_key.lower()
        if field_key_lower == 'void':
            return self.void_field
        elif field_key_lower in self.sephiroth_influencers:
            # Return the influencer object, not the void grid slice
            return self.sephiroth_influencers[field_key_lower]
        else:
            logger.error(f"Field key '{field_key}' not recognized.")
            return None

    def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Gets the properties from the VoidField at the given coordinates.
        These properties reflect the *blended* state after influences.
        Fails hard on bounds error.
        """
        try:
            return self.void_field.get_properties_at(coordinates)
        except IndexError as e:
            logger.error(f"Error getting properties: {e}")
            raise # Re-raise bounds error
        except Exception as e:
            logger.error(f"Unexpected error in get_properties_at: {e}", exc_info=True)
            raise RuntimeError("Failed to get field properties.") from e

    def get_dominant_sephiroth_at(self, coordinates: Tuple[int, int, int]) -> Optional[str]:
        """
        Determines the dominant Sephiroth influence at given coordinates.

        Returns:
            Optional[str]: Name of the dominant Sephirah, or None if in Void/ambiguous.
        """
        min_dist_sq = float('inf')
        dominant_sephirah = None

        for name, influencer in self.sephiroth_influencers.items():
            loc = influencer.location
            dist_sq = (coordinates[0] - loc[0])**2 + (coordinates[1] - loc[1])**2 + (coordinates[2] - loc[2])**2
            if dist_sq <= influencer.radius_sq: # Check if within radius
                if dist_sq < min_dist_sq: # If multiple overlap, pick closest
                    min_dist_sq = dist_sq
                    dominant_sephirah = name

        return dominant_sephirah

    # --- Soul Position & State Management ---

    def _coords_to_int_tuple(self, position: List[float]) -> Tuple[int, int, int]:
        """Converts float position to int grid coordinates, clamping to bounds."""
        coords = tuple(max(0, min(self.grid_size[i] - 1, int(round(p)))) for i, p in enumerate(position))
        return coords

    def move_soul(self, soul_spark: SoulSpark, new_position: List[float]) -> None:
        """
        Updates the soul's position and its current field key based on location.

        Args:
            soul_spark (SoulSpark): The soul spark object.
            new_position (List[float]): The new [x, y, z] position.
        """
        if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be SoulSpark.")
        if not isinstance(new_position, list) or len(new_position) != 3: raise ValueError("new_position must be list of 3 floats.")

        # Clamp position to be within grid bounds conceptually (although storage is int)
        clamped_pos = [max(0.0, min(float(self.grid_size[i]), p)) for i, p in enumerate(new_position)]

        setattr(soul_spark, 'position', clamped_pos)
        grid_coords = self._coords_to_int_tuple(clamped_pos)

        dominant_sephirah = self.get_dominant_sephiroth_at(grid_coords)
        new_field_key = dominant_sephirah if dominant_sephirah else 'void'
        setattr(soul_spark, 'current_field_key', new_field_key)
        # logger.debug(f"Moved soul {soul_spark.spark_id} to {clamped_pos}, Field: {new_field_key}")

    def place_soul_in_guff(self, soul_spark: SoulSpark) -> None:
        """Places the soul within the Guff region of Kether."""
        if not self.kether_field: raise RuntimeError("Kether field not initialized.")
        # Position randomly within Guff sphere for simplicity
        center = self.kether_field.location
        radius = self.kether_field.guff_radius
        phi = np.random.uniform(0, 2 * PI)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1) # For radius distribution
        theta = np.arccos(costheta)
        r = radius * np.cbrt(u) # Uniform volume distribution

        x = center[0] + r * np.sin(theta) * np.cos(phi)
        y = center[1] + r * np.sin(theta) * np.sin(phi)
        z = center[2] + r * np.cos(theta)

        guff_position = [x, y, z]
        self.move_soul(soul_spark, guff_position) # Use move_soul to set position and key
        setattr(soul_spark, 'current_field_key', 'kether') # Ensure key is Kether
        logger.info(f"Placed soul {soul_spark.spark_id} in Guff region at ~{self._coords_to_int_tuple(guff_position)}.")

    def release_soul_from_guff(self, soul_spark: SoulSpark) -> None:
        """Moves the soul just outside the Guff, still within Kether."""
        if not self.kether_field: raise RuntimeError("Kether field not initialized.")
        # Position just outside Guff radius, randomly on Kether sphere surface
        center = self.kether_field.location
        kether_radius = self.kether_field.radius
        guff_radius = self.kether_field.guff_radius
        release_radius = guff_radius + (kether_radius - guff_radius) * 0.1 # 10% outside guff

        phi = np.random.uniform(0, 2 * PI)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)

        x = center[0] + release_radius * np.sin(theta) * np.cos(phi)
        y = center[1] + release_radius * np.sin(theta) * np.sin(phi)
        z = center[2] + release_radius * np.cos(theta)

        release_position = [x, y, z]
        self.move_soul(soul_spark, release_position)
        setattr(soul_spark, 'current_field_key', 'kether') # Ensure key is Kether
        logger.info(f"Released soul {soul_spark.spark_id} from Guff to Kether region at ~{self._coords_to_int_tuple(release_position)}.")

# --- END OF FILE field_controller.py ---