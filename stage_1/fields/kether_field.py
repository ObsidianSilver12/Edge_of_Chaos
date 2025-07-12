# --- START OF FILE src/stage_1/fields/kether_field.py ---

"""
Kether Field Module (Refactored V4.1 - SEU/SU/CU Units, Production Concerns)

Implements Kether field including Guff region, using absolute units (SEU, SU, CU)
and targets defined in constants. Includes improved error handling.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# --- Use constants from the provided context ---
try:
    from shared.constants.constants import *
except ImportError:
    logging.critical("CRITICAL ERROR: constants.py not found or import failed in kether_field.py")
    # Define MINIMUM fallbacks
    FLOAT_EPSILON = 1e-9; GUFF_RADIUS_FACTOR = 0.3; GUFF_CAPACITY = 100
    MAX_SOUL_ENERGY_SEU = 1e6; MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0
    GUFF_TARGET_ENERGY_SEU = 800000.0; GUFF_TARGET_STABILITY_SU = 95.0; GUFF_TARGET_COHERENCE_CU = 95.0
    GRID_SIZE = (64, 64, 64); KETHER_FREQ = 963.0
    VOID_BASE_FREQUENCY_RANGE = (10.0, 1000.0) # For clamping

# Configure logger
logger = logging.getLogger(__name__)

# --- Base Class & Dependencies ---
try:
    from sephiroth_field import SephirothField
    from void_field import VoidField
    # field_harmonics is optional
    try: from field_harmonics import FieldHarmonics; FH_AVAILABLE = True
    except ImportError: FH_AVAILABLE = False; FieldHarmonics = None
except ImportError:
    try: # Fallback imports
        from sephiroth_field import SephirothField
        from void_field import VoidField
        try: from field_harmonics import FieldHarmonics; FH_AVAILABLE = True
        except ImportError: FH_AVAILABLE = False; FieldHarmonics = None
    except ImportError:
        logger.critical("CRITICAL ERROR: Cannot import base/void field in kether_field.py")
        class SephirothField: # Dummy
             def __init__(self, *args, **kwargs): self.location=(0,0,0); self.radius=1; self.sephirah_name='dummy'; self.target_frequency = 432.0
             def apply_sephiroth_influence(self, *args, **kwargs): pass
             def _validate_coordinates(self, *args): return True
        class VoidField: # Dummy
             def apply_geometric_pattern(self, *args, **kwargs): pass


class KetherField(SephirothField):
    """ Kether Field influence including the Guff. Uses absolute SEU/SU/CU targets. """

    # --- __init__ (Unchanged from V3.1 - uses constants) ---
    def __init__(self, location: Tuple[int, int, int], radius: float):
        super().__init__(sephirah_name='kether', location=location, radius=radius)
        self.guff_radius: float = self.radius * GUFF_RADIUS_FACTOR
        self.guff_radius_sq: float = self.guff_radius * self.guff_radius
        self.guff_min_offset: int = -int(np.ceil(self.guff_radius))
        self.guff_max_offset: int = int(np.ceil(self.guff_radius))
        self.guff_target_energy_seu: float = GUFF_TARGET_ENERGY_SEU
        self.guff_target_stability_su: float = GUFF_TARGET_STABILITY_SU
        self.guff_target_coherence_cu: float = GUFF_TARGET_COHERENCE_CU
        self.souls_in_guff: List[str] = []
        self.guff_capacity: int = GUFF_CAPACITY
        logger.info(f"KetherField initialized. Guff Radius: {self.guff_radius:.2f} (Targets: E={self.guff_target_energy_seu:.1f} SEU, S={self.guff_target_stability_su:.1f} SU, C={self.guff_target_coherence_cu:.1f} CU)")

    # --- apply_sephiroth_influence (Uses absolute targets, improved error checks) ---
    def apply_sephiroth_influence(self, void_field: VoidField) -> None:
        """ Applies Kether influence and modifies Guff region using absolute targets. """
        # 1. Apply base Kether influence
        super().apply_sephiroth_influence(void_field)
        logger.debug("Base Kether influence applied. Now applying Guff modifications...")

        # 2. Apply Guff-specific modifications
        center_x, center_y, center_z = self.location
        try:
            grid_shape = void_field.get_grid_shape()

            # Guff Bounding Box & Mask
            guff_min_x=max(0,center_x+self.guff_min_offset); guff_max_x=min(grid_shape[0],center_x+self.guff_max_offset+1)
            guff_min_y=max(0,center_y+self.guff_min_offset); guff_max_y=min(grid_shape[1],center_y+self.guff_max_offset+1)
            guff_min_z=max(0,center_z+self.guff_min_offset); guff_max_z=min(grid_shape[2],center_z+self.guff_max_offset+1)
            if guff_min_x>=guff_max_x or guff_min_y>=guff_max_y or guff_min_z>=guff_max_z: return

            gx_idx, gy_idx, gz_idx = np.meshgrid(np.arange(guff_min_x, guff_max_x), np.arange(guff_min_y, guff_max_y), np.arange(guff_min_z, guff_max_z), indexing='ij')
            guff_dist_sq = (gx_idx - center_x)**2 + (gy_idx - center_y)**2 + (gz_idx - center_z)**2
            guff_mask = guff_dist_sq <= self.guff_radius_sq
            if not np.any(guff_mask): return

            guff_slice = (slice(guff_min_x, guff_max_x), slice(guff_min_y, guff_max_y), slice(guff_min_z, guff_max_z))

            # Calculate influence strength within Guff
            guff_influence = 1.0 / (1.0 + guff_dist_sq[guff_mask] / (self.guff_radius_sq * 0.2))
            guff_influence = np.clip(guff_influence, 0.0, 1.0); inv_guff_influence = 1.0 - guff_influence

            # Ensure void_field grids exist before modification
            if void_field.energy is None or void_field.stability is None or void_field.coherence is None or \
               void_field.frequency is None or void_field.pattern_influence is None or void_field.order is None or void_field.chaos is None:
                 logger.error("Cannot apply Guff modifications: VoidField grids not initialized.")
                 return

            # Apply Guff Modifications (Blend towards Guff targets)
            current_e = void_field.energy[guff_slice][guff_mask]
            void_field.energy[guff_slice][guff_mask] = current_e * inv_guff_influence + self.guff_target_energy_seu * guff_influence
            current_s = void_field.stability[guff_slice][guff_mask]
            void_field.stability[guff_slice][guff_mask] = current_s * inv_guff_influence + self.guff_target_stability_su * guff_influence
            current_c = void_field.coherence[guff_slice][guff_mask]
            void_field.coherence[guff_slice][guff_mask] = current_c * inv_guff_influence + self.guff_target_coherence_cu * guff_influence
            current_f = void_field.frequency[guff_slice][guff_mask]
            void_field.frequency[guff_slice][guff_mask] = current_f * inv_guff_influence + self.target_frequency * guff_influence
            current_p = void_field.pattern_influence[guff_slice][guff_mask]
            void_field.pattern_influence[guff_slice][guff_mask] = current_p * inv_guff_influence + 0.95 * guff_influence

            # Apply Guff geometric pattern (Dodecahedron) via FieldBase method
            try:
                void_field.apply_geometric_pattern('dodecahedron', self.location, self.guff_radius, 0.9)
            except Exception as e: logger.error(f"Failed to apply Guff geometric pattern: {e}", exc_info=True)

            # Clamping & Derived Properties (Apply only to modified Guff cells)
            e_guff = void_field.energy[guff_slice][guff_mask]
            s_guff = void_field.stability[guff_slice][guff_mask]
            c_guff = void_field.coherence[guff_slice][guff_mask]
            f_guff = void_field.frequency[guff_slice][guff_mask]
            p_guff = void_field.pattern_influence[guff_slice][guff_mask]

            void_field.energy[guff_slice][guff_mask] = np.clip(e_guff, 0.0, MAX_SOUL_ENERGY_SEU * 5)
            void_field.stability[guff_slice][guff_mask] = np.clip(s_guff, 0.0, MAX_STABILITY_SU)
            void_field.coherence[guff_slice][guff_mask] = np.clip(c_guff, 0.0, MAX_COHERENCE_CU)
            void_field.frequency[guff_slice][guff_mask] = np.clip(f_guff, VOID_BASE_FREQUENCY_RANGE[0], VOID_BASE_FREQUENCY_RANGE[1]*2.0)
            void_field.pattern_influence[guff_slice][guff_mask] = np.clip(p_guff, 0.0, 1.0)

            norm_s = void_field.stability[guff_slice][guff_mask] / MAX_STABILITY_SU
            norm_c = void_field.coherence[guff_slice][guff_mask] / MAX_COHERENCE_CU
            new_order = np.clip(norm_s * 0.6 + norm_c * 0.4, 0.0, 1.0)
            void_field.order[guff_slice][guff_mask] = new_order
            void_field.chaos[guff_slice][guff_mask] = 1.0 - new_order

        except AttributeError as ae: logger.error(f"AttributeError applying Guff mods: {ae}. Void grids missing?", exc_info=True)
        except IndexError as ie: logger.error(f"IndexError applying Guff mods: {ie}. Slice/Mask invalid?", exc_info=True)
        except Exception as e: logger.error(f"Unexpected error applying Guff mods: {e}", exc_info=True)

    # --- is_in_guff (Added coordinate validation) ---
# In kether_field.py -> is_in_guff
    def is_in_guff(self, coords: Tuple[int, int, int]) -> bool:
        if not self._validate_coordinates(coords): return False
        dist_sq = sum((coords[i] - self.location[i])**2 for i in range(3))
        # Add a small tolerance, e.g., half diagonal of a cell squared (sqrt(3)/2)^2 = 0.75
        check_radius_sq = self.guff_radius_sq + 0.75
        # Or simpler: check against radius + 0.5 grid units squared
        # check_radius_sq = (self.guff_radius + 0.5)**2
        return dist_sq <= check_radius_sq # Use slightly larger radius for check

    # --- _validate_coordinates (Helper for bounds checking) ---
    def _validate_coordinates(self, coordinates: Tuple[int, int, int]) -> bool:
        # Needs grid size - assuming accessible via GRID_SIZE constant for now
        try:
            if not isinstance(coordinates, tuple) or len(coordinates) != 3: return False
            return all(0 <= coordinates[i] < GRID_SIZE[i] for i in range(3))
        except NameError: # GRID_SIZE not available
            logger.error("GRID_SIZE constant not found for coordinate validation in KetherField.")
            return False # Fail safe

    # --- register/remove_soul_from_guff (Unchanged) ---
    def register_soul_in_guff(self, soul_id: str) -> bool:
        if not isinstance(soul_id, str) or not soul_id: return False
        if len(self.souls_in_guff) >= self.guff_capacity:
            logger.warning(f"Guff at capacity ({self.guff_capacity}). Cannot register soul {soul_id}.")
            return False
        if soul_id not in self.souls_in_guff: self.souls_in_guff.append(soul_id)
        logger.debug(f"Soul {soul_id} registered in Guff ({len(self.souls_in_guff)}/{self.guff_capacity}).")
        return True
    def remove_soul_from_guff(self, soul_id: str) -> bool:
        if not isinstance(soul_id, str): return False
        if soul_id in self.souls_in_guff:
            self.souls_in_guff.remove(soul_id)
            logger.debug(f"Soul {soul_id} removed from Guff ({len(self.souls_in_guff)}/{self.guff_capacity}).")
            return True
        return False

    # --- get_guff_properties (Unchanged from V3.1) ---
    def get_guff_properties(self) -> Dict[str, Any]:
        return {
            'is_guff': True, 'center': self.location, 'radius': self.guff_radius,
            'capacity': self.guff_capacity, 'souls_stored_count': len(self.souls_in_guff),
            'target_energy_seu': self.guff_target_energy_seu,
            'target_stability_su': self.guff_target_stability_su,
            'target_coherence_cu': self.guff_target_coherence_cu,
            'target_frequency_hz': self.target_frequency,
        }

    # --- get_sound_parameters_for_guff (Placeholder - Needs Update) ---
    def get_sound_parameters_for_guff(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """ Get special sound parameters for the Guff region. """
        if not self.is_in_guff(coordinates): return {}
        logger.warning("get_sound_parameters_for_guff needs proper implementation using FieldHarmonics and SEU/SU/CU.")
        # Fallback using basic properties
        props = self.get_guff_properties()
        soul_density = props['souls_stored_count'] / max(1, props['capacity'])
        dist_sq = sum((coordinates[i] - self.location[i])**2 for i in range(3))
        norm_dist = np.sqrt(dist_sq) / max(FLOAT_EPSILON, self.guff_radius)

        return {
            'base_frequency': props.get('target_frequency_hz', KETHER_FREQ),
            'amplitude': np.clip(0.6 + 0.3 * soul_density - norm_dist * 0.4, 0.1, 0.9),
            'waveform': 'sine', 'reverb': 0.8, 'filter_cutoff': 1500 + soul_density * 1000,
            'resonance': 0.6 + soul_density * 0.3, 'is_guff': True
        }

    # --- String Representations (Unchanged from V3.1) ---
    def __str__(self) -> str: return f"KetherField(Loc: {self.location}, Rad: {self.radius:.1f}, GuffRad: {self.guff_radius:.1f}, Souls: {len(self.souls_in_guff)})"
    def __repr__(self) -> str: return f"<KetherField name='{self.sephirah_name}' location={self.location} radius={self.radius} guff_radius={self.guff_radius}>"

# --- END OF FILE src/stage_1/fields/kether_field.py ---