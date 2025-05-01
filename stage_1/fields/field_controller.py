# --- START OF FILE src/stage_1/fields/field_controller.py ---

"""
Field Controller Module (Refactored V4.3 - SEU/SU/CU Units, PEP8)

Manages VoidField/SephirothField/KetherField, uses SEU/SU/CU units from constants,
includes updated sound logic, performance guards, improved error handling, and
calls updated field/soul methods. Adheres to PEP 8 formatting.
"""

import logging
import numpy as np  # Import numpy with standard alias
from typing import Dict, Any, Tuple, Optional, List, Union
import json
from datetime import datetime
import os # Added os import

# --- Logger Initialization ---
logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately
try:
    import constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for FieldController logger."
    )
    logger.setLevel(logging.INFO)

# --- Constants Import (using alias 'const') ---
try:
    import constants.constants as const
except ImportError as e:
    logger.critical(
        "CRITICAL ERROR: constants.py failed import in field_controller.py"
    )
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Field & Soul Imports ---
try:
    # Base class (optional, but good for type hinting)
    from .field_base import FieldBase
    # Concrete field implementations
    from .void_field import VoidField
    from .sephiroth_field import SephirothField
    from .kether_field import KetherField
    # Soul representation
    from stage_1.soul_spark.soul_spark import SoulSpark
    # Aspect dictionary (needed for Sephiroth initialization)
    from .sephiroth_aspect_dictionary import aspect_dictionary
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary is None")
    # Field Harmonics (optional)
    try:
        from .field_harmonics import FieldHarmonics
        FH_AVAILABLE = True
    except ImportError:
        FH_AVAILABLE = False
        FieldHarmonics = None # Define as None if not available
        logger.warning("FieldHarmonics not found. Sound generation will use fallback.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import field/soul/aspect modules: {e}",
                    exc_info=True)
    raise ImportError(f"Core field/soul dependencies missing: {e}") from e


class FieldController:
    """
    Manages the simulation's field environment, including Void and Sephiroth.
    Handles interactions and soul placement using SEU/SU/CU units.
    """

    def __init__(self, grid_size: Tuple[int, int, int] = const.GRID_SIZE):
        """Initialize the FieldController."""
        if not (isinstance(grid_size, tuple) and len(grid_size) == 3 and
                all(isinstance(d, int) and d > 0 for d in grid_size)):
             raise ValueError("grid_size must be a tuple of 3 positive integers.")
        self.grid_size: Tuple[int, int, int] = grid_size
        self.dimensions: int = len(grid_size) 
        logger.info(f"Initializing FieldController with grid size {self.grid_size}...")

        # Check aspect dictionary early - critical dependency
        if aspect_dictionary is None:
             raise RuntimeError("Aspect Dictionary unavailable. FieldController "
                                "cannot initialize Sephiroth fields.")

        # 1. Create Void Field
        try:
            self.void_field: VoidField = VoidField(grid_size=self.grid_size)
            logger.info("VoidField created successfully.")
        except Exception as e:
            logger.critical("CRITICAL: VoidField initialization failed.", exc_info=True)
            raise RuntimeError("VoidField initialization failed.") from e

        # 2. Define Sephiroth Locations/Radii
        self._calculate_sephiroth_positions()

        # 3. Create Sephiroth Influencers
        self.sephiroth_influencers: Dict[str, SephirothField] = {}
        self.kether_field: Optional[KetherField] = None # Specifically track Kether
        self._create_sephiroth_influencers()

        # 4. Apply Initial Influences to Void Field
        self.apply_initial_influences()

        # 5. Initialize Tracking Variables
        self.update_counter: int = 0
        self.optimal_development_points: List[Tuple[int, int, int]] = []
        self.edge_values: Dict[Tuple[int, int, int], float] = {}
        # Initialize EoC tracking (call internal helper)
        self._initialize_edge_of_chaos_tracking()

        logger.info("FieldController initialized successfully and initial "
                    "influences applied.")

    def _calculate_sephiroth_positions(self) -> None:
        """Calculates grid positions and radii for Sephiroth."""
        logger.debug("Calculating Sephiroth positions and radii...")
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2
        center_z = self.grid_size[2] // 2
        # Scaling factors based on Tree of Life proportions relative to grid size
        # Example: 9 units wide, 16 units high in schematic
        scale_x = self.grid_size[0] / 9.0
        scale_y = self.grid_size[1] / 16.0
        # Default radius scales with the smallest grid dimension
        default_radius = const.SEPHIROTH_DEFAULT_RADIUS * (min(self.grid_size) / 64.0)

        # Positions based on a 9x16 grid (as per user specification)
        tol_base_positions = { # (Schematic X, Schematic Y)
            'kether': (4.5, 1), 'chokmah': (9, 3), 'binah': (0, 3),
            'daath': (4.5, 4.75), 'chesed': (9, 7.5), 'geburah': (0, 7.5),
            'tiphareth': (4.5, 9), 'netzach': (9, 12), 'hod': (0, 12),
            'yesod': (4.5, 14), 'malkuth': (4.5, 16)
        }
        self.sephiroth_locations: Dict[str, Tuple[int, int, int]] = {}
        self.sephiroth_radii: Dict[str, float] = {}

        for name, (tol_x, tol_y) in tol_base_positions.items():
            # Scale schematic coords to grid coords
            grid_x = int(tol_x * scale_x)
            grid_y = int(tol_y * scale_y)
            # Base Z is center, adjust for pillars
            grid_z = center_z
            if name in ['chokmah', 'chesed', 'netzach']: # Right pillar
                 grid_z += int(self.grid_size[2] * 0.1) # Slightly forward
            elif name in ['binah', 'geburah', 'hod']: # Left pillar
                 grid_z -= int(self.grid_size[2] * 0.1) # Slightly back

            # Clamp coordinates to be within grid boundaries
            grid_x = max(0, min(self.grid_size[0] - 1, grid_x))
            grid_y = max(0, min(self.grid_size[1] - 1, grid_y))
            grid_z = max(0, min(self.grid_size[2] - 1, grid_z))
            self.sephiroth_locations[name] = (grid_x, grid_y, grid_z)

            # Calculate radius (adjust multiplier based on importance/size)
            rad_multiplier = 1.0 # Default
            if name == 'kether': rad_multiplier = 1.2
            elif name == 'tiphareth': rad_multiplier = 1.1
            elif name == 'malkuth': rad_multiplier = 1.0
            elif name == 'daath': rad_multiplier = 0.8 # Daath is often smaller/hidden
            else: rad_multiplier = 0.9 # Other Sephiroth slightly smaller?
            self.sephiroth_radii[name] = default_radius * rad_multiplier
            logger.debug(f"  {name.capitalize()}: Pos={self.sephiroth_locations[name]}, "
                         f"Rad={self.sephiroth_radii[name]:.2f}")

    def _create_sephiroth_influencers(self) -> None:
        """Creates SephirothField/KetherField instances."""
        logger.debug("Creating Sephiroth influencer objects...")
        if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")

        seph_names_to_create = aspect_dictionary.sephiroth_names
        created_count = 0
        for name in seph_names_to_create:
             loc = self.sephiroth_locations.get(name)
             rad = self.sephiroth_radii.get(name)
             if loc is None or rad is None:
                 logger.error(f"Skipping Sephirah '{name}': Missing location or radius.")
                 continue
             try:
                 if name == 'kether':
                     self.kether_field = KetherField(location=loc, radius=rad)
                     self.sephiroth_influencers[name] = self.kether_field
                     logger.debug(f"  Created KetherField at {loc}, R={rad:.2f}")
                 else:
                     self.sephiroth_influencers[name] = SephirothField(
                         sephirah_name=name, location=loc, radius=rad
                     )
                     logger.debug(f"  Created SephirothField '{name}' at {loc}, R={rad:.2f}")
                 created_count += 1
             except Exception as e:
                 logger.error(f"Failed to create influencer '{name}': {e}", exc_info=True)

        if created_count != len(seph_names_to_create):
             logger.warning(f"Influencer count mismatch: "
                            f"Created {created_count}/{len(seph_names_to_create)}")
        if not self.kether_field: # Kether is essential for Guff logic
             raise RuntimeError("KetherField instance could not be created. Cannot proceed.")

    def _initialize_edge_of_chaos_tracking(self) -> None:
        """Initializes tracking of high EoC points."""
        logger.debug("Initializing Edge of Chaos tracking...")
        try:
            # Use the VoidField method to find optimal points
            self.optimal_development_points = self.void_field.find_optimal_development_points(count=10)
            # Calculate EoC values for these points
            self.edge_values = {
                p: self.void_field.calculate_edge_of_chaos(p)
                for p in self.optimal_development_points
                if self.void_field._validate_coordinates(p) # Ensure validity
            }
            if self.edge_values:
                max_eoc = max(self.edge_values.values())
                logger.info(f"EoC tracking initialized: {len(self.optimal_development_points)} points. "
                            f"Max EoC value: {max_eoc:.4f}")
            else:
                 logger.warning("EoC tracking initialized, but no optimal points found/valid.")
                 self.optimal_development_points = []
        except AttributeError as ae:
            logger.error(f"Failed EoC initialization: VoidField missing method? Error: {ae}", exc_info=True)
            self.optimal_development_points = []
            self.edge_values = {}
        except Exception as e:
            logger.error(f"Failed EoC initialization: {e}", exc_info=True)
            self.optimal_development_points = []
            self.edge_values = {}

    def apply_initial_influences(self) -> None:
        """Applies the initial static influence of all Sephiroth onto the Void."""
        logger.info("Applying initial Sephiroth influences to VoidField...")
        for name, influencer in self.sephiroth_influencers.items():
            try:
                # Call the influencer's method to modify the VoidField
                influencer.apply_sephiroth_influence(self.void_field)
                logger.debug(f"  Applied initial influence for '{name}'.")
            except Exception as e:
                logger.error(f"Error applying initial influence for '{name}': {e}",
                             exc_info=True)
        logger.info("Initial Sephiroth influences applied.")

    def update_fields(self, delta_time: float) -> None:
        """Updates the VoidField dynamics and reapplies Sephiroth influences."""
        if delta_time <= 0: return # No change if no time passed
        logger.debug(f"Updating fields (dt={delta_time:.3f}s)...")
        try:
            # 1. Update Void Field Dynamics (diffusion, dissipation, resonance)
            self.void_field.update_step(delta_time)

            # 2. Re-apply Sephiroth Influences (they modify the updated void state)
            # This could be optimized if influences are static, but allows for dynamic influencers later
            for name, influencer in self.sephiroth_influencers.items():
                try:
                    influencer.apply_sephiroth_influence(self.void_field)
                except Exception as e:
                     logger.error(f"Error reapplying influence for '{name}' during update: {e}",
                                  exc_info=True) # Log but continue update

            # 3. Update Tracking
            self.update_counter += 1
            # Periodically update EoC tracking
            if self.update_counter % 10 == 0:
                self._update_edge_of_chaos_tracking()

        except Exception as e:
            logger.error(f"Error during field update cycle: {e}", exc_info=True)
            # Depending on severity, could raise error or just log and attempt recovery

    def _update_edge_of_chaos_tracking(self) -> None:
        """Recalculates EoC values for tracked points and potentially finds new ones."""
        logger.debug("Refreshing Edge of Chaos tracking...")
        try:
            updated_edge_values = {}
            current_points = getattr(self, 'optimal_development_points', [])
            # Recalculate EoC for existing tracked points
            for point in current_points:
                if self.void_field._validate_coordinates(point):
                    updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

            # Periodically search for new high-EoC points
            if self.update_counter % 50 == 0:
                new_points = self.void_field.find_optimal_development_points(count=5) # Find 5 new candidates
                for point in new_points:
                    if point not in updated_edge_values and self.void_field._validate_coordinates(point):
                        updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

            if updated_edge_values:
                # Keep the top N points based on current EoC values
                sorted_points = sorted(updated_edge_values.keys(),
                                       key=lambda p: updated_edge_values.get(p, 0.0),
                                       reverse=True)
                self.optimal_development_points = sorted_points[:10] # Keep top 10
                self.edge_values = {p: updated_edge_values[p] for p in self.optimal_development_points}
                # Optional: Log only if values change significantly?
                # logger.debug(f"Refreshed EoC tracking. Max EoC: {max(self.edge_values.values()):.4f}")
            else:
                 # If no valid points found, reset tracking
                 self.optimal_development_points = []
                 self.edge_values = {}
        except Exception as e:
            logger.error(f"Error updating EoC tracking: {e}", exc_info=True)

    def get_field(self, field_key: str) -> Optional[Union[VoidField, SephirothField]]:
        """Retrieves a specific field object by key ('void' or sephirah name)."""
        field_key_lower = field_key.lower()
        if field_key_lower == 'void':
            return self.void_field
        elif field_key_lower in self.sephiroth_influencers:
            return self.sephiroth_influencers[field_key_lower]
        else:
             logger.warning(f"Field key '{field_key}' not recognized.")
             return None

    def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Gets combined field properties (SEU/SU/CU etc.) at specific coordinates,
        including dominant Sephirah, Guff status, and EoC value. Fails hard.
        """
        try:
            # Get base properties from VoidField (which holds the actual grid state)
            base_properties = self.void_field.get_properties_at(coordinates)

            # Determine dominant influencer at this location
            dominant_sephirah = self.get_dominant_sephiroth_at(coordinates)
            base_properties['dominant_sephirah'] = dominant_sephirah # Store name or None

            # Add specific properties if within a Sephirah influence zone
            if dominant_sephirah:
                sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
                if sephirah_field:
                     # Add geometric identifiers if available on the influencer
                     if getattr(sephirah_field, 'geometric_pattern', None):
                          base_properties['geometric_pattern'] = sephirah_field.geometric_pattern
                     if getattr(sephirah_field, 'platonic_affinity', None):
                          base_properties['platonic_affinity'] = sephirah_field.platonic_affinity

            # Calculate and add Edge of Chaos value
            base_properties['edge_of_chaos'] = self.void_field.calculate_edge_of_chaos(coordinates)

            # Check if within Guff (only possible if dominant is Kether)
            base_properties['in_guff'] = False
            if dominant_sephirah == 'kether' and self.kether_field:
                 if self.kether_field.is_in_guff(coordinates):
                     base_properties['in_guff'] = True
                     # Add Guff-specific target properties
                     guff_targets = self.kether_field.get_guff_properties()
                     base_properties['guff_target_energy_seu'] = guff_targets.get('target_energy_seu')
                     base_properties['guff_target_stability_su'] = guff_targets.get('target_stability_su')
                     base_properties['guff_target_coherence_cu'] = guff_targets.get('target_coherence_cu')

            return base_properties
        except IndexError as e: # Raised by get_properties_at if coords invalid
            logger.error(f"Coordinates out of bounds in get_properties_at({coordinates}).")
            raise # Re-raise index error
        except Exception as e:
            logger.error(f"Unexpected error in get_properties_at({coordinates}): {e}",
                         exc_info=True)
            raise # Re-raise other critical errors

    def get_dominant_sephiroth_at(self, coordinates: Tuple[int, int, int]) -> Optional[str]:
        """Determines which Sephirah influence is dominant at given coordinates."""
        min_dist_sq = float('inf')
        dominant_sephirah = None
        # Validate coordinates first
        if not self.void_field._validate_coordinates(coordinates):
             logger.warning(f"Invalid coordinates for dominant Sephirah check: {coordinates}")
             return None

        for name, influencer in self.sephiroth_influencers.items():
            loc = influencer.location
            # Calculate squared distance for efficiency
            dist_sq = sum((coordinates[i] - loc[i])**2 for i in range(self.dimensions)) # Use self.dimensions (set to 3 in __init__)
            # Check if within the influencer's radius (using squared distance)
            if dist_sq <= influencer.radius_sq:
                # If multiple overlaps, the closest one is dominant
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    dominant_sephirah = name
        return dominant_sephirah

    # --- Sound Parameter Generation ---
    def get_sound_parameters_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Get sound parameters for a location, preferring FieldHarmonics if available.
        Fallback uses direct mapping from SEU/SU/CU properties. Fails hard.
        """
        try:
            field_properties = self.get_properties_at(coordinates) # Fails hard if coords invalid
            field_type = field_properties.get('dominant_sephirah', 'void')

            # Special handling for Guff sound
            if field_properties.get('in_guff', False) and self.kether_field:
                try:
                    return self.kether_field.get_sound_parameters_for_guff(coordinates)
                except Exception as guff_sound_err:
                    logger.error(f"Error getting Guff sound params: {guff_sound_err}", exc_info=True)
                    # Fall through to generic fallback if Guff sound fails

            # Use FieldHarmonics if available
            if FH_AVAILABLE and FieldHarmonics:
                try:
                    # Assumes FieldHarmonics handles SEU/SU/CU inputs correctly
                    return FieldHarmonics.get_live_sound_parameters(field_type, field_properties)
                except Exception as fh_err:
                    logger.error(f"FieldHarmonics failed for {field_type} at {coordinates}: {fh_err}. Using fallback.", exc_info=True)
                    # Fall through to fallback if FieldHarmonics fails

            # --- Fallback Sound Parameter Generation (Uses SEU/SU/CU) ---
            logger.debug("Using fallback sound parameter generation.")
            base_freq = field_properties.get('frequency_hz', 432.0)
            if base_freq <= 0: base_freq = 432.0 # Ensure positive frequency

            # Map energy SEU (0 to MAX*0.5) to amplitude (0.1 to 0.8)
            energy_norm = np.clip(field_properties.get('energy_seu', 0.0) /
                                  max(const.FLOAT_EPSILON, const.MAX_SOUL_ENERGY_SEU * 0.5),
                                  0.0, 1.0)
            amplitude = 0.1 + energy_norm * 0.7

            # Map coherence CU (0-100) to waveform complexity/filter cutoff
            coherence_norm = np.clip(field_properties.get('coherence_cu', 0.0) /
                                     const.MAX_COHERENCE_CU, 0.0, 1.0)
            if coherence_norm > 0.7: waveform = 'sine'
            elif coherence_norm > 0.4: waveform = 'triangle'
            else: waveform = 'sawtooth'
            filter_cutoff = 500 + coherence_norm * 5000

            # Map stability SU (0-100) to modulation/detune (less stable = more mod)
            stability_norm = np.clip(field_properties.get('stability_su', 0.0) /
                                     const.MAX_STABILITY_SU, 0.0, 1.0)
            modulation = (1.0 - stability_norm) * 0.1 # Max 10% modulation

            return {
                'base_frequency': base_freq, 'amplitude': amplitude, 'waveform': waveform,
                'filter_cutoff': filter_cutoff, 'resonance': coherence_norm * 0.8,
                'modulation': modulation, 'field_type': field_type # Add field type for context
            }
        except IndexError: # Catch index errors specifically from get_properties_at
             logger.error(f"Coordinates out of bounds in get_sound_parameters_at({coordinates}).")
             raise # Re-raise
        except Exception as e:
            logger.error(f"Error in get_sound_parameters_at({coordinates}): {e}", exc_info=True)
            # Return minimal safe fallback on unexpected errors
            return {'base_frequency': 432.0, 'amplitude': 0.5, 'waveform': 'sine'}

    # --- Geometric Influence on Soul ---
    def apply_geometric_influence_to_soul(self, soul_spark: SoulSpark
                                          ) -> Dict[str, float]:
        """Applies geometric influences based on local field properties."""
        # Implementation unchanged from previous PEP8 version - it uses factors
        # and applies deltas based on GEOMETRY_EFFECTS constants.
        changes_applied_dict = {}
        try:
            position_coords = self._coords_to_int_tuple(soul_spark.position)
            local_props = self.get_properties_at(position_coords)
            dominant_sephirah = local_props.get('dominant_sephirah')
            if not dominant_sephirah: return {}
            sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
            if not sephirah_field: return {}

            geom_pattern = getattr(sephirah_field, 'geometric_pattern', None)
            plat_affinity = getattr(sephirah_field, 'platonic_affinity', None)
            # Use Edge of Chaos as resonance factor for geometric influence?
            resonance_factor = local_props.get('edge_of_chaos', 0.5)

            geom_effects = const.GEOMETRY_EFFECTS.get(geom_pattern, {})
            plat_effects = const.GEOMETRY_EFFECTS.get(plat_affinity, {})
            combined_effects = {}
            all_keys = set(geom_effects.keys()) | set(plat_effects.keys())
            for key in all_keys:
                 # Average effect if present in both, otherwise take the single value
                 effect_geom = geom_effects.get(key, 0.0)
                 effect_plat = plat_effects.get(key, 0.0)
                 count = (1 if effect_geom != 0.0 else 0) + (1 if effect_plat != 0.0 else 0)
                 combined_effects[key] = (effect_geom + effect_plat) / max(1.0, count)

            if not combined_effects: return {}

            transformation_occurred = False
            for effect_name, modifier in combined_effects.items():
                 # Target attribute determination logic copied from _apply_geometric_transformation
                 target_attr = effect_name
                 is_boost = '_boost' in effect_name or '_factor' in effect_name
                 is_push = '_push' in effect_name
                 if is_boost: target_attr = effect_name.split('_boost')[0].split('_factor')[0]
                 if is_push: target_attr = effect_name.split('_push')[0]

                 if not hasattr(soul_spark, target_attr): continue
                 current_value = getattr(soul_spark, target_attr)
                 if not isinstance(current_value, (int, float)): continue

                 change=0.0; max_clamp=1.0 # Assume 0-1 default
                 if target_attr=='stability': max_clamp=const.MAX_STABILITY_SU
                 elif target_attr=='coherence': max_clamp=const.MAX_COHERENCE_CU

                 if is_boost or not is_push: change = modifier * resonance_factor * 0.05
                 elif is_push:
                      push_target=0.5 if 'balance' in effect_name else 0.0
                      diff=push_target-current_value; change=diff*abs(modifier)*resonance_factor*0.05

                 new_value = current_value + change
                 clamped_new_value = max(0.0, min(max_clamp, new_value))
                 actual_change = clamped_new_value - current_value

                 if abs(actual_change) > const.FLOAT_EPSILON:
                     setattr(soul_spark, target_attr, float(clamped_new_value))
                     changes_applied_dict[f"{target_attr}_geom_delta"] = float(actual_change)
                     transformation_occurred = True

            if transformation_occurred:
                 setattr(soul_spark, 'last_modified', datetime.now().isoformat())
                 logger.debug(f"Applied geom effects to soul {soul_spark.spark_id} from {dominant_sephirah}.")
            return changes_applied_dict
        except Exception as e:
            logger.error(f"Error applying geom influence to soul {soul_spark.spark_id}: {e}",
                         exc_info=True)
            return {}


    # --- Soul Position & State Management ---
    def _coords_to_int_tuple(self, position: List[float]) -> Tuple[int, int, int]:
        """Converts float position list to clamped integer grid coordinate tuple."""
        if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
             logger.error(f"Invalid position format: {position}. Using grid center.")
             # Return clamped center coordinates
             return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))
        try:
            coords = []
            for i, p in enumerate(position):
                # Round to nearest int, then clamp to valid index range [0, size-1]
                coord_int = int(round(float(p)))
                clamped_coord = max(0, min(self.grid_size[i] - 1, coord_int))
                coords.append(clamped_coord)
            final_coords = tuple(coords)
            # Final validation check
            if not all(0 <= c < self.grid_size[i] for i, c in enumerate(final_coords)):
                 logger.error(f"Coordinate calculation resulted in out-of-bounds index {final_coords} "
                              f"for grid {self.grid_size}. Returning clamped center.")
                 return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))
            return final_coords
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting position {position} to int tuple: {e}. Using grid center.")
            return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))

    def move_soul(self, soul_spark: SoulSpark, new_position: List[float]) -> None:
        """Updates soul's position, field key, and applies transition effects."""
        if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
        if not isinstance(new_position, list) or len(new_position) != 3: raise ValueError("new_position invalid.")

        old_pos_coords = self._coords_to_int_tuple(getattr(soul_spark, 'position', new_position))
        old_field_key = getattr(soul_spark, 'current_field_key', None)

        # Clamp float position BEFORE setting attribute
        clamped_pos = [max(0.0, min(float(self.grid_size[i] - const.FLOAT_EPSILON), p))
                       for i, p in enumerate(new_position)]
        setattr(soul_spark, 'position', clamped_pos)
        new_grid_coords = self._coords_to_int_tuple(clamped_pos) # Convert clamped float pos

        dominant_sephirah = self.get_dominant_sephiroth_at(new_grid_coords)
        new_field_key = dominant_sephirah if dominant_sephirah else 'void'
        setattr(soul_spark, 'current_field_key', new_field_key)

        # --- Guff Registration Logic ---
        if self.kether_field: # Check if Kether field exists
            try:
                was_in_guff = (old_field_key == 'kether' and
                               self.kether_field.is_in_guff(old_pos_coords))
                is_in_guff = (new_field_key == 'kether' and
                              self.kether_field.is_in_guff(new_grid_coords))
                if is_in_guff and not was_in_guff:
                    self.kether_field.register_soul_in_guff(soul_spark.spark_id)
                elif was_in_guff and not is_in_guff:
                    self.kether_field.remove_soul_from_guff(soul_spark.spark_id)
            except Exception as guff_err:
                 logger.error(f"Error during Guff registration check: {guff_err}", exc_info=True)
        # --- End Guff Logic ---

        if old_field_key != new_field_key:
            self._apply_field_transition_effects(soul_spark, old_field_key,
                                               new_field_key, new_grid_coords)
            logger.info(f"Soul {soul_spark.spark_id} moved to {new_grid_coords}, "
                        f"field transition: {old_field_key} -> {new_field_key}")
        else:
             logger.debug(f"Soul {soul_spark.spark_id} moved within {new_field_key} "
                         f"to {new_grid_coords}")

        # Apply local geometric influence at new position
        self.apply_geometric_influence_to_soul(soul_spark)
        # Update soul's internal state (like S/C scores based on new position/influence)
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()


    def _apply_field_transition_effects(self, soul_spark: SoulSpark,
                                        old_field: Optional[str], new_field: str,
                                        coords: Tuple[int, int, int]) -> None:
        """Applies effects when a soul transitions between field types."""
        if old_field is None: return # No transition on first placement
        try:
            field_props = self.get_properties_at(coords)
            is_entering_guff = (new_field == 'kether' and
                                field_props.get('in_guff', False) and
                                (old_field != 'kether' or not field_props.get('was_in_guff_prev', False))) # Crude check

            if is_entering_guff:
                logger.debug(f"Soul {soul_spark.spark_id} entering Guff.")
                target_e = field_props.get('guff_target_energy_seu', soul_spark.energy)
                # Small immediate energy adjustment towards Guff target
                soul_spark.energy += (target_e - soul_spark.energy) * 0.05
                # Reduce freq variance history to allow faster stabilization in Guff
                if hasattr(soul_spark, 'frequency_history'):
                    soul_spark.frequency_history = soul_spark.frequency_history[-5:]

            # Log entry into any named Sephirah zone
            if new_field != 'void':
                logger.debug(f"Soul {soul_spark.spark_id} entering influence of {new_field}.")
                if hasattr(soul_spark, 'interaction_history'):
                     soul_spark.interaction_history.append({
                         'type': 'field_transition', 'timestamp': datetime.now().isoformat(),
                         'from': old_field, 'to': new_field, 'position': coords
                     })

            # Example: Apply effect based on Edge of Chaos value
            edge_value = field_props.get('edge_of_chaos', 0.0)
            if edge_value > 0.75 and hasattr(soul_spark, 'potential_realization'):
                 # Increase potential realization factor (0-1) in high EoC zones
                 current_potential = getattr(soul_spark, 'potential_realization', 0.0)
                 setattr(soul_spark, 'potential_realization', min(1.0, current_potential + 0.01))

        except Exception as e:
            logger.error(f"Error applying field transition effects for {soul_spark.spark_id}: {e}",
                         exc_info=True)

    def place_soul_in_guff(self, soul_spark: SoulSpark) -> None:
        """Places soul randomly within the Guff region."""
        if not self.kether_field: raise RuntimeError("Kether field not initialized.")
        center = self.kether_field.location
        guff_radius = self.kether_field.guff_radius
        if guff_radius <= 0: raise ValueError("Guff radius must be positive.")

        # Generate random point within the Guff sphere
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1) # Uniform distribution for radius cubed
        theta = np.arccos(costheta)
        r = guff_radius * np.cbrt(u) # Correct sampling for uniform volume

        x = center[0] + r * np.sin(theta) * np.cos(phi)
        y = center[1] + r * np.sin(theta) * np.sin(phi)
        z = center[2] + r * np.cos(theta)

        guff_position_float = [x, y, z]
        # Use move_soul to handle clamping, field key update, Guff registration
        self.move_soul(soul_spark, guff_position_float)

        # Force field key if move_soul failed detection (shouldn't happen often)
        if soul_spark.current_field_key != 'kether':
            logger.warning(f"Forcing field key to 'kether' after Guff placement.")
            setattr(soul_spark, 'current_field_key', 'kether')
            # Re-register just in case move_soul failed that part
            self.kether_field.register_soul_in_guff(soul_spark.spark_id)

        logger.info(f"Placed soul {soul_spark.spark_id} in Guff at "
                    f"{soul_spark.position} (Field: {soul_spark.current_field_key}).")

    def release_soul_from_guff(self, soul_spark: SoulSpark) -> None:
        """Moves soul just outside Guff radius but within Kether influence."""
        if not self.kether_field: raise RuntimeError("Kether field not initialized.")
        center = self.kether_field.location
        kether_radius = self.kether_field.radius
        guff_radius = self.kether_field.guff_radius
        if kether_radius <= guff_radius:
            logger.warning("Kether radius not larger than Guff radius, releasing just outside Guff.")
            release_radius = guff_radius + 1.0
        else:
             # Place slightly outside Guff, within Kether zone
             release_radius = guff_radius + (kether_radius - guff_radius) * 0.1
        release_radius = max(guff_radius + 0.5, release_radius) # Ensure distinctly outside

        # Generate random point on sphere surface at release_radius
        # Generate random point on sphere surface at release_radius
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)

        x = center[0] + release_radius * np.sin(theta) * np.cos(phi)
        y = center[1] + release_radius * np.sin(theta) * np.sin(phi)
        z = center[2] + release_radius * np.cos(theta)

        release_position_float = [x, y, z]
        # Use move_soul to handle clamping, field key update, Guff removal
        self.move_soul(soul_spark, release_position_float)

        # Force field key if needed
        if soul_spark.current_field_key != 'kether':
            logger.warning(f"Forcing field key to 'kether' after Guff release.")
            setattr(soul_spark, 'current_field_key', 'kether')
            # Ensure removed from Guff registration if move_soul missed it
            self.kether_field.remove_soul_from_guff(soul_spark.spark_id)

        logger.info(f"Released soul {soul_spark.spark_id} from Guff to Kether at "
                    f"{soul_spark.position} (Field: {soul_spark.current_field_key}).")


    def find_optimal_development_location(self, soul_spark: Optional[SoulSpark] = None
                                          ) -> List[float]:
        """
        Finds a location with high Edge of Chaos value, suitable for development.
        Selection from top N points is deterministic based on soul_spark ID if provided,
        otherwise random.
        """
        try:
            current_points = getattr(self, 'optimal_development_points', [])
            # Refresh EoC tracking if list is empty or seems stale
            if not current_points or not hasattr(self, 'edge_values') or not self.edge_values:
                self._initialize_edge_of_chaos_tracking()
                current_points = getattr(self, 'optimal_development_points', [])

            # If still no points, return random location within grid
            if not current_points or not self.edge_values:
                logger.warning("No optimal EoC points found, returning random location.")
                return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

            # Filter points that are still valid and have EoC values
            valid_points = [p for p in current_points if p in self.edge_values]
            if not valid_points:
                 logger.warning("Tracked EoC points are invalid, returning random location.")
                 return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

            # Select from top N points (e.g., top 5) based on EoC value
            sorted_points = sorted(valid_points,
                                   key=lambda p: self.edge_values.get(p, 0.0),
                                   reverse=True)
            num_top = min(5, len(sorted_points))
            idx = np.random.randint(0, num_top) # Default to random index in top N

            # Optional: Use soul ID hash for deterministic selection (if repeatable needed)
            if soul_spark and hasattr(soul_spark, 'spark_id'):
                try: idx = hash(soul_spark.spark_id) % num_top
                except Exception as hash_err:
                    logger.warning(f"Could not hash soul ID {soul_spark.spark_id}: {hash_err}. Using random index.")

            optimal_point_int = sorted_points[idx]
            # Return float coordinates near the center of the chosen grid cell
            return [float(c) + 0.5 for c in optimal_point_int]

        except Exception as e:
            logger.error(f"Error finding optimal development location: {e}", exc_info=True)
            # Fallback to random location on any error
            return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

    def get_visualization_data(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Gets data suitable for visualization at specific coordinates."""
        try:
            field_props = self.get_properties_at(coords) # Fails hard on invalid coords
            sound_params = self.get_sound_parameters_at(coords) # Uses fallback if needed

            # Combine properties and sound info
            # Prioritize FieldHarmonics visualization data if available
            if FH_AVAILABLE and FieldHarmonics:
                 try:
                     visualization = FieldHarmonics.generate_live_sound_visualization(
                         sound_params, field_props.get('dominant_sephirah', 'void'), field_props
                     )
                     # Add core field properties not included by FH visualization
                     core_props_to_add = ['energy_seu', 'frequency_hz', 'stability_su',
                                          'coherence_cu', 'edge_of_chaos', 'in_guff']
                     for prop in core_props_to_add:
                         if prop not in visualization:
                             visualization[prop] = field_props.get(prop)
                     return visualization
                 except Exception as fh_viz_err:
                      logger.warning(f"FieldHarmonics visualization failed: {fh_viz_err}. Using basic fallback.", exc_info=True)
                      # Fall through to basic fallback

            # Basic Fallback Visualization Data
            viz_data = {
                'energy_seu': field_props.get('energy_seu'),
                'frequency_hz': field_props.get('frequency_hz'),
                'stability_su': field_props.get('stability_su'),
                'coherence_cu': field_props.get('coherence_cu'),
                'dominant_sephirah': field_props.get('dominant_sephirah'),
                'edge_of_chaos': field_props.get('edge_of_chaos'),
                'color_rgb': field_props.get('color_rgb', [0.5, 0.5, 0.5]), # Grey fallback
                'in_guff': field_props.get('in_guff', False)
            }
            return viz_data

        except IndexError:
            logger.error(f"Coordinates {coords} out of bounds for visualization.")
            return {'error': 'Coordinates out of bounds', 'coords': coords}
        except Exception as e:
            logger.error(f"Error generating visualization data for {coords}: {e}",
                         exc_info=True)
            return {'error': str(e), 'coords': coords} # Return error dict


    def __str__(self) -> str:
        """String representation of the FieldController."""
        seph_count = len(self.sephiroth_influencers)
        void_status = "Initialized" if self.void_field and self.void_field.energy is not None else "Uninitialized"
        return (f"FieldController(Grid: {self.grid_size}, "
                f"Sephiroth: {seph_count}, Void Status: {void_status})")

# --- END OF FILE src/stage_1/fields/field_controller.py ---