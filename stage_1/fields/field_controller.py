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
    from shared.constants.constants import * 
    logger.setLevel(LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for FieldController logger."
    )
    logger.setLevel(logging.INFO)

try:
        from shared.sound.sound_generator import SoundGenerator
        SOUND_GENERATOR_AVAILABLE = True
except ImportError:
    SOUND_GENERATOR_AVAILABLE = False
    logger.warning("SoundGenerator not available. Sound generation will be disabled.")
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
    def __init__(self, grid_size: Tuple[int, int, int] = GRID_SIZE): # GRID_SIZE directly available
        """Initialize the FieldController."""
        if not (isinstance(grid_size, tuple) and len(grid_size) == 3 and
                all(isinstance(d, int) and d > 0 for d in grid_size)):
            raise ValueError("grid_size must be a tuple of 3 positive integers.")
        self.grid_size: Tuple[int, int, int] = grid_size
        self.dimensions: int = len(grid_size)
        logger.info(f"Initializing FieldController with grid size {self.grid_size}...")

        if aspect_dictionary is None:
            raise RuntimeError("Aspect Dictionary unavailable. FieldController "
                                "cannot initialize Sephiroth fields.")

        self._sound_saver: Optional['SoundGenerator'] = None # Forward reference if SoundGenerator defined later
        try:
            from shared.sound.sound_generator import SoundGenerator as LocalSoundGenerator # Local import is fine
            fc_sound_output_dir = os.path.join(DATA_DIR_BASE, "sounds", "field_controller_events")
            os.makedirs(fc_sound_output_dir, exist_ok=True)
            self._sound_saver = LocalSoundGenerator(output_dir=fc_sound_output_dir)
            logger.info(f"SoundGenerator initialized for FieldController, output to: {fc_sound_output_dir}")
        except ImportError:
            logger.warning("SoundGenerator not available for FieldController. Sound events will not be saved to file.")
        except Exception as sound_err:
            logger.error(f"Error initializing SoundGenerator for FieldController: {sound_err}", exc_info=True)

        try:
            self.void_field: VoidField = VoidField(grid_size=self.grid_size)
            logger.info("VoidField created successfully.")
        except Exception as e:
            logger.critical("CRITICAL: VoidField initialization failed.", exc_info=True)
            raise RuntimeError("VoidField initialization failed.") from e

        self._calculate_sephiroth_positions()
        self.sephiroth_influencers: Dict[str, SephirothField] = {}
        self.kether_field: Optional[KetherField] = None
        self._create_sephiroth_influencers()
        self.apply_initial_influences()

        self.update_counter: int = 0
        self.optimal_development_points: List[Tuple[int, int, int]] = []
        self.edge_values: Dict[Tuple[int, int, int], float] = {}
        self._initialize_edge_of_chaos_tracking()

        logger.info("FieldController initialized successfully and initial "
                    "influences applied.")

    def _calculate_sephiroth_positions(self) -> None:
        """Calculates grid positions and radii for Sephiroth."""
        logger.debug("Calculating Sephiroth positions and radii...")
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2
        center_z = self.grid_size[2] // 2
        scale_x = self.grid_size[0] / 9.0
        scale_y = self.grid_size[1] / 16.0
        default_radius = SEPHIROTH_DEFAULT_RADIUS * (min(self.grid_size) / 64.0)

        tol_base_positions = {
            'kether': (4.5, 1), 'chokmah': (9, 3), 'binah': (0, 3),
            'daath': (4.5, 4.75), 'chesed': (9, 7.5), 'geburah': (0, 7.5),
            'tiphareth': (4.5, 9), 'netzach': (9, 12), 'hod': (0, 12),
            'yesod': (4.5, 14), 'malkuth': (4.5, 16)
        }
        self.sephiroth_locations: Dict[str, Tuple[int, int, int]] = {}
        self.sephiroth_radii: Dict[str, float] = {}

        for name, (tol_x, tol_y) in tol_base_positions.items():
            grid_x = int(tol_x * scale_x)
            grid_y = int(tol_y * scale_y)
            grid_z = center_z
            if name in ['chokmah', 'chesed', 'netzach']: grid_z += int(self.grid_size[2] * 0.1)
            elif name in ['binah', 'geburah', 'hod']: grid_z -= int(self.grid_size[2] * 0.1)

            grid_x = max(0, min(self.grid_size[0] - 1, grid_x))
            grid_y = max(0, min(self.grid_size[1] - 1, grid_y))
            grid_z = max(0, min(self.grid_size[2] - 1, grid_z))
            self.sephiroth_locations[name] = (grid_x, grid_y, grid_z)

            rad_multiplier = 1.0
            if name == 'kether': rad_multiplier = 1.2
            elif name == 'tiphareth': rad_multiplier = 1.1
            elif name == 'daath': rad_multiplier = 0.8
            else: rad_multiplier = 0.9
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
                logger.critical(f"CRITICAL FAILURE: Cannot create influencer '{name}': {e}", exc_info=True)
                raise RuntimeError(f"Field controller initialization failed - cannot create {name} influencer") from e

        if created_count != len(seph_names_to_create):
            logger.warning(f"Influencer count mismatch: "
                            f"Created {created_count}/{len(seph_names_to_create)}")
        if not self.kether_field:
            raise RuntimeError("KetherField instance could not be created. Cannot proceed.")

    def _initialize_edge_of_chaos_tracking(self) -> None:
        """Initializes tracking of high EoC points."""
        logger.debug("Initializing Edge of Chaos tracking...")
        try:
            self.optimal_development_points = self.void_field.find_optimal_development_points(count=10)
            self.edge_values = {
                p: self.void_field.calculate_edge_of_chaos(p)
                for p in self.optimal_development_points
                if self.void_field._validate_coordinates(p)
            }
            if self.edge_values:
                max_eoc = max(self.edge_values.values()) if self.edge_values else 0.0
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
                influencer.apply_sephiroth_influence(self.void_field)
                logger.debug(f"  Applied initial influence for '{name}'.")
            except Exception as e:
                logger.error(f"Error applying initial influence for '{name}': {e}",
                            exc_info=True)
        logger.info("Initial Sephiroth influences applied.")

    def update_fields(self, delta_time: float) -> None:
        """Updates the VoidField dynamics and reapplies Sephiroth influences."""
        if delta_time <= 0: return
        logger.debug(f"Updating fields (dt={delta_time:.3f}s)...")
        try:
            self.void_field.update_step(delta_time)

            for name, influencer in self.sephiroth_influencers.items():
                try:
                    influencer.apply_sephiroth_influence(self.void_field)
                except Exception as e:
                    logger.critical(f"CRITICAL FAILURE: Cannot reapply influence for '{name}' during update: {e}", exc_info=True)
                    raise RuntimeError(f"Field update failed - cannot apply {name} influence") from e

            if self.update_counter % 5 == 0:
                try:
                    interaction_pairs = []
                    sephiroth_list = list(self.sephiroth_influencers.items())
                    for i in range(len(sephiroth_list)):
                        name1, field1 = sephiroth_list[i]
                        for j in range(i+1, len(sephiroth_list)):
                            name2, field2 = sephiroth_list[j]
                            dist_sq = sum((field1.location[k] - field2.location[k])**2 for k in range(3))
                            combined_radius = field1.radius + field2.radius
                            if dist_sq <= combined_radius**2 * 1.5:
                                interaction_pairs.append((name1, field1, name2, field2))

                    for name1, field1, name2, field2 in interaction_pairs:
                        dist = np.sqrt(sum((field1.location[k] - field2.location[k])**2 for k in range(3)))
                        max_dist = field1.radius + field2.radius
                        dist_factor = np.clip(1.0 - dist/max(FLOAT_EPSILON, max_dist), 0.0, 1.0)

                        freq1 = field1.target_frequency
                        freq2 = field2.target_frequency
                        resonance_factor = 0.1
                        if freq1 > FLOAT_EPSILON and freq2 > FLOAT_EPSILON:
                            ratio_val = max(freq1, freq2) / min(freq1, freq2) # Renamed to ratio_val
                            ratio_resonance = 0.0
                            for n_ratio in range(1, 6):
                                for d_ratio in range(1, 6):
                                    if abs(ratio_val - (n_ratio/d_ratio)) < 0.1:
                                        ratio_resonance = max(ratio_resonance, 1.0 - abs(ratio_val - (n_ratio/d_ratio))*10)
                            phi_resonance = max(0.0, 1.0 - min(abs(ratio_val - PHI), abs(ratio_val - (1/PHI))) * 10)
                            resonance_factor = max(ratio_resonance, phi_resonance)

                        interaction_strength = dist_factor * resonance_factor
                        if interaction_strength > 0.2:
                            midpoint = [int((field1.location[k] * field1.radius + field2.location[k] * field2.radius) /
                                            max(FLOAT_EPSILON, (field1.radius + field2.radius))) for k in range(3)]
                            interaction_radius = int((field1.radius + field2.radius) * 0.3)
                            min_x, max_x = max(0, midpoint[0]-interaction_radius), min(self.grid_size[0], midpoint[0]+interaction_radius+1)
                            min_y, max_y = max(0, midpoint[1]-interaction_radius), min(self.grid_size[1], midpoint[1]+interaction_radius+1)
                            min_z, max_z = max(0, midpoint[2]-interaction_radius), min(self.grid_size[2], midpoint[2]+interaction_radius+1)

                            if min_x < max_x and min_y < max_y and min_z < max_z:
                                x_idx, y_idx, z_idx = np.meshgrid(np.arange(min_x,max_x),np.arange(min_y,max_y),np.arange(min_z,max_z),indexing='ij')
                                dist_sq_interaction = (x_idx-midpoint[0])**2+(y_idx-midpoint[1])**2+(z_idx-midpoint[2])**2
                                mask_interaction = np.exp(-dist_sq_interaction / (2 * (max(FLOAT_EPSILON, interaction_radius)/2)**2)) # Renamed mask to mask_interaction
                                interaction_slice = (slice(min_x,max_x),slice(min_y,max_y),slice(min_z,max_z))

                                energy_boost = interaction_strength * 1000.0 * mask_interaction
                                self.void_field.energy[interaction_slice] += energy_boost
                                coherence_boost = interaction_strength * resonance_factor * 10.0 * mask_interaction
                                self.void_field.coherence[interaction_slice] += coherence_boost
                                avg_freq_interaction = (freq1 + freq2) / 2.0
                                freq_shift = avg_freq_interaction * 0.1 * interaction_strength * mask_interaction
                                self.void_field.frequency[interaction_slice] += freq_shift

                                if interaction_strength > 0.5 and self._sound_saver:
                                    logger.info(f"Strong light interaction: {name1}-{name2}, str={interaction_strength:.2f}, res={resonance_factor:.2f}")
                                    harmonics_list = [1.0, ratio_val if 'ratio_val' in locals() else 1.5] # Use calculated ratio if available
                                    sound_waveform = self._sound_saver.generate_harmonic_tone(
                                        base_frequency=avg_freq_interaction, duration=1.0,
                                        harmonics=harmonics_list, amplitudes=[0.7, 0.35 * resonance_factor],
                                        fade_in_out=0.1
                                    )
                                    filename = f"light_interaction_{name1}_{name2}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                                    saved_path = self._sound_saver.save_sound(sound_waveform, filename)
                                    if saved_path:
                                        logger.info(f"Saved light interaction sound: {saved_path}")
                except Exception as e:
                    logger.error(f"Error processing light-energy interactions: {e}", exc_info=True)
                    raise

            self.update_counter += 1
            if self.update_counter % 10 == 0:
                self._update_edge_of_chaos_tracking()
        except Exception as e:
            logger.error(f"Error during field update: {e}", exc_info=True)
            raise

    def _update_edge_of_chaos_tracking(self) -> None:
        """Recalculates EoC values for tracked points and potentially finds new ones."""
        logger.debug("Refreshing Edge of Chaos tracking...")
        try:
            updated_edge_values = {}
            current_points = getattr(self, 'optimal_development_points', [])
            for point in current_points:
                if self.void_field._validate_coordinates(point):
                    updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

            if self.update_counter % 50 == 0:
                new_points = self.void_field.find_optimal_development_points(count=5)
                for point in new_points:
                    if point not in updated_edge_values and self.void_field._validate_coordinates(point):
                        updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

            if updated_edge_values:
                sorted_points = sorted(updated_edge_values.keys(),
                                    key=lambda p: updated_edge_values.get(p, 0.0),
                                    reverse=True)
                self.optimal_development_points = sorted_points[:10]
                self.edge_values = {p: updated_edge_values[p] for p in self.optimal_development_points}
            else:
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
            base_properties = self.void_field.get_properties_at(coordinates)
            dominant_sephirah = self.get_dominant_sephiroth_at(coordinates)
            base_properties['dominant_sephirah'] = dominant_sephirah

            if dominant_sephirah:
                sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
                if sephirah_field:
                    if getattr(sephirah_field, 'geometric_pattern', None):
                        base_properties['geometric_pattern'] = sephirah_field.geometric_pattern
                    if getattr(sephirah_field, 'platonic_affinity', None):
                        base_properties['platonic_affinity'] = sephirah_field.platonic_affinity

            base_properties['edge_of_chaos'] = self.void_field.calculate_edge_of_chaos(coordinates)
            base_properties['in_guff'] = False
            if dominant_sephirah == 'kether' and self.kether_field:
                if self.kether_field.is_in_guff(coordinates):
                    base_properties['in_guff'] = True
                    guff_targets = self.kether_field.get_guff_properties()
                    base_properties['guff_target_energy_seu'] = guff_targets.get('target_energy_seu')
                    base_properties['guff_target_stability_su'] = guff_targets.get('target_stability_su')
                    base_properties['guff_target_coherence_cu'] = guff_targets.get('target_coherence_cu')
            return base_properties
        except IndexError as e:
            logger.error(f"Coordinates out of bounds in get_properties_at({coordinates}).")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_properties_at({coordinates}): {e}",
                        exc_info=True)
            raise

    def get_dominant_sephiroth_at(self, coordinates: Tuple[int, int, int]) -> Optional[str]:
        """Determines which Sephirah influence is dominant at given coordinates."""
        min_dist_sq = float('inf')
        dominant_sephirah = None
        if not self.void_field._validate_coordinates(coordinates):
            logger.warning(f"Invalid coordinates for dominant Sephirah check: {coordinates}")
            return None

        for name, influencer in self.sephiroth_influencers.items():
            loc = influencer.location
            dist_sq = sum((coordinates[i] - loc[i])**2 for i in range(self.dimensions))
            if dist_sq <= influencer.radius_sq:
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    dominant_sephirah = name
        return dominant_sephirah

    def get_sound_parameters_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Get sound parameters for a location, preferring FieldHarmonics if available.
        Fallback uses direct mapping from SEU/SU/CU properties. Fails hard.
        """
        try:
            field_properties = self.get_properties_at(coordinates)
            field_type = field_properties.get('dominant_sephirah', 'void')

            if field_properties.get('in_guff', False) and self.kether_field:
                try:
                    return self.kether_field.get_sound_parameters_for_guff(coordinates)
                except Exception as guff_sound_err:
                    logger.error(f"Error getting Guff sound params: {guff_sound_err}", exc_info=True)

            if FH_AVAILABLE and FieldHarmonics:
                try:
                    return FieldHarmonics.get_live_sound_parameters(field_type, field_properties)
                except Exception as fh_err:
                    logger.error(f"FieldHarmonics failed for {field_type} at {coordinates}: {fh_err}. Using fallback.", exc_info=True)

            logger.debug("Using fallback sound parameter generation.")
            base_freq = field_properties.get('frequency_hz', 432.0)
            if base_freq <= 0: base_freq = 432.0

            energy_norm = np.clip(field_properties.get('energy_seu', 0.0) /
                                max(FLOAT_EPSILON, MAX_SOUL_ENERGY_SEU * 0.5), 0.0, 1.0)
            amplitude = 0.1 + energy_norm * 0.7
            coherence_norm = np.clip(field_properties.get('coherence_cu', 0.0) / MAX_COHERENCE_CU, 0.0, 1.0)
            waveform = 'sine' if coherence_norm > 0.7 else ('triangle' if coherence_norm > 0.4 else 'sawtooth')
            filter_cutoff = 500 + coherence_norm * 5000
            stability_norm = np.clip(field_properties.get('stability_su', 0.0) / MAX_STABILITY_SU, 0.0, 1.0)
            modulation = (1.0 - stability_norm) * 0.1

            return {
                'base_frequency': base_freq, 'amplitude': amplitude, 'waveform': waveform,
                'filter_cutoff': filter_cutoff, 'resonance': coherence_norm * 0.8,
                'modulation': modulation, 'field_type': field_type
            }
        except IndexError:
            logger.error(f"Coordinates out of bounds in get_sound_parameters_at({coordinates}).")
            raise
        except Exception as e:
            logger.error(f"Error in get_sound_parameters_at({coordinates}): {e}", exc_info=True)
            return {'base_frequency': 432.0, 'amplitude': 0.5, 'waveform': 'sine'}

    def apply_geometric_influence_to_soul(self, soul_spark: SoulSpark) -> Dict[str, float]:
        """Applies geometric influences based on local field properties."""
        changes_applied_dict: Dict[str, float] = {}
        try:
            position_coords = self._coords_to_int_tuple(soul_spark.position)
            local_props = self.get_properties_at(position_coords)
            dominant_sephirah = local_props.get('dominant_sephirah')
            if not dominant_sephirah: return {}
            sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
            if not sephirah_field: return {}

            geom_pattern = getattr(sephirah_field, 'geometric_pattern', None)
            plat_affinity = getattr(sephirah_field, 'platonic_affinity', None)
            resonance_factor = local_props.get('edge_of_chaos', 0.5)

            geom_effects = GEOMETRY_EFFECTS.get(geom_pattern, {})
            plat_effects = GEOMETRY_EFFECTS.get(plat_affinity, {})
            combined_effects: Dict[str, float] = {}
            all_keys = set(geom_effects.keys()) | set(plat_effects.keys())
            for key in all_keys:
                effect_geom = geom_effects.get(key, 0.0)
                effect_plat = plat_effects.get(key, 0.0)
                count = (1 if effect_geom != 0.0 else 0) + (1 if effect_plat != 0.0 else 0)
                combined_effects[key] = (effect_geom + effect_plat) / max(1.0, float(count))

            if not combined_effects: return {}

            transformation_occurred = False
            for effect_name, modifier in combined_effects.items():
                target_attr = effect_name
                is_boost = '_boost' in effect_name or '_factor' in effect_name
                is_push = '_push' in effect_name
                if is_boost: target_attr = effect_name.split('_boost')[0].split('_factor')[0]
                if is_push: target_attr = effect_name.split('_push')[0]

                if not hasattr(soul_spark, target_attr): continue
                current_value = getattr(soul_spark, target_attr)
                if not isinstance(current_value, (int, float)): continue

                change=0.0; max_clamp_val=1.0
                if target_attr=='stability': max_clamp_val=MAX_STABILITY_SU
                elif target_attr=='coherence': max_clamp_val=MAX_COHERENCE_CU

                if is_boost or not is_push: change = modifier * resonance_factor * 0.05
                elif is_push:
                    push_target=0.5 if 'balance' in effect_name else 0.0
                    diff=push_target-current_value; change=diff*abs(modifier)*resonance_factor*0.05

                new_value = current_value + change
                clamped_new_value = max(0.0, min(max_clamp_val, new_value))
                actual_change = clamped_new_value - current_value

                if abs(actual_change) > FLOAT_EPSILON:
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

    def _coords_to_int_tuple(self, position: List[float]) -> Tuple[int, int, int]:
        """Converts float position list to clamped integer grid coordinate tuple."""
        if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
            logger.error(f"Invalid position format: {position}. Using grid center.")
            return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))
        try:
            coords = []
            for i, p_val in enumerate(position):
                coord_int = int(round(float(p_val)))
                clamped_coord = max(0, min(self.grid_size[i] - 1, coord_int))
                coords.append(clamped_coord)
            final_coords = tuple(coords)
            if not all(0 <= c < self.grid_size[i] for i, c in enumerate(final_coords)):
                logger.error(f"Coord calc out-of-bounds {final_coords} for grid {self.grid_size}. Clamped center.")
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

        clamped_pos = [max(0.0, min(float(self.grid_size[i] - FLOAT_EPSILON), p_val))
                    for i, p_val in enumerate(new_position)]
        setattr(soul_spark, 'position', clamped_pos)
        new_grid_coords = self._coords_to_int_tuple(clamped_pos)

        dominant_sephirah = self.get_dominant_sephiroth_at(new_grid_coords)
        new_field_key = dominant_sephirah if dominant_sephirah else 'void'
        setattr(soul_spark, 'current_field_key', new_field_key)

        if self.kether_field:
            try:
                was_in_guff = (old_field_key == 'kether' and self.kether_field.is_in_guff(old_pos_coords))
                is_in_guff = (new_field_key == 'kether' and self.kether_field.is_in_guff(new_grid_coords))
                if is_in_guff and not was_in_guff: self.kether_field.register_soul_in_guff(soul_spark.spark_id)
                elif was_in_guff and not is_in_guff: self.kether_field.remove_soul_from_guff(soul_spark.spark_id)
            except Exception as guff_err: logger.error(f"Error Guff registration: {guff_err}", exc_info=True)

        if old_field_key != new_field_key:
            self._apply_field_transition_effects(soul_spark, old_field_key, new_field_key, new_grid_coords)
            logger.info(f"Soul {soul_spark.spark_id} moved to {new_grid_coords}, transition: {old_field_key} -> {new_field_key}")
        else:
            logger.debug(f"Soul {soul_spark.spark_id} moved in {new_field_key} to {new_grid_coords}")

        self.apply_geometric_influence_to_soul(soul_spark)
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state()

        try:
            if self._sound_saver and hasattr(self, 'update_counter') and self.update_counter % 3 == 0:
                soul_freq = getattr(soul_spark, 'frequency', 0.0)
                field_props = self.get_properties_at(new_grid_coords)
                field_freq = field_props.get('frequency_hz', 0.0)

                if soul_freq > FLOAT_EPSILON and field_freq > FLOAT_EPSILON:
                    ratio = max(soul_freq, field_freq) / min(soul_freq, field_freq)
                    is_resonant = any(abs(ratio - (i/j)) < 0.1 for i in range(1,5) for j in range(1,5))
                    phi_resonant_flag = (abs(ratio - PHI) < 0.1 or abs(ratio - (1/PHI)) < 0.1) # Renamed to avoid conflict

                    if is_resonant or phi_resonant_flag:
                        resonance_sound = self._sound_saver.generate_harmonic_tone(
                            base_frequency=min(soul_freq, field_freq), duration=1.0,
                            harmonics=[1.0, ratio], amplitudes=[0.6, 0.3 * (0.5 + 0.5*int(phi_resonant_flag))],
                            fade_in_out=0.1
                        )
                        filename = f"soul_resonance_{soul_spark.spark_id[:8]}_{new_field_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                        saved_path = self._sound_saver.save_sound(resonance_sound, filename)
                        if saved_path and hasattr(soul_spark, 'interaction_history'):
                            soul_spark.interaction_history.append({
                                'type': 'resonance_sound', 'timestamp': datetime.now().isoformat(),
                                'field': new_field_key, 'ratio': ratio, 'sound_file': saved_path
                            })
                            logger.info(f"Generated resonance sound: {saved_path}")
        except Exception as sound_err:
            logger.error(f"Error generating soul-field resonance sound: {sound_err}", exc_info=True)

    def _apply_field_transition_effects(self, soul_spark: SoulSpark,
                                        old_field: Optional[str], new_field: str,
                                        coords: Tuple[int, int, int]) -> None:
        """Applies effects when a soul transitions between field types."""
        if old_field is None: return
        try:
            field_props = self.get_properties_at(coords)
            # Corrected check for Guff entry: ensure _was_in_guff_prev exists or default to False
            was_in_guff_prev = getattr(soul_spark, '_was_in_guff_prev', False)
            is_entering_guff = (new_field == 'kether' and field_props.get('in_guff', False) and
                                (old_field != 'kether' or not was_in_guff_prev))

            if is_entering_guff:
                logger.debug(f"Soul {soul_spark.spark_id} entering Guff.")
                target_e = field_props.get('guff_target_energy_seu', soul_spark.energy)
                soul_spark.energy += (target_e - soul_spark.energy) * 0.05
                if hasattr(soul_spark, 'frequency_history'):
                    soul_spark.frequency_history = soul_spark.frequency_history[-5:]
            # Update the _was_in_guff_prev attribute AFTER checking
            setattr(soul_spark, '_was_in_guff_prev', field_props.get('in_guff', False))


            if new_field != 'void' and hasattr(soul_spark, 'interaction_history'):
                logger.debug(f"Soul {soul_spark.spark_id} entering influence of {new_field}.")
                soul_spark.interaction_history.append({
                    'type': 'field_transition', 'timestamp': datetime.now().isoformat(),
                    'from': old_field, 'to': new_field, 'position': coords
                })

            edge_value = field_props.get('edge_of_chaos', 0.0)
            if edge_value > 0.75 and hasattr(soul_spark, 'potential_realization'):
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

        phi_rand = np.random.uniform(0, 2 * np.pi)
        costheta_rand = np.random.uniform(-1, 1)
        u_rand = np.random.uniform(0, 1)
        theta_rand = np.arccos(costheta_rand)
        r_rand = guff_radius * np.cbrt(u_rand)

        x = center[0] + r_rand * np.sin(theta_rand) * np.cos(phi_rand)
        y = center[1] + r_rand * np.sin(theta_rand) * np.sin(phi_rand)
        z = center[2] + r_rand * np.cos(theta_rand)
        guff_position_float = [x, y, z]
        self.move_soul(soul_spark, guff_position_float) # This will call _apply_field_transition_effects

        # After move_soul, Guff status should be correctly set.
        # One final check and explicit registration if something went wrong during move_soul's complex logic.
        if not (soul_spark.current_field_key == 'kether' and self.kether_field.is_in_guff(self._coords_to_int_tuple(soul_spark.position))):
            logger.warning(f"Guff placement check: Soul {soul_spark.spark_id} not correctly in Guff. Forcing status.")
            setattr(soul_spark, 'current_field_key', 'kether')
            self.kether_field.register_soul_in_guff(soul_spark.spark_id)
            setattr(soul_spark, '_was_in_guff_prev', True) # Explicitly set this as it's now in Guff


        logger.info(f"Placed soul {soul_spark.spark_id} in Guff at {soul_spark.position}. "
                    f"Field: {soul_spark.current_field_key}, InGuffCheck: {self.kether_field.is_in_guff(self._coords_to_int_tuple(soul_spark.position))}")


    def release_soul_from_guff(self, soul_spark: SoulSpark) -> None:
        """Moves soul just outside Guff radius but within Kether influence."""
        if not self.kether_field: raise RuntimeError("Kether field not initialized.")
        center = self.kether_field.location
        kether_radius = self.kether_field.radius
        guff_radius = self.kether_field.guff_radius
        release_radius = guff_radius + (kether_radius - guff_radius) * 0.1 if kether_radius > guff_radius else guff_radius + 1.0
        release_radius = max(guff_radius + 0.5, release_radius)

        phi_rand = np.random.uniform(0, 2 * np.pi)
        costheta_rand = np.random.uniform(-1, 1)
        theta_rand = np.arccos(costheta_rand)
        x = center[0] + release_radius * np.sin(theta_rand) * np.cos(phi_rand)
        y = center[1] + release_radius * np.sin(theta_rand) * np.sin(phi_rand)
        z = center[2] + release_radius * np.cos(theta_rand)
        release_position_float = [x, y, z]
        self.move_soul(soul_spark, release_position_float) # This will call _apply_field_transition_effects

        # After move_soul, Guff status should be correctly set (i.e., NOT in Guff).
        # One final check and explicit removal if something went wrong.
        if soul_spark.current_field_key != 'kether' or self.kether_field.is_in_guff(self._coords_to_int_tuple(soul_spark.position)):
            logger.warning(f"Guff release check: Soul {soul_spark.spark_id} still in Guff or wrong field. Forcing status.")
            setattr(soul_spark, 'current_field_key', 'kether') # Should be in Kether zone
            self.kether_field.remove_soul_from_guff(soul_spark.spark_id) # Ensure removal
            setattr(soul_spark, '_was_in_guff_prev', False) # Explicitly set this as it's now out of Guff

        logger.info(f"Released soul {soul_spark.spark_id} from Guff to Kether at {soul_spark.position}. "
                    f"Field: {soul_spark.current_field_key}, InGuffCheck: {self.kether_field.is_in_guff(self._coords_to_int_tuple(soul_spark.position))}")


    def find_optimal_development_location(self, soul_spark: Optional[SoulSpark] = None) -> List[float]:
        """
        Finds a location with high Edge of Chaos value, suitable for development.
        """
        try:
            current_points = getattr(self, 'optimal_development_points', [])
            if not current_points or not hasattr(self, 'edge_values') or not self.edge_values:
                self._initialize_edge_of_chaos_tracking()
                current_points = getattr(self, 'optimal_development_points', [])

            if not current_points or not self.edge_values:
                logger.warning("No optimal EoC points found, returning random grid center.")
                return [float(d // 2) + 0.5 for d in self.grid_size]


            valid_points = [p for p in current_points if p in self.edge_values]
            if not valid_points:
                logger.warning("Tracked EoC points are invalid, returning random grid center.")
                return [float(d // 2) + 0.5 for d in self.grid_size]


            sorted_points = sorted(valid_points, key=lambda p: self.edge_values.get(p, 0.0), reverse=True)
            num_top = min(5, len(sorted_points))
            idx = np.random.randint(0, num_top) if num_top > 0 else 0 # Ensure num_top is positive
            if num_top == 0: # Should not happen if valid_points is not empty
                logger.warning("No top points for EoC selection, returning first sorted or random.")
                return [float(c) + 0.5 for c in sorted_points[0]] if sorted_points else [float(d // 2) + 0.5 for d in self.grid_size]

            if soul_spark and hasattr(soul_spark, 'spark_id'):
                try: idx = hash(soul_spark.spark_id) % num_top
                except Exception as hash_err: logger.warning(f"Hash soul ID failed: {hash_err}. Random idx.")

            optimal_point_int = sorted_points[idx]
            return [float(c) + 0.5 for c in optimal_point_int]
        except Exception as e:
            logger.error(f"Error finding optimal location: {e}", exc_info=True)
            return [float(d // 2) + 0.5 for d in self.grid_size]


    def get_visualization_data(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Gets data suitable for visualization at specific coordinates."""
        try:
            field_props = self.get_properties_at(coords)
            sound_params = self.get_sound_parameters_at(coords)

            if FH_AVAILABLE and FieldHarmonics:
                try:
                    visualization = FieldHarmonics.generate_live_sound_visualization(
                        sound_params, field_props.get('dominant_sephirah', 'void'), field_props
                    )
                    core_props = ['energy_seu','frequency_hz','stability_su','coherence_cu','edge_of_chaos','in_guff']
                    for prop in core_props:
                        if prop not in visualization: visualization[prop] = field_props.get(prop)
                    return visualization
                except Exception as fh_viz_err:
                    logger.warning(f"FH viz failed: {fh_viz_err}. Basic fallback.", exc_info=True)

            return {
                'energy_seu': field_props.get('energy_seu'),
                'frequency_hz': field_props.get('frequency_hz'),
                'stability_su': field_props.get('stability_su'),
                'coherence_cu': field_props.get('coherence_cu'),
                'dominant_sephirah': field_props.get('dominant_sephirah'),
                'edge_of_chaos': field_props.get('edge_of_chaos'),
                'color_rgb': field_props.get('color_rgb', [0.5, 0.5, 0.5]),
                'in_guff': field_props.get('in_guff', False)
            }
        except IndexError:
            logger.error(f"Coords {coords} out of bounds for visualization.")
            return {'error': 'Coordinates out of bounds', 'coords': coords}
        except Exception as e:
            logger.error(f"Error gen viz data for {coords}: {e}", exc_info=True)
            return {'error': str(e), 'coords': coords}

    def __str__(self) -> str:
        """String representation of the FieldController."""
        seph_count = len(self.sephiroth_influencers)
        void_status = "Initialized" if self.void_field and self.void_field.energy is not None else "Uninitialized"
        return (f"FieldController(Grid: {self.grid_size}, "
                f"Sephiroth: {seph_count}, Void Status: {void_status})")




    # def __init__(self, grid_size: Tuple[int, int, int] = GRID_SIZE):
    #     """Initialize the FieldController."""
    #     if not (isinstance(grid_size, tuple) and len(grid_size) == 3 and
    #             all(isinstance(d, int) and d > 0 for d in grid_size)):
    #          raise ValueError("grid_size must be a tuple of 3 positive integers.")
    #     self.grid_size: Tuple[int, int, int] = grid_size
    #     self.dimensions: int = len(grid_size) 
    #     logger.info(f"Initializing FieldController with grid size {self.grid_size}...")

    #     # Check aspect dictionary early - critical dependency
    #     if aspect_dictionary is None:
    #          raise RuntimeError("Aspect Dictionary unavailable. FieldController "
    #                             "cannot initialize Sephiroth fields.")
    #     # Initialize sound generation components
    #     self._sound_saver = None
    #     try:
    #         from sound.sound_generator import SoundGenerator
    #         self._sound_saver = SoundGenerator(output_dir=os.path.join(DATA_DIR_BASE, "sounds", "field"))
    #         logger.info("Sound generation initialized for FieldController")
    #     except ImportError:
    #         logger.warning("SoundGenerator not available for FieldController")
    #     except Exception as sound_err:
    #         logger.error(f"Error initializing sound generator: {sound_err}", exc_info=True)
            
    #     # 1. Create Void Field
    #     try:
    #         self.void_field: VoidField = VoidField(grid_size=self.grid_size)
    #         logger.info("VoidField created successfully.")
    #     except Exception as e:
    #         logger.critical("CRITICAL: VoidField initialization failed.", exc_info=True)
    #         raise RuntimeError("VoidField initialization failed.") from e

    #     # 2. Define Sephiroth Locations/Radii
    #     self._calculate_sephiroth_positions()

    #     # 3. Create Sephiroth Influencers
    #     self.sephiroth_influencers: Dict[str, SephirothField] = {}
    #     self.kether_field: Optional[KetherField] = None # Specifically track Kether
    #     self._create_sephiroth_influencers()

    #     # 4. Apply Initial Influences to Void Field
    #     self.apply_initial_influences()

    #     # 5. Initialize Tracking Variables
    #     self.update_counter: int = 0
    #     self.optimal_development_points: List[Tuple[int, int, int]] = []
    #     self.edge_values: Dict[Tuple[int, int, int], float] = {}
    #     # Initialize EoC tracking (call internal helper)
    #     self._initialize_edge_of_chaos_tracking()

    #     logger.info("FieldController initialized successfully and initial "
    #                 "influences applied.")

    # def _calculate_sephiroth_positions(self) -> None:
    #     """Calculates grid positions and radii for Sephiroth."""
    #     logger.debug("Calculating Sephiroth positions and radii...")
    #     center_x = self.grid_size[0] // 2
    #     center_y = self.grid_size[1] // 2
    #     center_z = self.grid_size[2] // 2
    #     # Scaling factors based on Tree of Life proportions relative to grid size
    #     # Example: 9 units wide, 16 units high in schematic
    #     scale_x = self.grid_size[0] / 9.0
    #     scale_y = self.grid_size[1] / 16.0
    #     # Default radius scales with the smallest grid dimension
    #     default_radius = SEPHIROTH_DEFAULT_RADIUS * (min(self.grid_size) / 64.0)

    #     # Positions based on a 9x16 grid (as per user specification)
    #     tol_base_positions = { # (Schematic X, Schematic Y)
    #         'kether': (4.5, 1), 'chokmah': (9, 3), 'binah': (0, 3),
    #         'daath': (4.5, 4.75), 'chesed': (9, 7.5), 'geburah': (0, 7.5),
    #         'tiphareth': (4.5, 9), 'netzach': (9, 12), 'hod': (0, 12),
    #         'yesod': (4.5, 14), 'malkuth': (4.5, 16)
    #     }
    #     self.sephiroth_locations: Dict[str, Tuple[int, int, int]] = {}
    #     self.sephiroth_radii: Dict[str, float] = {}

    #     for name, (tol_x, tol_y) in tol_base_positions.items():
    #         # Scale schematic coords to grid coords
    #         grid_x = int(tol_x * scale_x)
    #         grid_y = int(tol_y * scale_y)
    #         # Base Z is center, adjust for pillars
    #         grid_z = center_z
    #         if name in ['chokmah', 'chesed', 'netzach']: # Right pillar
    #              grid_z += int(self.grid_size[2] * 0.1) # Slightly forward
    #         elif name in ['binah', 'geburah', 'hod']: # Left pillar
    #              grid_z -= int(self.grid_size[2] * 0.1) # Slightly back

    #         # Clamp coordinates to be within grid boundaries
    #         grid_x = max(0, min(self.grid_size[0] - 1, grid_x))
    #         grid_y = max(0, min(self.grid_size[1] - 1, grid_y))
    #         grid_z = max(0, min(self.grid_size[2] - 1, grid_z))
    #         self.sephiroth_locations[name] = (grid_x, grid_y, grid_z)

    #         # Calculate radius (adjust multiplier based on importance/size)
    #         rad_multiplier = 1.0 # Default
    #         if name == 'kether': rad_multiplier = 1.2
    #         elif name == 'tiphareth': rad_multiplier = 1.1
    #         elif name == 'malkuth': rad_multiplier = 1.0
    #         elif name == 'daath': rad_multiplier = 0.8 # Daath is often smaller/hidden
    #         else: rad_multiplier = 0.9 # Other Sephiroth slightly smaller?
    #         self.sephiroth_radii[name] = default_radius * rad_multiplier
    #         logger.debug(f"  {name.capitalize()}: Pos={self.sephiroth_locations[name]}, "
    #                      f"Rad={self.sephiroth_radii[name]:.2f}")

    # def _create_sephiroth_influencers(self) -> None:
    #     """Creates SephirothField/KetherField instances."""
    #     logger.debug("Creating Sephiroth influencer objects...")
    #     if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")

    #     seph_names_to_create = aspect_dictionary.sephiroth_names
    #     created_count = 0
    #     for name in seph_names_to_create:
    #          loc = self.sephiroth_locations.get(name)
    #          rad = self.sephiroth_radii.get(name)
    #          if loc is None or rad is None:
    #              logger.error(f"Skipping Sephirah '{name}': Missing location or radius.")
    #              continue
    #          try:
    #              if name == 'kether':
    #                  self.kether_field = KetherField(location=loc, radius=rad)
    #                  self.sephiroth_influencers[name] = self.kether_field
    #                  logger.debug(f"  Created KetherField at {loc}, R={rad:.2f}")
    #              else:
    #                  self.sephiroth_influencers[name] = SephirothField(
    #                      sephirah_name=name, location=loc, radius=rad
    #                  )
    #                  logger.debug(f"  Created SephirothField '{name}' at {loc}, R={rad:.2f}")
    #              created_count += 1
    #          except Exception as e:
    #              logger.error(f"Failed to create influencer '{name}': {e}", exc_info=True)

    #     if created_count != len(seph_names_to_create):
    #          logger.warning(f"Influencer count mismatch: "
    #                         f"Created {created_count}/{len(seph_names_to_create)}")
    #     if not self.kether_field: # Kether is essential for Guff logic
    #          raise RuntimeError("KetherField instance could not be created. Cannot proceed.")

    # def _initialize_edge_of_chaos_tracking(self) -> None:
    #     """Initializes tracking of high EoC points."""
    #     logger.debug("Initializing Edge of Chaos tracking...")
    #     try:
    #         # Use the VoidField method to find optimal points
    #         self.optimal_development_points = self.void_field.find_optimal_development_points(count=10)
    #         # Calculate EoC values for these points
    #         self.edge_values = {
    #             p: self.void_field.calculate_edge_of_chaos(p)
    #             for p in self.optimal_development_points
    #             if self.void_field._validate_coordinates(p) # Ensure validity
    #         }
    #         if self.edge_values:
    #             max_eoc = max(self.edge_values.values())
    #             logger.info(f"EoC tracking initialized: {len(self.optimal_development_points)} points. "
    #                         f"Max EoC value: {max_eoc:.4f}")
    #         else:
    #              logger.warning("EoC tracking initialized, but no optimal points found/valid.")
    #              self.optimal_development_points = []
    #     except AttributeError as ae:
    #         logger.error(f"Failed EoC initialization: VoidField missing method? Error: {ae}", exc_info=True)
    #         self.optimal_development_points = []
    #         self.edge_values = {}
    #     except Exception as e:
    #         logger.error(f"Failed EoC initialization: {e}", exc_info=True)
    #         self.optimal_development_points = []
    #         self.edge_values = {}

    # def apply_initial_influences(self) -> None:
    #     """Applies the initial static influence of all Sephiroth onto the Void."""
    #     logger.info("Applying initial Sephiroth influences to VoidField...")
    #     for name, influencer in self.sephiroth_influencers.items():
    #         try:
    #             # Call the influencer's method to modify the VoidField
    #             influencer.apply_sephiroth_influence(self.void_field)
    #             logger.debug(f"  Applied initial influence for '{name}'.")
    #         except Exception as e:
    #             logger.error(f"Error applying initial influence for '{name}': {e}",
    #                          exc_info=True)
    #     logger.info("Initial Sephiroth influences applied.")

    # def update_fields(self, delta_time: float) -> None:
    #     """Updates the VoidField dynamics and reapplies Sephiroth influences."""
    #     if delta_time <= 0: return # No change if no time passed
    #     logger.debug(f"Updating fields (dt={delta_time:.3f}s)...")
    #     try:
    #         # 1. Update Void Field Dynamics (diffusion, dissipation, resonance)
    #         self.void_field.update_step(delta_time)

    #         # 2. Re-apply Sephiroth Influences (they modify the updated void state)
    #         # This could be optimized if influences are static, but allows for dynamic influencers later
    #         for name, influencer in self.sephiroth_influencers.items():
    #             try:
    #                 influencer.apply_sephiroth_influence(self.void_field)
    #             except Exception as e:
    #                 logger.error(f"Error reapplying influence for '{name}' during update: {e}",
    #                             exc_info=True) # Log but continue update
    #         # 2.5 Apply light-energy interactions between Sephiroth fields
    #         if self.update_counter % 5 == 0:  # Only do occasionally for performance
    #             try:
    #                 # Find pairs of Sephiroth that are close enough for light interaction
    #                 interaction_pairs = []
    #                 sephiroth_list = list(self.sephiroth_influencers.items())
                    
    #                 for i in range(len(sephiroth_list)):
    #                     name1, field1 = sephiroth_list[i]
    #                     for j in range(i+1, len(sephiroth_list)):
    #                         name2, field2 = sephiroth_list[j]
                            
    #                         # Calculate squared distance between Sephiroth centers
    #                         dist_sq = sum((field1.location[k] - field2.location[k])**2 for k in range(3))
    #                         combined_radius = field1.radius + field2.radius
                            
    #                         # Check if fields are close enough to interact
    #                         if dist_sq <= combined_radius**2 * 1.5:  # Allow some extra distance
    #                             interaction_pairs.append((name1, field1, name2, field2))
                    
    #                 # Process each interaction pair
    #                 for name1, field1, name2, field2 in interaction_pairs:
    #                     # Calculate interaction strength based on distance and field properties
    #                     dist = np.sqrt(sum((field1.location[k] - field2.location[k])**2 for k in range(3)))
    #                     max_dist = field1.radius + field2.radius
    #                     dist_factor = np.clip(1.0 - dist/max_dist, 0.0, 1.0)
                        
    #                     # Frequency resonance affects interaction strength
    #                     freq1 = field1.target_frequency
    #                     freq2 = field2.target_frequency
    #                     if freq1 > FLOAT_EPSILON and freq2 > FLOAT_EPSILON:
    #                         # Calculate frequency ratio and check for resonance
    #                         ratio = max(freq1, freq2) / min(freq1, freq2)
    #                         # Check for integer or phi ratios
    #                         ratio_resonance = 0.0
    #                         for n in range(1, 6):
    #                             for d in range(1, 6):
    #                                 if abs(ratio - (n/d)) < 0.1:
    #                                     ratio_resonance = max(ratio_resonance, 1.0 - abs(ratio - (n/d))*10)
                            
    #                         # Also check phi resonance
    #                         phi_resonance = max(
    #                             0.0,
    #                             1.0 - min(abs(ratio - PHI), abs(ratio - (1/PHI))) * 10
    #                         )
                            
    #                         # Combined resonance factor
    #                         resonance_factor = max(ratio_resonance, phi_resonance)
    #                     else:
    #                         # Default if frequencies invalid
    #                         resonance_factor = 0.1
                        
    #                     # Combined interaction strength
    #                     interaction_strength = dist_factor * resonance_factor
                        
    #                     if interaction_strength > 0.2:  # Only process significant interactions
    #                         # Calculate midpoint between fields for interaction
    #                         midpoint = [
    #                             int((field1.location[k] * field1.radius + field2.location[k] * field2.radius) / 
    #                                 (field1.radius + field2.radius))
    #                             for k in range(3)
    #                         ]
                            
    #                         # Create bounding box around midpoint
    #                         interaction_radius = int((field1.radius + field2.radius) * 0.3)
    #                         min_x = max(0, midpoint[0] - interaction_radius)
    #                         max_x = min(self.grid_size[0], midpoint[0] + interaction_radius + 1)
    #                         min_y = max(0, midpoint[1] - interaction_radius)
    #                         max_y = min(self.grid_size[1], midpoint[1] + interaction_radius + 1)
    #                         min_z = max(0, midpoint[2] - interaction_radius)
    #                         max_z = min(self.grid_size[2], midpoint[2] + interaction_radius + 1)
                            
    #                         # Check if box has volume
    #                         if min_x < max_x and min_y < max_y and min_z < max_z:
    #                             # Create distance grid
    #                             x_idx, y_idx, z_idx = np.meshgrid(
    #                                 np.arange(min_x, max_x),
    #                                 np.arange(min_y, max_y),
    #                                 np.arange(min_z, max_z),
    #                                 indexing='ij'
    #                             )
                                
    #                             # Calculate squared distance from midpoint
    #                             dist_sq = (x_idx - midpoint[0])**2 + (y_idx - midpoint[1])**2 + (z_idx - midpoint[2])**2
                                
    #                             # Create influence mask based on distance
    #                             mask = np.exp(-dist_sq / (2 * (interaction_radius/2)**2))
                                
    #                             # Apply interaction effects to VoidField properties
    #                             interaction_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
                                
    #                             # 1. Boost energy in interaction zone
    #                             energy_boost = interaction_strength * 1000.0 * mask  # Scale by SEU constant if needed
    #                             self.void_field.energy[interaction_slice] += energy_boost
                                
    #                             # 2. Boost coherence in interaction zone based on resonance
    #                             coherence_boost = interaction_strength * resonance_factor * 10.0 * mask  # Scale by CU constant
    #                             self.void_field.coherence[interaction_slice] += coherence_boost
                                
    #                             # 3. Generate light-like frequency patterns
    #                             avg_freq = (freq1 + freq2) / 2.0
    #                             freq_shift = avg_freq * 0.1 * interaction_strength * mask
    #                             self.void_field.frequency[interaction_slice] += freq_shift
                                
    #                             # Log significant interactions
    #                             if interaction_strength > 0.5:
    #                                 logger.info(f"Strong light interaction between {name1} and {name2}, "
    #                                         f"strength={interaction_strength:.2f}, resonance={resonance_factor:.2f}")
                                    
    #                                 # Generate sound for significant light interactions
    #                                 try:
    #                                     # Import sound_generator if needed
    #                                     try:
    #                                         from sound.sound_generator import SoundGenerator
    #                                         sound_gen_available = True
    #                                     except ImportError:
    #                                         sound_gen_available = False
                                        
    #                                     if sound_gen_available:
    #                                         # Create sound generator
    #                                         sound_gen = SoundGenerator(output_dir="output/sounds/field_interactions")
                                            
    #                                         # Generate harmonic tone
    #                                         avg_freq = (freq1 + freq2) / 2.0
    #                                         base_frequency = avg_freq
    #                                         duration = 3.0
    #                                         harmonics = [1.0, ratio]
    #                                         amplitudes = [0.8, 0.4]
                                            
    #                                         # Create sound using the correct parameter names
    #                                         sound = sound_gen.generate_harmonic_tone(
    #                                             base_frequency=base_frequency,
    #                                             duration=duration,
    #                                             harmonics=harmonics,
    #                                             amplitudes=amplitudes,
    #                                             fade_in_out=0.5
    #                                         )
                                            
    #                                         # Save the sound
    #                                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                                         filename = f"light_interaction_{name1}_{name2}_{timestamp}.wav"
    #                                         sound_file = sound_gen.save_sound(sound, filename)
                                            
    #                                         if sound_file:
    #                                             logger.info(f"Generated sound file for light interaction: {sound_file}")
    #                                 except Exception as sound_err:
    #                                     logger.error(f"Error generating light interaction sound: {sound_err}", exc_info=True)
    #                                     # Non-critical, continue execution
    #             except Exception as e:
    #                 logger.error(f"Error processing light-energy interactions: {e}", exc_info=True)
    #                 # Hard fail as requested
    #                 raise

    #         # 3. Update Tracking
    #         self.update_counter += 1
    #         # Periodically update EoC tracking
    #         if self.update_counter % 10 == 0:
    #             self._update_edge_of_chaos_tracking()

    #     except Exception as e:
    #         logger.error(f"Error during field update: {e}", exc_info=True)
    #         # Hard fail as requested
    #         raise

    # def _update_edge_of_chaos_tracking(self) -> None:
    #     """Recalculates EoC values for tracked points and potentially finds new ones."""
    #     logger.debug("Refreshing Edge of Chaos tracking...")
    #     try:
    #         updated_edge_values = {}
    #         current_points = getattr(self, 'optimal_development_points', [])
    #         # Recalculate EoC for existing tracked points
    #         for point in current_points:
    #             if self.void_field._validate_coordinates(point):
    #                 updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

    #         # Periodically search for new high-EoC points
    #         if self.update_counter % 50 == 0:
    #             new_points = self.void_field.find_optimal_development_points(count=5) # Find 5 new candidates
    #             for point in new_points:
    #                 if point not in updated_edge_values and self.void_field._validate_coordinates(point):
    #                     updated_edge_values[point] = self.void_field.calculate_edge_of_chaos(point)

    #         if updated_edge_values:
    #             # Keep the top N points based on current EoC values
    #             sorted_points = sorted(updated_edge_values.keys(),
    #                                    key=lambda p: updated_edge_values.get(p, 0.0),
    #                                    reverse=True)
    #             self.optimal_development_points = sorted_points[:10] # Keep top 10
    #             self.edge_values = {p: updated_edge_values[p] for p in self.optimal_development_points}
    #             # Optional: Log only if values change significantly?
    #             # logger.debug(f"Refreshed EoC tracking. Max EoC: {max(self.edge_values.values()):.4f}")
    #         else:
    #              # If no valid points found, reset tracking
    #              self.optimal_development_points = []
    #              self.edge_values = {}
    #     except Exception as e:
    #         logger.error(f"Error updating EoC tracking: {e}", exc_info=True)

    # def get_field(self, field_key: str) -> Optional[Union[VoidField, SephirothField]]:
    #     """Retrieves a specific field object by key ('void' or sephirah name)."""
    #     field_key_lower = field_key.lower()
    #     if field_key_lower == 'void':
    #         return self.void_field
    #     elif field_key_lower in self.sephiroth_influencers:
    #         return self.sephiroth_influencers[field_key_lower]
    #     else:
    #          logger.warning(f"Field key '{field_key}' not recognized.")
    #          return None

    # def get_properties_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
    #     """
    #     Gets combined field properties (SEU/SU/CU etc.) at specific coordinates,
    #     including dominant Sephirah, Guff status, and EoC value. Fails hard.
    #     """
    #     try:
    #         # Get base properties from VoidField (which holds the actual grid state)
    #         base_properties = self.void_field.get_properties_at(coordinates)

    #         # Determine dominant influencer at this location
    #         dominant_sephirah = self.get_dominant_sephiroth_at(coordinates)
    #         base_properties['dominant_sephirah'] = dominant_sephirah # Store name or None

    #         # Add specific properties if within a Sephirah influence zone
    #         if dominant_sephirah:
    #             sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
    #             if sephirah_field:
    #                  # Add geometric identifiers if available on the influencer
    #                  if getattr(sephirah_field, 'geometric_pattern', None):
    #                       base_properties['geometric_pattern'] = sephirah_field.geometric_pattern
    #                  if getattr(sephirah_field, 'platonic_affinity', None):
    #                       base_properties['platonic_affinity'] = sephirah_field.platonic_affinity

    #         # Calculate and add Edge of Chaos value
    #         base_properties['edge_of_chaos'] = self.void_field.calculate_edge_of_chaos(coordinates)

    #         # Check if within Guff (only possible if dominant is Kether)
    #         base_properties['in_guff'] = False
    #         if dominant_sephirah == 'kether' and self.kether_field:
    #              if self.kether_field.is_in_guff(coordinates):
    #                  base_properties['in_guff'] = True
    #                  # Add Guff-specific target properties
    #                  guff_targets = self.kether_field.get_guff_properties()
    #                  base_properties['guff_target_energy_seu'] = guff_targets.get('target_energy_seu')
    #                  base_properties['guff_target_stability_su'] = guff_targets.get('target_stability_su')
    #                  base_properties['guff_target_coherence_cu'] = guff_targets.get('target_coherence_cu')

    #         return base_properties
    #     except IndexError as e: # Raised by get_properties_at if coords invalid
    #         logger.error(f"Coordinates out of bounds in get_properties_at({coordinates}).")
    #         raise # Re-raise index error
    #     except Exception as e:
    #         logger.error(f"Unexpected error in get_properties_at({coordinates}): {e}",
    #                      exc_info=True)
    #         raise # Re-raise other critical errors

    # def get_dominant_sephiroth_at(self, coordinates: Tuple[int, int, int]) -> Optional[str]:
    #     """Determines which Sephirah influence is dominant at given coordinates."""
    #     min_dist_sq = float('inf')
    #     dominant_sephirah = None
    #     # Validate coordinates first
    #     if not self.void_field._validate_coordinates(coordinates):
    #          logger.warning(f"Invalid coordinates for dominant Sephirah check: {coordinates}")
    #          return None

    #     for name, influencer in self.sephiroth_influencers.items():
    #         loc = influencer.location
    #         # Calculate squared distance for efficiency
    #         dist_sq = sum((coordinates[i] - loc[i])**2 for i in range(self.dimensions)) # Use self.dimensions (set to 3 in __init__)
    #         # Check if within the influencer's radius (using squared distance)
    #         if dist_sq <= influencer.radius_sq:
    #             # If multiple overlaps, the closest one is dominant
    #             if dist_sq < min_dist_sq:
    #                 min_dist_sq = dist_sq
    #                 dominant_sephirah = name
    #     return dominant_sephirah

    # # --- Sound Parameter Generation ---
    # def get_sound_parameters_at(self, coordinates: Tuple[int, int, int]) -> Dict[str, Any]:
    #     """
    #     Get sound parameters for a location, preferring FieldHarmonics if available.
    #     Fallback uses direct mapping from SEU/SU/CU properties. Fails hard.
    #     """
    #     try:
    #         field_properties = self.get_properties_at(coordinates) # Fails hard if coords invalid
    #         field_type = field_properties.get('dominant_sephirah', 'void')

    #         # Special handling for Guff sound
    #         if field_properties.get('in_guff', False) and self.kether_field:
    #             try:
    #                 return self.kether_field.get_sound_parameters_for_guff(coordinates)
    #             except Exception as guff_sound_err:
    #                 logger.error(f"Error getting Guff sound params: {guff_sound_err}", exc_info=True)
    #                 # Fall through to generic fallback if Guff sound fails

    #         # Use FieldHarmonics if available
    #         if FH_AVAILABLE and FieldHarmonics:
    #             try:
    #                 # Assumes FieldHarmonics handles SEU/SU/CU inputs correctly
    #                 return FieldHarmonics.get_live_sound_parameters(field_type, field_properties)
    #             except Exception as fh_err:
    #                 logger.error(f"FieldHarmonics failed for {field_type} at {coordinates}: {fh_err}. Using fallback.", exc_info=True)
    #                 # Fall through to fallback if FieldHarmonics fails

    #         # --- Fallback Sound Parameter Generation (Uses SEU/SU/CU) ---
    #         logger.debug("Using fallback sound parameter generation.")
    #         base_freq = field_properties.get('frequency_hz', 432.0)
    #         if base_freq <= 0: base_freq = 432.0 # Ensure positive frequency

    #         # Map energy SEU (0 to MAX*0.5) to amplitude (0.1 to 0.8)
    #         energy_norm = np.clip(field_properties.get('energy_seu', 0.0) /
    #                               max(FLOAT_EPSILON, MAX_SOUL_ENERGY_SEU * 0.5),
    #                               0.0, 1.0)
    #         amplitude = 0.1 + energy_norm * 0.7

    #         # Map coherence CU (0-100) to waveform complexity/filter cutoff
    #         coherence_norm = np.clip(field_properties.get('coherence_cu', 0.0) /
    #                                  MAX_COHERENCE_CU, 0.0, 1.0)
    #         if coherence_norm > 0.7: waveform = 'sine'
    #         elif coherence_norm > 0.4: waveform = 'triangle'
    #         else: waveform = 'sawtooth'
    #         filter_cutoff = 500 + coherence_norm * 5000

    #         # Map stability SU (0-100) to modulation/detune (less stable = more mod)
    #         stability_norm = np.clip(field_properties.get('stability_su', 0.0) /
    #                                  MAX_STABILITY_SU, 0.0, 1.0)
    #         modulation = (1.0 - stability_norm) * 0.1 # Max 10% modulation

    #         return {
    #             'base_frequency': base_freq, 'amplitude': amplitude, 'waveform': waveform,
    #             'filter_cutoff': filter_cutoff, 'resonance': coherence_norm * 0.8,
    #             'modulation': modulation, 'field_type': field_type # Add field type for context
    #         }
    #     except IndexError: # Catch index errors specifically from get_properties_at
    #          logger.error(f"Coordinates out of bounds in get_sound_parameters_at({coordinates}).")
    #          raise # Re-raise
    #     except Exception as e:
    #         logger.error(f"Error in get_sound_parameters_at({coordinates}): {e}", exc_info=True)
    #         # Return minimal safe fallback on unexpected errors
    #         return {'base_frequency': 432.0, 'amplitude': 0.5, 'waveform': 'sine'}

    # # --- Geometric Influence on Soul ---
    # def apply_geometric_influence_to_soul(self, soul_spark: SoulSpark
    #                                       ) -> Dict[str, float]:
    #     """Applies geometric influences based on local field properties."""
    #     # Implementation unchanged from previous PEP8 version - it uses factors
    #     # and applies deltas based on GEOMETRY_EFFECTS constants.
    #     changes_applied_dict = {}
    #     try:
    #         position_coords = self._coords_to_int_tuple(soul_spark.position)
    #         local_props = self.get_properties_at(position_coords)
    #         dominant_sephirah = local_props.get('dominant_sephirah')
    #         if not dominant_sephirah: return {}
    #         sephirah_field = self.sephiroth_influencers.get(dominant_sephirah)
    #         if not sephirah_field: return {}

    #         geom_pattern = getattr(sephirah_field, 'geometric_pattern', None)
    #         plat_affinity = getattr(sephirah_field, 'platonic_affinity', None)
    #         # Use Edge of Chaos as resonance factor for geometric influence?
    #         resonance_factor = local_props.get('edge_of_chaos', 0.5)

    #         geom_effects = GEOMETRY_EFFECTS.get(geom_pattern, {})
    #         plat_effects = GEOMETRY_EFFECTS.get(plat_affinity, {})
    #         combined_effects = {}
    #         all_keys = set(geom_effects.keys()) | set(plat_effects.keys())
    #         for key in all_keys:
    #              # Average effect if present in both, otherwise take the single value
    #              effect_geom = geom_effects.get(key, 0.0)
    #              effect_plat = plat_effects.get(key, 0.0)
    #              count = (1 if effect_geom != 0.0 else 0) + (1 if effect_plat != 0.0 else 0)
    #              combined_effects[key] = (effect_geom + effect_plat) / max(1.0, count)

    #         if not combined_effects: return {}

    #         transformation_occurred = False
    #         for effect_name, modifier in combined_effects.items():
    #              # Target attribute determination logic copied from _apply_geometric_transformation
    #              target_attr = effect_name
    #              is_boost = '_boost' in effect_name or '_factor' in effect_name
    #              is_push = '_push' in effect_name
    #              if is_boost: target_attr = effect_name.split('_boost')[0].split('_factor')[0]
    #              if is_push: target_attr = effect_name.split('_push')[0]

    #              if not hasattr(soul_spark, target_attr): continue
    #              current_value = getattr(soul_spark, target_attr)
    #              if not isinstance(current_value, (int, float)): continue

    #              change=0.0; max_clamp=1.0 # Assume 0-1 default
    #              if target_attr=='stability': max_clamp=MAX_STABILITY_SU
    #              elif target_attr=='coherence': max_clamp=MAX_COHERENCE_CU

    #              if is_boost or not is_push: change = modifier * resonance_factor * 0.05
    #              elif is_push:
    #                   push_target=0.5 if 'balance' in effect_name else 0.0
    #                   diff=push_target-current_value; change=diff*abs(modifier)*resonance_factor*0.05

    #              new_value = current_value + change
    #              clamped_new_value = max(0.0, min(max_clamp, new_value))
    #              actual_change = clamped_new_value - current_value

    #              if abs(actual_change) > FLOAT_EPSILON:
    #                  setattr(soul_spark, target_attr, float(clamped_new_value))
    #                  changes_applied_dict[f"{target_attr}_geom_delta"] = float(actual_change)
    #                  transformation_occurred = True

    #         if transformation_occurred:
    #              setattr(soul_spark, 'last_modified', datetime.now().isoformat())
    #              logger.debug(f"Applied geom effects to soul {soul_spark.spark_id} from {dominant_sephirah}.")
    #         return changes_applied_dict
    #     except Exception as e:
    #         logger.error(f"Error applying geom influence to soul {soul_spark.spark_id}: {e}",
    #                      exc_info=True)
    #         return {}


    # # --- Soul Position & State Management ---
    # def _coords_to_int_tuple(self, position: List[float]) -> Tuple[int, int, int]:
    #     """Converts float position list to clamped integer grid coordinate tuple."""
    #     if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
    #          logger.error(f"Invalid position format: {position}. Using grid center.")
    #          # Return clamped center coordinates
    #          return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))
    #     try:
    #         coords = []
    #         for i, p in enumerate(position):
    #             # Round to nearest int, then clamp to valid index range [0, size-1]
    #             coord_int = int(round(float(p)))
    #             clamped_coord = max(0, min(self.grid_size[i] - 1, coord_int))
    #             coords.append(clamped_coord)
    #         final_coords = tuple(coords)
    #         # Final validation check
    #         if not all(0 <= c < self.grid_size[i] for i, c in enumerate(final_coords)):
    #              logger.error(f"Coordinate calculation resulted in out-of-bounds index {final_coords} "
    #                           f"for grid {self.grid_size}. Returning clamped center.")
    #              return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))
    #         return final_coords
    #     except (TypeError, ValueError) as e:
    #         logger.error(f"Error converting position {position} to int tuple: {e}. Using grid center.")
    #         return tuple(min(self.grid_size[i]-1, max(0, self.grid_size[i]//2)) for i in range(3))

    # def move_soul(self, soul_spark: SoulSpark, new_position: List[float]) -> None:
    #     """Updates soul's position, field key, and applies transition effects."""
    #     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    #     if not isinstance(new_position, list) or len(new_position) != 3: raise ValueError("new_position invalid.")

    #     old_pos_coords = self._coords_to_int_tuple(getattr(soul_spark, 'position', new_position))
    #     old_field_key = getattr(soul_spark, 'current_field_key', None)

    #     # Clamp float position BEFORE setting attribute
    #     clamped_pos = [max(0.0, min(float(self.grid_size[i] - FLOAT_EPSILON), p))
    #                    for i, p in enumerate(new_position)]
    #     setattr(soul_spark, 'position', clamped_pos)
    #     new_grid_coords = self._coords_to_int_tuple(clamped_pos) # Convert clamped float pos

    #     dominant_sephirah = self.get_dominant_sephiroth_at(new_grid_coords)
    #     new_field_key = dominant_sephirah if dominant_sephirah else 'void'
    #     setattr(soul_spark, 'current_field_key', new_field_key)

    #     # --- Guff Registration Logic ---
    #     if self.kether_field: # Check if Kether field exists
    #         try:
    #             was_in_guff = (old_field_key == 'kether' and
    #                            self.kether_field.is_in_guff(old_pos_coords))
    #             is_in_guff = (new_field_key == 'kether' and
    #                           self.kether_field.is_in_guff(new_grid_coords))
    #             if is_in_guff and not was_in_guff:
    #                 self.kether_field.register_soul_in_guff(soul_spark.spark_id)
    #             elif was_in_guff and not is_in_guff:
    #                 self.kether_field.remove_soul_from_guff(soul_spark.spark_id)
    #         except Exception as guff_err:
    #              logger.error(f"Error during Guff registration check: {guff_err}", exc_info=True)
    #     # --- End Guff Logic ---

    #     if old_field_key != new_field_key:
    #         self._apply_field_transition_effects(soul_spark, old_field_key,
    #                                            new_field_key, new_grid_coords)
    #         logger.info(f"Soul {soul_spark.spark_id} moved to {new_grid_coords}, "
    #                     f"field transition: {old_field_key} -> {new_field_key}")
    #     else:
    #          logger.debug(f"Soul {soul_spark.spark_id} moved within {new_field_key} "
    #                      f"to {new_grid_coords}")

    #     # Apply local geometric influence at new position
    #     self.apply_geometric_influence_to_soul(soul_spark)
    #     # Update soul's internal state (like S/C scores based on new position/influence)
    #     if hasattr(soul_spark, 'update_state'):
    #         soul_spark.update_state()
            
    #     # Generate sound for soul-field resonance
    #     try:
    #         # Check if sound generation is available
    #         try:
    #             from sound.sound_generator import SoundGenerator
    #             sound_gen_available = True
    #         except ImportError:
    #             sound_gen_available = False
            
    #         # Only proceed if sound generation is available
    #         if sound_gen_available and hasattr(self, 'update_counter') and self.update_counter % 3 == 0:
    #             # Get frequencies
    #             soul_freq = getattr(soul_spark, 'frequency', 0.0)
    #             field_props = self.get_properties_at(new_grid_coords)
    #             field_freq = field_props.get('frequency_hz', 0.0)
                
    #             # Check for resonance
    #             if soul_freq > FLOAT_EPSILON and field_freq > FLOAT_EPSILON:
    #                 ratio = max(soul_freq, field_freq) / min(soul_freq, field_freq)
                    
    #                 # Check for resonant ratios
    #                 is_resonant = False
    #                 for i in range(1, 5):
    #                     for j in range(1, 5):
    #                         if abs(ratio - (i/j)) < 0.1:
    #                             is_resonant = True
    #                             break
    #                     if is_resonant:
    #                         break
                    
    #                 # Also check for phi resonance
    #                 phi_resonant = (abs(ratio - PHI) < 0.1 or abs(ratio - (1/PHI)) < 0.1)
                    
    #                 # Generate sound if resonant
    #                 if is_resonant or phi_resonant:
    #                     # Create sound generator
    #                     sound_gen = SoundGenerator(output_dir="output/sounds/soul_field")
                        
    #                     # Generate a resonance sound
    #                     sound_params = {
    #                         "base_frequency": min(soul_freq, field_freq),
    #                         "amplitudes": 0.7,
    #                         "duration": 3.0,
    #                         "harmonics": [1.0, ratio],  # Include the resonant ratio
    #                         "fade_in_out": 0.5
    #                     }
                        
    #                     # Generate and save the sound
    #                     resonance_sound = sound_gen.generate_harmonic_tone(**sound_params)
    #                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                     soul_id = getattr(soul_spark, 'spark_id', 'unknown')[:8]
    #                     field_key = getattr(soul_spark, 'current_field_key', 'void')
    #                     filename = f"soul_resonance_{soul_id}_{field_key}_{timestamp}.wav"
    #                     sound_file = sound_gen.save_sound(resonance_sound, filename)
                        
    #                     if sound_file and hasattr(soul_spark, 'interaction_history'):
    #                         # Record the resonance in soul's history
    #                         soul_spark.interaction_history.append({
    #                             'type': 'resonance_sound',
    #                             'timestamp': datetime.now().isoformat(),
    #                             'field': field_key,
    #                             'ratio': ratio,
    #                             'sound_file': sound_file
    #                         })
                            
    #                         logger.info(f"Generated resonance sound for soul {soul_id} in {field_key} field")
    #     except Exception as sound_err:
    #         logger.error(f"Error generating soul-field resonance sound: {sound_err}", exc_info=True)
    #         # Non-critical, continue execution


    # def _apply_field_transition_effects(self, soul_spark: SoulSpark,
    #                                     old_field: Optional[str], new_field: str,
    #                                     coords: Tuple[int, int, int]) -> None:
    #     """Applies effects when a soul transitions between field types."""
    #     if old_field is None: return # No transition on first placement
    #     try:
    #         field_props = self.get_properties_at(coords)
    #         is_entering_guff = (new_field == 'kether' and
    #                             field_props.get('in_guff', False) and
    #                             (old_field != 'kether' or not field_props.get('was_in_guff_prev', False))) # Crude check

    #         if is_entering_guff:
    #             logger.debug(f"Soul {soul_spark.spark_id} entering Guff.")
    #             target_e = field_props.get('guff_target_energy_seu', soul_spark.energy)
    #             # Small immediate energy adjustment towards Guff target
    #             soul_spark.energy += (target_e - soul_spark.energy) * 0.05
    #             # Reduce freq variance history to allow faster stabilization in Guff
    #             if hasattr(soul_spark, 'frequency_history'):
    #                 soul_spark.frequency_history = soul_spark.frequency_history[-5:]

    #         # Log entry into any named Sephirah zone
    #         if new_field != 'void':
    #             logger.debug(f"Soul {soul_spark.spark_id} entering influence of {new_field}.")
    #             if hasattr(soul_spark, 'interaction_history'):
    #                  soul_spark.interaction_history.append({
    #                      'type': 'field_transition', 'timestamp': datetime.now().isoformat(),
    #                      'from': old_field, 'to': new_field, 'position': coords
    #                  })

    #         # Example: Apply effect based on Edge of Chaos value
    #         edge_value = field_props.get('edge_of_chaos', 0.0)
    #         if edge_value > 0.75 and hasattr(soul_spark, 'potential_realization'):
    #              # Increase potential realization factor (0-1) in high EoC zones
    #              current_potential = getattr(soul_spark, 'potential_realization', 0.0)
    #              setattr(soul_spark, 'potential_realization', min(1.0, current_potential + 0.01))

    #     except Exception as e:
    #         logger.error(f"Error applying field transition effects for {soul_spark.spark_id}: {e}",
    #                      exc_info=True)

    # def place_soul_in_guff(self, soul_spark: SoulSpark) -> None:
    #     """Places soul randomly within the Guff region."""
    #     if not self.kether_field: raise RuntimeError("Kether field not initialized.")
    #     center = self.kether_field.location
    #     guff_radius = self.kether_field.guff_radius
    #     if guff_radius <= 0: raise ValueError("Guff radius must be positive.")

    #     # Generate random point within the Guff sphere
    #     phi = np.random.uniform(0, 2 * np.pi)
    #     costheta = np.random.uniform(-1, 1)
    #     u = np.random.uniform(0, 1) # Uniform distribution for radius cubed
    #     theta = np.arccos(costheta)
    #     r = guff_radius * np.cbrt(u) # Correct sampling for uniform volume

    #     x = center[0] + r * np.sin(theta) * np.cos(phi)
    #     y = center[1] + r * np.sin(theta) * np.sin(phi)
    #     z = center[2] + r * np.cos(theta)

    #     guff_position_float = [x, y, z]
    #     # Use move_soul to handle clamping, field key update, Guff registration
    #     self.move_soul(soul_spark, guff_position_float)

    #     # Force field key if move_soul failed detection (shouldn't happen often)
    #     if soul_spark.current_field_key != 'kether':
    #         logger.warning(f"Forcing field key to 'kether' after Guff placement.")
    #         setattr(soul_spark, 'current_field_key', 'kether')
    #         # Re-register just in case move_soul failed that part
    #         self.kether_field.register_soul_in_guff(soul_spark.spark_id)

    #     logger.info(f"Placed soul {soul_spark.spark_id} in Guff at "
    #                 f"{soul_spark.position} (Field: {soul_spark.current_field_key}).")

    # def release_soul_from_guff(self, soul_spark: SoulSpark) -> None:
    #     """Moves soul just outside Guff radius but within Kether influence."""
    #     if not self.kether_field: raise RuntimeError("Kether field not initialized.")
    #     center = self.kether_field.location
    #     kether_radius = self.kether_field.radius
    #     guff_radius = self.kether_field.guff_radius
    #     if kether_radius <= guff_radius:
    #         logger.warning("Kether radius not larger than Guff radius, releasing just outside Guff.")
    #         release_radius = guff_radius + 1.0
    #     else:
    #          # Place slightly outside Guff, within Kether zone
    #          release_radius = guff_radius + (kether_radius - guff_radius) * 0.1
    #     release_radius = max(guff_radius + 0.5, release_radius) # Ensure distinctly outside

    #     # Generate random point on sphere surface at release_radius
    #     # Generate random point on sphere surface at release_radius
    #     phi = np.random.uniform(0, 2 * np.pi)
    #     costheta = np.random.uniform(-1, 1)
    #     theta = np.arccos(costheta)

    #     x = center[0] + release_radius * np.sin(theta) * np.cos(phi)
    #     y = center[1] + release_radius * np.sin(theta) * np.sin(phi)
    #     z = center[2] + release_radius * np.cos(theta)

    #     release_position_float = [x, y, z]
    #     # Use move_soul to handle clamping, field key update, Guff removal
    #     self.move_soul(soul_spark, release_position_float)

    #     # Force field key if needed
    #     if soul_spark.current_field_key != 'kether':
    #         logger.warning(f"Forcing field key to 'kether' after Guff release.")
    #         setattr(soul_spark, 'current_field_key', 'kether')
    #         # Ensure removed from Guff registration if move_soul missed it
    #         self.kether_field.remove_soul_from_guff(soul_spark.spark_id)

    #     logger.info(f"Released soul {soul_spark.spark_id} from Guff to Kether at "
    #                 f"{soul_spark.position} (Field: {soul_spark.current_field_key}).")


    # def find_optimal_development_location(self, soul_spark: Optional[SoulSpark] = None
    #                                       ) -> List[float]:
    #     """
    #     Finds a location with high Edge of Chaos value, suitable for development.
    #     Selection from top N points is deterministic based on soul_spark ID if provided,
    #     otherwise random.
    #     """
    #     try:
    #         current_points = getattr(self, 'optimal_development_points', [])
    #         # Refresh EoC tracking if list is empty or seems stale
    #         if not current_points or not hasattr(self, 'edge_values') or not self.edge_values:
    #             self._initialize_edge_of_chaos_tracking()
    #             current_points = getattr(self, 'optimal_development_points', [])

    #         # If still no points, return random location within grid
    #         if not current_points or not self.edge_values:
    #             logger.warning("No optimal EoC points found, returning random location.")
    #             return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

    #         # Filter points that are still valid and have EoC values
    #         valid_points = [p for p in current_points if p in self.edge_values]
    #         if not valid_points:
    #              logger.warning("Tracked EoC points are invalid, returning random location.")
    #              return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

    #         # Select from top N points (e.g., top 5) based on EoC value
    #         sorted_points = sorted(valid_points,
    #                                key=lambda p: self.edge_values.get(p, 0.0),
    #                                reverse=True)
    #         num_top = min(5, len(sorted_points))
    #         idx = np.random.randint(0, num_top) # Default to random index in top N

    #         # Optional: Use soul ID hash for deterministic selection (if repeatable needed)
    #         if soul_spark and hasattr(soul_spark, 'spark_id'):
    #             try: idx = hash(soul_spark.spark_id) % num_top
    #             except Exception as hash_err:
    #                 logger.warning(f"Could not hash soul ID {soul_spark.spark_id}: {hash_err}. Using random index.")

    #         optimal_point_int = sorted_points[idx]
    #         # Return float coordinates near the center of the chosen grid cell
    #         return [float(c) + 0.5 for c in optimal_point_int]

    #     except Exception as e:
    #         logger.error(f"Error finding optimal development location: {e}", exc_info=True)
    #         # Fallback to random location on any error
    #         return [np.random.uniform(0, d - 1.0) for d in self.grid_size]

    # def get_visualization_data(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
    #     """Gets data suitable for visualization at specific coordinates."""
    #     try:
    #         field_props = self.get_properties_at(coords) # Fails hard on invalid coords
    #         sound_params = self.get_sound_parameters_at(coords) # Uses fallback if needed

    #         # Combine properties and sound info
    #         # Prioritize FieldHarmonics visualization data if available
    #         if FH_AVAILABLE and FieldHarmonics:
    #              try:
    #                  visualization = FieldHarmonics.generate_live_sound_visualization(
    #                      sound_params, field_props.get('dominant_sephirah', 'void'), field_props
    #                  )
    #                  # Add core field properties not included by FH visualization
    #                  core_props_to_add = ['energy_seu', 'frequency_hz', 'stability_su',
    #                                       'coherence_cu', 'edge_of_chaos', 'in_guff']
    #                  for prop in core_props_to_add:
    #                      if prop not in visualization:
    #                          visualization[prop] = field_props.get(prop)
    #                  return visualization
    #              except Exception as fh_viz_err:
    #                   logger.warning(f"FieldHarmonics visualization failed: {fh_viz_err}. Using basic fallback.", exc_info=True)
    #                   # Fall through to basic fallback

    #         # Basic Fallback Visualization Data
    #         viz_data = {
    #             'energy_seu': field_props.get('energy_seu'),
    #             'frequency_hz': field_props.get('frequency_hz'),
    #             'stability_su': field_props.get('stability_su'),
    #             'coherence_cu': field_props.get('coherence_cu'),
    #             'dominant_sephirah': field_props.get('dominant_sephirah'),
    #             'edge_of_chaos': field_props.get('edge_of_chaos'),
    #             'color_rgb': field_props.get('color_rgb', [0.5, 0.5, 0.5]), # Grey fallback
    #             'in_guff': field_props.get('in_guff', False)
    #         }
    #         return viz_data

    #     except IndexError:
    #         logger.error(f"Coordinates {coords} out of bounds for visualization.")
    #         return {'error': 'Coordinates out of bounds', 'coords': coords}
    #     except Exception as e:
    #         logger.error(f"Error generating visualization data for {coords}: {e}",
    #                      exc_info=True)
    #         return {'error': str(e), 'coords': coords} # Return error dict


    # def __str__(self) -> str:
    #     """String representation of the FieldController."""
    #     seph_count = len(self.sephiroth_influencers)
    #     void_status = "Initialized" if self.void_field and self.void_field.energy is not None else "Uninitialized"
    #     return (f"FieldController(Grid: {self.grid_size}, "
    #             f"Sephiroth: {seph_count}, Void Status: {void_status})")

# --- END OF FILE src/stage_1/fields/field_controller.py ---