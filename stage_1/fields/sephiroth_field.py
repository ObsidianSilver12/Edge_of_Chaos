# --- START OF FILE src/stage_1/fields/sephiroth_field.py ---

"""
Sephiroth Field Module (Refactored V4.3 - SEU/SU/CU Units, PEP8 Fix)

Implements Sephiroth influences using SEU/SU/CU targets from constants.
Includes improved error handling and calls updated VoidField methods.
Corrected default frequency constant name. Adheres to PEP 8 formatting.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# --- Logging ---
logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately
try:
    import shared.constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for SephirothField logger."
    )
    logger.setLevel(logging.INFO)

# --- Constants Import (using alias 'const') ---
try:
    import shared.constants.constants as const
except ImportError as e:
    logger.critical(
        "CRITICAL ERROR: constants.py failed import in sephiroth_field.py"
    )
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Base Class & Dependencies ---
try:
    from .field_base import FieldBase
    from .void_field import VoidField
    from .sephiroth_aspect_dictionary import aspect_dictionary
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary is None")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Cannot import base/void field or aspect dict "
                    f"in sephiroth_field.py: {e}", exc_info=True)
    # Define dummy classes only if needed for script structure, but raise error ideally
    class FieldBase: pass
    class VoidField: pass
    aspect_dictionary = None
    raise ImportError(f"Core dependencies missing: {e}") from e


class SephirothField:
    """
    Represents the influence zone of a single Sephirah.
    Applies modifications to the base VoidField based on its properties.
    Uses SEU/SU/CU targets loaded from constants.
    """

    def __init__(self, sephirah_name: str, location: Tuple[int, int, int],
                 radius: float):
        """
        Initialize a SephirothField influencer.

        Args:
            sephirah_name (str): The name of the Sephirah (e.g., 'kether', 'chesed').
            location (Tuple[int, int, int]): Grid coordinates of the center.
            radius (float): Radius of influence in grid units.

        Raises:
            ValueError: If input parameters are invalid or aspect data is missing.
            RuntimeError: If Aspect Dictionary is unavailable.
            TypeError: If input types are incorrect.
        """
        if not isinstance(sephirah_name, str) or not sephirah_name:
             raise ValueError("Sephirah name must be a non-empty string.")
        if not isinstance(location, tuple) or len(location) != 3 or \
           not all(isinstance(i, int) for i in location):
            raise TypeError("Location must be a tuple of 3 integers.")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")

        self.sephirah_name: str = sephirah_name.lower()
        self.location: Tuple[int, int, int] = location
        self.radius: float = radius
        self.radius_sq: float = radius * radius # Pre-calculate for efficiency

        # --- Load Aspect Data ---
        if aspect_dictionary is None:
            # This check should ideally be done before instantiation, but good safeguard
            raise RuntimeError("Aspect Dictionary unavailable.")
        try:
            self.aspect_data: Dict[str, Any] = aspect_dictionary.get_aspects(self.sephirah_name)
            if not self.aspect_data: # Check if dict is empty
                raise ValueError(f"No aspect data found for '{self.sephirah_name}' "
                                 f"in Aspect Dictionary.")
        except Exception as e:
             logger.error(f"Failed to load aspect data for {self.sephirah_name}", exc_info=True)
             raise ValueError(f"Error loading aspect data for '{self.sephirah_name}'") from e

        # --- Initialize Properties from Aspect Data & Constants ---
        # Use aspect_data.get() with a valid fallback FROM CONSTANTS
        self.target_frequency: float = self.aspect_data.get(
            'base_frequency', const.INITIAL_SPARK_BASE_FREQUENCY_HZ # *** CORRECTED CONSTANT ***
        )
        self.primary_color_str: str = self.aspect_data.get('primary_color', 'white')
        self.target_color_rgb: np.ndarray = self._get_target_color_vector(
            self.primary_color_str
        ) # Expects [r,g,b] 0-1 float32
        self.geometric_pattern: Optional[str] = self.aspect_data.get(
            'geometric_correspondence'
        )
        self.platonic_affinity: Optional[str] = self.aspect_data.get(
            'platonic_affinity'
        )
        self.resonance_multiplier: float = self.aspect_data.get(
            'resonance_multiplier', 1.0
        )
        self.harmonic_signature_params: Dict[str, Any] = self.aspect_data.get(
            'harmonic_signature_params', {'ratios': [1.0, 2.0], 'falloff': 0.1}
        ) # Default harmonic signature

        # Validate calculated/loaded properties
        if self.target_frequency <= 0:
             logger.warning(f"Target frequency for {self.sephirah_name} is non-positive "
                           f"({self.target_frequency}). Using default.")
             self.target_frequency = const.INITIAL_SPARK_BASE_FREQUENCY_HZ

        logger.info(
            f"SephirothField influencer '{self.sephirah_name.capitalize()}' "
            f"initialized at {self.location} with radius {self.radius:.1f}. "
            f"Target Freq: {self.target_frequency:.1f} Hz."
        )

    # --- Properties for Field Visualization Compatibility ---
    @property
    def name(self) -> str:
        """Name property for visualization compatibility."""
        return self.sephirah_name
    
    @property 
    def base_frequency(self) -> float:
        """Base frequency property for visualization compatibility."""
        return self.target_frequency
    
    @property
    def color(self) -> np.ndarray:
        """Color property for visualization compatibility."""
        return self.target_color_rgb
    
    @property
    def pattern_type(self) -> str:
        """Pattern type property for visualization compatibility."""
        return self.geometric_pattern or self.platonic_affinity or "sacred_geometry"
    
    @property 
    def pattern_influence(self) -> Optional[np.ndarray]:
        """Pattern influence property for visualization compatibility - returns None for symbolic visualization."""
        return None

    # --- Target Value Retrieval Methods ---
    def get_target_energy_seu(self) -> float:
        """Gets target energy potential (SEU) from constants."""
        # Use VOID_BASE as fallback if specific value missing
        return const.SEPHIROTH_ENERGY_POTENTIALS_SEU.get(
            self.sephirah_name, const.VOID_BASE_ENERGY_SEU * 2.0
        )

    def get_target_stability_su(self) -> float:
        """Gets target stability score (SU) from constants."""
        return const.SEPHIROTH_TARGET_STABILITY_SU.get(
            self.sephirah_name, const.VOID_BASE_STABILITY_SU * 1.5
        )

    def get_target_coherence_cu(self) -> float:
        """Gets target coherence score (CU) from constants."""
        return const.SEPHIROTH_TARGET_COHERENCE_CU.get(
            self.sephirah_name, const.VOID_BASE_COHERENCE_CU * 1.5
        )

    # --- Helper: Get Color Vector ---
    def _get_target_color_vector(self, color_name_or_hex: str) -> np.ndarray:
        """Converts color name or hex string to a [R,G,B] numpy array (0-1 float)."""
        # 1. Check for Hex Code
        if isinstance(color_name_or_hex, str) and color_name_or_hex.startswith('#'):
            hex_code = color_name_or_hex.lstrip('#')
            if len(hex_code) == 6:
                try:
                    rgb_int = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
                    # Normalize to 0-1 float32
                    return np.array(rgb_int, dtype=np.float32) / 255.0
                except ValueError:
                    logger.warning(f"Invalid hex code '{color_name_or_hex}' for "
                                   f"{self.sephirah_name}. Using grey.")
                    # Fall through to grey default

        # 2. Check for Color Name in Spectrum Constant
        elif isinstance(color_name_or_hex, str):
            color_data = const.COLOR_SPECTRUM.get(color_name_or_hex.lower())
            if color_data and 'hex' in color_data:
                 # Recursively call with the hex code found
                 return self._get_target_color_vector(color_data['hex'])
            else:
                 logger.warning(f"Color name '{color_name_or_hex}' not found in "
                                f"COLOR_SPECTRUM for {self.sephirah_name}. Using grey.")
                 # Fall through to grey default
        else:
            logger.warning(f"Invalid color format '{color_name_or_hex}' "
                           f"(type: {type(color_name_or_hex)}) for {self.sephirah_name}. "
                           f"Using grey.")
            # Fall through to grey default

        # 3. Fallback Default (Grey)
        return np.array([0.7, 0.7, 0.7], dtype=np.float32)

    # --- apply_sephiroth_influence ---
    def apply_sephiroth_influence(self, void_field: VoidField) -> None:
        """Applies this Sephirah's influence onto the provided VoidField."""
        if not isinstance(void_field, VoidField):
            raise TypeError("void_field must be an instance of VoidField.")
        # Check if void field grids are initialized
        if void_field.energy is None or void_field.stability is None or \
           void_field.coherence is None or void_field.frequency is None or \
           void_field.color is None or void_field.pattern_influence is None or \
           void_field.order is None or void_field.chaos is None:
            raise AttributeError("VoidField grids are not initialized. Cannot apply influence.")

        logger.debug(f"Applying influence of '{self.sephirah_name}' onto VoidField...")
        center_x, center_y, center_z = self.location
        grid_shape = void_field.get_grid_shape()

        try: # Wrap main calculation block for robust error handling
            # --- Bounding Box & Mask Calculation ---
            # Calculate integer radius for slicing efficiency
            int_radius = int(np.ceil(self.radius))
            # Define bounding box, clamped to grid dimensions
            min_x = max(0, center_x - int_radius)
            max_x = min(grid_shape[0], center_x + int_radius + 1)
            min_y = max(0, center_y - int_radius)
            max_y = min(grid_shape[1], center_y + int_radius + 1)
            min_z = max(0, center_z - int_radius)
            max_z = min(grid_shape[2], center_z + int_radius + 1)

            # Check if bounding box is valid (has volume)
            if min_x >= max_x or min_y >= max_y or min_z >= max_z:
                logger.debug(f"Bounding box for {self.sephirah_name} has zero volume. Skipping.")
                return # No points to affect

            # Create index grids ONLY for the bounding box slice
            x_idx, y_idx, z_idx = np.meshgrid(
                np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z),
                indexing='ij' # Use 'ij' indexing for consistency with NumPy array access
            )
            # Calculate squared distance from center within the box
            dist_sq = ((x_idx - center_x)**2 + (y_idx - center_y)**2 +
                       (z_idx - center_z)**2)
            # Create mask for points within the sphere's radius (squared)
            within_radius_mask = dist_sq <= self.radius_sq
            # If no points are within the radius in this box, skip
            if not np.any(within_radius_mask):
                logger.debug(f"No grid points within radius for {self.sephirah_name} in bounding box. Skipping.")
                return

            # --- Calculate Influence Strength ---
            # Influence decreases with distance from the center (inverse distance based)
            # Use a sharper falloff near the edge? Example: 1 / (1 + (dist/radius)^4)
            norm_dist_sq = dist_sq / max(const.FLOAT_EPSILON, self.radius_sq)
            influence_strength = 1.0 / (1.0 + norm_dist_sq**2 * 5.0) # Example falloff
            # Apply radius mask - zero influence outside radius
            influence_strength[~within_radius_mask] = 0.0
            # Further mask to only consider cells with significant influence
            influence_slice_mask = influence_strength > const.FLOAT_EPSILON

            # If no points have significant influence, skip
            if not np.any(influence_slice_mask):
                 logger.debug(f"No significant influence points for {self.sephirah_name}. Skipping.")
                 return

            # --- Apply Influence to Void Field Grids ---
            # Define the slice corresponding to the bounding box
            box_slice = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))

            # Get influence values ONLY for the affected cells
            influence_vals = influence_strength[influence_slice_mask]
            inv_influence_vals = 1.0 - influence_vals

            # Get Target Values (SEU/SU/CU/Hz/RGB)
            target_e = self.get_target_energy_seu()
            target_s = self.get_target_stability_su()
            target_c = self.get_target_coherence_cu()
            target_f = self.target_frequency
            target_col = self.target_color_rgb # Already [r,g,b] 0-1

            # Blend Core Properties (Vectorized on the masked slice)
            # Ensures we only read/write the necessary cells
            # Syntax: grid[slice][mask] accesses the masked elements within the slice
            void_field.energy[box_slice][influence_slice_mask] = (
                void_field.energy[box_slice][influence_slice_mask] * inv_influence_vals +
                target_e * influence_vals
            )
            void_field.stability[box_slice][influence_slice_mask] = (
                void_field.stability[box_slice][influence_slice_mask] * inv_influence_vals +
                target_s * influence_vals
            )
            void_field.coherence[box_slice][influence_slice_mask] = (
                void_field.coherence[box_slice][influence_slice_mask] * inv_influence_vals +
                target_c * influence_vals
            )
            void_field.frequency[box_slice][influence_slice_mask] = (
                void_field.frequency[box_slice][influence_slice_mask] * inv_influence_vals +
                target_f * influence_vals
            )
            # Color blending requires broadcasting influence
            influence_broadcast = influence_vals[:, np.newaxis]
            inv_influence_broadcast = 1.0 - influence_broadcast
            void_field.color[box_slice][influence_slice_mask] = (
                void_field.color[box_slice][influence_slice_mask] * inv_influence_broadcast +
                target_col * influence_broadcast # Broadcast target_col might be needed if shape mismatch: target_col[np.newaxis, :] * influence_broadcast
            )

            # Apply Geometric & Harmonic Influences (Modify the blended state)
            self._apply_geometric_influence(void_field, box_slice,
                                           influence_slice_mask, influence_vals)
            self._apply_harmonic_resonance(void_field, box_slice,
                                          influence_slice_mask, influence_vals)

            # --- Recalculate Derived Order/Chaos for Affected Cells ---
            # Ensure calculations only use affected cells to avoid unnecessary work
            affected_indices = tuple(idx[influence_slice_mask] for idx in (x_idx, y_idx, z_idx))
            # Use affected_indices for direct access
            norm_s_affected = void_field.stability[affected_indices] / const.MAX_STABILITY_SU
            norm_c_affected = void_field.coherence[affected_indices] / const.MAX_COHERENCE_CU
            new_order_affected = np.clip((norm_s_affected * 0.6 + norm_c_affected * 0.4), 0.0, 1.0)
            void_field.order[affected_indices] = new_order_affected
            void_field.chaos[affected_indices] = 1.0 - new_order_affected

            # --- Final Clamping (Apply only to the cells modified by this influencer) ---
            # Using affected_indices ensures we only clamp modified cells
            void_field.energy[affected_indices] = np.clip(void_field.energy[affected_indices], 0.0, const.MAX_SOUL_ENERGY_SEU * 5)
            void_field.stability[affected_indices] = np.clip(void_field.stability[affected_indices], 0.0, const.MAX_STABILITY_SU)
            void_field.coherence[affected_indices] = np.clip(void_field.coherence[affected_indices], 0.0, const.MAX_COHERENCE_CU)
            void_field.frequency[affected_indices] = np.clip(void_field.frequency[affected_indices], const.VOID_BASE_FREQUENCY_RANGE[0], const.VOID_BASE_FREQUENCY_RANGE[1]*2.0) # Wider range allowed near Sephiroth?
            void_field.order[affected_indices] = np.clip(void_field.order[affected_indices], 0.0, 1.0)
            void_field.chaos[affected_indices] = np.clip(void_field.chaos[affected_indices], 0.0, 1.0)
            void_field.color[affected_indices] = np.clip(void_field.color[affected_indices], 0.0, 1.0)
            void_field.pattern_influence[affected_indices] = np.clip(void_field.pattern_influence[affected_indices], 0.0, 1.0)

        # --- Error Handling ---
        except AttributeError as ae: # Catch errors if void_field grids are None mid-process
            logger.error(f"AttributeError applying influence for '{self.sephirah_name}': {ae}. "
                         f"VoidField grids might be missing.", exc_info=True)
            # Don't re-raise here, allow simulation to potentially continue if possible
        except IndexError as ie:
             logger.error(f"IndexError applying influence for '{self.sephirah_name}': {ie}. "
                          f"Coordinates likely invalid or slice error.", exc_info=True)
             # Don't re-raise here
        except Exception as e:
            logger.error(f"Unexpected error applying influence for '{self.sephirah_name}': {e}",
                         exc_info=True)
            # Don't re-raise here


    def _apply_geometric_influence(self, void_field: VoidField, box_slice: Tuple,
                                   influence_slice_mask: np.ndarray,
                                   influence_vals: np.ndarray) -> None:
        """Applies geometric/platonic influence based on aspect data."""
        try:
            # Combine strengths (geometric pattern + platonic affinity)
            geom_strength = const.GEOMETRY_VOID_INFLUENCE_STRENGTH if self.geometric_pattern else 0.0
            plat_strength = const.PLATONIC_VOID_INFLUENCE_STRENGTH if self.platonic_affinity else 0.0
            # Avoid division by zero if both are zero
            norm_factor = (1.0 if geom_strength == 0.0 or plat_strength == 0.0 else 2.0)
            combined_strength = (geom_strength + plat_strength) / norm_factor

            effective_influence = influence_vals * combined_strength
            if effective_influence.size == 0 or np.max(effective_influence) <= const.FLOAT_EPSILON:
                return # No significant influence

            # --- 1. Modify Pattern Influence Grid ---
            # Check grid exists before modifying
            if void_field.pattern_influence is None: return
            # Get affected indices for direct access
            affected_indices = tuple(idx[influence_slice_mask] for idx in np.indices(influence_slice_mask.shape))
            # Apply influence using the correct indices for the slice
            void_field.pattern_influence[box_slice][influence_slice_mask] += effective_influence * 0.5 # Example: Additive effect
            void_field.pattern_influence[affected_indices] = np.clip(
                void_field.pattern_influence[affected_indices], 0.0, 1.0
            )

            # --- 2. Apply Subtle Bias based on GEOMETRY_EFFECTS ---
            # Combine effects from pattern and platonic affinity
            geom_effects = const.GEOMETRY_EFFECTS.get(self.geometric_pattern, {})
            plat_effects = const.GEOMETRY_EFFECTS.get(self.platonic_affinity, {})
            combined_mods = {}
            all_keys = set(geom_effects.keys()) | set(plat_effects.keys())
            for key in all_keys:
                effect_geom = geom_effects.get(key, 0.0); effect_plat = plat_effects.get(key, 0.0)
                count = (1 if effect_geom != 0.0 else 0) + (1 if effect_plat != 0.0 else 0)
                combined_mods[key] = (effect_geom + effect_plat) / max(1.0, count)

            if combined_mods:
                # Apply stability boost/penalty (to SU)
                if 'stability_factor_boost' in combined_mods and void_field.stability is not None:
                    stability_push = (combined_mods['stability_factor_boost'] *
                                      effective_influence * 0.1) # Gentle push factor
                    # Apply change scaled by MAX_STABILITY_SU
                    void_field.stability[affected_indices] += stability_push * const.MAX_STABILITY_SU

                # Apply coherence boost/penalty (to CU)
                if 'coherence_factor_boost' in combined_mods and void_field.coherence is not None:
                    coherence_push = (combined_mods['coherence_factor_boost'] *
                                      effective_influence * 0.1)
                    void_field.coherence[affected_indices] += coherence_push * const.MAX_COHERENCE_CU

                # Apply frequency pull towards associated base frequency (Hz)
                if void_field.frequency is not None:
                    assoc_geom = self.platonic_affinity or self.geometric_pattern
                    target_freq = const.PLATONIC_BASE_FREQUENCIES.get(assoc_geom, self.target_frequency)
                    current_freq = void_field.frequency[affected_indices]
                    freq_push = (target_freq - current_freq) * effective_influence * 0.05 # Gentle pull
                    void_field.frequency[affected_indices] += freq_push

                # Note: Clamping happens *after* all influences are applied in apply_sephiroth_influence

        except Exception as e:
            logger.error(f"Error applying geometric influence for {self.sephirah_name}: {e}",
                         exc_info=True)


    def _apply_harmonic_resonance(self, void_field: VoidField, box_slice: Tuple,
                                  influence_slice_mask: np.ndarray,
                                  influence_vals: np.ndarray) -> None:
        """Applies harmonic resonance effects based on Sephirah signature."""
        try:
            params = self.harmonic_signature_params
            ratios = params.get('ratios', [1.0, 2.0]) # Default ratios
            falloff = params.get('falloff', 0.1) # Default falloff
            base_freq = self.target_frequency
            resonance_mult = self.resonance_multiplier

            # Check grid exists and get affected cell frequencies
            if void_field.frequency is None: return
            # Get affected indices for direct access
            affected_indices = tuple(idx[influence_slice_mask] for idx in np.indices(influence_slice_mask.shape))
            cell_freqs = void_field.frequency[affected_indices]
            if cell_freqs.size == 0: return # No cells to affect

            # Calculate resonance boost based on proximity to harmonic frequencies
            resonance_boost = np.zeros_like(cell_freqs)
            # Normalization factor based on weights (1/ratio)
            max_possible_boost = sum(1.0 / max(const.FLOAT_EPSILON, abs(r)) for r in ratios) if ratios else 1.0
            if max_possible_boost <= const.FLOAT_EPSILON: max_possible_boost = 1.0

            for ratio in ratios:
                harmonic_freq = base_freq * ratio
                if harmonic_freq <= const.FLOAT_EPSILON: continue
                # Relative frequency distance
                freq_distance = np.abs(cell_freqs - harmonic_freq) / harmonic_freq
                # Calculate resonance score (exponential decay)
                harmonic_resonance = np.exp(-freq_distance / max(const.FLOAT_EPSILON, falloff))
                # Weight by inverse ratio (fundamental has highest weight)
                resonance_weight = 1.0 / max(const.FLOAT_EPSILON, abs(ratio))
                resonance_boost += harmonic_resonance * resonance_weight

            # Normalize and scale boost
            resonance_boost = (resonance_boost / max_possible_boost) * resonance_mult
            resonance_boost = np.clip(resonance_boost, 0.0, 1.0)

            # Apply effects scaled by influence and resonance boost
            effective_boost = resonance_boost * influence_vals # Combine resonance with spatial influence

            # Apply additive effects to SEU, CU, SU, pattern influence (0-1)
            if void_field.energy is not None:
                energy_effect = effective_boost * 0.05 * self.get_target_energy_seu() # Small energy gain
                void_field.energy[affected_indices] += energy_effect
            if void_field.coherence is not None:
                coherence_effect = effective_boost * 0.15 * const.MAX_COHERENCE_CU # Coherence gain
                void_field.coherence[affected_indices] += coherence_effect
            if void_field.stability is not None:
                 stability_effect = effective_boost * 0.05 * const.MAX_STABILITY_SU # Small stability gain
                 void_field.stability[affected_indices] += stability_effect
            if void_field.pattern_influence is not None:
                 pattern_effect = effective_boost * 0.2 # Pattern influence gain
                 void_field.pattern_influence[affected_indices] += pattern_effect

            # Note: Clamping happens *after* all influences are applied

        except Exception as e:
            logger.error(f"Error applying harmonic resonance for {self.sephirah_name}: {e}",
                         exc_info=True)

    def get_aspects(self) -> Dict[str, Any]:
        """Returns a copy of the aspect data for this Sephirah."""
        # Ensure aspect_data was loaded correctly
        if not hasattr(self, 'aspect_data') or not isinstance(self.aspect_data, dict):
            logger.error(f"Aspect data not loaded for {self.sephirah_name}. Returning empty.")
            return {}
        return self.aspect_data.copy() # Return a copy

    # --- String Representations ---
    def __str__(self) -> str:
        """String representation of the SephirothField."""
        return (f"SephirothField(Name: {self.sephirah_name.capitalize()}, "
                f"Loc: {self.location}, Rad: {self.radius:.1f})")

    def __repr__(self) -> str:
        """Detailed representation of the SephirothField."""
        return (f"<SephirothField name='{self.sephirah_name}' "
                f"location={self.location} radius={self.radius}>")

# --- END OF FILE src/stage_1/fields/sephiroth_field.py ---