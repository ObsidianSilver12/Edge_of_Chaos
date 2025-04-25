# --- START OF FILE earth_harmonisation.py ---

"""
Earth Harmonization Functions (Refactored - Operates on SoulSpark Object, Uses Constants)

Provides functions for harmonizing the soul spark with Earth's frequencies,
elements, and cycles, preparing it for physical incarnation. Modifies the
SoulSpark object instance directly. Uses centrally defined constants.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from src.stage_1.soul_formation.soul_spark import SoulSpark
from src.constants.constants import *


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Import necessary constants FROM THE CENTRAL FILE
    # Extract specific Earth freq if needed
    SCHUMANN_FREQUENCY = EARTH_FREQUENCIES.get("schumann", 7.83)
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. Earth Harmonization cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from src.stage_1.soul_formation.soul_spark import SoulSpark
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    DEPENDENCIES_AVAILABLE = True
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark or aspect_dictionary: {e}. Earth Harmonization cannot function.")
    raise ImportError(f"Core dependencies missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs):
            pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """Checks if the soul meets prerequisites for Earth harmonization using constants."""
    logger.debug(f"Checking Earth harmonization prerequisites for soul {soul_spark.spark_id}...")
    # Requires life cord formation completion flag using constant
    if HARMONY_PREREQ_CORD_COMPLETE and not getattr(soul_spark, "cord_formation_complete", False):
        logger.error("Prerequisite failed: Life cord formation not complete.")
        return False
    # Requires sufficient cord integrity using constant
    cord_integrity = getattr(soul_spark, "cord_integrity", 0.0)
    if cord_integrity < HARMONY_PREREQ_CORD_INTEGRITY_MIN:
        logger.error(f"Prerequisite failed: Life cord integrity ({cord_integrity:.3f}) below threshold ({HARMONY_PREREQ_CORD_INTEGRITY_MIN}).")
        return False
    # Requires minimum stability and coherence using constants
    stability = getattr(soul_spark, "stability", 0.0)
    if stability < HARMONY_PREREQ_STABILITY_MIN:
        logger.error(f"Prerequisite failed: Soul stability ({stability:.3f}) below threshold ({HARMONY_PREREQ_STABILITY_MIN}).")
        return False
    coherence = getattr(soul_spark, "coherence", 0.0)
    if coherence < HARMONY_PREREQ_COHERENCE_MIN:
        logger.error(f"Prerequisite failed: Soul coherence ({coherence:.3f}) below threshold ({HARMONY_PREREQ_COHERENCE_MIN}).")
        return False
    # Optionally check if already harmonized
    if getattr(soul_spark, "earth_harmonized", False):
        logger.warning(f"Soul {soul_spark.spark_id} is already marked as earth_harmonized. Proceeding with re-harmonization.")

    logger.debug("Earth harmonization prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary numeric properties for harmonization. Fails hard."""
    logger.debug(f"Ensuring properties exist for soul {soul_spark.spark_id}...")
    attributes_to_check = [
        ("frequency", SOUL_SPARK_DEFAULT_FREQ),
        ("stability", SOUL_SPARK_DEFAULT_STABILITY),
        ("coherence", SOUL_SPARK_DEFAULT_COHERENCE),
        ("elements", {}), # Dict expected
        ("earth_cycles", {}), # Dict expected
        ("planetary_resonance", 0.3), # Default low
        ("gaia_connection", 0.2) # Default low
    ]
    for attr, default in attributes_to_check:
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
            logger.warning(f"Soul {soul_spark.spark_id} missing '{attr}'. Initializing to default: {default}")
            setattr(soul_spark, attr, default)

    # Validate numerical types and ranges
    for attr, _ in attributes_to_check:
        val = getattr(soul_spark, attr)
        if attr in ['elements', 'earth_cycles']:
             if not isinstance(val, dict):
                  raise TypeError(f"Soul attribute '{attr}' must be a dictionary, found {type(val)}.")
        elif not isinstance(val, (int, float)):
             raise TypeError(f"Soul attribute '{attr}' must be numeric, found {type(val)}.")
        elif not np.isfinite(val):
             raise ValueError(f"Soul attribute '{attr}' has non-finite value {val}.")
        elif attr == 'frequency' and val <= FLOAT_EPSILON:
             raise ValueError(f"Soul frequency ({val}) must be positive.")
        elif attr != 'frequency' and not (0.0 <= val <= 1.0):
             logger.warning(f"Clamping soul attribute '{attr}' ({val}) to 0-1 range.")
             setattr(soul_spark, attr, max(0.0, min(1.0, val)))
    logger.debug("Soul properties ensured for Earth Harmonization.")


def _calculate_earth_resonance(soul_spark: SoulSpark) -> float:
    """Calculates the soul's current resonance with Earth using constants."""
    logger.debug(f"Calculating initial Earth resonance for soul {soul_spark.spark_id}...")
    try:
        _ensure_soul_properties(soul_spark) # Make sure needed attrs exist
        soul_freq = getattr(soul_spark, "frequency")
        soul_elements = getattr(soul_spark, "elements")

        # --- Frequency Resonance Component ---
        freq_factors = []
        valid_earth_freqs = {k:v for k,v in EARTH_FREQUENCIES.items() if v > FLOAT_EPSILON}
        for name, earth_freq in valid_earth_freqs.items():
            ratio = min(earth_freq / soul_freq, soul_freq / earth_freq)
            freq_factors.append(max(0.0, 1.0 - abs(1.0 - ratio))**2) # Squared closeness to 1

        schumann_res = 0.0; other_res = []
        for i, name in enumerate(valid_earth_freqs.keys()):
             if i < len(freq_factors): # Safety check
                  if name == "schumann": schumann_res = freq_factors[i]
                  else: other_res.append(freq_factors[i])
        avg_other_res = np.mean(other_res) if other_res else 0.0
        # Use constants for weighting
        freq_resonance = (HARMONY_FREQ_RES_WEIGHT_SCHUMANN * schumann_res +
                          HARMONY_FREQ_RES_WEIGHT_OTHER * avg_other_res)
        logger.debug(f"  Frequency Resonance component: {freq_resonance:.4f}")

        # --- Elemental Resonance Component ---
        elem_resonance = 0.0
        if soul_elements:
            try: malkuth_element = aspect_dictionary.get_aspects("malkuth").get("element", "earth").lower()
            except Exception: malkuth_element = "earth"

            total_strength = 0.0; earth_strength = 0.0; valid_elements_count = 0
            for elem in EARTH_ELEMENTS:
                 strength = soul_elements.get(elem.lower(), 0.0)
                 if isinstance(strength, (int, float)):
                      total_strength += strength
                      if elem.lower() == malkuth_element: earth_strength = strength
                      valid_elements_count += 1
            if valid_elements_count > 0:
                 avg_strength = total_strength / valid_elements_count
                 # Use constants for weighting
                 elem_resonance = (HARMONY_ELEM_RES_WEIGHT_PRIMARY * earth_strength +
                                   HARMONY_ELEM_RES_WEIGHT_AVERAGE * avg_strength)
        logger.debug(f"  Elemental Resonance component: {elem_resonance:.4f}")

        # Calculate overall resonance (keep 60/40 weighting or make constants?)
        overall_resonance = freq_resonance * 0.6 + elem_resonance * 0.4
        overall_resonance = max(0.0, min(1.0, overall_resonance))
        logger.debug(f"  Overall Earth Resonance calculated: {overall_resonance:.4f}")
        return float(overall_resonance)

    except Exception as e:
        logger.error(f"Error calculating Earth resonance for soul {soul_spark.spark_id}: {e}", exc_info=True)
        return 0.1 # Return default low resonance on error


def _perform_frequency_attunement(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """Attunes frequency towards Earth using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting frequency attunement for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    _ensure_soul_properties(soul_spark) # Ensure attributes exist

    try:
        current_freq = getattr(soul_spark, "frequency")
        logger.debug(f"  Current frequency: {current_freq:.2f} Hz")

        # Target frequency blends Schumann and current soul state using constants
        target_freq = (SCHUMANN_FREQUENCY * HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT +
                       current_freq * HARMONY_FREQ_TARGET_SOUL_WEIGHT)
        logger.debug(f"  Target harmonic frequency: {target_freq:.2f} Hz")

        # Calculate tuning amount using constants
        freq_diff = target_freq - current_freq
        tuning_amount = freq_diff * intensity * HARMONY_FREQ_TUNING_FACTOR * duration_factor # Include duration factor
        new_freq = current_freq + tuning_amount
        new_freq = max(SCHUMANN_FREQUENCY * 0.5, new_freq) # Ensure positive and not too low
        logger.debug(f"  Calculated frequency shift: {tuning_amount:.2f} Hz, New frequency: {new_freq:.2f} Hz")

        # --- Update SoulSpark ---
        setattr(soul_spark, "frequency", float(new_freq))
        h_count = getattr(soul_spark, 'harmonic_count', HARMONY_FREQ_UPDATE_HARMONIC_COUNT) # Use constant
        harmonics_list = [new_freq * n for n in range(1, h_count + 1)]
        setattr(soul_spark, "harmonics", harmonics_list)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        attunement_level = 0.0; target_reached = False
        if abs(freq_diff) > FLOAT_EPSILON: attunement_level = abs(tuning_amount) / abs(freq_diff)
        target_reached = abs(new_freq - target_freq) < HARMONY_FREQ_TUNING_TARGET_REACH_HZ # Use constant

        phase_metrics = {
            "original_frequency": float(current_freq), "target_frequency": float(target_freq), "new_frequency": float(new_freq),
            "frequency_shift": float(tuning_amount), "attunement_level": float(attunement_level), "target_reached": target_reached,
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('earth_harmony_frequency', phase_metrics)
        except Exception as e: logger.error(f"Failed to record frequency attunement metrics: {e}")

        logger.info(f"Frequency attunement complete. New Frequency: {new_freq:.2f} Hz (Level: {attunement_level:.2f}, TargetReached: {target_reached})")
        return phase_metrics

    except Exception as e: logger.error(f"Error during frequency attunement: {e}", exc_info=True); raise RuntimeError("Frequency attunement failed.") from e


def _perform_elemental_alignment(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Aligns elements towards Earth using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting elemental alignment for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    _ensure_soul_properties(soul_spark) # Ensures 'elements' dict exists

    try:
        soul_elements = getattr(soul_spark, "elements")
        try: malkuth_element = aspect_dictionary.get_aspects("malkuth").get("element", "earth").lower()
        except Exception: malkuth_element = "earth"

        alignment_results = {}
        logger.debug(f"  Initial elements: {soul_elements}")

        for element in EARTH_ELEMENTS: # Use constant list
            element_lower = element.lower()
            current_strength = soul_elements.get(element_lower, 0.0)
            # Use constants for targets
            target_strength = ELEMENTAL_TARGET_EARTH if element_lower == malkuth_element else ELEMENTAL_TARGET_OTHER

            # Use constant factor for adjustment
            adjustment = (target_strength - current_strength) * intensity * ELEMENTAL_ALIGN_INTENSITY_FACTOR
            new_strength = max(0.0, min(1.0, current_strength + adjustment)) # Clamp 0-1

            soul_elements[element_lower] = float(new_strength) # Update soul's dict

            alignment_level = 1.0 - abs(target_strength - new_strength) / max(FLOAT_EPSILON, target_strength)
            alignment_results[element_lower] = {"original": float(current_strength), "target": float(target_strength),
                                               "final": float(new_strength), "alignment": float(alignment_level)}

        overall_alignment = np.mean([r["alignment"] for r in alignment_results.values()]) if alignment_results else 0.0
        logger.debug(f"  Final elements: {soul_elements}")
        logger.debug(f"  Overall alignment: {overall_alignment:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "elemental_alignment", float(overall_alignment))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "overall_alignment": float(overall_alignment), "element_details": alignment_results,
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('earth_harmony_elements', phase_metrics)
        except Exception as e: logger.error(f"Failed to record elemental alignment metrics: {e}")

        logger.info(f"Elemental alignment complete. Overall Level: {overall_alignment:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during elemental alignment: {e}", exc_info=True); raise RuntimeError("Elemental alignment failed.") from e


def _perform_cycle_synchronization(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """Synchronizes soul with Earth cycles using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting cycle synchronization for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    _ensure_soul_properties(soul_spark) # Ensures 'earth_cycles' dict exists

    try:
        soul_cycles = getattr(soul_spark, "earth_cycles")
        sync_results = {}
        logger.debug(f"  Initial cycles sync: {soul_cycles}")

        # Use constants for cycles and importance
        for cycle_name in HARMONY_CYCLE_NAMES:
            if cycle_name not in EARTH_FREQUENCIES: logger.warning(f"Cycle '{cycle_name}' not in EARTH_FREQUENCIES. Skipping."); continue

            current_sync = soul_cycles.get(cycle_name, 0.3)
            target_sync = HARMONY_CYCLE_SYNC_TARGET_BASE * HARMONY_CYCLE_IMPORTANCE.get(cycle_name, 0.7)

            # Use constants for adjustment factors
            adjustment = (target_sync - current_sync) * intensity * duration_factor * HARMONY_CYCLE_SYNC_INTENSITY_FACTOR * HARMONY_CYCLE_SYNC_DURATION_FACTOR
            new_sync = max(0.0, min(1.0, current_sync + adjustment)) # Clamp 0-1

            soul_cycles[cycle_name] = float(new_sync) # Update soul's dict

            sync_level = 1.0 - abs(target_sync - new_sync) / max(FLOAT_EPSILON, target_sync)
            sync_results[cycle_name] = {"original": float(current_sync), "target": float(target_sync),
                                        "final": float(new_sync), "sync_level": float(sync_level)}

        overall_sync = np.mean([r["sync_level"] for r in sync_results.values()]) if sync_results else 0.0
        logger.debug(f"  Final cycles sync: {soul_cycles}")
        logger.debug(f"  Overall sync level: {overall_sync:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "cycle_synchronization", float(overall_sync))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = { "overall_sync": float(overall_sync), "cycle_details": sync_results, "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('earth_harmony_cycles', phase_metrics)
        except Exception as e: logger.error(f"Failed to record cycle synchronization metrics: {e}")

        logger.info(f"Cycle synchronization complete. Overall Level: {overall_sync:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during cycle synchronization: {e}", exc_info=True); raise RuntimeError("Cycle synchronization failed.") from e


def _perform_planetary_resonance(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Attunes soul to planetary resonance using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting planetary resonance for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    _ensure_soul_properties(soul_spark) # Ensures attr exists

    try:
        current_resonance = getattr(soul_spark, "planetary_resonance")
        logger.debug(f"  Initial Planetary Resonance: {current_resonance:.4f}")

        # Use constants
        target_resonance = HARMONY_PLANETARY_RESONANCE_TARGET
        adjustment = (target_resonance - current_resonance) * intensity * HARMONY_PLANETARY_RESONANCE_FACTOR
        new_resonance = max(0.0, min(1.0, current_resonance + adjustment))
        logger.debug(f"  Calculated Adjustment: {adjustment:.4f}, New Planetary Resonance: {new_resonance:.4f}")

        resonance_level = 1.0 - abs(target_resonance - new_resonance) / max(FLOAT_EPSILON, target_resonance)

        # --- Update SoulSpark ---
        setattr(soul_spark, "planetary_resonance", float(new_resonance))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "original_resonance": float(current_resonance), "target_resonance": float(target_resonance),
            "new_resonance": float(new_resonance), "resonance_level": float(resonance_level),
            "adjustment": float(adjustment), "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('earth_harmony_planetary', phase_metrics)
        except Exception as e: logger.error(f"Failed to record planetary resonance metrics: {e}")

        logger.info(f"Planetary resonance complete. New Level: {new_resonance:.4f} (Attainment: {resonance_level:.2f})")
        return phase_metrics

    except Exception as e: logger.error(f"Error during planetary resonance: {e}", exc_info=True); raise RuntimeError("Planetary resonance failed.") from e


def _perform_gaia_connection(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """Establishes Gaia connection using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting Gaia connection for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    _ensure_soul_properties(soul_spark) # Ensures attr exists

    try:
        current_connection = getattr(soul_spark, "gaia_connection")
        logger.debug(f"  Initial Gaia Connection: {current_connection:.4f}")

        # Use constants
        target_connection = HARMONY_GAIA_CONNECTION_TARGET
        adjustment = (target_connection - current_connection) * intensity * duration_factor * HARMONY_GAIA_CONNECTION_FACTOR
        new_connection = max(0.0, min(1.0, current_connection + adjustment))
        logger.debug(f"  Calculated Adjustment: {adjustment:.4f}, New Gaia Connection: {new_connection:.4f}")

        connection_level = 1.0 - abs(target_connection - new_connection) / max(FLOAT_EPSILON, target_connection)

        # --- Update SoulSpark ---
        setattr(soul_spark, "gaia_connection", float(new_connection))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "original_connection": float(current_connection), "target_connection": float(target_connection),
            "new_connection": float(new_connection), "connection_level": float(connection_level),
            "adjustment": float(adjustment), "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('earth_harmony_gaia', phase_metrics)
        except Exception as e: logger.error(f"Failed to record Gaia connection metrics: {e}")

        logger.info(f"Gaia connection complete. New Level: {new_connection:.4f} (Attainment: {connection_level:.2f})")
        return phase_metrics

    except Exception as e: logger.error(f"Error during Gaia connection: {e}", exc_info=True); raise RuntimeError("Gaia connection failed.") from e


def _update_soul_final_properties(soul_spark: SoulSpark, final_earth_resonance: float):
    """Updates soul with final harmonization properties using constants. Fails hard."""
    logger.info(f"Updating soul {soul_spark.spark_id} with final harmonization properties...")
    _ensure_soul_properties(soul_spark) # Ensure base attrs exist

    try:
        setattr(soul_spark, "earth_harmonized", True)
        setattr(soul_spark, "earth_resonance", float(final_earth_resonance))

        # Adjust stability and coherence using constants
        current_stability = getattr(soul_spark, "stability")
        current_coherence = getattr(soul_spark, "coherence")
        stability_bonus = final_earth_resonance * HARMONY_FINAL_STABILITY_BONUS
        coherence_bonus = final_earth_resonance * HARMONY_FINAL_COHERENCE_BONUS
        new_stability = min(1.0, current_stability + stability_bonus)
        new_coherence = min(1.0, current_coherence + coherence_bonus)

        setattr(soul_spark, "stability", float(new_stability))
        setattr(soul_spark, "coherence", float(new_coherence))
        setattr(soul_spark, "ready_for_birth", True) # Mark ready for next stage
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        logger.info(f"Soul updated. Final Earth Resonance: {final_earth_resonance:.4f}, New Stability: {new_stability:.4f}, New Coherence: {new_coherence:.4f}")
        # Log memory echo
        logger.info(f"Memory echo created: Earth harmonization complete for soul {soul_spark.spark_id}.")
        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
            soul_spark.memory_echoes.append(f"Earth harmonization complete @ {getattr(soul_spark, 'last_modified')}")


    except Exception as e:
        logger.error(f"Error updating final soul properties after harmonization: {e}", exc_info=True)
        raise RuntimeError("Failed to update final soul properties.") from e


# --- Orchestration Function ---

def perform_earth_harmonization(soul_spark: SoulSpark, intensity: float = 0.7, duration_factor: float = 1.0) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the complete Earth harmonization process. Modifies SoulSpark. Fails hard. Uses constants.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of harmonization (0.1-1.0).
        duration_factor (float): Relative duration multiplier (0.1-2.0).

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the harmonization process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If parameters invalid or prerequisites not met.
        RuntimeError: If any phase fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor must be between 0.1 and 2.0")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Earth Harmonization for Soul {spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        # --- Check Prerequisites (raises ValueError if failed) ---
        if not _check_prerequisites(soul_spark):
            # This path should not be reached if _check raises ValueError
            raise ValueError("Soul prerequisites for Earth harmonization not met.")
        logger.info("Prerequisites checked successfully.")

        # --- Store Initial State ---
        initial_resonance = _calculate_earth_resonance(soul_spark)
        initial_state = {
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'earth_resonance': initial_resonance,
            'elemental_alignment': getattr(soul_spark, 'elemental_alignment', 0.0),
            'cycle_synchronization': getattr(soul_spark, 'cycle_synchronization', 0.0),
            'planetary_resonance': getattr(soul_spark, 'planetary_resonance', 0.0),
            'gaia_connection': getattr(soul_spark, 'gaia_connection', 0.0) }
        logger.info(f"Initial State: EarthRes={initial_state['earth_resonance']:.4f}, Freq={initial_state['frequency']:.2f}, Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}")

        # --- Run Phases (Fail hard within each) ---
        logger.info("Step 1: Frequency Attunement...")
        metrics1 = _perform_frequency_attunement(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['frequency'] = metrics1

        logger.info("Step 2: Elemental Alignment...")
        metrics2 = _perform_elemental_alignment(soul_spark, intensity)
        process_metrics_summary['steps']['elements'] = metrics2

        logger.info("Step 3: Cycle Synchronization...")
        metrics3 = _perform_cycle_synchronization(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['cycles'] = metrics3

        logger.info("Step 4: Planetary Resonance...")
        metrics4 = _perform_planetary_resonance(soul_spark, intensity)
        process_metrics_summary['steps']['planetary'] = metrics4

        logger.info("Step 5: Gaia Connection...")
        metrics5 = _perform_gaia_connection(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['gaia'] = metrics5

        # --- Final Update & Metrics ---
        final_earth_resonance = _calculate_earth_resonance(soul_spark) # Recalculate final value
        _update_soul_final_properties(soul_spark, final_earth_resonance) # Apply final bonuses, flags

        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Capture relevant final state metrics
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'earth_resonance': final_earth_resonance,
            'elemental_alignment': getattr(soul_spark, 'elemental_alignment', 0.0),
            'cycle_synchronization': getattr(soul_spark, 'cycle_synchronization', 0.0),
            'planetary_resonance': getattr(soul_spark, 'planetary_resonance', 0.0),
            'gaia_connection': getattr(soul_spark, 'gaia_connection', 0.0),
            'earth_harmonized': getattr(soul_spark, 'earth_harmonized', False),
            'ready_for_birth': getattr(soul_spark, 'ready_for_birth', False) }

        overall_metrics = {
            'action': 'earth_harmonization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'intensity_setting': intensity, 'duration_factor': duration_factor,
            'initial_state': initial_state, 'final_state': final_state,
            'earth_resonance_change': final_earth_resonance - initial_resonance, 'success': True, }
        try: metrics.record_metrics('earth_harmonization_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for earth harmonization: {e}")

        logger.info(f"--- Earth Harmonization Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final Earth Resonance: {final_state['earth_resonance']:.4f} (Change: {overall_metrics['earth_resonance_change']:.4f})")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Earth harmonization process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"; steps_completed = list(process_metrics_summary['steps'].keys())
        if steps_completed: failed_step = steps_completed[-1]
        # Mark soul as failed this stage
        setattr(soul_spark, "earth_harmonized", False); setattr(soul_spark, "ready_for_birth", False)
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('earth_harmonization_summary', {
                  'action': 'earth_harmonization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'intensity_setting': intensity, 'duration_factor': duration_factor, 'success': False, 'error': str(e), 'failed_step': failed_step })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Earth harmonization process failed at step '{failed_step}'.") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Earth Harmonization Module Example (with Constants)...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        test_soul = SoulSpark()
        test_soul.spark_id="test_earthharm_const_002"
        # Set state *after* life cord formation
        test_soul.stability = 0.85
        test_soul.coherence = 0.88
        test_soul.frequency = 432.0
        test_soul.formation_complete = True
        test_soul.harmonically_strengthened = True
        test_soul.cord_formation_complete = True # Prerequisite
        test_soul.cord_integrity = 0.88 # Prerequisite (must meet CORD_INTEGRITY_THRESHOLD_EARTH)
        test_soul.elements = {'earth': 0.7, 'water': 0.6, 'aether': 0.5}
        test_soul.last_modified = datetime.now().isoformat()
        test_soul.memory_echoes = []

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Cord Integrity: {test_soul.cord_integrity:.4f}")
        print(f"  Cord Formed: {getattr(test_soul, 'cord_formation_complete', False)}")

        try:
            print("\n--- Running Earth Harmonization Process ---")
            modified_soul, summary_metrics_result = perform_earth_harmonization(
                soul_spark=test_soul,
                intensity=0.8,
                duration_factor=1.0
            )

            print("\n--- Harmonization Complete ---")
            print(f"  Earth Harmonized Flag: {getattr(modified_soul, 'earth_harmonized', False)}")
            print(f"  Earth Resonance: {getattr(modified_soul, 'earth_resonance', 'N/A'):.4f}")
            print(f"  Ready for Birth Flag: {getattr(modified_soul, 'ready_for_birth', False)}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Final Coherence: {getattr(modified_soul, 'coherence', 'N/A'):.4f}")
            print(f"  Final Frequency: {getattr(modified_soul, 'frequency', 'N/A'):.2f} Hz")
            print(f"  Final Elements: {getattr(modified_soul, 'elements', {})}")
            print(f"  Memory Echoes: {getattr(modified_soul, 'memory_echoes', [])}")

            print("\nOverall Process Metrics:")
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")
            print(f"  Earth Resonance Change: {summary_metrics_result.get('earth_resonance_change', 'N/A'):.4f}")

        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Earth Harmonization Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Earth Harmonization Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nEarth Harmonization Module Example Finished.")

# --- END OF FILE earth_harmonisation.py ---
