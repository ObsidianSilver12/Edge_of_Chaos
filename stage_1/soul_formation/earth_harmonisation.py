# --- START OF FILE src/stage_1/soul_formation/earth_harmonisation.py ---

"""
Earth Harmonization Functions (Refactored V4.1 - SEU/SU/CU Units)

Harmonizes soul with Earth frequencies/elements/cycles after Life Cord formation.
Uses absolute SU/CU prerequisites. Applies stability/coherence bonuses (SU/CU).
Earth resonance and related factors remain 0-1 scores. Frequency in Hz.
Modifies SoulSpark directly. Uses constants.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
    # Extract specific Earth freq if needed
    SCHUMANN_FREQUENCY = EARTH_FREQUENCIES.get("schumann", 7.83)
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    # Aspect dictionary optional for Malkuth element lookup, provide fallback
    try: from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
    except ImportError: aspect_dictionary = None
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.error("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds and cord integrity factor. """
    logger.debug(f"Checking Earth harmonization prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): return False

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_EARTH, False): # Set by Life Cord
        logger.error(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_EARTH}.")
        return False

    # 2. Minimum State Thresholds (Absolute SU/CU, Factor 0-1)
    cord_integrity = getattr(soul_spark, "cord_integrity", 0.0) # 0-1 factor
    stability_su = getattr(soul_spark, "stability", 0.0) # SU score
    coherence_cu = getattr(soul_spark, "coherence", 0.0) # CU score

    if cord_integrity < HARMONY_PREREQ_CORD_INTEGRITY_MIN:
        logger.error(f"Prerequisite failed: Cord integrity ({cord_integrity:.3f}) < {HARMONY_PREREQ_CORD_INTEGRITY_MIN}.")
        return False
    if stability_su < HARMONY_PREREQ_STABILITY_MIN_SU: # Use SU threshold
        logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {HARMONY_PREREQ_STABILITY_MIN_SU} SU.")
        return False
    if coherence_cu < HARMONY_PREREQ_COHERENCE_MIN_CU: # Use CU threshold
        logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {HARMONY_PREREQ_COHERENCE_MIN_CU} CU.")
        return False

    if getattr(soul_spark, FLAG_EARTH_HARMONIZED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_EARTH_HARMONIZED}. Re-running.")

    logger.debug("Earth harmonization prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. """
    logger.debug(f"Ensuring properties for earth harmonization (Soul {soul_spark.spark_id})...")
    # Check core attributes exist
    required = ['frequency', 'stability', 'coherence', 'elements', 'earth_cycles', 'planetary_resonance', 'gaia_connection', 'earth_resonance']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Earth Harmony: {missing}")

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.elements, dict): raise TypeError("elements must be dict.")
    if not isinstance(soul_spark.earth_cycles, dict): raise TypeError("earth_cycles must be dict.")
    logger.debug("Soul properties ensured for Earth Harmonization.")

# --- _calculate_earth_resonance (Logic mostly unchanged - calculates 0-1 score) ---
def _calculate_earth_resonance(soul_spark: SoulSpark) -> float:
    """ Calculates the soul's resonance score (0-1) with Earth. """
    logger.debug(f"Calculating Earth resonance for soul {soul_spark.spark_id}...")
    try:
        soul_freq = soul_spark.frequency # Hz
        soul_elements = getattr(soul_spark, "elements", {}) # 0-1 strengths

        # Frequency Component (Resonance with EARTH_FREQUENCIES)
        freq_resonance_score = 0.0
        total_weight = 0.0
        weights = {'schumann': HARMONY_FREQ_RES_WEIGHT_SCHUMANN, 'other': HARMONY_FREQ_RES_WEIGHT_OTHER}
        valid_earth_freqs = {k:v for k,v in EARTH_FREQUENCIES.items() if v > FLOAT_EPSILON}

        for name, earth_freq in valid_earth_freqs.items():
            # Calculate proximity resonance score (0-1)
            ratio = min(earth_freq / soul_freq, soul_freq / earth_freq) if min(earth_freq, soul_freq) > FLOAT_EPSILON else 0.0
            freq_prox_res = max(0.0, 1.0 - abs(1.0 - ratio))**2 # Simple proximity match score
            # Use detailed resonance calculation? Overkill here? Let's stick to simple proximity.
            # freq_detail_res = calculate_resonance(soul_freq, earth_freq) # Using detailed helper
            # Use weighted average of proximity and detailed? Let's just use proximity for simplicity here.

            weight = weights.get(name, weights['other']) # Use specific weight or 'other' weight
            freq_resonance_score += freq_prox_res * weight
            total_weight += weight

        if total_weight > FLOAT_EPSILON: freq_resonance_score /= total_weight
        else: freq_resonance_score = 0.0

        # Elemental Component (Match with EARTH_ELEMENTS, weighted towards primary Earth element)
        elem_resonance_score = 0.0
        malkuth_element = "earth" # Default
        if aspect_dictionary:
             try: malkuth_element = aspect_dictionary.get_aspects("malkuth").get("element", "earth").lower()
             except Exception: pass # Use default if lookup fails

        total_strength = 0.0; earth_strength = 0.0; valid_elements_count = 0
        for elem in EARTH_ELEMENTS:
             strength = float(soul_elements.get(elem.lower(), 0.0)) # Strength 0-1
             if np.isfinite(strength) and 0.0 <= strength <= 1.0:
                  total_strength += strength
                  if elem.lower() == malkuth_element: earth_strength = strength
                  valid_elements_count += 1

        if valid_elements_count > 0:
             avg_strength = total_strength / valid_elements_count
             elem_resonance_score = (HARMONY_ELEM_RES_WEIGHT_PRIMARY * earth_strength +
                                     HARMONY_ELEM_RES_WEIGHT_AVERAGE * avg_strength)
        elem_resonance_score = max(0.0, min(1.0, elem_resonance_score))

        # Combine Frequency and Elemental Scores (Weighted average)
        overall_resonance = freq_resonance_score * 0.6 + elem_resonance_score * 0.4
        overall_resonance = max(0.0, min(1.0, overall_resonance))
        logger.debug(f"  Overall Earth Resonance score calculated: {overall_resonance:.4f}")
        return float(overall_resonance)

    except Exception as e: logger.error(f"Error calculating Earth resonance: {e}", exc_info=True); return 0.0


# --- Core Harmonization Functions (Modify Hz, SU/CU, or 0-1 factors) ---

def _perform_frequency_attunement(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """ Attunes frequency (Hz) towards Earth targets. Modifies SoulSpark. """
    logger.info(f"Starting frequency attunement (Int={intensity:.2f}, DurF={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    try:
        current_freq = soul_spark.frequency # Hz
        # Target is weighted average of Schumann, Earth Core, and Soul's current freq
        target_freq = (SCHUMANN_FREQUENCY * HARMONY_FREQ_TARGET_SCHUMANN_WEIGHT +
                       EARTH_FREQUENCY * HARMONY_FREQ_TARGET_CORE_WEIGHT + # Added core weight
                       current_freq * HARMONY_FREQ_TARGET_SOUL_WEIGHT)
        freq_diff = target_freq - current_freq
        tuning_amount = freq_diff * intensity * HARMONY_FREQ_TUNING_FACTOR * duration_factor # Hz change
        new_freq = current_freq + tuning_amount
        new_freq = max(SCHUMANN_FREQUENCY * 0.5, new_freq) # Clamp lower bound (Hz)

        soul_spark.frequency = float(new_freq)
        soul_spark.generate_harmonic_structure() # Update signature based on new freq
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        attunement_level = abs(tuning_amount) / max(FLOAT_EPSILON, abs(freq_diff)) if abs(freq_diff) > FLOAT_EPSILON else 1.0
        target_reached = abs(new_freq - target_freq) < HARMONY_FREQ_TUNING_TARGET_REACH_HZ
        phase_metrics = { "new_frequency_hz": new_freq, "attunement_level": attunement_level, "target_reached": target_reached, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmony_frequency', phase_metrics)
        logger.info(f"Frequency attunement complete. New Freq: {new_freq:.2f} Hz")
        return phase_metrics
    except Exception as e: logger.error(f"Error frequency attunement: {e}", exc_info=True); raise RuntimeError("Freq attunement failed.") from e

def _perform_elemental_alignment(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Aligns elements (0-1 factors) towards Earth targets. Modifies SoulSpark. """
    logger.info(f"Starting elemental alignment (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    try:
        soul_elements = getattr(soul_spark, "elements", {})
        malkuth_element = "earth"
        if aspect_dictionary:
             try: malkuth_element = aspect_dictionary.get_aspects("malkuth").get("element", "earth").lower()
             except Exception: pass

        alignment_results = {}
        for element in EARTH_ELEMENTS:
            element_lower = element.lower()
            current_strength = float(soul_elements.get(element_lower, 0.0)) # 0-1 factor
            target_strength = ELEMENTAL_TARGET_EARTH if element_lower == malkuth_element else ELEMENTAL_TARGET_OTHER # Target 0-1 factor
            adjustment = (target_strength - current_strength) * intensity * ELEMENTAL_ALIGN_INTENSITY_FACTOR # Delta for 0-1 factor
            new_strength = max(0.0, min(1.0, current_strength + adjustment))
            soul_elements[element_lower] = float(new_strength) # Update soul's dict
            alignment_level = 1.0 - abs(target_strength - new_strength) / max(FLOAT_EPSILON, target_strength) if target_strength > FLOAT_EPSILON else (1.0 if abs(new_strength)<FLOAT_EPSILON else 0.0)
            alignment_results[element_lower] = {"final": new_strength, "alignment_level": alignment_level}

        overall_alignment = np.mean([r["alignment_level"] for r in alignment_results.values()]) if alignment_results else 0.0
        setattr(soul_spark, "elemental_alignment", float(overall_alignment)) # Store overall 0-1 score
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        phase_metrics = { "overall_alignment_score": overall_alignment, "element_details": alignment_results, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmony_elements', phase_metrics)
        logger.info(f"Elemental alignment complete. Overall Score: {overall_alignment:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error elemental alignment: {e}", exc_info=True); raise RuntimeError("Elemental alignment failed.") from e

def _perform_cycle_synchronization(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """ Synchronizes soul cycle factors (0-1) with Earth cycles. Modifies SoulSpark. """
    logger.info(f"Starting cycle synchronization (Int={intensity:.2f}, DurF={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    try:
        soul_cycles = getattr(soul_spark, "earth_cycles", {})
        sync_results = {}
        for cycle_name in HARMONY_CYCLE_NAMES:
            current_sync = float(soul_cycles.get(cycle_name, 0.3)) # 0-1 factor
            target_sync = HARMONY_CYCLE_SYNC_TARGET_BASE * HARMONY_CYCLE_IMPORTANCE.get(cycle_name, 0.7) # Target 0-1 factor
            adjustment = (target_sync - current_sync) * intensity * duration_factor * HARMONY_CYCLE_SYNC_INTENSITY_FACTOR * HARMONY_CYCLE_SYNC_DURATION_FACTOR # Delta for 0-1 factor
            new_sync = max(0.0, min(1.0, current_sync + adjustment))
            soul_cycles[cycle_name] = float(new_sync)
            sync_level = 1.0 - abs(target_sync - new_sync) / max(FLOAT_EPSILON, target_sync) if target_sync > FLOAT_EPSILON else (1.0 if abs(new_sync)<FLOAT_EPSILON else 0.0)
            sync_results[cycle_name] = {"final": new_sync, "sync_level": sync_level}

        overall_sync = np.mean([r["sync_level"] for r in sync_results.values()]) if sync_results else 0.0
        setattr(soul_spark, "cycle_synchronization", float(overall_sync)) # Store overall 0-1 score
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        phase_metrics = { "overall_sync_score": overall_sync, "cycle_details": sync_results, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmony_cycles', phase_metrics)
        logger.info(f"Cycle synchronization complete. Overall Score: {overall_sync:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error cycle synchronization: {e}", exc_info=True); raise RuntimeError("Cycle sync failed.") from e

def _perform_planetary_resonance(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Attunes soul planetary resonance factor (0-1). Modifies SoulSpark. """
    logger.info(f"Starting planetary resonance (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    try:
        current_resonance = getattr(soul_spark, "planetary_resonance", 0.0) # 0-1 factor
        target_resonance = HARMONY_PLANETARY_RESONANCE_TARGET # Target 0-1 factor
        adjustment = (target_resonance - current_resonance) * intensity * HARMONY_PLANETARY_RESONANCE_FACTOR # Delta for 0-1 factor
        new_resonance = max(0.0, min(1.0, current_resonance + adjustment))

        setattr(soul_spark, "planetary_resonance", float(new_resonance))
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        resonance_level = 1.0 - abs(target_resonance - new_resonance) / max(FLOAT_EPSILON, target_resonance) if target_resonance > FLOAT_EPSILON else (1.0 if abs(new_resonance)<FLOAT_EPSILON else 0.0)
        phase_metrics = { "new_resonance_factor": new_resonance, "resonance_level": resonance_level, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmony_planetary', phase_metrics)
        logger.info(f"Planetary resonance complete. New Factor: {new_resonance:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error planetary resonance: {e}", exc_info=True); raise RuntimeError("Planetary resonance failed.") from e

def _perform_gaia_connection(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """ Establishes Gaia connection factor (0-1). Modifies SoulSpark. """
    logger.info(f"Starting Gaia connection (Int={intensity:.2f}, DurF={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    try:
        current_connection = getattr(soul_spark, "gaia_connection", 0.0) # 0-1 factor
        target_connection = HARMONY_GAIA_CONNECTION_TARGET # Target 0-1 factor
        adjustment = (target_connection - current_connection) * intensity * duration_factor * HARMONY_GAIA_CONNECTION_FACTOR # Delta for 0-1 factor
        new_connection = max(0.0, min(1.0, current_connection + adjustment))

        setattr(soul_spark, "gaia_connection", float(new_connection))
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        connection_level = 1.0 - abs(target_connection - new_connection) / max(FLOAT_EPSILON, target_connection) if target_connection > FLOAT_EPSILON else (1.0 if abs(new_connection)<FLOAT_EPSILON else 0.0)
        phase_metrics = { "new_connection_factor": new_connection, "connection_level": connection_level, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmony_gaia', phase_metrics)
        logger.info(f"Gaia connection complete. New Factor: {new_connection:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error Gaia connection: {e}", exc_info=True); raise RuntimeError("Gaia connection failed.") from e

def _update_soul_final_properties(soul_spark: SoulSpark, final_earth_resonance: float):
    """ Updates final properties. Applies stability/coherence bonus (SU/CU). """
    logger.info(f"Updating soul {soul_spark.spark_id} final harmonization properties...")
    try:
        setattr(soul_spark, FLAG_EARTH_HARMONIZED, True)
        setattr(soul_spark, "earth_resonance", float(final_earth_resonance)) # Final 0-1 score

        current_stability_su = soul_spark.stability
        current_coherence_cu = soul_spark.coherence

        # Calculate absolute SU/CU bonus based on 0-1 earth resonance score
        stability_bonus_su = final_earth_resonance * HARMONY_FINAL_STABILITY_BONUS * MAX_STABILITY_SU
        coherence_bonus_cu = final_earth_resonance * HARMONY_FINAL_COHERENCE_BONUS * MAX_COHERENCE_CU
        new_stability_su = min(MAX_STABILITY_SU, current_stability_su + stability_bonus_su)
        new_coherence_cu = min(MAX_COHERENCE_CU, current_coherence_cu + coherence_bonus_cu)

        setattr(soul_spark, "stability", float(new_stability_su))
        setattr(soul_spark, "coherence", float(new_coherence_cu))
        setattr(soul_spark, FLAG_READY_FOR_IDENTITY, True)
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        if hasattr(soul_spark, 'add_memory_echo'):
            echo_msg = f"Earth harmonization complete. EarthRes:{final_earth_resonance:.3f}, Stab:{new_stability_su:.1f}, Coh:{new_coherence_cu:.1f}"
            soul_spark.add_memory_echo(echo_msg)
            logger.info(f"Memory echo created: {echo_msg}")
        logger.info(f"Soul updated post-harmonization. Ready for Identity.")

    except Exception as e: logger.error(f"Error updating final properties: {e}", exc_info=True); raise RuntimeError("Final property update failed.") from e


# --- Orchestration Function ---
def perform_earth_harmonization(soul_spark: SoulSpark, intensity: float = EARTH_HARMONY_INTENSITY_DEFAULT, duration_factor: float = EARTH_HARMONY_DURATION_FACTOR_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete Earth harmonization. Modifies SoulSpark. Uses SU/CU. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Earth Harmonization for Soul {spark_id} (Int={intensity:.2f}, DurF={duration_factor:.2f}) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark): # Uses SU/CU thresholds
            raise ValueError("Soul prerequisites for Earth harmonization not met.")

        # Store Initial State (Absolute SU/CU and Factors)
        initial_resonance = _calculate_earth_resonance(soul_spark) # Initial 0-1 score
        initial_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'frequency_hz': soul_spark.frequency, 'earth_resonance': initial_resonance,
            'elemental_alignment': soul_spark.elemental_alignment, 'cycle_synchronization': soul_spark.cycle_synchronization,
            'planetary_resonance': soul_spark.planetary_resonance, 'gaia_connection': soul_spark.gaia_connection
        }
        logger.info(f"Initial State: S={initial_state['stability_su']:.1f}SU, C={initial_state['coherence_cu']:.1f}CU, EarthRes={initial_state['earth_resonance']:.3f}")

        # Run Phases
        metrics1 = _perform_frequency_attunement(soul_spark, intensity, duration_factor)
        metrics2 = _perform_elemental_alignment(soul_spark, intensity)
        metrics3 = _perform_cycle_synchronization(soul_spark, intensity, duration_factor)
        metrics4 = _perform_planetary_resonance(soul_spark, intensity)
        metrics5 = _perform_gaia_connection(soul_spark, intensity, duration_factor)
        # Store step metrics if needed

        # Final Update & Metrics
        final_earth_resonance = _calculate_earth_resonance(soul_spark) # Recalculate final 0-1 score
        _update_soul_final_properties(soul_spark, final_earth_resonance)
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Update derived S/C scores

        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report final state in correct units/scores
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'frequency_hz': soul_spark.frequency, 'earth_resonance': final_earth_resonance,
            'elemental_alignment': soul_spark.elemental_alignment, 'cycle_synchronization': soul_spark.cycle_synchronization,
            'planetary_resonance': soul_spark.planetary_resonance, 'gaia_connection': soul_spark.gaia_connection,
            FLAG_EARTH_HARMONIZED: getattr(soul_spark, FLAG_EARTH_HARMONIZED),
            FLAG_READY_FOR_IDENTITY: getattr(soul_spark, FLAG_READY_FOR_IDENTITY) }
        overall_metrics = {
            'action': 'earth_harmonization', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'intensity_setting': intensity, 'duration_factor': duration_factor,
            'initial_state': initial_state, 'final_state': final_state,
            'earth_resonance_change': final_earth_resonance - initial_resonance, # Change in 0-1 score
            'stability_gain_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_gain_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('earth_harmonization_summary', overall_metrics)

        logger.info(f"--- Earth Harmonization Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Earth Harmonization failed for {spark_id} due to validation error: {e_val}", exc_info=True)
         record_eh_failure(spark_id, start_time_iso, 'prerequisites/validation', str(e_val))
         raise
    except RuntimeError as e_rt:
         logger.critical(f"Earth Harmonization failed critically for {spark_id}: {e_rt}", exc_info=True)
         record_eh_failure(spark_id, start_time_iso, 'runtime', str(e_rt))
         setattr(soul_spark, FLAG_EARTH_HARMONIZED, False); setattr(soul_spark, FLAG_READY_FOR_IDENTITY, False)
         raise
    except Exception as e:
         logger.critical(f"Unexpected error during Earth Harmonization for {spark_id}: {e}", exc_info=True)
         record_eh_failure(spark_id, start_time_iso, 'unexpected', str(e))
         setattr(soul_spark, FLAG_EARTH_HARMONIZED, False); setattr(soul_spark, FLAG_READY_FOR_IDENTITY, False)
         raise RuntimeError(f"Unexpected Earth Harmonization failure: {e}") from e

# --- Failure Metric Helper ---
def record_eh_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('earth_harmonization_summary', {
                'action': 'earth_harmonization', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record EH failure metrics for {spark_id}: {metric_e}")


# --- END OF FILE src/stage_1/soul_formation/earth_harmonisation.py ---
