# --- START OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---

"""
Harmonic Strengthening Functions (Refactored V4.1 - SEU/SU/CU Units)

Enhances soul stability (SU), coherence (CU) after Creator Entanglement
by applying direct deltas based on harmonic/phi resonance principles.
Modifies the SoulSpark object instance directly. Uses constants.
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
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}.")
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
    """ Checks prerequisites using SU/CU thresholds. """
    logger.debug(f"Checking prerequisites for harmonic strengthening (Soul: {soul_spark.spark_id})...")
    if not isinstance(soul_spark, SoulSpark): return False

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, False): # Set by entanglement
        logger.error(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_STRENGTHENING}.")
        return False

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', 0.0)
    coherence_cu = getattr(soul_spark, 'coherence', 0.0)
    # Use specific HS prerequisite constants
    if stability_su < HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU:
        logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU} SU.")
        return False
    if coherence_cu < HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU:
        logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU} CU.")
        return False

    if getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_HARMONICALLY_STRENGTHENED}. Re-running.")

    logger.debug("Harmonic strengthening prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary numeric properties. """
    logger.debug(f"Ensuring properties for harmonic strengthening (Soul {soul_spark.spark_id})...")
    # Check core attributes exist (should be guaranteed by init)
    required = ['frequency', 'stability', 'coherence', 'harmonics', 'phi_resonance', 'pattern_coherence', 'harmony', 'aspects']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for HS: {missing}")

    # Initialize field radius/strength if missing (treat as 0-1 factor for now)
    if not hasattr(soul_spark, 'field_radius'): setattr(soul_spark, 'field_radius', 1.0) # Example default size factor
    if not hasattr(soul_spark, 'field_strength'): setattr(soul_spark, 'field_strength', 0.5) # Example default strength factor

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    logger.debug("Soul properties ensured for harmonic strengthening.")


# --- Core Strengthening Functions (Applying Deltas to SU/CU) ---

def _perform_frequency_tuning(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Tunes frequency towards target harmonics. Modifies SoulSpark frequency (Hz). """
    logger.info(f"Starting frequency tuning (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        current_freq = soul_spark.frequency
        target_freq = min(HARMONIC_STRENGTHENING_TARGET_FREQS, key=lambda f: abs(f - current_freq))
        freq_diff = target_freq - current_freq
        tuning_amount = freq_diff * intensity * HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR
        new_freq = max(FLOAT_EPSILON * 10, current_freq + tuning_amount) # Ensure positive Hz

        soul_spark.frequency = float(new_freq)
        # Regenerate harmonics and frequency signature based on new frequency
        soul_spark.generate_harmonic_structure() # This updates self.harmonics and self.frequency_signature
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        # Calculate Metrics
        attunement_level = abs(tuning_amount) / max(FLOAT_EPSILON, abs(freq_diff)) if abs(freq_diff) > FLOAT_EPSILON else 1.0
        target_reached = abs(new_freq - target_freq) < HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ
        phase_metrics = {
            "original_frequency_hz": float(current_freq), "target_frequency_hz": float(target_freq), "new_frequency_hz": float(new_freq),
            "frequency_shift_hz": float(tuning_amount), "attunement_level": float(attunement_level), "target_reached": target_reached,
            "timestamp": timestamp
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_tuning', phase_metrics)
        logger.info(f"Frequency tuning complete. New Freq: {new_freq:.2f} Hz")
        return phase_metrics

    except Exception as e: logger.error(f"Error frequency tuning: {e}", exc_info=True); raise RuntimeError("Freq tuning failed.") from e


def _perform_phi_resonance_amplification(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """ Amplifies phi resonance factor (0-1) and boosts stability (SU). """
    logger.info(f"Starting phi resonance amplification (Int={intensity:.2f}, DurF={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

    try:
        current_phi_resonance = soul_spark.phi_resonance # 0-1 factor
        current_stability_su = soul_spark.stability # SU score

        phi_increase_factor = HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR * intensity * duration_factor
        new_phi_resonance = min(1.0, current_phi_resonance + phi_increase_factor)

        # Stability boost is proportional to phi increase and max potential SU gain
        stability_boost_su = phi_increase_factor * HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR * MAX_STABILITY_SU
        new_stability_su = min(MAX_STABILITY_SU, current_stability_su + stability_boost_su)
        actual_stability_gain = new_stability_su - current_stability_su

        soul_spark.phi_resonance = float(new_phi_resonance)
        soul_spark.stability = float(new_stability_su)
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        # Calculate Metrics
        phi_change = new_phi_resonance - current_phi_resonance
        phase_metrics = {
            "original_phi_resonance": float(current_phi_resonance), "new_phi_resonance": float(new_phi_resonance),
            "phi_resonance_change": float(phi_change),
            "initial_stability_su": float(current_stability_su), "stability_boost_su": float(stability_boost_su),
            "final_stability_su": float(new_stability_su), "actual_stability_gain_su": float(actual_stability_gain),
            "timestamp": timestamp
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_phi', phase_metrics)
        logger.info(f"Phi resonance amplified. New PhiRes: {new_phi_resonance:.4f}, Stability: {new_stability_su:.1f} SU")
        return phase_metrics

    except Exception as e: logger.error(f"Error phi resonance amplification: {e}", exc_info=True); raise RuntimeError("Phi resonance amp failed.") from e


def _perform_pattern_stabilization(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Stabilizes patterns (boosts pattern coherence factor 0-1) and stability (SU). """
    logger.info(f"Starting pattern stabilization (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        current_stability_su = soul_spark.stability # SU score
        current_pattern_coherence = soul_spark.pattern_coherence # 0-1 factor
        aspect_count = len(getattr(soul_spark, 'aspects', {}))

        # Calculate increase factor for pattern coherence (0-1)
        aspect_influence = min(HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP, aspect_count * HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR)
        base_increase = HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR * intensity
        total_increase_factor = base_increase + aspect_influence * intensity
        new_pattern_coherence = min(1.0, current_pattern_coherence + total_increase_factor)
        actual_pcoh_gain = new_pattern_coherence - current_pattern_coherence

        # Stability boost (SU) based on pattern coherence improvement
        stability_boost_su = actual_pcoh_gain * HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST * MAX_STABILITY_SU # Scale by max SU
        new_stability_su = min(MAX_STABILITY_SU, current_stability_su + stability_boost_su)
        actual_stability_gain = new_stability_su - current_stability_su

        soul_spark.pattern_coherence = float(new_pattern_coherence)
        soul_spark.stability = float(new_stability_su)
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        # Calculate Metrics
        phase_metrics = {
            "original_pattern_coherence": float(current_pattern_coherence), "new_pattern_coherence": float(new_pattern_coherence),
            "pattern_coherence_change": float(actual_pcoh_gain),
            "initial_stability_su": float(current_stability_su), "stability_boost_su": float(stability_boost_su),
            "final_stability_su": float(new_stability_su), "actual_stability_gain_su": float(actual_stability_gain),
            "aspect_influence": float(aspect_influence), "timestamp": timestamp
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_pattern', phase_metrics)
        logger.info(f"Pattern stabilization complete. New P.Coh: {new_pattern_coherence:.4f}, Stability: {new_stability_su:.1f} SU")
        return phase_metrics

    except Exception as e: logger.error(f"Error pattern stabilization: {e}", exc_info=True); raise RuntimeError("Pattern stabilization failed.") from e

def _perform_coherence_enhancement(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """ Enhances overall coherence (CU score) and harmony (0-1 factor). """
    logger.info(f"Starting coherence enhancement (Int={intensity:.2f}, DurF={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

    try:
        current_coherence_cu = soul_spark.coherence # CU score
        current_harmony = soul_spark.harmony # 0-1 factor
        harmonics_list = getattr(soul_spark, 'harmonics', [])

        # Coherence boost based on intensity, duration, and harmonic complexity
        harmonic_factor = min(1.0, len(harmonics_list) / HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM)
        base_increase = HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR * intensity * duration_factor
        harmonic_bonus = harmonic_factor * intensity * HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR
        total_coherence_boost_cu = (base_increase + harmonic_bonus) * MAX_COHERENCE_CU # Scale boost by max CU
        new_coherence_cu = min(MAX_COHERENCE_CU, current_coherence_cu + total_coherence_boost_cu)
        actual_coherence_gain = new_coherence_cu - current_coherence_cu

        # Harmony factor (0-1) boost based on coherence gain
        harmony_increase_factor = (actual_coherence_gain / MAX_COHERENCE_CU) * HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST
        new_harmony = min(1.0, current_harmony + harmony_increase_factor)
        actual_harmony_gain = new_harmony - current_harmony

        soul_spark.coherence = float(new_coherence_cu)
        soul_spark.harmony = float(new_harmony)
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        # Calculate Metrics
        phase_metrics = {
            "original_coherence_cu": float(current_coherence_cu), "new_coherence_cu": float(new_coherence_cu),
            "coherence_boost_cu": float(total_coherence_boost_cu), "actual_coherence_gain_cu": float(actual_coherence_gain),
            "original_harmony": float(current_harmony), "new_harmony": float(new_harmony),
            "harmony_gain": float(actual_harmony_gain), "harmonic_factor": float(harmonic_factor),
            "timestamp": timestamp
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_coherence', phase_metrics)
        logger.info(f"Coherence enhancement complete. New Coherence: {new_coherence_cu:.1f} CU, Harmony: {new_harmony:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error coherence enhancement: {e}", exc_info=True); raise RuntimeError("Coherence enhancement failed.") from e


def _perform_field_expansion(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """ Expands resonance field radius (absolute units?) and strength (0-1 factor). """
    logger.info(f"Starting field expansion (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")

    try:
        # Ensure field attributes exist
        if not hasattr(soul_spark, 'field_radius'): setattr(soul_spark, 'field_radius', 1.0) # Default relative size?
        if not hasattr(soul_spark, 'field_strength'): setattr(soul_spark, 'field_strength', 0.5) # Default 0-1 factor
        current_radius = float(soul_spark.field_radius) # Assuming absolute units now? Let's keep relative for now.
        current_strength = float(soul_spark.field_strength) # 0-1 factor
        # Use normalized stability/coherence scores
        coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU
        stability_norm = soul_spark.stability / MAX_STABILITY_SU

        expand_factor = ((coherence_norm + stability_norm) / 2.0) * HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR
        # Increase radius additively (or multiplicatively?)
        radius_increase = HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR * intensity * expand_factor # Additive increase?
        strength_increase_factor = HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR * intensity * expand_factor * HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR

        new_radius = current_radius + radius_increase # Additive increase
        new_strength = min(1.0, current_strength + strength_increase_factor) # Additive increase to 0-1 factor

        soul_spark.field_radius = float(new_radius)
        soul_spark.field_strength = float(new_strength)
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        # Calculate Metrics
        phase_metrics = {
            "original_radius": float(current_radius), "new_radius": float(new_radius), "radius_change": float(radius_increase),
            "original_strength": float(current_strength), "new_strength": float(new_strength),
            "strength_change": float(new_strength - current_strength), "timestamp": timestamp
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_expansion', phase_metrics)
        logger.info(f"Field expansion complete. New Radius: {new_radius:.4f}, Strength: {new_strength:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error field expansion: {e}", exc_info=True); raise RuntimeError("Field expansion failed.") from e


# --- Orchestration Function ---
def perform_harmonic_strengthening(soul_spark: SoulSpark, intensity: float = HARMONIC_STRENGTHENING_INTENSITY_DEFAULT, duration_factor: float = HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete harmonic strengthening. Modifies SoulSpark. Uses SU/CU units. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Harmonic Strengthening for Soul {spark_id} (Int={intensity:.2f}, DurF={duration_factor:.2f}) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark): # Uses SU/CU thresholds
            raise ValueError("Soul prerequisites for harmonic strengthening not met.")

        # Store Initial State (Absolute SU/CU and Factors)
        initial_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'frequency_hz': soul_spark.frequency, 'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence, 'harmony': soul_spark.harmony,
            'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength
        }
        logger.info(f"Initial State: S={initial_state['stability_su']:.1f}SU, C={initial_state['coherence_cu']:.1f}CU, Freq={initial_state['frequency_hz']:.1f}Hz")

        # Run Phases
        metrics1 = _perform_frequency_tuning(soul_spark, intensity)
        metrics2 = _perform_phi_resonance_amplification(soul_spark, intensity, duration_factor)
        metrics3 = _perform_pattern_stabilization(soul_spark, intensity)
        metrics4 = _perform_coherence_enhancement(soul_spark, intensity, duration_factor)
        metrics5 = _perform_field_expansion(soul_spark, intensity)
        # Store step metrics if needed: process_metrics_summary['steps']['tuning'] = metrics1 ...

        # Finalize
        setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, True)
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Final update of S/C scores
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Harmonically strengthened. S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")

        # Compile Overall Metrics
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report final state in correct units
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'frequency_hz': soul_spark.frequency, 'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence, 'harmony': soul_spark.harmony,
            'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength,
            FLAG_HARMONICALLY_STRENGTHENED: getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED)
        }
        # Calculate overall improvement based on Stability/Coherence SU/CU gain
        initial_avg_score = (initial_state['stability_su'] / MAX_STABILITY_SU + initial_state['coherence_cu'] / MAX_COHERENCE_CU) / 2.0
        final_avg_score = (final_state['stability_su'] / MAX_STABILITY_SU + final_state['coherence_cu'] / MAX_COHERENCE_CU) / 2.0
        improvement_pct = ((final_avg_score - initial_avg_score) / max(FLOAT_EPSILON, initial_avg_score)) * 100.0 if initial_avg_score > FLOAT_EPSILON else (100.0 if final_avg_score > 0 else 0.0)

        overall_metrics = {
            'action': 'harmonic_strengthening', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'intensity_setting': intensity, 'duration_factor': duration_factor,
            'initial_state': initial_state, 'final_state': final_state,
            'stability_gain_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_gain_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'overall_improvement_pct': float(improvement_pct), 'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_summary', overall_metrics)

        logger.info(f"--- Harmonic Strengthening Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val: # Catch setup/attribute errors
         logger.error(f"Harmonic Strengthening failed for {spark_id} due to validation error: {e_val}", exc_info=True)
         record_hs_failure(spark_id, start_time_iso, 'prerequisites/validation', str(e_val))
         raise
    except RuntimeError as e_rt: # Catch errors from sub-functions
         logger.critical(f"Harmonic Strengthening failed critically for {spark_id}: {e_rt}", exc_info=True)
         # Determine failed step based on which metrics were recorded? More complex. Assume 'runtime'.
         record_hs_failure(spark_id, start_time_iso, 'runtime', str(e_rt))
         # Reset flags?
         setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
         setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
         raise
    except Exception as e: # Catch unexpected errors
         logger.critical(f"Unexpected error during Harmonic Strengthening for {spark_id}: {e}", exc_info=True)
         record_hs_failure(spark_id, start_time_iso, 'unexpected', str(e))
         setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
         setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
         raise RuntimeError(f"Unexpected Harmonic Strengthening failure: {e}") from e

# --- Failure Metric Helper ---
def record_hs_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('harmonic_strengthening_summary', {
                'action': 'harmonic_strengthening', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record HS failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---