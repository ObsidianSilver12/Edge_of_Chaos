# --- START OF FILE harmonic_strengthening.py ---

"""
Harmonic Strengthening Functions (Refactored - Operates on SoulSpark Object, Uses Constants)

Provides functions for enhancing soul stability, coherence, and resonance
through harmonic frequencies and phi-resonance amplification. Modifies the
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

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    # Import necessary constants FROM THE CENTRAL FILE
    from src.constants import * # Import all for convenience
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. Harmonic Strengthening cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.void.soul_spark import SoulSpark
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}. Harmonic Strengthening cannot function.")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder: def record_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """Checks if the soul meets prerequisites for harmonic strengthening."""
    logger.debug(f"Checking prerequisites for soul {soul_spark.spark_id}...")
    # Requires formation completion (from Sephiroth journey or Guff stage depending on flow)
    # Let's assume a flag like 'sephiroth_journey_complete' exists or check 'formation_complete'
    if not getattr(soul_spark, "sephiroth_journey_complete", getattr(soul_spark, "formation_complete", False)):
        logger.error("Prerequisite failed: Soul formation/journey not complete.")
        return False

    # Requires minimum stability and coherence using constants
    stability = getattr(soul_spark, "stability", 0.0)
    if stability < HARMONIC_STRENGTHENING_PREREQ_STABILITY:
        logger.error(f"Prerequisite failed: Soul stability ({stability:.3f}) below threshold ({HARMONIC_STRENGTHENING_PREREQ_STABILITY}).")
        return False
    coherence = getattr(soul_spark, "coherence", 0.0)
    if coherence < HARMONIC_STRENGTHENING_PREREQ_COHERENCE:
        logger.error(f"Prerequisite failed: Soul coherence ({coherence:.3f}) below threshold ({HARMONIC_STRENGTHENING_PREREQ_COHERENCE}).")
        return False

    logger.debug("Harmonic strengthening prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary numeric properties, initializing if missing. Fails hard."""
    # This helper is important for robustness before calculations
    logger.debug(f"Ensuring properties exist for soul {soul_spark.spark_id}...")
    attributes_to_check = [
        ("frequency", SOUL_SPARK_DEFAULT_FREQ),
        ("stability", SOUL_SPARK_DEFAULT_STABILITY),
        ("coherence", SOUL_SPARK_DEFAULT_COHERENCE),
        ("harmonics", []), # Default to empty list, populated later if needed
        ("phi_resonance", 0.5),
        ("pattern_stability", 0.5),
        ("harmony", 0.5),
        ("field_radius", 3.0),
        ("field_strength", 0.5)
    ]
    for attr, default in attributes_to_check:
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
            logger.warning(f"Soul {soul_spark.spark_id} missing '{attr}'. Initializing to default: {default}")
            setattr(soul_spark, attr, default)

    # Ensure harmonics list is populated if frequency exists but harmonics doesn't
    if hasattr(soul_spark, 'frequency') and getattr(soul_spark, 'frequency') > FLOAT_EPSILON and (not hasattr(soul_spark, 'harmonics') or not soul_spark.harmonics):
        logger.warning(f"Soul {soul_spark.spark_id} missing 'harmonics'. Initializing based on frequency.")
        h_count = getattr(soul_spark, 'harmonic_count', HARMONIC_STRENGTHENING_HARMONIC_COUNT) # Use soul's count or constant
        setattr(soul_spark, 'harmonics', [soul_spark.frequency * n for n in range(1, h_count + 1)])

    # Validate numerical types and ranges
    for attr, _ in attributes_to_check:
        val = getattr(soul_spark, attr)
        if attr == 'harmonics': # Special check for list
             if not isinstance(val, list) or not all(isinstance(f, (int, float)) and f > FLOAT_EPSILON for f in val):
                  raise TypeError(f"Soul attribute 'harmonics' must be a list of positive numbers, found {val}.")
        elif not isinstance(val, (int, float)):
             raise TypeError(f"Soul attribute '{attr}' must be numeric, found {type(val)}.")
        elif not np.isfinite(val):
             raise ValueError(f"Soul attribute '{attr}' has non-finite value {val}.")
        elif attr == 'frequency' and val <= FLOAT_EPSILON:
             raise ValueError(f"Soul frequency ({val}) must be positive.")
        elif attr != 'frequency' and attr != 'field_radius' and not (0.0 <= val <= 1.0):
             logger.warning(f"Clamping soul attribute '{attr}' ({val}) to 0-1 range.")
             setattr(soul_spark, attr, max(0.0, min(1.0, val)))
    logger.debug("Soul properties ensured.")


def _perform_frequency_tuning(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Tunes frequency using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting frequency tuning for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    _ensure_soul_properties(soul_spark) # Ensure necessary attributes exist and are valid

    try:
        current_freq = getattr(soul_spark, "frequency")
        logger.debug(f"  Current frequency: {current_freq:.2f} Hz")

        # Find closest target frequency using constant list
        target_freq = min(HARMONIC_STRENGTHENING_TARGET_FREQS, key=lambda f: abs(f - current_freq))
        logger.debug(f"  Target harmonic frequency: {target_freq:.2f} Hz")

        # Calculate tuning amount using constant factor
        freq_diff = target_freq - current_freq
        tuning_amount = freq_diff * intensity * HARMONIC_STRENGTHENING_TUNING_INTENSITY_FACTOR
        new_freq = current_freq + tuning_amount
        new_freq = max(FLOAT_EPSILON, new_freq) # Ensure positive
        logger.debug(f"  Calculated frequency shift: {tuning_amount:.2f} Hz, New frequency: {new_freq:.2f} Hz")

        # --- Update SoulSpark ---
        setattr(soul_spark, "frequency", float(new_freq))
        # Update basic harmonic list based on new frequency
        h_count = getattr(soul_spark, 'harmonic_count', HARMONIC_STRENGTHENING_HARMONIC_COUNT)
        harmonics_list = [new_freq * n for n in range(1, h_count + 1)]
        setattr(soul_spark, "harmonics", harmonics_list)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        attunement_level = 0.0; target_reached = False
        if abs(freq_diff) > FLOAT_EPSILON: attunement_level = abs(tuning_amount) / abs(freq_diff)
        target_reached = abs(new_freq - target_freq) < HARMONIC_STRENGTHENING_TUNING_TARGET_REACH_HZ # Use constant

        phase_metrics = {
            "original_frequency": float(current_freq), "target_frequency": float(target_freq), "new_frequency": float(new_freq),
            "frequency_shift": float(tuning_amount), "attunement_level": float(attunement_level), "target_reached": target_reached,
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('harmonic_strengthening_tuning', phase_metrics)
        except Exception as e: logger.error(f"Failed to record tuning metrics: {e}")

        logger.info(f"Frequency tuning complete. New frequency: {new_freq:.2f} Hz (Level: {attunement_level:.2f}, TargetReached: {target_reached})")
        return phase_metrics

    except Exception as e: logger.error(f"Error during frequency tuning: {e}", exc_info=True); raise RuntimeError("Frequency tuning failed.") from e


def _perform_phi_resonance_amplification(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """Amplifies phi resonance using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting phi resonance amplification for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    _ensure_soul_properties(soul_spark)

    try:
        current_phi_resonance = getattr(soul_spark, "phi_resonance")
        current_stability = getattr(soul_spark, "stability")
        logger.debug(f"  Initial Phi Resonance: {current_phi_resonance:.4f}, Stability: {current_stability:.4f}")

        # Calculate increase using constants
        phi_increase = HARMONIC_STRENGTHENING_PHI_AMP_INTENSITY_FACTOR * intensity * HARMONIC_STRENGTHENING_PHI_AMP_DURATION_FACTOR * duration_factor
        new_phi_resonance = min(1.0, current_phi_resonance + phi_increase)
        logger.debug(f"  Calculated phi resonance increase: {phi_increase:.4f}, New Phi Resonance: {new_phi_resonance:.4f}")

        # Use constant for stability boost factor
        stability_increase = phi_increase * HARMONIC_STRENGTHENING_PHI_STABILITY_BOOST_FACTOR
        new_stability = min(1.0, current_stability + stability_increase)
        logger.debug(f"  Calculated stability increase: {stability_increase:.4f}, New Stability: {new_stability:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "phi_resonance", float(new_phi_resonance))
        setattr(soul_spark, "stability", float(new_stability))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        improvement = (phi_increase / max(FLOAT_EPSILON, 1.0 - current_phi_resonance)) if current_phi_resonance < 1.0 else 0.0
        phase_metrics = {
            "original_phi_resonance": float(current_phi_resonance), "new_phi_resonance": float(new_phi_resonance),
            "phi_resonance_change": float(phi_increase), "stability_increase": float(stability_increase),
            "final_stability": float(new_stability), "improvement_factor": float(improvement),
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('harmonic_strengthening_phi', phase_metrics)
        except Exception as e: logger.error(f"Failed to record phi resonance metrics: {e}")

        logger.info(f"Phi resonance amplified. New Resonance: {new_phi_resonance:.4f}, Stability: {new_stability:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during phi resonance amplification: {e}", exc_info=True); raise RuntimeError("Phi resonance amplification failed.") from e


def _perform_pattern_stabilization(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Stabilizes harmonic patterns using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting pattern stabilization for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    _ensure_soul_properties(soul_spark)

    try:
        current_stability = getattr(soul_spark, "stability")
        current_pattern_stability = getattr(soul_spark, "pattern_stability")
        aspects = getattr(soul_spark, "aspects", {})
        logger.debug(f"  Initial Pattern Stability: {current_pattern_stability:.4f}, Soul Stability: {current_stability:.4f}")

        # Calculate aspect influence using constant factors
        aspect_influence = min(HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_CAP, len(aspects) * HARMONIC_STRENGTHENING_PATTERN_STAB_ASPECT_FACTOR)
        logger.debug(f"  Aspect influence factor: {aspect_influence:.4f}")

        # Calculate stabilization amount using constants
        base_increase = HARMONIC_STRENGTHENING_PATTERN_STAB_INTENSITY_FACTOR * intensity
        aspect_bonus = aspect_influence * intensity # Aspect influence scales with intensity
        total_increase = base_increase + aspect_bonus
        new_pattern_stability = min(1.0, current_pattern_stability + total_increase)
        logger.debug(f"  Calculated pattern stability increase: {total_increase:.4f}, New Pattern Stability: {new_pattern_stability:.4f}")

        # Use constant for stability boost
        stability_increase = total_increase * HARMONIC_STRENGTHENING_PATTERN_STAB_STABILITY_BOOST
        new_stability = min(1.0, current_stability + stability_increase)
        logger.debug(f"  Calculated soul stability increase: {stability_increase:.4f}, New Soul Stability: {new_stability:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "pattern_stability", float(new_pattern_stability))
        setattr(soul_spark, "stability", float(new_stability))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        improvement = (total_increase / max(FLOAT_EPSILON, 1.0 - current_pattern_stability)) if current_pattern_stability < 1.0 else 0.0
        phase_metrics = {
            "original_pattern_stability": float(current_pattern_stability), "new_pattern_stability": float(new_pattern_stability),
            "pattern_stability_change": float(total_increase), "stability_increase": float(stability_increase),
            "final_stability": float(new_stability), "aspect_influence": float(aspect_influence),
            "improvement_factor": float(improvement), "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('harmonic_strengthening_pattern', phase_metrics)
        except Exception as e: logger.error(f"Failed to record pattern stabilization metrics: {e}")

        logger.info(f"Pattern stabilization complete. New Pattern Stability: {new_pattern_stability:.4f}, Soul Stability: {new_stability:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during pattern stabilization: {e}", exc_info=True); raise RuntimeError("Pattern stabilization failed.") from e


def _perform_coherence_enhancement(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """Enhances coherence using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting coherence enhancement for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")
    _ensure_soul_properties(soul_spark)

    try:
        current_coherence = getattr(soul_spark, "coherence")
        current_harmony = getattr(soul_spark, "harmony")
        harmonics_list = getattr(soul_spark, "harmonics", [])
        logger.debug(f"  Initial Coherence: {current_coherence:.4f}, Harmony: {current_harmony:.4f}, Harmonics Count: {len(harmonics_list)}")

        # Use constants for calculation
        harmonic_factor = min(1.0, len(harmonics_list) / HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_COUNT_NORM)
        base_increase = HARMONIC_STRENGTHENING_COHERENCE_INTENSITY_FACTOR * intensity * HARMONIC_STRENGTHENING_COHERENCE_DURATION_FACTOR * duration_factor
        harmonic_bonus = harmonic_factor * intensity * HARMONIC_STRENGTHENING_COHERENCE_HARMONIC_FACTOR
        total_increase = base_increase + harmonic_bonus
        new_coherence = min(1.0, current_coherence + total_increase)
        logger.debug(f"  Harmonic richness factor: {harmonic_factor:.4f}")
        logger.debug(f"  Calculated coherence increase: {total_increase:.4f}, New Coherence: {new_coherence:.4f}")

        # Use constant for harmony boost
        harmony_increase = total_increase * HARMONIC_STRENGTHENING_COHERENCE_HARMONY_BOOST
        new_harmony = min(1.0, current_harmony + harmony_increase)
        logger.debug(f"  Calculated harmony increase: {harmony_increase:.4f}, New Harmony: {new_harmony:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "coherence", float(new_coherence))
        setattr(soul_spark, "harmony", float(new_harmony))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        improvement = (total_increase / max(FLOAT_EPSILON, 1.0 - current_coherence)) if current_coherence < 1.0 else 0.0
        phase_metrics = {
            "original_coherence": float(current_coherence), "new_coherence": float(new_coherence),
            "coherence_change": float(total_increase), "harmony_increase": float(harmony_increase),
            "final_harmony": float(new_harmony), "harmonic_factor": float(harmonic_factor),
            "improvement_factor": float(improvement), "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('harmonic_strengthening_coherence', phase_metrics)
        except Exception as e: logger.error(f"Failed to record coherence enhancement metrics: {e}")

        logger.info(f"Coherence enhancement complete. New Coherence: {new_coherence:.4f}, Harmony: {new_harmony:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during coherence enhancement: {e}", exc_info=True); raise RuntimeError("Coherence enhancement failed.") from e


def _perform_field_expansion(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """Expands resonance field using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Starting field expansion for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    _ensure_soul_properties(soul_spark)

    try:
        current_radius = getattr(soul_spark, "field_radius")
        current_strength = getattr(soul_spark, "field_strength")
        coherence = getattr(soul_spark, "coherence")
        stability = getattr(soul_spark, "stability")
        logger.debug(f"  Initial Field Radius: {current_radius:.4f}, Strength: {current_strength:.4f}")
        logger.debug(f"  State factors: Coherence={coherence:.3f}, Stability={stability:.3f}")


        # Calculate expansion using constants
        expand_factor = ((coherence + stability) / 2.0) * HARMONIC_STRENGTHENING_EXPANSION_STATE_FACTOR
        radius_increase = HARMONIC_STRENGTHENING_EXPANSION_INTENSITY_FACTOR * intensity * expand_factor
        strength_increase = HARMONIC_STRENGTHENING_EXPANSION_STR_INTENSITY_FACTOR * intensity * expand_factor * HARMONIC_STRENGTHENING_EXPANSION_STR_STATE_FACTOR

        new_radius = current_radius + radius_increase
        new_strength = min(1.0, current_strength + strength_increase)
        logger.debug(f"  ExpandFactor={expand_factor:.3f}, RadiusInc={radius_increase:.4f}, StrengthInc={strength_increase:.4f}")
        logger.debug(f"  New Field Radius: {new_radius:.4f}, New Strength: {new_strength:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "field_radius", float(new_radius))
        setattr(soul_spark, "field_strength", float(new_strength))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        radius_improvement_pct = (radius_increase / max(FLOAT_EPSILON, current_radius)) * 100.0 if current_radius > FLOAT_EPSILON else (100.0 if radius_increase > 0 else 0.0)
        strength_improvement_factor = (strength_increase / max(FLOAT_EPSILON, 1.0 - current_strength)) if current_strength < 1.0 else 0.0
        phase_metrics = {
            "original_radius": float(current_radius), "new_radius": float(new_radius), "radius_change": float(radius_increase),
            "radius_improvement_pct": float(radius_improvement_pct), "original_strength": float(current_strength),
            "new_strength": float(new_strength), "strength_change": float(strength_increase),
            "strength_improvement_factor": float(strength_improvement_factor),
            "coherence_factor": float(coherence), "stability_factor": float(stability),
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('harmonic_strengthening_expansion', phase_metrics)
        except Exception as e: logger.error(f"Failed to record field expansion metrics: {e}")

        logger.info(f"Field expansion complete. New Radius: {new_radius:.4f}, Strength: {new_strength:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error during field expansion: {e}", exc_info=True); raise RuntimeError("Field expansion failed.") from e


# --- Orchestration Function ---

def perform_harmonic_strengthening(soul_spark: SoulSpark, intensity: float = 0.7, duration_factor: float = 1.0) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the complete harmonic strengthening process. Modifies SoulSpark. Fails hard. Uses constants.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the strengthening process (0.1-1.0).
        duration_factor (float): Relative duration multiplier for phases (0.1-2.0).

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the strengthening process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If parameters invalid or prerequisites not met.
        RuntimeError: If any phase fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor must be between 0.1 and 2.0")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Harmonic Strengthening for Soul {spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        # --- Check Prerequisites ---
        if not _check_prerequisites(soul_spark):
            raise ValueError("Soul prerequisites for harmonic strengthening not met.")
        logger.info("Prerequisites checked successfully.")

        # --- Store Initial State ---
        initial_state = {
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_stability': getattr(soul_spark, 'pattern_stability', 0.0), 'harmony': getattr(soul_spark, 'harmony', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0), 'field_strength': getattr(soul_spark, 'field_strength', 0.0),}
        logger.info(f"Initial State: Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}, Freq={initial_state['frequency']:.2f}, PhiRes={initial_state['phi_resonance']:.4f}")

        # --- Run Phases (Fail hard within each) ---
        logger.info("Step 1: Frequency Tuning...")
        metrics1 = _perform_frequency_tuning(soul_spark, intensity)
        process_metrics_summary['steps']['tuning'] = metrics1

        logger.info("Step 2: Phi Resonance Amplification...")
        metrics2 = _perform_phi_resonance_amplification(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['phi_resonance'] = metrics2

        logger.info("Step 3: Pattern Stabilization...")
        metrics3 = _perform_pattern_stabilization(soul_spark, intensity)
        process_metrics_summary['steps']['pattern_stabilization'] = metrics3

        logger.info("Step 4: Coherence Enhancement...")
        metrics4 = _perform_coherence_enhancement(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['coherence'] = metrics4

        logger.info("Step 5: Field Expansion...")
        metrics5 = _perform_field_expansion(soul_spark, intensity)
        process_metrics_summary['steps']['expansion'] = metrics5

        # --- Finalize ---
        setattr(soul_spark, 'harmonically_strengthened', True) # Set flag
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Capture final state metrics
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0), 'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_stability': getattr(soul_spark, 'pattern_stability', 0.0), 'harmony': getattr(soul_spark, 'harmony', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0), 'field_strength': getattr(soul_spark, 'field_strength', 0.0),
            'harmonically_strengthened': getattr(soul_spark, 'harmonically_strengthened', False) }
        # Calculate overall improvement %
        initial_avg = (initial_state['stability'] + initial_state['coherence']) / 2.0
        final_avg = (final_state['stability'] + final_state['coherence']) / 2.0
        improvement_pct = ((final_avg - initial_avg) / max(FLOAT_EPSILON, initial_avg)) * 100.0

        overall_metrics = {
            'action': 'harmonic_strengthening', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'intensity_setting': intensity, 'duration_factor': duration_factor,
            'initial_state': initial_state, 'final_state': final_state, 'overall_improvement_pct': float(improvement_pct), 'success': True, }
        try: metrics.record_metrics('harmonic_strengthening_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for strengthening: {e}")

        logger.info(f"--- Harmonic Strengthening Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s, Overall Improvement: {improvement_pct:.2f}%")
        logger.info(f"Final State: Stab={final_state['stability']:.4f}, Coh={final_state['coherence']:.4f}, Freq={final_state['frequency']:.2f}")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Harmonic strengthening process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"; steps_completed = list(process_metrics_summary['steps'].keys())
        if steps_completed: failed_step = steps_completed[-1]
        # Mark soul as failed this stage? Or just log? Let's just log.
        setattr(soul_spark, "harmonically_strengthened", False) # Mark as failed

        if METRICS_AVAILABLE:
             try: metrics.record_metrics('harmonic_strengthening_summary', {
                  'action': 'harmonic_strengthening', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'intensity_setting': intensity, 'duration_factor': duration_factor, 'success': False, 'error': str(e), 'failed_step': failed_step })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Harmonic strengthening process failed at step '{failed_step}'.") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Harmonic Strengthening Module Example...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        test_soul = SoulSpark()
        test_soul.spark_id="test_harmonic_constants_001"
        # Set state *after* Sephiroth journey might occur
        test_soul.stability = 0.70
        test_soul.coherence = 0.65
        test_soul.frequency = 600.0 # Example frequency
        test_soul.formation_complete = True
        test_soul.sephiroth_journey_complete = True # Example flag set by previous controller
        test_soul.aspects = {'unity': {'strength': 0.8}, 'love': {'strength': 0.6}, 'wisdom': {'strength': 0.5}}
        test_soul.last_modified = datetime.now().isoformat()

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Frequency: {test_soul.frequency:.4f}")
        print(f"  Aspects Count: {len(test_soul.aspects)}")

        try:
            print("\n--- Running Harmonic Strengthening Process ---")
            modified_soul, summary_metrics_result = perform_harmonic_strengthening(
                soul_spark=test_soul,
                intensity=0.85,
                duration_factor=1.0
            )

            print("\n--- Strengthening Complete ---")
            print("Final Soul State Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Harmonically Strengthened Flag: {getattr(modified_soul, 'harmonically_strengthened', False)}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Final Coherence: {getattr(modified_soul, 'coherence', 'N/A'):.4f}")
            print(f"  Final Frequency: {getattr(modified_soul, 'frequency', 'N/A'):.2f} Hz")
            print(f"  Final Phi Resonance: {getattr(modified_soul, 'phi_resonance', 'N/A'):.4f}")
            print(f"  Final Pattern Stability: {getattr(modified_soul, 'pattern_stability', 'N/A'):.4f}")
            print(f"  Final Harmony: {getattr(modified_soul, 'harmony', 'N/A'):.4f}")


            print("\nOverall Process Metrics:")
            # print(json.dumps(summary_metrics_result, indent=2, default=str))
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")
            print(f"  Overall Improvement % (Stab+Coh): {summary_metrics_result.get('overall_improvement_pct', 'N/A'):.2f}%")

        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Harmonic Strengthening Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Harmonic Strengthening Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nHarmonic Strengthening Module Example Finished.")

# --- END OF FILE harmonic_strengthening.py ---
