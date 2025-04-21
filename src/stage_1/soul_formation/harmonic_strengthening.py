# --- START OF FILE harmonic_strengthening.py ---

"""
Harmonic Strengthening Functions (Refactored - Operates on SoulSpark Object)

Provides functions for enhancing soul stability, coherence, and resonance
through harmonic frequencies and phi-resonance amplification. Modifies the
SoulSpark object instance directly.

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
    # Import necessary constants
    from src.constants import (
        GOLDEN_RATIO, PHI, # Aliases are fine
        SOLFEGGIO_FREQUENCIES, FIBONACCI_SEQUENCE,
        FLOAT_EPSILON, LOG_LEVEL # Use constants
    )
except ImportError as e:
    # Fallback logging setup if constants unavailable during standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.warning(f"Could not import constants: {e}. Using fallback values.")
    GOLDEN_RATIO = 1.618033988749895
    PHI = GOLDEN_RATIO
    # Provide fallback for Solfeggio and Fibonacci if absolutely needed for structure
    SOLFEGGIO_FREQUENCIES = {'MI': 528.0, 'LA': 852.0} # Minimal fallback
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    FLOAT_EPSILON = 1e-9

# --- Dependency Imports ---
try:
    from stage_1.void.soul_spark import SoulSpark
    # No need for aspect dictionary here directly
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
    # Define dummy metrics functions
    class MetricsPlaceholder:
        def record_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder()

# --- Module Specific Constants ---
# Base frequencies considered optimal for harmonic tuning
HARMONIC_TARGET_FREQUENCIES = list(SOLFEGGIO_FREQUENCIES.values()) + [432.0] # Add 432Hz

# --- Helper Functions ---

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary properties, initializing if missing. Fails hard if critical init fails."""
    logger.debug(f"Ensuring properties for soul {soul_spark.spark_id}...")
    # Check critical prerequisite
    if not getattr(soul_spark, "formation_complete", False):
        # This stage requires formation completion. Should it fail or just warn? Let's fail.
        raise ValueError("Cannot perform harmonic strengthening: Soul formation is not marked complete.")

    # Initialize numerical properties if missing, log warning
    defaults = {
        "frequency": SOLFEGGIO_FREQUENCIES.get('MI', 528.0), # Default to transformation freq
        "stability": 0.6,
        "coherence": 0.6,
        "harmonics": [], # Will be populated based on frequency
        "phi_resonance": 0.5,
        "pattern_stability": 0.5,
        "harmony": 0.5,
        "field_radius": 3.0,
        "field_strength": 0.5
    }
    for attr, default_val in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
            logger.warning(f"Soul {soul_spark.spark_id} missing '{attr}'. Initializing to default: {default_val}")
            setattr(soul_spark, attr, default_val)
            # Ensure harmonics list is populated if frequency was missing
            if attr == 'frequency' and not soul_spark.harmonics:
                 setattr(soul_spark, 'harmonics', [soul_spark.frequency * n for n in range(1, 6)]) # Simple harmonic series

    # Ensure harmonics list is populated if empty but frequency exists
    if hasattr(soul_spark, 'frequency') and (not hasattr(soul_spark, 'harmonics') or not soul_spark.harmonics):
         logger.warning(f"Soul {soul_spark.spark_id} missing 'harmonics'. Initializing based on frequency.")
         setattr(soul_spark, 'harmonics', [soul_spark.frequency * n for n in range(1, 6)])

    # Validate existing numerical properties
    for attr in ["frequency", "stability", "coherence", "phi_resonance", "pattern_stability", "harmony", "field_radius", "field_strength"]:
         val = getattr(soul_spark, attr, None)
         if not isinstance(val, (int, float)):
              raise TypeError(f"Soul attribute '{attr}' must be numeric, found {type(val)}.")
         # Add range checks if necessary (e.g., frequency > 0, others 0-1)
         if attr == 'frequency' and val <= FLOAT_EPSILON: raise ValueError(f"Soul frequency ({val}) must be positive.")
         if attr != 'frequency' and attr != 'field_radius' and not (0.0 <= val <= 1.0):
              logger.warning(f"Soul attribute '{attr}' ({val}) outside typical 0-1 range. Clamping.")
              setattr(soul_spark, attr, max(0.0, min(1.0, val)))


def _perform_frequency_tuning(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """
    Tunes the soul's frequency towards an optimal harmonic base. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the tuning (0.1-1.0).

    Returns:
        Dict[str, Any]: Metrics for this phase.

    Raises:
        ValueError: If intensity is invalid or critical soul attributes missing/invalid.
    """
    logger.info(f"Starting frequency tuning for soul {soul_spark.spark_id} with intensity {intensity:.2f}...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    _ensure_soul_properties(soul_spark) # Ensure necessary attributes exist and are valid

    current_freq = getattr(soul_spark, "frequency") # Already validated by ensure
    logger.debug(f"  Current frequency: {current_freq:.2f} Hz")

    # Find closest target frequency (Solfeggio or 432)
    target_freq = min(HARMONIC_TARGET_FREQUENCIES, key=lambda f: abs(f - current_freq))
    logger.debug(f"  Target harmonic frequency: {target_freq:.2f} Hz")

    # Calculate tuning amount (move partway towards target based on intensity)
    freq_diff = target_freq - current_freq
    tuning_amount = freq_diff * intensity * 0.7 # Don't jump fully in one go
    new_freq = current_freq + tuning_amount
    logger.debug(f"  Calculated frequency shift: {tuning_amount:.2f} Hz, New frequency: {new_freq:.2f} Hz")

    # --- Update SoulSpark ---
    setattr(soul_spark, "frequency", new_freq)
    # Update basic harmonic list based on new frequency
    harmonics_list = [new_freq * n for n in range(1, 6)] # Recalculate simple harmonics
    setattr(soul_spark, "harmonics", harmonics_list)
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # --- Calculate & Record Metrics ---
    tuning_improvement = 0.0
    target_reached = False
    if abs(freq_diff) > FLOAT_EPSILON:
        # Improvement is how much closer we got relative to the initial difference
        tuning_improvement = abs(tuning_amount) / abs(freq_diff)
        target_reached = abs(new_freq - target_freq) < 1.0 # Consider tuned if within 1 Hz
    else:
        tuning_improvement = 1.0 # Already at target
        target_reached = True

    phase_metrics = {
        "original_frequency": float(current_freq), "target_frequency": float(target_freq),
        "new_frequency": float(new_freq), "tuning_improvement": float(tuning_improvement),
        "target_reached": target_reached, "timestamp": getattr(soul_spark, 'last_modified')
    }
    try: metrics.record_metrics('harmonic_strengthening_tuning', phase_metrics)
    except Exception as e: logger.error(f"Failed to record tuning metrics: {e}")

    logger.info(f"Frequency tuning complete. New frequency: {new_freq:.2f} Hz (Improvement: {tuning_improvement:.2f})")
    return phase_metrics


def _perform_phi_resonance_amplification(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """
    Amplifies the soul's phi resonance patterns. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the amplification (0.1-1.0).
        duration_factor (float): Relative duration factor (0.1-2.0).

    Returns:
        Dict[str, Any]: Metrics for this phase.

    Raises:
        ValueError: If intensity/duration invalid or critical soul attributes missing/invalid.
    """
    logger.info(f"Starting phi resonance amplification for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor must be between 0.1 and 2.0")
    _ensure_soul_properties(soul_spark)

    current_phi_resonance = getattr(soul_spark, "phi_resonance")
    current_stability = getattr(soul_spark, "stability")
    logger.debug(f"  Initial Phi Resonance: {current_phi_resonance:.4f}, Stability: {current_stability:.4f}")

    # Calculate increase based on intensity and duration
    phi_increase = 0.15 * intensity * duration_factor # Base increase
    new_phi_resonance = min(1.0, current_phi_resonance + phi_increase)
    logger.debug(f"  Calculated phi resonance increase: {phi_increase:.4f}, New Phi Resonance: {new_phi_resonance:.4f}")

    # Phi resonance directly improves stability
    stability_increase = phi_increase * 0.4 # Phi resonance is key to stability
    new_stability = min(1.0, current_stability + stability_increase)
    logger.debug(f"  Calculated stability increase: {stability_increase:.4f}, New Stability: {new_stability:.4f}")

    # --- Update SoulSpark ---
    setattr(soul_spark, "phi_resonance", new_phi_resonance)
    setattr(soul_spark, "stability", new_stability)
    # Recalculate phi harmonics based on current frequency (optional, depends on design)
    # If harmonics should reflect phi resonance, recalculate them here or in generate_harmonic_structure
    # For now, just update the phi_resonance attribute value.
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # --- Calculate & Record Metrics ---
    improvement = (new_phi_resonance - current_phi_resonance) / max(FLOAT_EPSILON, 1.0 - current_phi_resonance) # Improvement relative to potential gain
    phase_metrics = {
        "original_phi_resonance": float(current_phi_resonance), "new_phi_resonance": float(new_phi_resonance),
        "phi_resonance_change": float(new_phi_resonance - current_phi_resonance),
        "stability_increase": float(stability_increase), "final_stability": float(new_stability),
        "improvement_factor": float(improvement), "timestamp": getattr(soul_spark, 'last_modified')
    }
    try: metrics.record_metrics('harmonic_strengthening_phi', phase_metrics)
    except Exception as e: logger.error(f"Failed to record phi resonance metrics: {e}")

    logger.info(f"Phi resonance amplified. New Resonance: {new_phi_resonance:.4f}, Stability: {new_stability:.4f}")
    return phase_metrics


def _perform_pattern_stabilization(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """
    Stabilizes the soul's harmonic patterns. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the stabilization (0.1-1.0).

    Returns:
        Dict[str, Any]: Metrics for this phase.

    Raises:
        ValueError: If intensity invalid or critical soul attributes missing/invalid.
    """
    logger.info(f"Starting pattern stabilization for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    _ensure_soul_properties(soul_spark)

    current_stability = getattr(soul_spark, "stability")
    current_pattern_stability = getattr(soul_spark, "pattern_stability")
    aspects = getattr(soul_spark, "aspects", {}) # Use aspects dict if available
    logger.debug(f"  Initial Pattern Stability: {current_pattern_stability:.4f}, Soul Stability: {current_stability:.4f}")

    # Calculate influence of acquired aspects (more aspects = easier stabilization)
    aspect_influence = min(0.3, len(aspects) * 0.02) # Capped influence
    logger.debug(f"  Aspect influence factor: {aspect_influence:.4f}")

    # Calculate stabilization amount
    base_increase = 0.1 * intensity # Base stabilization rate
    aspect_bonus = aspect_influence * intensity * 0.5 # Bonus from aspects
    total_increase = base_increase + aspect_bonus
    new_pattern_stability = min(1.0, current_pattern_stability + total_increase)
    logger.debug(f"  Calculated pattern stability increase: {total_increase:.4f}, New Pattern Stability: {new_pattern_stability:.4f}")

    # Pattern stability contributes to overall soul stability
    stability_increase = total_increase * 0.6 # Strong link
    new_stability = min(1.0, current_stability + stability_increase)
    logger.debug(f"  Calculated soul stability increase: {stability_increase:.4f}, New Soul Stability: {new_stability:.4f}")

    # --- Update SoulSpark ---
    setattr(soul_spark, "pattern_stability", new_pattern_stability)
    setattr(soul_spark, "stability", new_stability)
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # --- Calculate & Record Metrics ---
    improvement = (new_pattern_stability - current_pattern_stability) / max(FLOAT_EPSILON, 1.0 - current_pattern_stability)
    phase_metrics = {
        "original_pattern_stability": float(current_pattern_stability), "new_pattern_stability": float(new_pattern_stability),
        "pattern_stability_change": float(new_pattern_stability - current_pattern_stability),
        "stability_increase": float(stability_increase), "final_stability": float(new_stability),
        "aspect_influence": float(aspect_influence), "improvement_factor": float(improvement),
        "timestamp": getattr(soul_spark, 'last_modified')
    }
    try: metrics.record_metrics('harmonic_strengthening_pattern', phase_metrics)
    except Exception as e: logger.error(f"Failed to record pattern stabilization metrics: {e}")

    logger.info(f"Pattern stabilization complete. New Pattern Stability: {new_pattern_stability:.4f}, Soul Stability: {new_stability:.4f}")
    return phase_metrics


def _perform_coherence_enhancement(soul_spark: SoulSpark, intensity: float, duration_factor: float) -> Dict[str, Any]:
    """
    Enhances soul coherence through harmonic resonance. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the enhancement (0.1-1.0).
        duration_factor (float): Relative duration factor (0.1-2.0).

    Returns:
        Dict[str, Any]: Metrics for this phase.

    Raises:
        ValueError: If intensity/duration invalid or critical soul attributes missing/invalid.
    """
    logger.info(f"Starting coherence enhancement for soul {soul_spark.spark_id} (Intensity={intensity:.2f}, DurationFactor={duration_factor:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor must be between 0.1 and 2.0")
    _ensure_soul_properties(soul_spark)

    current_coherence = getattr(soul_spark, "coherence")
    current_harmony = getattr(soul_spark, "harmony")
    harmonics_list = getattr(soul_spark, "harmonics", [])
    logger.debug(f"  Initial Coherence: {current_coherence:.4f}, Harmony: {current_harmony:.4f}, Harmonics Count: {len(harmonics_list)}")

    # Calculate harmonic richness factor (more harmonics = better potential coherence)
    harmonic_factor = min(1.0, len(harmonics_list) / 10.0) # Normalized against typical max (10)
    logger.debug(f"  Harmonic richness factor: {harmonic_factor:.4f}")

    # Calculate coherence increase
    base_increase = 0.12 * intensity * duration_factor # Base rate
    harmonic_bonus = harmonic_factor * intensity * 0.08 # Bonus from richness
    total_increase = base_increase + harmonic_bonus
    new_coherence = min(1.0, current_coherence + total_increase)
    logger.debug(f"  Calculated coherence increase: {total_increase:.4f}, New Coherence: {new_coherence:.4f}")

    # Coherence strongly influences harmony
    harmony_increase = total_increase * 0.9
    new_harmony = min(1.0, current_harmony + harmony_increase)
    logger.debug(f"  Calculated harmony increase: {harmony_increase:.4f}, New Harmony: {new_harmony:.4f}")

    # --- Update SoulSpark ---
    setattr(soul_spark, "coherence", new_coherence)
    setattr(soul_spark, "harmony", new_harmony)
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # --- Calculate & Record Metrics ---
    improvement = (new_coherence - current_coherence) / max(FLOAT_EPSILON, 1.0 - current_coherence)
    phase_metrics = {
        "original_coherence": float(current_coherence), "new_coherence": float(new_coherence),
        "coherence_change": float(new_coherence - current_coherence),
        "harmony_increase": float(harmony_increase), "final_harmony": float(new_harmony),
        "harmonic_factor": float(harmonic_factor), "improvement_factor": float(improvement),
        "timestamp": getattr(soul_spark, 'last_modified')
    }
    try: metrics.record_metrics('harmonic_strengthening_coherence', phase_metrics)
    except Exception as e: logger.error(f"Failed to record coherence enhancement metrics: {e}")

    logger.info(f"Coherence enhancement complete. New Coherence: {new_coherence:.4f}, Harmony: {new_harmony:.4f}")
    return phase_metrics


def _perform_field_expansion(soul_spark: SoulSpark, intensity: float) -> Dict[str, Any]:
    """
    Expands the soul's resonance field. Modifies SoulSpark. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        intensity (float): Intensity of the expansion (0.1-1.0).

    Returns:
        Dict[str, Any]: Metrics for this phase.

    Raises:
        ValueError: If intensity invalid or critical soul attributes missing/invalid.
    """
    logger.info(f"Starting field expansion for soul {soul_spark.spark_id} (Intensity={intensity:.2f})...")
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity must be between 0.1 and 1.0")
    _ensure_soul_properties(soul_spark)

    current_radius = getattr(soul_spark, "field_radius")
    current_strength = getattr(soul_spark, "field_strength")
    logger.debug(f"  Initial Field Radius: {current_radius:.4f}, Strength: {current_strength:.4f}")

    # Calculate expansion based on intensity and current coherence/stability
    coherence = getattr(soul_spark, "coherence")
    stability = getattr(soul_spark, "stability")
    # Expansion is easier with higher coherence and stability
    expand_factor = (coherence + stability) / 2.0
    radius_increase = 1.5 * intensity * expand_factor # Base increase of up to 1.5 units
    strength_increase = 0.15 * intensity * expand_factor # Base increase of up to 0.15

    new_radius = current_radius + radius_increase
    new_strength = min(1.0, current_strength + strength_increase)
    logger.debug(f"  Calculated Radius Increase: {radius_increase:.4f}, Strength Increase: {strength_increase:.4f}")
    logger.debug(f"  New Field Radius: {new_radius:.4f}, New Strength: {new_strength:.4f}")

    # --- Update SoulSpark ---
    setattr(soul_spark, "field_radius", new_radius)
    setattr(soul_spark, "field_strength", new_strength)
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # --- Calculate & Record Metrics ---
    radius_improvement = (radius_increase / max(FLOAT_EPSILON, current_radius)) if current_radius > FLOAT_EPSILON else (1.0 if radius_increase > 0 else 0.0)
    strength_improvement = (strength_increase / max(FLOAT_EPSILON, 1.0 - current_strength)) if current_strength < 1.0 else 0.0
    phase_metrics = {
        "original_radius": float(current_radius), "new_radius": float(new_radius),
        "radius_change": float(radius_increase), "radius_improvement_pct": float(radius_improvement * 100.0),
        "original_strength": float(current_strength), "new_strength": float(new_strength),
        "strength_change": float(strength_increase), "strength_improvement_factor": float(strength_improvement),
        "coherence_factor": float(coherence), "stability_factor": float(stability),
        "timestamp": getattr(soul_spark, 'last_modified')
    }
    try: metrics.record_metrics('harmonic_strengthening_expansion', phase_metrics)
    except Exception as e: logger.error(f"Failed to record field expansion metrics: {e}")

    logger.info(f"Field expansion complete. New Radius: {new_radius:.4f}, Strength: {new_strength:.4f}")
    return phase_metrics

# --- Orchestration Function ---

def perform_harmonic_strengthening(soul_spark: SoulSpark, intensity: float = 0.7, duration_factor: float = 1.0) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the complete harmonic strengthening process on a SoulSpark object.
    Modifies the SoulSpark object in place. Fails hard on critical errors.

    Args:
        soul_spark (SoulSpark): The soul spark object to be strengthened (will be modified).
        intensity (float): Intensity of the strengthening process (0.1-1.0).
        duration_factor (float): Relative duration multiplier for phases (0.1-2.0).

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the entire strengthening process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If parameters are invalid or prerequisites not met.
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
        # --- Store Initial State ---
        initial_state = {
            'stability': getattr(soul_spark, 'stability', 0.0),
            'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0),
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_stability': getattr(soul_spark, 'pattern_stability', 0.0),
            'harmony': getattr(soul_spark, 'harmony', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0),
            'field_strength': getattr(soul_spark, 'field_strength', 0.0),
        }
        logger.info(f"Initial State: Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}, Freq={initial_state['frequency']:.2f}, PhiRes={initial_state['phi_resonance']:.4f}")

        # --- Run Phases ---
        logger.info("Step 1: Frequency Tuning...")
        metrics1 = _perform_frequency_tuning(soul_spark, intensity)
        process_metrics_summary['steps']['tuning'] = metrics1
        logger.info("Step 1 Complete.")

        logger.info("Step 2: Phi Resonance Amplification...")
        metrics2 = _perform_phi_resonance_amplification(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['phi_resonance'] = metrics2
        logger.info("Step 2 Complete.")

        logger.info("Step 3: Pattern Stabilization...")
        metrics3 = _perform_pattern_stabilization(soul_spark, intensity)
        process_metrics_summary['steps']['pattern_stabilization'] = metrics3
        logger.info("Step 3 Complete.")

        logger.info("Step 4: Coherence Enhancement...")
        metrics4 = _perform_coherence_enhancement(soul_spark, intensity, duration_factor)
        process_metrics_summary['steps']['coherence'] = metrics4
        logger.info("Step 4 Complete.")

        logger.info("Step 5: Field Expansion...")
        metrics5 = _perform_field_expansion(soul_spark, intensity)
        process_metrics_summary['steps']['expansion'] = metrics5
        logger.info("Step 5 Complete.")

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability': getattr(soul_spark, 'stability', 0.0),
            'coherence': getattr(soul_spark, 'coherence', 0.0),
            'frequency': getattr(soul_spark, 'frequency', 0.0),
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_stability': getattr(soul_spark, 'pattern_stability', 0.0),
            'harmony': getattr(soul_spark, 'harmony', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0),
            'field_strength': getattr(soul_spark, 'field_strength', 0.0),
        }
        # Calculate overall improvement
        initial_avg = (initial_state['stability'] + initial_state['coherence']) / 2
        final_avg = (final_state['stability'] + final_state['coherence']) / 2
        improvement_pct = ((final_avg - initial_avg) / max(FLOAT_EPSILON, initial_avg)) * 100.0

        overall_metrics = {
            'action': 'harmonic_strengthening', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'overall_improvement_pct': float(improvement_pct),
            'success': True,
            # 'steps_metrics': process_metrics_summary['steps'] # Optional detail
        }
        try: metrics.record_metrics('harmonic_strengthening_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for strengthening: {e}")

        # Update soul state marker
        setattr(soul_spark, 'harmonically_strengthened', True)
        setattr(soul_spark, 'last_modified', end_time_iso)

        logger.info(f"--- Harmonic Strengthening Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final State: Stab={final_state['stability']:.4f}, Coh={final_state['coherence']:.4f}, Freq={final_state['frequency']:.2f}, PhiRes={final_state['phi_resonance']:.4f}")
        logger.info(f"Overall Improvement (Stab+Coh): {improvement_pct:.2f}%")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Harmonic strengthening process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown" # Determine last successful step if needed
        if 'expansion' in process_metrics_summary['steps']: failed_step = 'expansion'
        elif 'coherence' in process_metrics_summary['steps']: failed_step = 'coherence'
        # ... and so on

        if METRICS_AVAILABLE:
             try: metrics.record_metrics('harmonic_strengthening_summary', {
                  'action': 'harmonic_strengthening', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'success': False, 'error': str(e), 'failed_step': failed_step
             })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError("Harmonic strengthening process failed.") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Harmonic Strengthening Module Example...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        # Create a SoulSpark instance
        test_soul = SoulSpark()
        test_soul.spark_id="test_harmonic_001"
        # Set some initial state suitable for this stage
        test_soul.stability = 0.68
        test_soul.coherence = 0.72
        test_soul.frequency = 500.0
        test_soul.formation_complete = True # Prerequisite
        test_soul.aspects = {'unity': {'strength': 0.5}} # Example aspect
        test_soul.generate_harmonic_structure() # Ensure harmonics exist
        test_soul.phi_resonance = 0.45
        test_soul.pattern_stability = 0.55
        test_soul.harmony = 0.60
        test_soul.field_radius = 2.8
        test_soul.field_strength = 0.58

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Frequency: {test_soul.frequency:.4f}")
        print(f"  Phi Resonance: {test_soul.phi_resonance:.4f}")

        try:
            print("\n--- Running Harmonic Strengthening Process ---")
            # Run the process, modifying the test_soul object
            modified_soul, summary_metrics_result = perform_harmonic_strengthening(
                soul_spark=test_soul, # Pass the object
                intensity=0.8,
                duration_factor=1.1
            )

            print("\n--- Strengthening Complete ---")
            print("Final Soul State Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Coherence: {getattr(modified_soul, 'coherence', 'N/A'):.4f}")
            print(f"  Frequency: {getattr(modified_soul, 'frequency', 'N/A'):.4f}")
            print(f"  Phi Resonance: {getattr(modified_soul, 'phi_resonance', 'N/A'):.4f}")
            print(f"  Pattern Stability: {getattr(modified_soul, 'pattern_stability', 'N/A'):.4f}")
            print(f"  Harmony: {getattr(modified_soul, 'harmony', 'N/A'):.4f}")
            print(f"  Field Radius: {getattr(modified_soul, 'field_radius', 'N/A'):.4f}")
            print(f"  Field Strength: {getattr(modified_soul, 'field_strength', 'N/A'):.4f}")
            print(f"  Harmonically Strengthened Flag: {getattr(modified_soul, 'harmonically_strengthened', False)}")


            print("\nOverall Process Metrics:")
            # print(json.dumps(summary_metrics_result, indent=2, default=str))
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")
            print(f"  Overall Improvement % (Stab+Coh): {summary_metrics_result.get('overall_improvement_pct', 'N/A'):.2f}%")


        except (ValueError, TypeError, RuntimeError, ImportError) as e:
            print(f"\n--- ERROR during Harmonic Strengthening Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Harmonic Strengthening Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\nHarmonic Strengthening Module Example Finished.")


# --- END OF FILE harmonic_strengthening.py ---
