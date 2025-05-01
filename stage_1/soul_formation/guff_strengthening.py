# --- START OF FILE src/stage_1/soul_formation/guff_strengthening.py ---

"""
Guff Strengthening Functions (Refactored V4.3 - Principle-Driven S/C, PEP8)

Handles initial strengthening within the Guff using SEU energy transfer
and applying an *influence factor* increment to SoulSpark instead of direct
SU/CU boosts. Adheres to PEP 8 formatting.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from math import exp

logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately
try:
    import constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for Guff logger."
    )
    logger.setLevel(logging.INFO)

# --- Constants Import (using alias 'const') ---
try:
    import constants.constants as const
except ImportError as e:
    logger.critical(
        "CRITICAL ERROR: constants.py failed import in guff_strengthening.py"
    )
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.field_controller import FieldController
    from stage_1.fields.kether_field import KetherField
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}",
                    exc_info=True)
    raise ImportError(f"Core dependencies missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.error("Metrics tracking module not found.")
    METRICS_AVAILABLE = False
    # Define placeholder if module not found
    class MetricsPlaceholder: record_metrics = lambda *a, **kw: None
    metrics = MetricsPlaceholder()

# --- Helper: Check Prerequisites ---
def _check_prerequisites(soul_spark: SoulSpark,
                         field_controller: FieldController) -> bool:
    """Checks if the soul and environment meet Guff strengthening criteria."""
    logger.debug(
        f"Checking Guff strengthening prerequisites for soul "
        f"{soul_spark.spark_id}..."
    )
    if not isinstance(soul_spark, SoulSpark):
        logger.error("Prereq fail: Invalid SoulSpark object.")
        return False
    if not isinstance(field_controller, FieldController):
        logger.error("Prereq fail: Invalid FieldController object.")
        return False

    # Check flags using constants
    if getattr(soul_spark, const.FLAG_GUFF_STRENGTHENED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already Guff strengthened.")
        # Allow re-strengthening? For now, return True but log warning.
        # If re-strengthening is disallowed, return False here.
    if not getattr(soul_spark, const.FLAG_READY_FOR_GUFF, False):
        logger.error(f"Prereq fail: Soul not marked {const.FLAG_READY_FOR_GUFF}.")
        return False

    # Check essential attributes
    required_attrs = ['energy', 'stability', 'coherence', 'frequency', 'position']
    if not all(hasattr(soul_spark, attr) for attr in required_attrs):
        missing = [a for a in required_attrs if not hasattr(soul_spark, a)]
        logger.error(f"Prereq fail: SoulSpark missing attributes: {missing}.")
        return False

    # Check Kether field presence
    if not field_controller.kether_field or \
       not isinstance(field_controller.kether_field, KetherField):
        logger.error("Prereq fail: FieldController missing valid KetherField.")
        return False

    # Check location
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    if current_field != 'kether':
        logger.error(f"Prereq fail: Soul must be in Kether field "
                     f"(current: {current_field}).")
        return False
    position = getattr(soul_spark, 'position')
    try:
        # Use FieldController helper to convert float coords to int tuple
        grid_coords = field_controller._coords_to_int_tuple(position)
        if not field_controller.kether_field.is_in_guff(grid_coords):
            logger.error(f"Prereq fail: Soul at {grid_coords} not in Guff.")
            return False
    except AttributeError:
         logger.error("FieldController missing '_coords_to_int_tuple' method.")
         return False
    except Exception as e:
        logger.error(f"Prereq fail: Error checking Guff location {position}: {e}")
        return False

    logger.debug("Guff strengthening prerequisites met.")
    return True

# --- Helper: Calculate Resonance ---
def _calculate_guff_resonance(soul_freq: float, guff_freq: float) -> float:
    """Calculates resonance factor (0.1 to 1.0) based on frequency diff."""
    if soul_freq <= const.FLOAT_EPSILON or guff_freq <= const.FLOAT_EPSILON:
        return 0.1 # Minimum resonance if frequencies are zero/negative
    # Normalized frequency difference
    freq_diff_norm = abs(soul_freq - guff_freq) / max(guff_freq, soul_freq,
                                                     const.FLOAT_EPSILON)
    # Exponential decay: resonance drops as difference increases
    resonance = exp(-freq_diff_norm * 3.0) # Tuning parameter '3.0' controls sensitivity
    return max(0.1, float(resonance)) # Ensure minimum resonance factor

# --- Core Calculation (Returns energy boost and INFLUENCE increment) ---
def _calculate_guff_boosts(soul_spark: SoulSpark,
                           guff_properties: Dict[str, Any],
                           delta_time: float) -> Tuple[float, float]:
    """
    Calculates energy boost (SEU) and Guff influence factor increment (0-1).
    Influence factor affects stability/coherence calculation within SoulSpark.
    """
    try:
        target_e_seu = guff_properties.get('guff_target_energy_seu',
                                          const.GUFF_TARGET_ENERGY_SEU)
        guff_freq = guff_properties.get('frequency_hz', const.KETHER_FREQ)

        # Calculate resonance and coupling
        resonance = _calculate_guff_resonance(soul_spark.frequency, guff_freq)
        # Coupling influenced by soul's coherence (more coherent = better coupling)
        coupling_factor = resonance * (soul_spark.coherence / const.MAX_COHERENCE_CU)
        coupling_factor = max(0.0, min(1.0, coupling_factor)) # Clamp 0-1

        # Energy Transfer (SEU)
        energy_diff = target_e_seu - soul_spark.energy
        # Use specific Guff energy transfer rate constant
        energy_boost = (energy_diff * const.GUFF_ENERGY_TRANSFER_RATE_K *
                        coupling_factor * delta_time)

        # Calculate Guff Influence Factor Increment
        # Base rate modified by coupling and time. Stronger coupling -> faster influence gain.
        influence_increment = (coupling_factor * delta_time *
                               const.GUFF_INFLUENCE_RATE_K) # Use new rate

        logger.debug(f"  Guff Calc: Resonance={resonance:.3f}, "
                     f"Coupling={coupling_factor:.3f}")
        logger.debug(f"  Guff Calc: dE={energy_boost:.3f} SEU, "
                     f"InfluenceIncrement={influence_increment:.5f}")
        return energy_boost, influence_increment

    except Exception as e:
        logger.error(f"Error calculating Guff boosts: {e}", exc_info=True)
        return 0.0, 0.0 # Return zero boost/increment on error

# --- Main Orchestration Function ---
def perform_guff_strengthening(
        soul_spark: SoulSpark,
        field_controller: FieldController,
        duration: float = const.GUFF_STRENGTHENING_DURATION
        ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs Guff strengthening using SEU energy transfer and influence factor
    increments. Modifies SoulSpark attributes directly.

    Args:
        soul_spark: The SoulSpark object to strengthen.
        field_controller: The FieldController managing the environment.
        duration: The simulated duration of the strengthening process (seconds).

    Returns:
        Tuple containing the modified SoulSpark and a dictionary of metrics.

    Raises:
        TypeError: If input object types are invalid.
        ValueError: If duration is invalid or prerequisites are not met.
        RuntimeError: If critical errors occur during processing.
    """
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(field_controller, FieldController): raise TypeError("field_controller invalid.")
    if not isinstance(duration, (int, float)) or duration <= 0: raise ValueError("Duration must be positive.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    log_msg = (f"--- Starting Guff Strengthening [Principle-Driven] "
               f"for Soul {spark_id} (Dur: {duration}) ---")
    logger.info(log_msg)

    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    time_step = 0.1 # Simulation step duration
    num_steps = max(1, int(duration / time_step))

    try:
        # --- Prerequisites Check ---
        if not _check_prerequisites(soul_spark, field_controller):
            raise ValueError("Guff strengthening prerequisites not met.")

        # --- Initial State & Properties ---
        initial_state = {
            'energy_seu': soul_spark.energy,
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'guff_influence_factor': getattr(soul_spark, 'guff_influence_factor', 0.0)
        }
        log_initial = (f"Initial State: E={initial_state['energy_seu']:.2f}SEU, "
                       f"S={initial_state['stability_su']:.1f}SU, "
                       f"C={initial_state['coherence_cu']:.1f}CU, "
                       f"GuffInf={initial_state['guff_influence_factor']:.3f}")
        logger.info(log_initial)

        # Get Guff properties at soul's location
        soul_pos_coords = field_controller._coords_to_int_tuple(soul_spark.position)
        try:
            guff_props = field_controller.get_properties_at(soul_pos_coords)
        except Exception as e:
             raise RuntimeError(f"Could not retrieve Guff properties at "
                                f"{soul_pos_coords}: {e}") from e
        if not guff_props.get('in_guff', False):
            # This check should ideally be redundant due to prerequisites
            raise RuntimeError(f"Consistency check failed: Soul position "
                               f"{soul_pos_coords} not reported as in Guff.")
        guff_freq_hz = guff_props.get('frequency_hz', const.KETHER_FREQ)
        logger.debug(f"Guff props at {soul_pos_coords}: Freq={guff_freq_hz:.1f}Hz")

        # --- Simulation Loop ---
        total_e_gain = 0.0
        total_influence_gain = 0.0
        for step in range(num_steps):
            e_boost, influence_increment = _calculate_guff_boosts(
                soul_spark, guff_props, time_step
            )

            # Apply Energy Boost
            energy_before = soul_spark.energy
            soul_spark.energy = min(const.MAX_SOUL_ENERGY_SEU,
                                    max(0.0, soul_spark.energy + e_boost))
            total_e_gain += (soul_spark.energy - energy_before)

            # Apply Guff Influence Factor Increment
            current_influence = getattr(soul_spark, 'guff_influence_factor', 0.0)
            new_influence = min(1.0, current_influence + influence_increment)
            setattr(soul_spark, 'guff_influence_factor', new_influence)
            total_influence_gain += (new_influence - current_influence)

            # Update Derived S/C Scores using soul_spark's internal method
            # This now incorporates the updated guff_influence_factor
            if hasattr(soul_spark, 'update_state'):
                soul_spark.update_state()
            else:
                 logger.error("SoulSpark object missing 'update_state' method!")
                 # Decide how to handle - maybe raise error or just log?
                 # raise AttributeError("SoulSpark needs 'update_state' method.")

            # Log progress periodically
            if step % (num_steps // 5 or 1) == 0:
                 log_step = (f"  Guff Step {step+1}/{num_steps}: "
                             f"E={soul_spark.energy:.1f}, "
                             f"S={soul_spark.stability:.1f}, "
                             f"C={soul_spark.coherence:.1f}, "
                             f"GuffInf={new_influence:.3f}")
                 logger.debug(log_step)

        # --- Final Update & Metrics ---
        setattr(soul_spark, const.FLAG_GUFF_STRENGTHENED, True)
        setattr(soul_spark, const.FLAG_READY_FOR_JOURNEY, True)
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        end_time_iso = last_mod_time
        end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'energy_seu': soul_spark.energy,
            'stability_su': soul_spark.stability,
            'coherence_cu': soul_spark.coherence,
            'guff_influence_factor': getattr(soul_spark, 'guff_influence_factor', 0.0),
            const.FLAG_GUFF_STRENGTHENED: True,
            const.FLAG_READY_FOR_JOURNEY: True
        }
        overall_metrics = {
            'action': 'guff_strengthening', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'simulated_duration': duration,
            'initial_state': initial_state, 'final_state': final_state,
            'total_energy_gain_seu': float(final_state['energy_seu'] -
                                           initial_state['energy_seu']),
            'total_guff_influence_gain': float(final_state['guff_influence_factor'] -
                                                initial_state['guff_influence_factor']),
            'final_stability_su': final_state['stability_su'], # Report final emergent score
            'final_coherence_cu': final_state['coherence_cu'], # Report final emergent score
            # Guff targets used (for information)
            'guff_targets_used': {k:v for k,v in guff_props.items()
                                  if 'guff_target' in k},
            'success': True,
        }
        if METRICS_AVAILABLE:
             metrics.record_metrics('guff_strengthening_summary', overall_metrics)

        # Add Memory Echo
        echo_msg = (f"Strengthened in Guff. "
                    f"E:{final_state['energy_seu']:.1f}, "
                    f"S:{final_state['stability_su']:.1f}, "
                    f"C:{final_state['coherence_cu']:.1f}, "
                    f"GuffInf:{final_state['guff_influence_factor']:.3f}")
        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(echo_msg)
            logger.info(f"Memory echo created: {echo_msg}")

        log_final = (f"--- Guff Strengthening Completed Successfully "
                     f"for Soul {spark_id} ---")
        logger.info(log_final)
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final State: E={final_state['energy_seu']:.1f}SEU, "
                    f"S={final_state['stability_su']:.1f}SU, "
                    f"C={final_state['coherence_cu']:.1f}CU")
        return soul_spark, overall_metrics

    # --- Error Handling ---
    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Guff strengthening failed for {spark_id}: {e_val}",
                      exc_info=True)
         record_failure_metric(spark_id, start_time_iso,
                               'prerequisites/validation', str(e_val))
         raise # Re-raise validation errors
    except RuntimeError as e_rt:
         logger.critical(f"Guff strengthening failed critically "
                         f"for {spark_id}: {e_rt}", exc_info=True)
         record_failure_metric(spark_id, start_time_iso, 'runtime', str(e_rt))
         raise # Re-raise runtime errors
    except Exception as e:
         logger.critical(f"Unexpected error during Guff strengthening "
                         f"for {spark_id}: {e}", exc_info=True)
         record_failure_metric(spark_id, start_time_iso, 'unexpected', str(e))
         raise RuntimeError(f"Unexpected Guff strengthening failure: {e}") from e

# --- Failure Metric Helper ---
def record_failure_metric(spark_id: str, start_time_iso: str,
                          failed_step: str, error_msg: str):
    """Helper to record failure metrics consistently."""
    if METRICS_AVAILABLE:
        try:
            end_time_iso = datetime.now().isoformat()
            duration_secs = (datetime.fromisoformat(end_time_iso) -
                             datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('guff_strengthening_summary', {
                'action': 'guff_strengthening', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time_iso,
                'duration_seconds': duration_secs, 'success': False,
                'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed record failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/guff_strengthening.py ---