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
    import shared.constants.constants as const # Use alias
    logger.setLevel(const.LOG_LEVEL)
except ImportError:
    logger.warning(
        "Constants not loaded, using default INFO level for Guff logger."
    )
    logger.setLevel(logging.INFO)

# --- Constants Import (using alias 'const') ---
try:
    import shared.constants.constants as const
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
    print(f"DEBUG: Inside _check_prerequisites for {getattr(soul_spark, 'spark_id', 'unknown')}")
    logger.info(
        f"Checking Guff strengthening prerequisites for soul "
        f"{soul_spark.spark_id}..."
    )
    print("DEBUG: Past logger.info statement")
    if not isinstance(soul_spark, SoulSpark):
        logger.error("Prereq fail: Invalid SoulSpark object.")
        print("DEBUG: Failed SoulSpark instance check")
        return False
    print("DEBUG: SoulSpark instance check passed")
    if not isinstance(field_controller, FieldController):
        logger.error("Prereq fail: Invalid FieldController object.")
        print("DEBUG: Failed FieldController instance check")
        return False
    print("DEBUG: FieldController instance check passed")

    # Check flags using constants
    print("DEBUG: About to check flags")
    try:
        guff_strengthened_flag = getattr(soul_spark, const.FLAG_GUFF_STRENGTHENED, False)
        ready_for_guff_flag = getattr(soul_spark, const.FLAG_READY_FOR_GUFF, False)
        print(f"DEBUG: Flags retrieved - Guff Strengthened: {guff_strengthened_flag}, Ready for Guff: {ready_for_guff_flag}")
        logger.info(f"Flags - Guff Strengthened: {guff_strengthened_flag}, Ready for Guff: {ready_for_guff_flag}")
    except Exception as flag_e:
        print(f"DEBUG: Error getting flags: {flag_e}")
        logger.error(f"Error accessing flag constants: {flag_e}")
        return False
    
    if guff_strengthened_flag:
        logger.warning(f"Soul {soul_spark.spark_id} already Guff strengthened.")
        # Allow re-strengthening? For now, return True but log warning.
        # If re-strengthening is disallowed, return False here.
    if not ready_for_guff_flag:
        logger.error(f"Prereq fail: Soul not marked {const.FLAG_READY_FOR_GUFF}. Flag value: {ready_for_guff_flag}")
        return False

    # Check essential attributes
    required_attrs = ['energy', 'stability', 'coherence', 'frequency', 'position']
    print(f"DEBUG: Checking required attributes: {required_attrs}")
    missing_attrs = [a for a in required_attrs if not hasattr(soul_spark, a)]
    if missing_attrs:
        print(f"DEBUG: Missing attributes: {missing_attrs}")
        logger.error(f"Prereq fail: SoulSpark missing attributes: {missing_attrs}.")
        return False
    print("DEBUG: All required attributes present")

    # Check Kether field presence
    print("DEBUG: Checking Kether field presence")
    if not field_controller.kether_field or \
       not isinstance(field_controller.kether_field, KetherField):
        print("DEBUG: Kether field check failed")
        logger.error("Prereq fail: FieldController missing valid KetherField.")
        return False
    print("DEBUG: Kether field check passed")

    # Check location
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    logger.info(f"Current field: {current_field}")
    if current_field != 'kether':
        logger.error(f"Prereq fail: Soul must be in Kether field "
                     f"(current: {current_field}).")
        return False
    position = getattr(soul_spark, 'position')
    logger.info(f"Soul position: {position}")
    try:
        # Use FieldController helper to convert float coords to int tuple
        print(f"DEBUG: Checking for _coords_to_int_tuple method. hasattr: {hasattr(field_controller, '_coords_to_int_tuple')}")
        if hasattr(field_controller, "_coords_to_int_tuple"):
            print(f"DEBUG: Method exists, callable: {callable(getattr(field_controller, '_coords_to_int_tuple'))}")
        if hasattr(field_controller, "_coords_to_int_tuple") and callable(getattr(field_controller, "_coords_to_int_tuple")):
            grid_coords = field_controller._coords_to_int_tuple(position)
        else:
            print("DEBUG: _coords_to_int_tuple method not found or not callable")
            logger.error("FieldController must provide a public '_coords_to_int_tuple' method for coordinate conversion.")
            return False
        logger.info(f"Grid coordinates: {grid_coords}")
        in_guff = field_controller.kether_field.is_in_guff(grid_coords)
        logger.info(f"Is in Guff: {in_guff}")
        if not in_guff:
            logger.error(f"Prereq fail: Soul at {grid_coords} not in Guff.")
            return False
    except KeyError as e:
        logger.error(f"Prereq fail: Error checking Guff location {position}: {e}")
        return False
    except ValueError as e:
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
                               const.GUFF_INFLUENCE_RATE_K) # Use rate constant

        logger.debug(f"  Guff Calc: Resonance={resonance:.3f}, "
                     f"Coupling={coupling_factor:.3f}")
        logger.debug(f"  Guff Calc: dE={energy_boost:.3f} SEU, "
                     f"InfluenceIncrement={influence_increment:.5f}")
        return energy_boost, influence_increment

    except (AttributeError, KeyError, ValueError) as e:
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
        print(f"DEBUG: About to check prerequisites for {spark_id}")
        prereq_result = _check_prerequisites(soul_spark, field_controller)
        print(f"DEBUG: Prerequisites check result: {prereq_result}")
        if not prereq_result:
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
        if hasattr(field_controller, "_coords_to_int_tuple") and callable(getattr(field_controller, "_coords_to_int_tuple")):
            soul_pos_coords = field_controller._coords_to_int_tuple(soul_spark.position)
        else:
            raise AttributeError("FieldController must provide a public '_coords_to_int_tuple' method for coordinate conversion.")
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
            # --- Capture state BEFORE calculation for this step ---
            e_before_step = soul_spark.energy
            s_before_step = soul_spark.stability # Get S/C BEFORE update_state for this step
            c_before_step = soul_spark.coherence
            inf_before_step = getattr(soul_spark, 'guff_influence_factor', 0.0)

            # --- Calculate boosts for this step ---
            e_boost, influence_increment = _calculate_guff_boosts(
                soul_spark, guff_props, time_step
            )

            # --- Log pre-application state and calculated deltas ---
            log_pre = (f"  Guff Step {step+1}/{num_steps}: BEFORE - "
                    f"E={e_before_step:.1f}, S={s_before_step:.1f}, C={c_before_step:.1f}, "
                    f"GuffInf={inf_before_step:.4f}")
            logger.debug(log_pre)
            log_calc = (f"  Guff Step {step+1}/{num_steps}: CALC   - "
                        f"dE={e_boost:+.3f}, dInf={influence_increment:+.5f}") # Added '+' for sign
            logger.debug(log_calc)

            # --- Apply Energy Boost ---
            soul_spark.energy = min(const.MAX_SOUL_ENERGY_SEU,
                                    max(0.0, soul_spark.energy + e_boost))
            actual_e_gain_step = soul_spark.energy - e_before_step
            total_e_gain += actual_e_gain_step

            # --- Apply Guff Influence Factor Increment ---
            new_influence = min(1.0, inf_before_step + influence_increment)
            setattr(soul_spark, 'guff_influence_factor', new_influence)
            actual_inf_gain_step = new_influence - inf_before_step
            total_influence_gain += actual_inf_gain_step

            # --- Update Derived S/C Scores using soul_spark's internal method ---
            if hasattr(soul_spark, 'update_state'):
                soul_spark.update_state() # This recalculates S/C based on *new* influence factor
            else:
                logger.error("SoulSpark object missing 'update_state' method!")
                raise AttributeError("SoulSpark needs 'update_state' method.")

            # --- Capture coherence change AFTER update ---
            c_after_step = soul_spark.coherence
            c_delta_step = c_after_step - c_before_step

            # --- NEW: Override stability change to be proportional to coherence change ---
            # Use the proportional factor you specified (0.2 * coherence gain or loss)
            proportional_s_delta = 0.2 * c_delta_step
            
            # Apply the proportional stability change, replacing whatever change came from update_state
            # This makes stability directly follow coherence with the specified factor
            soul_spark.stability = min(const.MAX_STABILITY_SU, 
                                    max(0.0, s_before_step + proportional_s_delta))
            
            # Recalculate the actual stability change for logging
            s_after_step = soul_spark.stability
            s_delta_step = s_after_step - s_before_step
            
            # --- Continue with Capture and Log state AFTER application and update ---
            e_after_step = soul_spark.energy
            inf_after_step = new_influence # Already calculated

            log_post = (f"  Guff Step {step+1}/{num_steps}: AFTER  - "
                        f"E={e_after_step:.1f}({actual_e_gain_step:+.1f}), " # Show change
                        f"S={s_after_step:.1f}({s_delta_step:+.2f}), " # Show change
                        f"C={c_after_step:.1f}({c_delta_step:+.2f}), " # Show change
                        f"GuffInf={inf_after_step:.4f}({actual_inf_gain_step:+.5f})") # Show change
            logger.debug(log_post)

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
        except RuntimeError as metric_e:
            logger.error(f"Failed to record failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/guff_strengthening.py ---