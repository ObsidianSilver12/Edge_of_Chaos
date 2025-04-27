# --- START OF FILE src/stage_1/soul_formation/guff_strengthening.py ---

"""
Guff Strengthening Functions

Handles the initial strengthening and energizing of a newly sparked soul
within the Guff (Hall of Souls) region of the Kether field. Modifies the
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
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Guff Strengthening cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_formation.soul_spark import SoulSpark
    # Need FieldController to get properties at soul's location
    from stage_1.fields.field_controller import FieldController
    from stage_1.fields.kether_field import KetherField # For type hint and check
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies (SoulSpark, FieldController, KetherField): {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark, field_controller: FieldController) -> bool:
    """Checks if the soul is ready and correctly positioned for Guff strengthening."""
    logger.debug(f"Checking Guff strengthening prerequisites for soul {soul_spark.spark_id}...")

    # 1. Soul State Check
    if getattr(soul_spark, "guff_strengthened", False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked as Guff strengthened. Proceeding with re-strengthening.")
        # return False # Or allow re-strengthening? Let's allow for now.
    if not getattr(soul_spark, "ready_for_guff", False):
        logger.error("Prerequisite failed: Soul not marked ready for Guff (should be set after initial spark).")
        return False

    # 2. Field Controller Check
    if not isinstance(field_controller, FieldController):
         logger.error("Prerequisite failed: Invalid FieldController instance provided.")
         return False
    if not field_controller.kether_field:
        logger.error("Prerequisite failed: FieldController does not have a valid KetherField instance.")
        return False

    # 3. Location Check
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    if current_field != 'kether':
        logger.error(f"Prerequisite failed: Soul is not in Kether field (current: {current_field}).")
        return False
    position = getattr(soul_spark, 'position', None)
    if position is None:
        logger.error("Prerequisite failed: Soul position attribute is missing.")
        return False
    # Convert float position to int grid coordinates for check
    grid_coords = field_controller._coords_to_int_tuple(position)
    if not field_controller.kether_field.is_in_guff(grid_coords):
        logger.error(f"Prerequisite failed: Soul position {grid_coords} is not within the Guff region.")
        return False

    logger.debug("Guff strengthening prerequisites met.")
    return True

def _calculate_guff_boosts(soul_spark: SoulSpark, guff_properties: Dict[str, Any], delta_time: float) -> Tuple[float, float, float]:
    """Calculates the energy, stability, and coherence boosts for one time step."""
    # Energy boost based on difference and Guff energy level
    energy_diff = guff_properties.get('energy', VOID_BASE_ENERGY) - soul_spark.energy
    energy_boost = energy_diff * GUFF_STRENGTHENING_ENERGY_RATE * delta_time

    # Stability boost towards target
    stability_diff = GUFF_STABILITY_TARGET - soul_spark.stability
    stability_boost = stability_diff * GUFF_STRENGTHENING_STABILITY_RATE * delta_time

    # Coherence boost towards target
    coherence_diff = GUFF_COHERENCE_TARGET - soul_spark.coherence
    coherence_boost = coherence_diff * GUFF_STRENGTHENING_COHERENCE_RATE * delta_time

    return energy_boost, stability_boost, coherence_boost


# --- Main Orchestration Function ---

def perform_guff_strengthening(soul_spark: SoulSpark, field_controller: FieldController, duration: float = GUFF_STRENGTHENING_DURATION) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the Guff strengthening process over a simulated duration.
    Modifies the SoulSpark object in place. Fails hard on critical errors.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        field_controller (FieldController): The controller managing the fields.
        duration (float): The simulated duration (time units) spent in the Guff.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the strengthening process.

    Raises:
        TypeError: If soul_spark or field_controller are not the correct types.
        ValueError: If duration is invalid or prerequisites not met.
        RuntimeError: If getting Guff properties fails or process fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be SoulSpark.")
    if not isinstance(field_controller, FieldController): raise TypeError("field_controller must be FieldController.")
    if not isinstance(duration, (int, float)) or duration <= 0: raise ValueError("Duration must be positive.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Guff Strengthening for Soul {spark_id} (Duration: {duration}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)

    # Use a small internal time step for simulation stability
    time_step = 0.1 # Simulation time units per step
    num_steps = max(1, int(duration / time_step))

    try:
        # --- Check Prerequisites ---
        if not _check_prerequisites(soul_spark, field_controller):
            raise ValueError("Soul prerequisites for Guff strengthening not met.")
        logger.info("Prerequisites checked successfully.")

        # --- Store Initial State ---
        initial_state = {
            'energy': getattr(soul_spark, 'energy', 0.0),
            'stability': getattr(soul_spark, 'stability', 0.0),
            'coherence': getattr(soul_spark, 'coherence', 0.0),
        }
        logger.info(f"Initial State: Energy={initial_state['energy']:.4f}, Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}")

        # --- Simulate Strengthening Over Duration ---
        # Get Guff properties once (assume they are static for this process)
        soul_pos_coords = field_controller._coords_to_int_tuple(soul_spark.position)
        try:
            guff_props = field_controller.get_properties_at(soul_pos_coords)
            if not field_controller.kether_field or not field_controller.kether_field.is_in_guff(soul_pos_coords):
                 raise RuntimeError("Failed consistency check: Soul position not in Guff according to KetherField.")
            logger.debug(f"Guff properties at {soul_pos_coords}: E={guff_props.get('energy'):.3f}, Coh={guff_props.get('coherence'):.3f}")
        except (IndexError, RuntimeError) as e:
            logger.error(f"Failed to get Guff properties at soul position {soul_pos_coords}: {e}")
            raise RuntimeError("Could not retrieve properties from Guff region.") from e

        total_energy_gain = 0.0
        total_stability_gain = 0.0
        total_coherence_gain = 0.0

        for step in range(num_steps):
            e_boost, s_boost, c_boost = _calculate_guff_boosts(soul_spark, guff_props, time_step)

            # Apply boosts and clamp
            soul_spark.energy = min(GUFF_ENERGY_MULTIPLIER, max(0.0, soul_spark.energy + e_boost)) # Cap energy gain relative to Guff multiplier
            soul_spark.stability = min(GUFF_STABILITY_TARGET, max(0.0, soul_spark.stability + s_boost)) # Move towards target
            soul_spark.coherence = min(GUFF_COHERENCE_TARGET, max(0.0, soul_spark.coherence + c_boost)) # Move towards target

            total_energy_gain += e_boost
            total_stability_gain += s_boost
            total_coherence_gain += c_boost

            # Optional: Log progress periodically
            if step % (num_steps // 5) == 0: # Log 5 times during process
                 logger.debug(f"  Guff Step {step}/{num_steps}: E={soul_spark.energy:.4f}, S={soul_spark.stability:.4f}, C={soul_spark.coherence:.4f}")

        # --- Final Update & Metrics ---
        setattr(soul_spark, 'guff_strengthened', True)
        setattr(soul_spark, 'ready_for_journey', True) # Mark ready for next stage
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'energy': soul_spark.energy,
            'stability': soul_spark.stability,
            'coherence': soul_spark.coherence,
            'guff_strengthened': soul_spark.guff_strengthened,
            'ready_for_journey': soul_spark.ready_for_journey,
        }

        overall_metrics = {
            'action': 'guff_strengthening', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'simulated_duration': duration,
            'initial_state': initial_state, 'final_state': final_state,
            'total_energy_gain': float(total_energy_gain),
            'total_stability_gain': float(total_stability_gain),
            'total_coherence_gain': float(total_coherence_gain),
            'guff_properties_used': guff_props, # Record properties used for boost
            'success': True,
        }
        # Record metrics
        if METRICS_AVAILABLE:
            try: metrics.record_metrics('guff_strengthening_summary', overall_metrics)
            except Exception as e: logger.error(f"Failed to record summary metrics for guff strengthening: {e}")

        # Log memory echo
        echo_msg = f"Strengthened in Guff. E:{final_state['energy']:.3f}, S:{final_state['stability']:.3f}, C:{final_state['coherence']:.3f}"
        logger.info(f"Memory echo created: {echo_msg}")
        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
            soul_spark.memory_echoes.append(f"{getattr(soul_spark, 'last_modified')}: {echo_msg}")

        logger.info(f"--- Guff Strengthening Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final State: E={final_state['energy']:.4f}, S={final_state['stability']:.4f}, C={final_state['coherence']:.4f}")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Guff strengthening process failed critically for soul {spark_id}: {e}", exc_info=True)
        # Mark soul as failed this stage?
        setattr(soul_spark, "guff_strengthened", False)
        setattr(soul_spark, "ready_for_journey", False)
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('guff_strengthening_summary', {
                  'action': 'guff_strengthening', 'soul_id': spark_id,
                  'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - start_time_dt).total_seconds(),
                  'success': False, 'error': str(e)
             })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Guff strengthening process failed critically.") from e

# --- END OF FILE src/stage_1/soul_formation/guff_strengthening.py ---