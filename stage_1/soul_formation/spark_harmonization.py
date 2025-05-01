# --- START OF FILE src/stage_1/soul_formation/spark_harmonization.py ---

"""
Spark Harmonization Module (New V4.3.7 - Emergence Principle, PEP8)

Implements the initial self-harmonization phase for a newly emerged SoulSpark.
This stage iteratively refines internal coherence, stability factors, and energy
based on principles like toroidal flow and harmonic resonance, preparing the
spark for external interactions (like Guff). Modifies SoulSpark attributes.
Adheres strictly to PEP 8 formatting. Assumes `from constants.constants import *`.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple
import logging

# --- Constants Import ---
# Assume constants are imported at the top level
try:
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: constants.py failed import in spark_harmonization.py")
    # Define minimal fallbacks to allow import, but execution will likely fail
    HARMONIZATION_ITERATIONS = 144
    HARMONIZATION_PATTERN_COHERENCE_RATE = 0.003
    HARMONIZATION_PHI_RESONANCE_RATE = 0.002
    HARMONIZATION_HARMONY_RATE = 0.0015
    HARMONIZATION_ENERGY_GAIN_RATE = 0.1
    MAX_SOUL_ENERGY_SEU = 1e6
    MAX_STABILITY_SU = 100.0 # Needed for logging/metrics only here
    MAX_COHERENCE_CU = 100.0 # Needed for logging/metrics only here
    LOG_LEVEL = logging.INFO # Fallback Log Level
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Fallback Log Format
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logging.error("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder: record_metrics = lambda *a, **kw: None
    metrics = MetricsPlaceholder()

# --- Logger Setup ---
logger = logging.getLogger(__name__)
try:
    logger.setLevel(LOG_LEVEL)
except NameError:
    logger.warning("LOG_LEVEL constant not found, using default INFO level.")
    logger.setLevel(logging.INFO)
if not logger.hasHandlers(): # Ensure handlers aren't added multiple times
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)


def perform_spark_harmonization(
    soul_spark: SoulSpark,
    iterations: int = HARMONIZATION_ITERATIONS # Use constant default
) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs initial self-harmonization and structuring of the newly created spark.
    Operates iteratively on SoulSpark attributes before Guff, simulating internal
    processes like toroidal flow and resonance stabilization.

    Args:
        soul_spark (SoulSpark): The spark to harmonize.
        iterations (int): Number of harmonization cycles to run.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: The harmonized spark and metrics summary.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If iterations is not a positive integer.
        RuntimeError: If critical errors occur during harmonization.
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Input must be a SoulSpark instance.")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("Iterations must be a positive integer.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Spark Harmonization for {spark_id} "
                f"({iterations} iterations) ---")
    start_time = time.time()

    # --- Record Initial State ---
    initial_s = soul_spark.stability
    initial_c = soul_spark.coherence
    initial_e = soul_spark.energy
    initial_p_coh = getattr(soul_spark, 'pattern_coherence', 0.0)
    initial_phi_res = getattr(soul_spark, 'phi_resonance', 0.0)
    initial_harmony = getattr(soul_spark, 'harmony', 0.0)

    logger.info(f"  Initial Harmonization State: S={initial_s:.1f}, C={initial_c:.1f}, "
                f"E={initial_e:.1f}, P.Coh={initial_p_coh:.3f}, PhiRes={initial_phi_res:.3f}")

    # --- Harmonization Loop ---
    try:
        for i in range(iterations):
            # --- 1. Simulate Toroidal Flow / Structure Building ---
            # Principle: Increase internal organization / pattern coherence factor (0-1).
            # Rate slows as it approaches 1.0 and over iterations.
            rate_p_coh = HARMONIZATION_PATTERN_COHERENCE_RATE
            current_p_coh = getattr(soul_spark, 'pattern_coherence', 0.0)
            increase_p_coh = rate_p_coh * (1.0 - current_p_coh) * (1.0 - (i/iterations)*0.5)
            soul_spark.pattern_coherence = min(1.0, current_p_coh + increase_p_coh)

            # --- 2. Simulate Harmonic / Phi Resonance / Phase Alignment ---
            # Principle: Increase internal harmonic order and phase alignment factors (0-1).
            # Increase phi_resonance towards 1.0
            rate_phi = HARMONIZATION_PHI_RESONANCE_RATE
            current_phi = getattr(soul_spark, 'phi_resonance', 0.0)
            increase_phi = rate_phi * (1.0 - current_phi) * (1.0 - (i/iterations)*0.5)
            soul_spark.phi_resonance = min(1.0, current_phi + increase_phi)

            # Increase general harmony factor towards 1.0
            rate_harm = HARMONIZATION_HARMONY_RATE
            current_harm = getattr(soul_spark, 'harmony', 0.0)
            increase_harm = rate_harm * (1.0 - current_harm) * (1.0 - (i/iterations)*0.5)
            soul_spark.harmony = min(1.0, current_harm + increase_harm)

            # --- Placeholder for actual harmonic/phase adjustment ---
            # A deeper implementation would modify soul_spark.harmonics and
            # soul_spark.frequency_signature['phases'] here based on resonance principles,
            # which would then directly influence the calculated coherence score.
            # For now, we rely on pattern_coherence, phi_resonance, and harmony
            # increases feeding into the existing _calculate_coherence_score method.

            # --- 3. Simulate Energy gain from internal harmony ---
            # Energy gain is proportional to the achieved internal order.
            # Use pattern_coherence and phi_resonance as proxies for this order.
            internal_order_proxy = (soul_spark.pattern_coherence + soul_spark.phi_resonance) / 2.0
            energy_gain = internal_order_proxy * HARMONIZATION_ENERGY_GAIN_RATE
            soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, soul_spark.energy + energy_gain)

            # --- 4. Update State (Recalculates SU/CU based on factor changes) ---
            soul_spark.update_state()

            # --- Logging (Periodic) ---
            if (i + 1) % (iterations // 10 or 1) == 0:
                logger.debug(f"  Harmonization {i+1}/{iterations}: S={soul_spark.stability:.1f}, "
                             f"C={soul_spark.coherence:.1f}, E={soul_spark.energy:.1f}, "
                             f"P.Coh={soul_spark.pattern_coherence:.3f}, PhiRes={soul_spark.phi_resonance:.3f}")

    except Exception as e:
        logger.error(f"Error during harmonization loop for {spark_id}: {e}", exc_info=True)
        # Package minimal failure metrics
        metrics = {
            'action': 'spark_harmonization', 'soul_id': spark_id,
            'duration_sec': time.time() - start_time, 'iterations_run': i + 1,
            'success': False, 'error': str(e)
        }
        # Re-raise as RuntimeError
        raise RuntimeError(f"Spark harmonization failed during loop: {e}") from e

    # --- Finalization & Metrics ---
    final_s = soul_spark.stability
    final_c = soul_spark.coherence
    final_e = soul_spark.energy
    final_p_coh = soul_spark.pattern_coherence
    final_phi_res = soul_spark.phi_resonance
    final_harmony = soul_spark.harmony
    duration = time.time() - start_time

    metrics_summary = {
        'action': 'spark_harmonization',
        'soul_id': spark_id,
        'duration_sec': duration,
        'iterations_run': iterations,
        'initial_stability_su': initial_s,
        'final_stability_su': final_s,
        'stability_gain_su': final_s - initial_s,
        'initial_coherence_cu': initial_c,
        'final_coherence_cu': final_c,
        'coherence_gain_cu': final_c - initial_c,
        'initial_energy_seu': initial_e,
        'final_energy_seu': final_e,
        'energy_gain_seu': final_e - initial_e,
        'initial_pattern_coherence': initial_p_coh,
        'final_pattern_coherence': final_p_coh,
        'initial_phi_resonance': initial_phi_res,
        'final_phi_resonance': final_phi_res,
        'initial_harmony': initial_harmony,
        'final_harmony': final_harmony,
        'success': True
    }
    if METRICS_AVAILABLE:
        metrics.record_metrics('spark_harmonization_summary', metrics_summary)

    logger.info(f"--- Spark Harmonization Complete for {spark_id}. ---")
    logger.info(f"  Final State: S={final_s:.1f}, C={final_c:.1f}, E={final_e:.1f}, "
                f"P.Coh={final_p_coh:.3f}, PhiRes={final_phi_res:.3f}")

    return soul_spark, metrics_summary

# --- END OF FILE src/stage_1/soul_formation/spark_harmonization.py ---