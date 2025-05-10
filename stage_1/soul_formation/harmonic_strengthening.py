# --- START OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---

"""
Harmonic Strengthening Functions (Refactored V4.3.8 - Principle-Driven)

Enhances soul factors (phi_resonance, pattern_coherence, harmony, phase, harmonics, torus)
after Creator Entanglement through iterative, targeted refinement based on thresholds.
Stability/Coherence emerge via update_state. No direct S/C boosts.
Modifies the SoulSpark object instance directly. Uses constants. Hard fails on critical errors.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from math import sqrt, pi as PI, exp, atan2
from constants.constants import *
# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Harmonic Strengthening cannot function.")
    # Define minimal fallbacks ONLY to allow script parsing

    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}.")
    raise ImportError(f"Core dependency SoulSpark missing: {e}") from e

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# ... (logger configuration as before) ...
if not logger.handlers:
    try: logger.setLevel(LOG_LEVEL)
    except NameError: logger.warning("LOG_LEVEL constant not found."); logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter(LOG_FORMAT)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(log_formatter); logger.addHandler(ch)

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()

# --- Helper: Calculate Circular Variance ---
def _calculate_circular_variance(phases_array: np.ndarray) -> float:
    """Calculates circular variance (0=sync, 1=uniform)."""
    if phases_array is None or not isinstance(phases_array, np.ndarray) or phases_array.size < 2:
        return 1.0 # Max variance if no/one phase or invalid input
    if not np.all(np.isfinite(phases_array)):
        logger.warning("Non-finite values found in phases array during variance calculation.")
        return 1.0 # Max variance if data invalid
    mean_cos = np.mean(np.cos(phases_array))
    mean_sin = np.mean(np.sin(phases_array))
    r_len_sq = mean_cos**2 + mean_sin**2
    if r_len_sq < 0: # Protect against precision errors
        r_len_sq = 0.0
    r_len = sqrt(r_len_sq)
    return 1.0 - r_len

# --- Helper: Calculate Harmonic Deviation ---
def _calculate_harmonic_deviation(harmonics: List[float], base_freq: float) -> float:
    """Calculates average deviation of harmonics from ideal ratios."""
    if not harmonics or not isinstance(harmonics, list) or base_freq <= FLOAT_EPSILON:
        return 1.0 # Max deviation if no harmonics/base_freq
    valid_harmonics = [h for h in harmonics if isinstance(h, (int,float)) and h > FLOAT_EPSILON]
    if not valid_harmonics: return 1.0

    ratios = [h / base_freq for h in valid_harmonics]
    if not ratios: return 1.0

    deviations = []
    for r in ratios:
        if r <= 0 or not np.isfinite(r): continue
        # Check simple integer ratios (1 to 5)
        int_dists = [abs(r - n) for n in range(1, 6)]
        # Check Phi ratios (Phi, 1/Phi, Phi^2, 1/Phi^2)
        phi_powers = [PHI**n for n in [1, -1, 2, -2]]
        phi_dists = [abs(r - p) for p in phi_powers]

        # Find the minimum deviation to *any* ideal ratio
        min_deviation = min(int_dists + phi_dists)
        # Normalize deviation relative to the ratio itself for scale independence?
        # normalized_deviation = min_deviation / r
        # deviations.append(normalized_deviation)
        deviations.append(min_deviation) # Using absolute deviation for now

    # Return average deviation, capped at 1.0
    avg_dev = np.mean(deviations) if deviations else 1.0
    return min(1.0, max(0.0, avg_dev))


# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
    logger.debug(f"Checking prerequisites for harmonic strengthening (Soul: {soul_spark.spark_id})...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, False):
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_STRENGTHENING}."
        logger.error(msg)
        raise ValueError(msg)

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing stability or coherence attributes."
        logger.error(msg)
        raise AttributeError(msg)

    # Use specific HS prerequisite constants
    if stability_su < HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {HARMONIC_STRENGTHENING_PREREQ_STABILITY_SU} SU."
        logger.error(msg)
        raise ValueError(msg)
    if coherence_cu < HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {HARMONIC_STRENGTHENING_PREREQ_COHERENCE_CU} CU."
        logger.error(msg)
        raise ValueError(msg)

    if getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_HARMONICALLY_STRENGTHENED}. Re-running.")

    logger.debug("Harmonic strengthening prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary numeric properties. Raises AttributeError if missing. """
    logger.debug(f"Ensuring properties for harmonic strengthening (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'harmonics', 'phi_resonance',
                'pattern_coherence', 'harmony', 'aspects', 'frequency_signature',
                'toroidal_flow_strength'] # Added torus
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for HS: {missing}")

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    if not isinstance(soul_spark.frequency_signature, dict): raise TypeError("frequency_signature must be dict.")
    if 'phases' not in soul_spark.frequency_signature: raise ValueError("frequency_signature missing 'phases'.")
    if not isinstance(soul_spark.harmonics, list): raise TypeError("harmonics must be a list.")

    # Initialize field radius/strength if missing
    if not hasattr(soul_spark, 'field_radius'): setattr(soul_spark, 'field_radius', 1.0)
    if not hasattr(soul_spark, 'field_strength'): setattr(soul_spark, 'field_strength', 0.5)

    # Validate existing values (optional strict check)
    # if not (0.0 <= soul_spark.phi_resonance <= 1.0): raise ValueError("Invalid phi_resonance")
    # ... etc for other factors

    logger.debug("Soul properties ensured for harmonic strengthening.")



# --- Targeted Refinement Cycle ---
def _run_hs_refinement_cycle(soul_spark: SoulSpark, cycle_num: int) -> Dict[str, float]:
    """ Performs one cycle of targeted refinement based on weakest area. """
    changes = {
        'delta_phi': 0.0, 'delta_pcoh': 0.0, 'delta_harmony': 0.0,
        'delta_torus': 0.0, 'delta_phase_var': 0.0, 'delta_harm_dev': 0.0,
        'delta_energy': 0.0, 'targeted_factor': 'none'
    }
    initial_harmony = soul_spark.harmony # Track for energy calc

    # --- Identify Weakest Area ---
    weakness_scores = {}

    # Phase Coherence
    phases = np.array(soul_spark.frequency_signature.get('phases', []))
    current_phase_var = _calculate_circular_variance(phases)
    current_phase_coh = 1.0 - current_phase_var
    weakness_scores['phase'] = HS_TRIGGER_PHASE_COHERENCE - current_phase_coh

    # Harmonic Purity
    current_harm_dev = _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)
    current_harm_purity = 1.0 - current_harm_dev
    weakness_scores['harmonics'] = HS_TRIGGER_HARMONIC_PURITY - current_harm_purity

    # Factors
    weakness_scores['phi'] = HS_TRIGGER_FACTOR_THRESHOLD - soul_spark.phi_resonance
    weakness_scores['pcoh'] = HS_TRIGGER_FACTOR_THRESHOLD - soul_spark.pattern_coherence
    weakness_scores['harmony'] = HS_TRIGGER_FACTOR_THRESHOLD - soul_spark.harmony
    weakness_scores['torus'] = HS_TRIGGER_FACTOR_THRESHOLD - soul_spark.toroidal_flow_strength

    # Find the factor with the largest positive weakness score (most below threshold)
    max_weakness = -1.0
    target_factor_key = 'none'
    for key, score in weakness_scores.items():
        if score > FLOAT_EPSILON and score > max_weakness:
            max_weakness = score
            target_factor_key = key

    changes['targeted_factor'] = target_factor_key
    logger.debug(f"  HS Cycle {cycle_num}: Weakness Scores={ {k: f'{v:.3f}' for k,v in weakness_scores.items()} } -> Targeting: {target_factor_key}")

    # --- Apply Targeted Action ---
    if target_factor_key == 'phase':
        if hasattr(soul_spark, '_optimize_phase_coherence'):
            improvement = soul_spark._optimize_phase_coherence(HS_PHASE_ADJUST_RATE)
            changes['delta_phase_var'] = -improvement # Improvement reduces variance
            logger.debug(f"    Phase Opt Action: Rate={HS_PHASE_ADJUST_RATE:.4f}, Improvement={improvement:.4f}")
        else: logger.warning("    Phase Opt Skipped: Method missing.")

    elif target_factor_key == 'harmonics':
        harmonics_list = soul_spark.harmonics
        base_freq = soul_spark.frequency
        if harmonics_list and base_freq > FLOAT_EPSILON:
            new_harmonics = []
            total_adjustment = 0
            for h in harmonics_list:
                ratio = h / base_freq
                if ratio <= 0 or not np.isfinite(ratio): new_harmonics.append(h); continue
                int_dists = [abs(ratio - n) for n in range(1, 6)]
                phi_powers = [PHI**n for n in [1, -1, 2, -2]]
                phi_dists = [abs(ratio - p) for p in phi_powers]
                all_dists = int_dists + phi_dists
                best_dist_idx = np.argmin(all_dists)
                if best_dist_idx < len(int_dists): nearest_ideal_ratio = float(range(1, 6)[best_dist_idx])
                else: nearest_ideal_ratio = phi_powers[best_dist_idx - len(int_dists)]

                target_harmonic = base_freq * nearest_ideal_ratio
                adjustment = (target_harmonic - h) * current_harm_dev * HS_HARMONIC_ADJUST_RATE # Scale by deviation
                adjusted_h = h + adjustment
                new_harmonics.append(adjusted_h)
                total_adjustment += abs(adjustment)
            # Update harmonics lists
            soul_spark.harmonics = sorted(new_harmonics)
            soul_spark.frequency_signature['frequencies'] = sorted(new_harmonics)
            final_harm_dev = _calculate_harmonic_deviation(new_harmonics, base_freq)
            changes['delta_harm_dev'] = final_harm_dev - current_harm_dev
            logger.debug(f"    Harmonic Nudge Action: Rate={HS_HARMONIC_ADJUST_RATE:.4f}, AvgDev {current_harm_dev:.3f}->{final_harm_dev:.3f}, TotalAdj={total_adjustment:.2f}")

    elif target_factor_key == 'phi':
        current_phi = soul_spark.phi_resonance
        # Scale rate by current torus and pattern coherence
        rate_mod = 0.5 + (soul_spark.toroidal_flow_strength + soul_spark.pattern_coherence) / 4.0
        delta = HS_BASE_RATE_PHI * rate_mod * (1.0 - current_phi) # Diminishing returns
        soul_spark.phi_resonance = min(1.0, current_phi + delta)
        changes['delta_phi'] = soul_spark.phi_resonance - current_phi
        logger.debug(f"    Phi Action: RateMod={rate_mod:.3f}, Delta={delta:.5f}")

    elif target_factor_key == 'pcoh':
        current_pcoh = soul_spark.pattern_coherence
        # Scale rate by current torus and phi resonance
        rate_mod = 0.5 + (soul_spark.toroidal_flow_strength + soul_spark.phi_resonance) / 4.0
        delta = HS_BASE_RATE_PCOH * rate_mod * (1.0 - current_pcoh)
        soul_spark.pattern_coherence = min(1.0, current_pcoh + delta)
        changes['delta_pcoh'] = soul_spark.pattern_coherence - current_pcoh
        logger.debug(f"    P.Coh Action: RateMod={rate_mod:.3f}, Delta={delta:.5f}")

    elif target_factor_key == 'harmony':
        current_harmony = soul_spark.harmony
        # Scale rate by phase coherence, harmonic purity, pcoh, phi
        rate_mod = 0.2 + (current_phase_coh + current_harm_purity + soul_spark.pattern_coherence + soul_spark.phi_resonance) / 5.0
        delta = HS_BASE_RATE_HARMONY * rate_mod * (1.0 - current_harmony)
        soul_spark.harmony = min(1.0, current_harmony + delta)
        changes['delta_harmony'] = soul_spark.harmony - current_harmony
        logger.debug(f"    Harmony Action: RateMod={rate_mod:.3f}, Delta={delta:.5f}")

    elif target_factor_key == 'torus':
        current_torus = soul_spark.toroidal_flow_strength
        # Scale rate by current stability, coherence, harmony
        rate_mod = 0.3 + (soul_spark.stability/MAX_STABILITY_SU + soul_spark.coherence/MAX_COHERENCE_CU + soul_spark.harmony) / 4.0
        delta = HS_BASE_RATE_TORUS * rate_mod * (1.0 - current_torus)
        soul_spark.toroidal_flow_strength = min(1.0, current_torus + delta)
        changes['delta_torus'] = soul_spark.toroidal_flow_strength - current_torus
        logger.debug(f"    Torus Action: RateMod={rate_mod:.3f}, Delta={delta:.5f}")

    # --- Energy Adjustment based on Harmony Change ---
    harmony_change = soul_spark.harmony - initial_harmony
    energy_adjustment = harmony_change * HS_ENERGY_ADJUST_FACTOR # Can be positive or negative
    soul_spark.energy = min(MAX_SOUL_ENERGY_SEU, max(0.0, soul_spark.energy + energy_adjustment))
    changes['delta_energy'] = energy_adjustment
    logger.debug(f"    Energy Adj: HarmonyChange={harmony_change:+.4f}, EnergyDelta={energy_adjustment:+.2f}")

    return changes

# --- Orchestration Function ---
def perform_harmonic_strengthening(soul_spark: SoulSpark, intensity: float = HARMONIC_STRENGTHENING_INTENSITY_DEFAULT, duration_factor: float = HARMONIC_STRENGTHENING_DURATION_FACTOR_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete harmonic strengthening using targeted refinement cycles. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    # Intensity/Duration factor might be repurposed or removed if cycles drive duration
    if not (0.1 <= intensity <= 1.0): raise ValueError("Intensity out of range.")
    if not (0.1 <= duration_factor <= 2.0): raise ValueError("Duration factor out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    # Adjust max cycles based on duration factor? Or remove duration factor?
    # Let's keep it simple for now: use fixed max cycles.
    max_cycles = HS_MAX_CYCLES
    logger.info(f"--- Starting Harmonic Strengthening [Targeted Cycles] for Soul {spark_id} (Max Cycles={max_cycles}) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    cycle_changes_log = [] # Store changes per cycle

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark):
            # Hard fail - prerequisites not met
            raise ValueError("Soul prerequisites for harmonic strengthening not met.")

        # Store Initial State
        initial_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy, 'frequency_hz': soul_spark.frequency,
            'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence, 'harmony': soul_spark.harmony,
            'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
            # Add initial phase/harmonic metrics
            'phase_coherence_initial': 1.0 - _calculate_circular_variance(np.array(soul_spark.frequency_signature.get('phases', []))),
            'harmonic_purity_initial': 1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)
        }
        logger.info(f"HS Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, Phi={initial_state['phi_resonance']:.3f}, P.Coh={initial_state['pattern_coherence']:.3f}, Harm={initial_state['harmony']:.3f}, Torus={initial_state['toroidal_flow_strength']:.3f}, PhaseCoh={initial_state['phase_coherence_initial']:.3f}, HarmPur={initial_state['harmonic_purity_initial']:.3f}")

        # --- Refinement Loop ---
        cycles_run = 0
        for cycle in range(max_cycles):
            cycles_run = cycle + 1
            logger.debug(f"--- HS Cycle {cycles_run}/{max_cycles} ---")
            cycle_changes = _run_hs_refinement_cycle(soul_spark, cycles_run)
            cycle_changes_log.append(cycle_changes)

            # Periodic State Update & Check Exit Condition
            if cycle % HS_UPDATE_STATE_INTERVAL == (HS_UPDATE_STATE_INTERVAL - 1):
                logger.debug(f"  Updating S/C at end of cycle {cycles_run}")
                if hasattr(soul_spark, 'update_state'):
                    soul_spark.update_state()
                    logger.debug(f"  S/C after update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
                else:
                    raise AttributeError("SoulSpark missing 'update_state' method.")

                # Check if all targets met
                phases = np.array(soul_spark.frequency_signature.get('phases', []))
                current_phase_coh = 1.0 - _calculate_circular_variance(phases)
                current_harm_purity = 1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency)

                all_met = (soul_spark.stability >= HS_TRIGGER_STABILITY_SU and
                           soul_spark.coherence >= HS_TRIGGER_COHERENCE_CU and
                           current_phase_coh >= HS_TRIGGER_PHASE_COHERENCE and
                           current_harm_purity >= HS_TRIGGER_HARMONIC_PURITY and
                           soul_spark.phi_resonance >= HS_TRIGGER_FACTOR_THRESHOLD and
                           soul_spark.pattern_coherence >= HS_TRIGGER_FACTOR_THRESHOLD and
                           soul_spark.harmony >= HS_TRIGGER_FACTOR_THRESHOLD and
                           soul_spark.toroidal_flow_strength >= HS_TRIGGER_FACTOR_THRESHOLD)

                if all_met:
                    logger.info(f"Harmonic Strengthening convergence thresholds met after {cycles_run} cycles.")
                    break
        else: # Loop completed without breaking
            logger.warning(f"Harmonic Strengthening reached max cycles ({max_cycles}) without full convergence.")

        # --- Final Update & Metrics ---
        logger.info("Performing final state update after HS cycles...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
        else:
            raise AttributeError("SoulSpark missing 'update_state' method.")

        setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, True)
        setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, True)
        
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(f"Harmonically strengthened (Cycles: {cycles_run}). S:{soul_spark.stability:.1f}, C:{soul_spark.coherence:.1f}")

        # Compile Overall Metrics
        end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy, 'frequency_hz': soul_spark.frequency,
            'phi_resonance': soul_spark.phi_resonance,
            'pattern_coherence': soul_spark.pattern_coherence, 'harmony': soul_spark.harmony,
            'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
            'phase_coherence_final': 1.0 - _calculate_circular_variance(np.array(soul_spark.frequency_signature.get('phases', []))),
            'harmonic_purity_final': 1.0 - _calculate_harmonic_deviation(soul_spark.harmonics, soul_spark.frequency),
            FLAG_HARMONICALLY_STRENGTHENED: getattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED)
        }
        stability_gain = final_state['stability_su'] - initial_state['stability_su']
        coherence_gain = final_state['coherence_cu'] - initial_state['coherence_cu']

        overall_metrics = {
            'action': 'harmonic_strengthening', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'cycles_run': cycles_run, 'max_cycles': max_cycles,
            'initial_state': initial_state, 'final_state': final_state,
            'stability_gain_su': float(stability_gain),
            'coherence_gain_cu': float(coherence_gain),
            'success': True,
            # 'cycle_details': cycle_changes_log # Optional: log detailed changes per cycle
        }
        if METRICS_AVAILABLE: metrics.record_metrics('harmonic_strengthening_summary', overall_metrics)

        logger.info(f"--- Harmonic Strengthening Completed Successfully for Soul {spark_id} ({cycles_run} cycles) ---")
        logger.info(f"  Final State: S={soul_spark.stability:.1f} SU (+{stability_gain:.1f}), C={soul_spark.coherence:.1f} CU (+{coherence_gain:.1f})")
        logger.info(f"  Final Factors: Phi={final_state['phi_resonance']:.3f}, P.Coh={final_state['pattern_coherence']:.3f}, Harm={final_state['harmony']:.3f}, Torus={final_state['toroidal_flow_strength']:.3f}")
        logger.info(f"  Final Freq/Phase/Harm: Freq={final_state['frequency_hz']:.1f}, PhaseCoh={final_state['phase_coherence_final']:.3f}, HarmPur={final_state['harmonic_purity_final']:.3f}")

        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Harmonic Strengthening failed for {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
         failed_step = 'prerequisites/validation'
         record_hs_failure(spark_id, start_time_iso, failed_step, str(e_val))
         # Hard fail
         raise e_val
    except RuntimeError as e_rt:
         logger.critical(f"Harmonic Strengthening failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = 'runtime'
         record_hs_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
         setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
         # Hard fail
         raise e_rt
    except Exception as e:
         logger.critical(f"Unexpected error during Harmonic Strengthening for {spark_id}: {e}", exc_info=True)
         failed_step = 'unexpected'
         record_hs_failure(spark_id, start_time_iso, failed_step, str(e))
         setattr(soul_spark, FLAG_HARMONICALLY_STRENGTHENED, False)
         setattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False)
         # Hard fail
         raise RuntimeError(f"Unexpected Harmonic Strengthening failure: {e}") from e

# --- Failure Metric Helper ---
def record_hs_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('harmonic_strengthening_summary', {
                'action': 'harmonic_strengthening', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time,
                'duration_seconds': duration,
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record HS failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/harmonic_strengthening.py ---