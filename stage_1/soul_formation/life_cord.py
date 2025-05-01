# --- START OF FILE src/stage_1/soul_formation/life_cord.py ---

"""
Life Cord Formation Functions (Refactored V4.1 - SEU/SU/CU Units)

Forms the life cord, using absolute SU/CU prerequisites and applying stability bonus.
Cord integrity and related factors remain 0-1 scores. Bandwidth in Hz.
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
    logger.debug(f"Checking life cord prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): return False

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False): # Set by HS
        logger.error(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_LIFE_CORD}.")
        return False

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', 0.0)
    coherence_cu = getattr(soul_spark, 'coherence', 0.0)
    if stability_su < CORD_STABILITY_THRESHOLD_SU: # Use new constant name if changed
        logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {CORD_STABILITY_THRESHOLD_SU} SU.")
        return False
    if coherence_cu < CORD_COHERENCE_THRESHOLD_CU: # Use new constant name if changed
        logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {CORD_COHERENCE_THRESHOLD_CU} CU.")
        return False

    if getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_CORD_FORMATION_COMPLETE}. Re-running.")

    logger.debug("Life cord prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. """
    logger.debug(f"Ensuring properties for life cord formation (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'position', 'field_radius', 'field_strength']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Life Cord: {missing}")

    # Initialize life_cord structure if missing
    if not hasattr(soul_spark, 'life_cord') or getattr(soul_spark, 'life_cord') is None:
         setattr(soul_spark, 'life_cord', {})
    if not hasattr(soul_spark, 'cord_integrity'): # Keep 0-1 score
         setattr(soul_spark, 'cord_integrity', 0.0)

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    pos = getattr(soul_spark, 'position')
    if not isinstance(pos, list) or len(pos)!=3: raise ValueError(f"Invalid position: {pos}")
    logger.debug("Soul properties ensured for Life Cord.")

# --- Core Cord Formation Functions (Logic mostly unchanged, uses factors/scores) ---

def _establish_anchor_points(soul_spark: SoulSpark) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """ Establishes anchor points. Connection strength is a 0-1 factor. """
    logger.info(f"Establishing anchor points for soul {soul_spark.spark_id}...")
    try:
        # Use normalized Stability/Coherence scores (0-1) for anchor strength/resonance
        soul_stability_norm = soul_spark.stability / MAX_STABILITY_SU
        soul_coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU

        soul_anchor_strength = soul_stability_norm * ANCHOR_STRENGTH_MODIFIER + (1.0 - ANCHOR_STRENGTH_MODIFIER)
        soul_anchor_resonance = soul_coherence_norm * ANCHOR_STRENGTH_MODIFIER + (1.0 - ANCHOR_STRENGTH_MODIFIER)
        soul_anchor = {
            "position": [float(p) for p in soul_spark.position],
            "frequency": float(soul_spark.frequency),
            "strength": float(max(0.0, min(1.0, soul_anchor_strength))), # 0-1 factor
            "resonance": float(max(0.0, min(1.0, soul_anchor_resonance))) # 0-1 factor
        }

        earth_anchor_freq = EARTH_FREQUENCY # Absolute Hz
        earth_anchor_pos = [float(soul_spark.position[0]), float(soul_spark.position[1]), float(soul_spark.position[2] - 100.0)] # Conceptual offset
        earth_anchor = {
            "position": earth_anchor_pos, "frequency": float(earth_anchor_freq),
            "strength": float(EARTH_ANCHOR_STRENGTH),   # 0-1 factor from constant
            "resonance": float(EARTH_ANCHOR_RESONANCE) # 0-1 factor from constant
        }

        # Calculate connection viability (0-1 factor based on anchor strengths and freq resonance)
        freq_res_score = calculate_resonance(soul_anchor["frequency"], earth_anchor["frequency"]) # Use detailed resonance
        connection_strength = (soul_anchor["strength"] * 0.4 + earth_anchor["strength"] * 0.4 + freq_res_score * 0.2)
        connection_strength = max(0.0, min(1.0, connection_strength)) # 0-1 factor

        logger.info(f"Anchor points established. Connection Strength Factor: {connection_strength:.4f}")
        return soul_anchor, earth_anchor, float(connection_strength)

    except Exception as e: logger.error(f"Error establishing anchor points: {e}", exc_info=True); raise RuntimeError("Anchor point setup failed.") from e

# --- calculate_resonance helper (Duplicate from Journey, consider utility) ---
def calculate_resonance(freq1: float, freq2: float) -> float:
    if min(freq1, freq2) <= FLOAT_EPSILON: return 0.0
    ratio = max(freq1, freq2) / min(freq1, freq2)
    int_res = 0.0
    for i in range(1, 5):
        for j in range(1, 5):
            target = float(max(i,j))/float(min(i,j))
            dev = abs(ratio-target)
            if dev < RESONANCE_INTEGER_RATIO_TOLERANCE * target: int_res = max(int_res, (1.0 - dev / (RESONANCE_INTEGER_RATIO_TOLERANCE * target))**2)
    phi_res = 0.0
    for i in [1, 2]:
        phi_pow = GOLDEN_RATIO ** i
        dev_phi = abs(ratio - phi_pow); dev_inv = abs(ratio - (1.0/phi_pow))
        if dev_phi < RESONANCE_PHI_RATIO_TOLERANCE * phi_pow: phi_res = max(phi_res, (1.0 - dev_phi / (RESONANCE_PHI_RATIO_TOLERANCE * phi_pow))**2)
        if dev_inv < RESONANCE_PHI_RATIO_TOLERANCE * (1.0/phi_pow): phi_res = max(phi_res, (1.0 - dev_inv / (RESONANCE_PHI_RATIO_TOLERANCE * (1.0/phi_pow)))**2)
    return max(int_res, phi_res)

# --- _form_primary_channel, _create_harmonic_nodes, _add_secondary_channels ---
# These functions primarily calculate factors (stability 0-1, resistance 0-1, elasticity 0-1)
# and absolute values (bandwidth Hz, frequency Hz). The logic can remain largely the same,
# just ensure inputs like connection_strength are the 0-1 factors.

def _form_primary_channel(soul_anchor: Dict[str, Any], earth_anchor: Dict[str, Any],
                         connection_strength: float, complexity: float) -> Dict[str, Any]:
    """ Forms primary channel. Bandwidth/Freq in Hz, others 0-1 factors. """
    logger.info("Forming primary channel...")
    if not (0.0 <= connection_strength <= 1.0): raise ValueError("connection_strength must be 0-1")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")
    try:
        channel_bandwidth = connection_strength * complexity * PRIMARY_CHANNEL_BANDWIDTH_FACTOR # Hz
        channel_stability = connection_strength * PRIMARY_CHANNEL_STABILITY_FACTOR_CONN + complexity * PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX # 0-1
        interference_resistance = connection_strength * PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN + complexity * PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX # 0-1
        primary_frequency = (soul_anchor['frequency'] * 0.8 + earth_anchor['frequency'] * 0.2) # Hz
        elasticity = PRIMARY_CHANNEL_ELASTICITY_BASE + complexity * PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX # 0-1

        primary_channel_data = {
            "bandwidth_hz": float(max(0.0, channel_bandwidth)),
            "stability_factor": float(max(0.0, min(1.0, channel_stability))),
            "interference_resistance": float(max(0.0, min(1.0, interference_resistance))),
            "primary_frequency_hz": float(primary_frequency),
            "elasticity_factor": float(max(0.0, min(1.0, elasticity))),
            "channel_count": 1
        }
        logger.info(f"Primary channel formed: Freq={primary_frequency:.1f}Hz, BW={channel_bandwidth:.1f}Hz, Stab={primary_channel_data['stability_factor']:.3f}")
        return primary_channel_data
    except Exception as e: logger.error(f"Error forming primary channel: {e}", exc_info=True); raise RuntimeError("Primary channel form failed.") from e

def _create_harmonic_nodes(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """ Creates harmonic nodes. Modifies cord_structure bandwidth (Hz). """
    logger.info("Creating harmonic nodes...")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")
    primary_freq = cord_structure.get("primary_frequency_hz")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON: raise ValueError("Invalid primary frequency in cord.")
    try:
        num_nodes = HARMONIC_NODE_COUNT_BASE + int(complexity * HARMONIC_NODE_COUNT_FACTOR)
        nodes = []
        for i in range(num_nodes):
            position = (i + 1.0) / (num_nodes + 1.0)
            freq = 0.0; harmonic_type = "unknown"
            type_idx = i % 3
            if type_idx == 0: phi_exp = (i//3)%5+1; freq = primary_freq * (PHI**phi_exp); harmonic_type = f"phi^{phi_exp}"
            elif type_idx == 1: int_mult = (i//3)%7+2; freq = primary_freq * int_mult; harmonic_type = f"int*{int_mult}"
            else: silver_exp = (i//3)%3+1; freq = primary_freq * (SILVER_RATIO**silver_exp); harmonic_type = f"silver^{silver_exp}"
            amplitude = (HARMONIC_NODE_AMP_BASE + complexity * HARMONIC_NODE_AMP_FACTOR_COMPLEX) * (1.0 - position * HARMONIC_NODE_AMP_FALLOFF) # 0-1 factor

            if freq > FLOAT_EPSILON and amplitude > FLOAT_EPSILON:
                 nodes.append({"position": float(position), "frequency_hz": float(freq), "type": harmonic_type, "amplitude": float(max(0.0, min(1.0, amplitude)))})

        cord_structure["harmonic_nodes"] = nodes
        bandwidth_increase_hz = len(nodes) * complexity * HARMONIC_NODE_BW_INCREASE_FACTOR # Hz increase
        cord_structure["bandwidth_hz"] = cord_structure.get("bandwidth_hz", 0.0) + bandwidth_increase_hz

        phase_metrics = { "num_nodes": len(nodes), "bandwidth_increase_hz": bandwidth_increase_hz, "final_bandwidth_hz": cord_structure["bandwidth_hz"], "timestamp": datetime.now().isoformat() }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_nodes', phase_metrics)
        logger.info(f"Created {len(nodes)} nodes. Total BW: {cord_structure['bandwidth_hz']:.1f} Hz.")
        return phase_metrics
    except Exception as e: logger.error(f"Error creating nodes: {e}", exc_info=True); raise RuntimeError("Node creation failed.") from e

def _add_secondary_channels(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """ Adds secondary channels. Modifies cord_structure bandwidth (Hz). """
    logger.info("Adding secondary channels...")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")
    primary_freq = cord_structure.get("primary_frequency_hz")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON: raise ValueError("Invalid primary frequency.")
    try:
        num_channels = min(MAX_CORD_CHANNELS - 1, int(complexity * SECONDARY_CHANNEL_COUNT_FACTOR))
        secondary_channels = []
        total_secondary_bw_hz = 0.0
        types=["emotional","mental","spiritual"]; bw_configs={"e":SECONDARY_CHANNEL_BW_EMOTIONAL,"m":SECONDARY_CHANNEL_BW_MENTAL,"s":SECONDARY_CHANNEL_BW_SPIRITUAL}; resist_configs={"e":SECONDARY_CHANNEL_RESIST_EMOTIONAL,"m":SECONDARY_CHANNEL_RESIST_MENTAL,"s":SECONDARY_CHANNEL_RESIST_SPIRITUAL}

        for i in range(num_channels):
            ch_type_key = types[i % len(types)][0] # Use first letter for key lookup
            ch_type = types[i % len(types)]
            bw_base, bw_factor = bw_configs[ch_type_key]; resist_base, resist_factor = resist_configs[ch_type_key]
            bandwidth = bw_base + complexity * bw_factor; resistance = resist_base + complexity * resist_factor # 0-1 factor
            frequency = primary_freq * (1.0 + (i + 1) * SECONDARY_CHANNEL_FREQ_FACTOR) # Hz

            if bandwidth > FLOAT_EPSILON and frequency > FLOAT_EPSILON:
                 secondary_channels.append({ "type": ch_type, "bandwidth_hz": float(bandwidth), "interference_resistance": float(max(0.0, min(1.0, resistance))), "frequency_hz": float(frequency) })
                 total_secondary_bw_hz += bandwidth

        cord_structure["secondary_channels"] = secondary_channels
        cord_structure["channel_count"] = 1 + len(secondary_channels)
        cord_structure["bandwidth_hz"] += total_secondary_bw_hz

        phase_metrics = { "num_secondary": len(secondary_channels), "secondary_bw_added_hz": total_secondary_bw_hz, "final_bandwidth_hz": cord_structure["bandwidth_hz"], "final_channel_count": cord_structure["channel_count"], "timestamp": datetime.now().isoformat() }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_secondary', phase_metrics)
        logger.info(f"Added {len(secondary_channels)} secondary channels. Total BW: {cord_structure['bandwidth_hz']:.1f} Hz.")
        return phase_metrics
    except Exception as e: logger.error(f"Error adding secondary channels: {e}", exc_info=True); raise RuntimeError("Secondary channel add failed.") from e

def _integrate_with_soul_field(soul_spark: SoulSpark, cord_structure: Dict[str, Any], connection_strength: float) -> Dict[str, Any]:
    """ Integrates cord. Modifies SoulSpark field_radius(relative?), field_strength(0-1), field_integration(0-1). """
    logger.info(f"Integrating cord with soul field for {soul_spark.spark_id}...")
    try:
        current_radius = getattr(soul_spark, "field_radius", 1.0) # Relative size? Or should this be absolute? Let's keep relative for now.
        current_strength = getattr(soul_spark, "field_strength", 0.5) # 0-1 factor

        # Integration strength factor (0-1)
        integration_strength = (current_strength * FIELD_INTEGRATION_FACTOR_FIELD_STR + connection_strength * FIELD_INTEGRATION_FACTOR_CONN_STR)
        integration_strength = max(0.0, min(1.0, integration_strength))

        # Field expands slightly (relative radius increases)
        new_radius = current_radius * FIELD_EXPANSION_FACTOR
        # Field strength factor (0-1) increases slightly
        new_strength = min(1.0, current_strength + 0.05 * integration_strength)

        soul_spark.field_radius = float(new_radius)
        soul_spark.field_strength = float(new_strength)
        soul_spark.field_integration = float(integration_strength) # Store 0-1 integration score
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        phase_metrics = { "integration_strength": integration_strength, "field_radius_after": new_radius, "field_strength_after": new_strength, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_integration', phase_metrics)
        logger.info(f"Integrated cord with soul field. Integration Strength: {integration_strength:.4f}")
        return phase_metrics
    except Exception as e: logger.error(f"Error integrating cord with field: {e}", exc_info=True); raise RuntimeError("Cord field integration failed.") from e

def _establish_earth_connection(cord_structure: Dict[str, Any], connection_strength: float) -> Tuple[float, float, Dict[str, Any]]:
    """ Establishes Earth connection. Returns earth_connection(0-1) and final_cord_integrity(0-1). """
    logger.info("Establishing Earth connection...")
    try:
        elasticity = cord_structure.get("elasticity_factor") # 0-1
        primary_stability = cord_structure.get("stability_factor") # 0-1
        if elasticity is None or primary_stability is None: raise ValueError("Missing factors in cord.")

        # Earth connection factor (0-1)
        earth_connection = (connection_strength * EARTH_CONN_FACTOR_CONN_STR + elasticity * EARTH_CONN_FACTOR_ELASTICITY + EARTH_CONN_BASE_FACTOR)
        earth_connection = max(0.1, min(0.95, earth_connection)) # Clamp 0.1-0.95
        cord_structure["earth_connection_factor"] = float(earth_connection)

        # Final cord integrity score (0-1)
        final_cord_integrity = (connection_strength * CORD_INTEGRITY_FACTOR_CONN_STR + primary_stability * CORD_INTEGRITY_FACTOR_STABILITY + earth_connection * CORD_INTEGRITY_FACTOR_EARTH_CONN)
        final_cord_integrity = max(0.0, min(1.0, final_cord_integrity))

        timestamp = datetime.now().isoformat()
        phase_metrics = { "earth_connection_factor": earth_connection, "final_cord_integrity": final_cord_integrity, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_earth_connection', phase_metrics)
        logger.info(f"Established Earth connection. Strength Factor: {earth_connection:.4f}, Integrity: {final_cord_integrity:.4f}")
        return float(earth_connection), float(final_cord_integrity), phase_metrics

    except Exception as e: logger.error(f"Error establishing Earth connection: {e}", exc_info=True); raise RuntimeError("Earth connection failed.") from e

# --- Orchestration Function ---
def form_life_cord(soul_spark: SoulSpark, complexity: float = LIFE_CORD_COMPLEXITY_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Forms complete life cord. Modifies SoulSpark. Applies stability (SU) bonus. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Life Cord Formation for Soul {spark_id} (Complexity={complexity:.2f}) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    final_cord_structure = {}

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark): # Uses SU/CU thresholds
            raise ValueError("Soul prerequisites for life cord formation not met.")

        # Store Initial State (SU and factors)
        initial_state = { 'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence, 'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength, 'cord_integrity': soul_spark.cord_integrity }

        # --- Run Phases ---
        soul_anchor, earth_anchor, connection_strength = _establish_anchor_points(soul_spark)
        process_metrics_summary['steps']['anchors'] = {'connection_strength_factor': connection_strength}

        final_cord_structure = _form_primary_channel(soul_anchor, earth_anchor, connection_strength, complexity)
        metrics_nodes = _create_harmonic_nodes(final_cord_structure, complexity)
        metrics_secondary = _add_secondary_channels(final_cord_structure, complexity)
        metrics_integration = _integrate_with_soul_field(soul_spark, final_cord_structure, connection_strength)
        earth_conn_strength, final_integrity, metrics_earth = _establish_earth_connection(final_cord_structure, connection_strength)
        process_metrics_summary['steps'].update({ 'nodes': metrics_nodes, 'secondary': metrics_secondary, 'integration': metrics_integration, 'earth_conn': metrics_earth })

        # --- Update SoulSpark ---
        setattr(soul_spark, "life_cord", final_cord_structure.copy())
        setattr(soul_spark, "cord_integrity", final_integrity) # 0-1 score
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, True)

        # Apply stability bonus (SU) based on final cord integrity (0-1 score)
        stability_bonus_su = final_integrity * FINAL_STABILITY_BONUS_FACTOR * MAX_STABILITY_SU # Scale by max SU
        new_stability_su = min(MAX_STABILITY_SU, initial_state['stability_su'] + stability_bonus_su)
        setattr(soul_spark, "stability", new_stability_su)
        stability_gain_su = new_stability_su - initial_state['stability_su']

        setattr(soul_spark, FLAG_READY_FOR_EARTH, True)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Recalc SU/CU scores if needed
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Life cord formed. Integrity:{final_integrity:.3f}, BW:{final_cord_structure.get('bandwidth_hz', 0.0):.1f} Hz, Stab Gain:{stability_gain_su:.1f}SU")

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report state in correct units
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength,
            'cord_integrity': final_integrity, 'final_bandwidth_hz': final_cord_structure.get('bandwidth_hz', 0.0),
            FLAG_CORD_FORMATION_COMPLETE: getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE) }
        overall_metrics = {
            'action': 'form_life_cord', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'complexity_setting': complexity,
            'initial_state': initial_state, 'final_state': final_state,
            'final_cord_integrity': final_integrity, 'final_bandwidth_hz': final_state['final_bandwidth_hz'],
            'stability_gain_su': stability_gain_su, 'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_summary', overall_metrics)

        logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Life Cord formation failed for {spark_id} due to validation error: {e_val}", exc_info=True)
         record_lc_failure(spark_id, start_time_iso, 'prerequisites/validation', str(e_val))
         raise
    except RuntimeError as e_rt:
         logger.critical(f"Life Cord formation failed critically for {spark_id}: {e_rt}", exc_info=True)
         record_lc_failure(spark_id, start_time_iso, 'runtime', str(e_rt))
         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_EARTH, False)
         raise
    except Exception as e:
         logger.critical(f"Unexpected error during Life Cord formation for {spark_id}: {e}", exc_info=True)
         record_lc_failure(spark_id, start_time_iso, 'unexpected', str(e))
         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_EARTH, False)
         raise RuntimeError(f"Unexpected Life Cord formation failure: {e}") from e

# --- Failure Metric Helper ---
def record_lc_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('life_cord_summary', {
                'action': 'form_life_cord', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record LC failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/life_cord.py ---