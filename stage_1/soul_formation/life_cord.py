# --- START OF FILE src/stage_1/soul_formation/life_cord.py ---

"""
Life Cord Formation Functions (Refactored V4.3.8 - Principle-Driven)

Manifests the life cord based on Creator connection quality, anchored by the soul.
Soul activates its anchor point with minimal energy cost. Cord integrity/bandwidth emerge.
Stability might be slightly boosted via update_state due to organized anchoring.
Modifies the SoulSpark object instance directly. Uses constants. Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional
from constants.constants import *
# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Life Cord cannot function.")
    # Define minimal fallbacks for parsing
    # FLAG_READY_FOR_LIFE_CORD = "ready_for_life_cord"; CORD_STABILITY_THRESHOLD_SU = 80.0; CORD_COHERENCE_THRESHOLD_CU = 80.0
    # FLAG_CORD_FORMATION_COMPLETE = "cord_formation_complete"; MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0
    # FLOAT_EPSILON = 1e-9; EARTH_FREQUENCY = 136.1; ANCHOR_STRENGTH_MODIFIER = 0.6; EARTH_ANCHOR_STRENGTH = 0.9
    # EARTH_ANCHOR_RESONANCE = 0.9; GOLDEN_RATIO = 1.618; RESONANCE_INTEGER_RATIO_TOLERANCE = 0.02; RESONANCE_PHI_RATIO_TOLERANCE = 0.03
    # PRIMARY_CHANNEL_BANDWIDTH_FACTOR = 200.0; PRIMARY_CHANNEL_STABILITY_FACTOR_CONN = 0.7; PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX = 0.3
    # PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN = 0.6; PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX = 0.4
    # PRIMARY_CHANNEL_ELASTICITY_BASE = 0.5; PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX = 0.3
    # HARMONIC_NODE_COUNT_BASE = 3; HARMONIC_NODE_COUNT_FACTOR = 15.0; PHI = GOLDEN_RATIO; SILVER_RATIO = 2.414
    # HARMONIC_NODE_AMP_BASE = 0.4; HARMONIC_NODE_AMP_FACTOR_COMPLEX = 0.4; HARMONIC_NODE_AMP_FALLOFF = 0.6
    # HARMONIC_NODE_BW_INCREASE_FACTOR = 5.0; MAX_CORD_CHANNELS = 7; SECONDARY_CHANNEL_COUNT_FACTOR = 6.0
    # SECONDARY_CHANNEL_FREQ_FACTOR = 0.1; SECONDARY_CHANNEL_BW_EMOTIONAL = (10.0, 30.0); SECONDARY_CHANNEL_RESIST_EMOTIONAL = (0.4, 0.3)
    # SECONDARY_CHANNEL_BW_MENTAL = (15.0, 40.0); SECONDARY_CHANNEL_RESIST_MENTAL = (0.5, 0.3)
    # SECONDARY_CHANNEL_BW_SPIRITUAL = (20.0, 50.0); SECONDARY_CHANNEL_RESIST_SPIRITUAL = (0.6, 0.3)
    # FIELD_INTEGRATION_FACTOR_FIELD_STR = 0.6; FIELD_INTEGRATION_FACTOR_CONN_STR = 0.4; FIELD_EXPANSION_FACTOR = 1.05
    # EARTH_CONN_FACTOR_CONN_STR = 0.5; EARTH_CONN_FACTOR_ELASTICITY = 0.3; EARTH_CONN_BASE_FACTOR = 0.1
    # CORD_INTEGRITY_FACTOR_CONN_STR = 0.4; CORD_INTEGRITY_FACTOR_STABILITY = 0.3; CORD_INTEGRITY_FACTOR_EARTH_CONN = 0.3
    # FLAG_READY_FOR_EARTH = "ready_for_earth"; LIFE_CORD_COMPLEXITY_DEFAULT = 0.7
    # CORD_ACTIVATION_ENERGY_COST = 50.0 # SEU cost for soul to activate anchor
    # # Need LIFE_CORD_FREQUENCIES if not defined elsewhere
    # if 'LIFE_CORD_FREQUENCIES' not in globals(): LIFE_CORD_FREQUENCIES = {'primary': 528.0}
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    # Import resonance calculation
    from .creator_entanglement import calculate_resonance
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}.")
    raise ImportError(f"Core dependencies missing: {e}") from e

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

# --- Helper Functions ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
    logger.debug(f"Checking life cord prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_READY_FOR_LIFE_CORD, False):
        msg = f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_LIFE_CORD}."
        logger.error(msg); raise ValueError(msg)

    # 2. Minimum Stability and Coherence (Absolute SU/CU)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < 0 or coherence_cu < 0:
        msg = "Prerequisite failed: Soul missing stability or coherence attributes."
        logger.error(msg); raise AttributeError(msg)

    if stability_su < CORD_STABILITY_THRESHOLD_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {CORD_STABILITY_THRESHOLD_SU} SU."
        logger.error(msg); raise ValueError(msg)
    if coherence_cu < CORD_COHERENCE_THRESHOLD_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {CORD_COHERENCE_THRESHOLD_CU} CU."
        logger.error(msg); raise ValueError(msg)

    # 3. Energy Check (Done in main function before cost)

    if getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_CORD_FORMATION_COMPLETE}. Re-running.")

    logger.debug("Life cord prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Raises error if missing/invalid. """
    logger.debug(f"Ensuring properties for life cord formation (Soul {soul_spark.spark_id})...")
    required = ['frequency', 'stability', 'coherence', 'position', 'field_radius',
                'field_strength', 'creator_connection_strength', 'energy']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Life Cord: {missing}")

    if not hasattr(soul_spark, 'life_cord'): setattr(soul_spark, 'life_cord', {})
    if not hasattr(soul_spark, 'cord_integrity'): setattr(soul_spark, 'cord_integrity', 0.0)

    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Soul frequency must be positive.")
    pos = getattr(soul_spark, 'position')
    if not isinstance(pos, list) or len(pos)!=3: raise ValueError(f"Invalid position: {pos}")
    if soul_spark.energy < CORD_ACTIVATION_ENERGY_COST:
        raise ValueError(f"Insufficient energy ({soul_spark.energy:.1f} SEU) for cord activation cost ({CORD_ACTIVATION_ENERGY_COST} SEU).")

    logger.debug("Soul properties ensured for Life Cord.")

# --- Core Cord Formation Functions ---

def _establish_anchor_points(soul_spark: SoulSpark) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """ Establishes anchor points. Connection strength factor emerges from resonance. """
    logger.info("LC Step: Establishing Anchor Points...")
    try:
        soul_stability_norm = soul_spark.stability / MAX_STABILITY_SU
        soul_coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU
        soul_anchor_strength = (soul_stability_norm * 0.6 + soul_coherence_norm * 0.4) * ANCHOR_STRENGTH_MODIFIER
        soul_anchor_resonance = (soul_coherence_norm * 0.7 + soul_stability_norm * 0.3) * ANCHOR_STRENGTH_MODIFIER
        soul_anchor = {
            "position": [float(p) for p in soul_spark.position],
            "frequency": float(soul_spark.frequency),
            "strength": float(max(0.1, min(1.0, soul_anchor_strength))),
            "resonance": float(max(0.1, min(1.0, soul_anchor_resonance)))
        }
        logger.debug(f"  Soul Anchor: Strength={soul_anchor['strength']:.3f}, Resonance={soul_anchor['resonance']:.3f}")

        earth_anchor_freq = EARTH_FREQUENCY
        earth_anchor_pos = [float(p) for p in soul_anchor['position']]
        earth_anchor_pos[2] -= 100.0 # Conceptual offset
        earth_anchor = {
            "position": earth_anchor_pos, "frequency": float(earth_anchor_freq),
            "strength": float(EARTH_ANCHOR_STRENGTH),
            "resonance": float(EARTH_ANCHOR_RESONANCE)
        }
        logger.debug(f"  Earth Anchor: Strength={earth_anchor['strength']:.3f}, Resonance={earth_anchor['resonance']:.3f}")

        freq_res_score = calculate_resonance(soul_anchor["frequency"], earth_anchor["frequency"])
        connection_strength = freq_res_score * (soul_anchor["resonance"] * 0.6 + earth_anchor["resonance"] * 0.4)
        connection_strength = max(0.0, min(1.0, connection_strength))
        logger.info(f"Anchor points established. FreqRes={freq_res_score:.3f} -> ConnStrengthFactor={connection_strength:.4f}")

        return soul_anchor, earth_anchor, float(connection_strength)

    except Exception as e:
        logger.error(f"Error establishing anchor points: {e}", exc_info=True)
        raise RuntimeError("Anchor point setup failed critically.") from e

def _form_primary_channel(soul_anchor: Dict[str, Any], earth_anchor: Dict[str, Any],
                         connection_strength: float, creator_connection_strength: float,
                         complexity: float) -> Dict[str, Any]:
    """ Forms primary channel based on Creator-initiated template and connection quality. """
    logger.info("LC Step: Forming Primary Channel...")
    # Validation
    if not (0.0 <= connection_strength <= 1.0): raise ValueError("connection_strength invalid.")
    if not (0.0 <= creator_connection_strength <= 1.0): raise ValueError("creator_connection_strength invalid.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity invalid.")
    try:
        # Properties emerge based on connection qualities and complexity
        channel_bandwidth = (connection_strength * 0.5 + creator_connection_strength * 0.5) * complexity * PRIMARY_CHANNEL_BANDWIDTH_FACTOR
        channel_stability = (soul_anchor['strength'] * 0.3 + earth_anchor['strength'] * 0.3 + creator_connection_strength * 0.4)
        interference_resistance = (soul_anchor['resonance'] * 0.3 + earth_anchor['resonance'] * 0.3 + creator_connection_strength * 0.4)
        soul_weight = soul_anchor['resonance']; earth_weight = earth_anchor['resonance']
        total_weight = soul_weight + earth_weight
        primary_frequency = ((soul_anchor['frequency'] * soul_weight + earth_anchor['frequency'] * earth_weight) / total_weight if total_weight > FLOAT_EPSILON else soul_anchor['frequency'])
        elasticity = PRIMARY_CHANNEL_ELASTICITY_BASE + complexity * 0.2 + creator_connection_strength * 0.3

        primary_channel_data = {
            "bandwidth_hz": float(max(FLOAT_EPSILON, channel_bandwidth)),
            "stability_factor": float(max(0.0, min(1.0, channel_stability))),
            "interference_resistance": float(max(0.0, min(1.0, interference_resistance))),
            "primary_frequency_hz": float(primary_frequency),
            "elasticity_factor": float(max(0.0, min(1.0, elasticity))),
            "channel_count": 1,
            "creator_influence": float(creator_connection_strength)
        }
        logger.info(f"Primary channel formed: Freq={primary_frequency:.1f}Hz, BW={primary_channel_data['bandwidth_hz']:.1f}Hz, StabFactor={primary_channel_data['stability_factor']:.3f}")
        return primary_channel_data
    except Exception as e:
        logger.error(f"Error forming primary channel: {e}", exc_info=True)
        raise RuntimeError("Primary channel formation failed critically.") from e

def _create_harmonic_nodes(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """ Creates harmonic nodes based on primary freq & complexity. Modifies cord_structure. """
    logger.info("LC Step: Creating Harmonic Nodes...")
    if not isinstance(cord_structure, dict): raise TypeError("cord_structure must be a dict.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")
    primary_freq = cord_structure.get("primary_frequency_hz")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON:
        raise ValueError(f"Invalid primary frequency in cord: {primary_freq}")
    try:
        num_nodes = HARMONIC_NODE_COUNT_BASE + int(complexity * HARMONIC_NODE_COUNT_FACTOR)
        nodes = []
        logger.debug(f"  Creating {num_nodes} harmonic nodes (Complexity={complexity:.2f})")
        for i in range(num_nodes):
            position = (i + 1.0) / (num_nodes + 1.0)
            freq = 0.0; harmonic_type = "unknown"; ratio = 1.0
            type_idx = i % 3
            if type_idx == 0: phi_exp=(i//3)%5+1; ratio=PHI**phi_exp; harmonic_type=f"phi^{phi_exp}"
            elif type_idx == 1: int_mult=(i//3)%7+2; ratio=float(int_mult); harmonic_type=f"int*{int_mult}"
            else: silver_exp=(i//3)%3+1; ratio=SILVER_RATIO**silver_exp; harmonic_type=f"silver^{silver_exp}"
            freq = primary_freq * ratio
            amplitude = (HARMONIC_NODE_AMP_BASE + complexity * HARMONIC_NODE_AMP_FACTOR_COMPLEX) * (1.0 - position * HARMONIC_NODE_AMP_FALLOFF)

            if freq > FLOAT_EPSILON and amplitude > FLOAT_EPSILON:
                 nodes.append({"position": float(position), "frequency_hz": float(freq), "type": harmonic_type, "amplitude": float(max(0.0, min(1.0, amplitude)))})
                 # logger.debug(f"    Node {i+1}: Type={harmonic_type}, Freq={freq:.1f}, Amp={amplitude:.3f}")

        cord_structure["harmonic_nodes"] = nodes
        bandwidth_increase_hz = len(nodes) * complexity * HARMONIC_NODE_BW_INCREASE_FACTOR
        new_bandwidth = cord_structure.get("bandwidth_hz", 0.0) + bandwidth_increase_hz
        cord_structure["bandwidth_hz"] = max(FLOAT_EPSILON, new_bandwidth)

        phase_metrics = { "num_nodes": len(nodes), "bandwidth_increase_hz": bandwidth_increase_hz, "final_bandwidth_hz": cord_structure["bandwidth_hz"], "timestamp": datetime.now().isoformat() }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_nodes', phase_metrics)
        logger.info(f"Created {len(nodes)} nodes. BW increase: {bandwidth_increase_hz:.1f} Hz. Total BW: {cord_structure['bandwidth_hz']:.1f} Hz.")
        return phase_metrics
    except Exception as e:
        logger.error(f"Error creating harmonic nodes: {e}", exc_info=True)
        raise RuntimeError("Harmonic node creation failed critically.") from e

def _add_secondary_channels(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """ Adds secondary channels based on complexity. Modifies cord_structure. """
    logger.info("LC Step: Adding Secondary Channels...")
    if not isinstance(cord_structure, dict): raise TypeError("cord_structure must be a dict.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")
    primary_freq = cord_structure.get("primary_frequency_hz")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON:
        raise ValueError(f"Invalid primary frequency in cord: {primary_freq}")
    try:
        num_channels = min(MAX_CORD_CHANNELS - 1, int(complexity * SECONDARY_CHANNEL_COUNT_FACTOR))
        secondary_channels = []
        total_secondary_bw_hz = 0.0
        types=["emotional","mental","spiritual"]; bw_configs={"e":SECONDARY_CHANNEL_BW_EMOTIONAL,"m":SECONDARY_CHANNEL_BW_MENTAL,"s":SECONDARY_CHANNEL_BW_SPIRITUAL}; resist_configs={"e":SECONDARY_CHANNEL_RESIST_EMOTIONAL,"m":SECONDARY_CHANNEL_RESIST_MENTAL,"s":SECONDARY_CHANNEL_RESIST_SPIRITUAL}
        logger.debug(f"  Adding {num_channels} secondary channels (Complexity={complexity:.2f})")

        for i in range(num_channels):
            ch_type_key = types[i % len(types)][0]
            ch_type = types[i % len(types)]
            bw_base, bw_factor = bw_configs[ch_type_key]; resist_base, resist_factor = resist_configs[ch_type_key]
            bandwidth = bw_base + complexity * bw_factor; resistance = resist_base + complexity * resist_factor
            frequency = primary_freq * (1.0 + (i + 1) * SECONDARY_CHANNEL_FREQ_FACTOR)

            if bandwidth > FLOAT_EPSILON and frequency > FLOAT_EPSILON:
                 secondary_channels.append({ "type": ch_type, "bandwidth_hz": float(bandwidth), "interference_resistance": float(max(0.0, min(1.0, resistance))), "frequency_hz": float(frequency) })
                 total_secondary_bw_hz += bandwidth
                 # logger.debug(f"    Channel {i+1}: Type={ch_type}, Freq={frequency:.1f}, BW={bandwidth:.1f}, Resist={resistance:.3f}")


        cord_structure["secondary_channels"] = secondary_channels
        cord_structure["channel_count"] = 1 + len(secondary_channels)
        new_bandwidth = cord_structure.get("bandwidth_hz", 0.0) + total_secondary_bw_hz
        cord_structure["bandwidth_hz"] = max(FLOAT_EPSILON, new_bandwidth)

        phase_metrics = { "num_secondary": len(secondary_channels), "secondary_bw_added_hz": total_secondary_bw_hz, "final_bandwidth_hz": cord_structure["bandwidth_hz"], "final_channel_count": cord_structure["channel_count"], "timestamp": datetime.now().isoformat() }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_secondary', phase_metrics)
        logger.info(f"Added {len(secondary_channels)} secondary channels. BW increase: {total_secondary_bw_hz:.1f} Hz. Total BW: {cord_structure['bandwidth_hz']:.1f} Hz.")
        return phase_metrics
    except Exception as e:
        logger.error(f"Error adding secondary channels: {e}", exc_info=True)
        raise RuntimeError("Secondary channel addition failed critically.") from e

def _integrate_with_soul_field(soul_spark: SoulSpark, cord_structure: Dict[str, Any], connection_strength: float) -> Dict[str, Any]:
    """ Integrates cord with soul field factors. Modifies SoulSpark. """
    logger.info("LC Step: Integrating Cord with Soul Field...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(cord_structure, dict): raise TypeError("cord_structure invalid.")
    if not (0.0 <= connection_strength <= 1.0): raise ValueError("connection_strength invalid.")
    try:
        current_radius = getattr(soul_spark, "field_radius", 1.0)
        current_strength_factor = getattr(soul_spark, "field_strength", 0.5) # 0-1 factor

        integration_strength = (current_strength_factor * FIELD_INTEGRATION_FACTOR_FIELD_STR +
                                connection_strength * FIELD_INTEGRATION_FACTOR_CONN_STR)
        integration_strength = max(0.0, min(1.0, integration_strength))
        logger.debug(f"  Field Integration: CurrentStrengthFactor={current_strength_factor:.3f}, ConnStrength={connection_strength:.3f} -> IntegrationFactor={integration_strength:.4f}")

        new_radius = current_radius * FIELD_EXPANSION_FACTOR
        strength_gain = 0.05 * integration_strength
        new_strength_factor = min(1.0, current_strength_factor + strength_gain)
        logger.debug(f"  Field Integration Update: Radius {current_radius:.3f}->{new_radius:.3f}. StrengthFactor {current_strength_factor:.3f}->{new_strength_factor:.3f} (+{strength_gain:.4f})")

        soul_spark.field_radius = float(new_radius)
        soul_spark.field_strength = float(new_strength_factor) # Store updated 0-1 factor
        soul_spark.field_integration = float(integration_strength) # Store 0-1 integration score
        timestamp = datetime.now().isoformat(); soul_spark.last_modified = timestamp

        phase_metrics = { "integration_strength_factor": integration_strength, "field_radius_after": new_radius, "field_strength_after": new_strength_factor, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_integration', phase_metrics)
        logger.info(f"Integrated cord with soul field. Integration Strength Factor: {integration_strength:.4f}")
        return phase_metrics
    except Exception as e:
        logger.error(f"Error integrating cord with field: {e}", exc_info=True)
        raise RuntimeError("Cord field integration failed critically.") from e

def _establish_earth_connection(cord_structure: Dict[str, Any], connection_strength: float) -> Tuple[float, float, Dict[str, Any]]:
    """ Establishes Earth connection factor (0-1) & final cord integrity (0-1). """
    logger.info("LC Step: Establishing Earth Connection & Final Integrity...")
    if not isinstance(cord_structure, dict): raise TypeError("cord_structure invalid.")
    if not (0.0 <= connection_strength <= 1.0): raise ValueError("connection_strength invalid.")
    try:
        elasticity = cord_structure.get("elasticity_factor")
        primary_stability = cord_structure.get("stability_factor")
        if elasticity is None or primary_stability is None:
            raise ValueError("Missing elasticity or stability factors in cord structure.")

        # Earth connection factor (0-1)
        earth_connection = (connection_strength * EARTH_CONN_FACTOR_CONN_STR +
                           elasticity * EARTH_CONN_FACTOR_ELASTICITY +
                           EARTH_CONN_BASE_FACTOR)
        earth_connection = max(0.1, min(0.95, earth_connection))
        cord_structure["earth_connection_factor"] = float(earth_connection)
        logger.debug(f"  Earth Connection Factor: {earth_connection:.4f}")

        # Final cord integrity score (0-1)
        final_cord_integrity = (connection_strength * CORD_INTEGRITY_FACTOR_CONN_STR +
                                primary_stability * CORD_INTEGRITY_FACTOR_STABILITY +
                                earth_connection * CORD_INTEGRITY_FACTOR_EARTH_CONN)
        final_cord_integrity = max(0.0, min(1.0, final_cord_integrity))
        logger.debug(f"  Final Cord Integrity Factor: {final_cord_integrity:.4f}")

        timestamp = datetime.now().isoformat()
        phase_metrics = { "earth_connection_factor": earth_connection, "final_cord_integrity": final_cord_integrity, "timestamp": timestamp }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_earth_connection', phase_metrics)
        logger.info(f"Established Earth connection. Strength Factor: {earth_connection:.4f}, Final Integrity Factor: {final_cord_integrity:.4f}")
        return float(earth_connection), float(final_cord_integrity), phase_metrics

    except Exception as e:
        logger.error(f"Error establishing Earth connection: {e}", exc_info=True)
        raise RuntimeError("Earth connection establishment failed critically.") from e

# --- Orchestration Function ---
def form_life_cord(soul_spark: SoulSpark, complexity: float = LIFE_CORD_COMPLEXITY_DEFAULT) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Forms complete life cord based on Creator connection and soul state. Modifies SoulSpark. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity out of range.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Life Cord Formation for Soul {spark_id} (Complexity={complexity:.2f}) ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    final_cord_structure = {}

    try:
        # Ensure properties exist and energy is sufficient BEFORE cost deduction
        _ensure_soul_properties(soul_spark)
        # Check prerequisites AFTER energy check
        _check_prerequisites(soul_spark)

        # --- Activation Cost ---
        logger.debug(f"Applying cord activation energy cost: {CORD_ACTIVATION_ENERGY_COST} SEU")
        initial_energy_seu = soul_spark.energy
        soul_spark.energy -= CORD_ACTIVATION_ENERGY_COST
        activation_cost_applied = initial_energy_seu - soul_spark.energy
        logger.debug(f"  Energy after activation cost: {soul_spark.energy:.1f} SEU (-{activation_cost_applied:.1f} SEU)")

        # Store Initial State (After activation cost)
        initial_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy,
            'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength,
            'cord_integrity': soul_spark.cord_integrity, # Should be 0 unless re-run
            'creator_connection_strength': soul_spark.creator_connection_strength
        }
        logger.info(f"LC Initial State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, E={initial_state['energy_seu']:.1f}, CreatorConn={initial_state['creator_connection_strength']:.3f}")

        # --- Run Phases ---
        soul_anchor, earth_anchor, connection_strength = _establish_anchor_points(soul_spark)
        process_metrics_summary['steps']['anchors'] = {'connection_strength_factor': connection_strength}

        final_cord_structure = _form_primary_channel(soul_anchor, earth_anchor, connection_strength,
                                                    initial_state['creator_connection_strength'], complexity)
        metrics_nodes = _create_harmonic_nodes(final_cord_structure, complexity)
        process_metrics_summary['steps']['nodes'] = metrics_nodes
        metrics_secondary = _add_secondary_channels(final_cord_structure, complexity)
        process_metrics_summary['steps']['secondary'] = metrics_secondary
        metrics_integration = _integrate_with_soul_field(soul_spark, final_cord_structure, connection_strength)
        process_metrics_summary['steps']['integration'] = metrics_integration
        earth_conn_strength, final_integrity, metrics_earth = _establish_earth_connection(final_cord_structure, connection_strength)
        process_metrics_summary['steps']['earth_conn'] = metrics_earth

        # --- Update SoulSpark ---
        setattr(soul_spark, "life_cord", final_cord_structure.copy())
        setattr(soul_spark, "cord_integrity", final_integrity) # Final 0-1 score
        setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, True)
        setattr(soul_spark, FLAG_READY_FOR_EARTH, True)

        # --- No Direct Stability Boost ---
        stability_gain_su = 0.0 # Initialize emergent gain tracker

        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        # Call update_state to recalculate S/C based on new factors
        if hasattr(soul_spark, 'update_state'):
            logger.debug("  Calling final update_state for Life Cord...")
            soul_spark.update_state()
            logger.debug(f"  S/C after update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
            stability_gain_su = soul_spark.stability - initial_state['stability_su'] # Record emergent gain
        else:
            # Hard fail
            raise AttributeError("SoulSpark missing 'update_state' method.")

        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(f"Life cord formed. Integrity:{final_integrity:.3f}, BW:{final_cord_structure.get('bandwidth_hz', 0.0):.1f} Hz, StabGain(Emergent):{stability_gain_su:+.1f}SU")

        # --- Compile Overall Metrics ---
        end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'energy_seu': soul_spark.energy,
            'field_radius': soul_spark.field_radius, 'field_strength': soul_spark.field_strength,
            'cord_integrity': final_integrity, 'final_bandwidth_hz': final_cord_structure.get('bandwidth_hz', 0.0),
            FLAG_CORD_FORMATION_COMPLETE: getattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE)
        }
        overall_metrics = {
            'action': 'form_life_cord', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'complexity_setting': complexity,
            'initial_state': initial_state, 'final_state': final_state,
            'activation_energy_cost_seu': activation_cost_applied,
            'final_cord_integrity': final_integrity, 'final_bandwidth_hz': final_state['final_bandwidth_hz'],
            'stability_gain_su': float(stability_gain_su), # Emergent gain
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'], # Emergent change
            'success': True,
            'step_metrics': process_metrics_summary['steps']
        }
        if METRICS_AVAILABLE: metrics.record_metrics('life_cord_summary', overall_metrics)

        logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: Integrity={final_integrity:.3f}, BW={final_state['final_bandwidth_hz']:.1f}Hz, S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        return soul_spark, overall_metrics

    # --- Error Handling (Ensure Hard Fail) ---
    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Life Cord formation failed for {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites/validation'
         record_lc_failure(spark_id, start_time_iso, failed_step, str(e_val))
         # Hard fail
         raise e_val
    except RuntimeError as e_rt:
         logger.critical(f"Life Cord formation failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = 'runtime' # Determine more specifically if possible
         record_lc_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_EARTH, False)
         # Hard fail
         raise e_rt
    except Exception as e:
         logger.critical(f"Unexpected error during Life Cord formation for {spark_id}: {e}", exc_info=True)
         failed_step = 'unexpected'
         record_lc_failure(spark_id, start_time_iso, failed_step, str(e))
         setattr(soul_spark, FLAG_CORD_FORMATION_COMPLETE, False); setattr(soul_spark, FLAG_READY_FOR_EARTH, False)
         # Hard fail
         raise RuntimeError(f"Unexpected Life Cord formation failure: {e}") from e

# --- Failure Metric Helper ---
def record_lc_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('life_cord_summary', {
                'action': 'form_life_cord', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time,
                'duration_seconds': duration,
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record LC failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/life_cord.py ---