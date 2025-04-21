# --- START OF FILE life_cord.py ---

"""
Life Cord Formation Functions (Refactored - Operates on SoulSpark Object, Uses Constants)

Provides functions for forming the life cord, the energetic connection between
the soul and its future physical anchor point (representing the potential for
incarnation). Modifies the SoulSpark object instance directly.
Uses centrally defined constants for calculations.

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
    from src.constants import (
        GOLDEN_RATIO, SILVER_RATIO, EARTH_FREQUENCIES,
        FLOAT_EPSILON, LOG_LEVEL, # Basic utils
        # Life Cord Specific Constants (Ensure these are defined in src.constants.py)
        CORD_STABILITY_THRESHOLD, CORD_COHERENCE_THRESHOLD,
        MAX_CORD_CHANNELS, ANCHOR_STRENGTH_MODIFIER,
        EARTH_ANCHOR_STRENGTH, EARTH_ANCHOR_RESONANCE,
        PRIMARY_CHANNEL_BANDWIDTH_FACTOR,
        PRIMARY_CHANNEL_STABILITY_FACTOR_CONN, PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX,
        PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN, PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX,
        PRIMARY_CHANNEL_ELASTICITY_BASE, PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX,
        HARMONIC_NODE_COUNT_BASE, HARMONIC_NODE_COUNT_FACTOR,
        HARMONIC_NODE_AMP_BASE, HARMONIC_NODE_AMP_FACTOR_COMPLEX, HARMONIC_NODE_AMP_FALLOFF,
        HARMONIC_NODE_BW_INCREASE_FACTOR, SECONDARY_CHANNEL_COUNT_FACTOR,
        SECONDARY_CHANNEL_BW_EMOTIONAL, SECONDARY_CHANNEL_BW_MENTAL, SECONDARY_CHANNEL_BW_SPIRITUAL,
        SECONDARY_CHANNEL_RESIST_EMOTIONAL, SECONDARY_CHANNEL_RESIST_MENTAL, SECONDARY_CHANNEL_RESIST_SPIRITUAL,
        SECONDARY_CHANNEL_FREQ_FACTOR, FIELD_INTEGRATION_FACTOR_FIELD_STR,
        FIELD_INTEGRATION_FACTOR_CONN_STR, FIELD_EXPANSION_FACTOR,
        EARTH_CONN_FACTOR_CONN_STR, EARTH_CONN_FACTOR_ELASTICITY, EARTH_CONN_BASE_FACTOR,
        CORD_INTEGRITY_FACTOR_CONN_STR, CORD_INTEGRITY_FACTOR_STABILITY,
        CORD_INTEGRITY_FACTOR_EARTH_CONN, FINAL_STABILITY_BONUS_FACTOR,
        # Other needed constants
        PI
    )
    # Extract specific Earth freq if needed
    EARTH_FREQUENCY = EARTH_FREQUENCIES.get("schumann", 7.83) # Use Schumann as anchor freq
except ImportError as e:
    # Fallback logging setup if constants unavailable during standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. Life Cord formation cannot function correctly.")
    # Raise error - this module is unusable without constants
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.void.soul_spark import SoulSpark
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import SoulSpark: {e}. Life Cord formation cannot function.")
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
    """Checks if the soul meets prerequisites for cord formation using constants."""
    logger.debug(f"Checking prerequisites for soul {soul_spark.spark_id}...")
    # Requires harmonic strengthening completion flag
    if not getattr(soul_spark, "harmonically_strengthened", False):
        logger.error("Prerequisite failed: Soul harmonic strengthening not complete.")
        return False

    # Requires minimum stability and coherence using constants
    stability = getattr(soul_spark, "stability", 0.0)
    if stability < CORD_STABILITY_THRESHOLD:
        logger.error(f"Prerequisite failed: Soul stability ({stability:.3f}) below threshold ({CORD_STABILITY_THRESHOLD}).")
        return False
    coherence = getattr(soul_spark, "coherence", 0.0)
    if coherence < CORD_COHERENCE_THRESHOLD:
        logger.error(f"Prerequisite failed: Soul coherence ({coherence:.3f}) below threshold ({CORD_COHERENCE_THRESHOLD}).")
        return False

    logger.debug("Prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary numeric properties, initializing if missing. Fails hard."""
    # Reuse helper from harmonic strengthening (or make a shared one)
    logger.debug(f"Ensuring properties exist for soul {soul_spark.spark_id}...")
    attributes_to_check = [
        ("frequency", SOUL_SPARK_DEFAULT_FREQ),
        ("stability", SOUL_SPARK_DEFAULT_STABILITY),
        ("coherence", SOUL_SPARK_DEFAULT_COHERENCE),
        ("position", [0.0, 0.0, 0.0]), # Need position
        ("field_radius", 3.0),
        ("field_strength", 0.5)
        # Add others if life cord logic depends on them
    ]
    for attr, default in attributes_to_check:
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
            logger.warning(f"Soul {soul_spark.spark_id} missing '{attr}'. Initializing to default: {default}")
            setattr(soul_spark, attr, default)

    # Validate numerical types and ranges
    for attr, _ in attributes_to_check:
        val = getattr(soul_spark, attr)
        if attr == 'position':
             if not isinstance(val, (list, tuple)) or len(val) != 3 or not all(isinstance(v, (int, float)) for v in val):
                  logger.error(f"Soul attribute '{attr}' must be list/tuple of 3 numbers. Resetting.")
                  setattr(soul_spark, attr, [0.0, 0.0, 0.0])
        elif not isinstance(val, (int, float)):
             raise TypeError(f"Soul attribute '{attr}' must be numeric, found {type(val)}.")
        elif not np.isfinite(val):
             raise ValueError(f"Soul attribute '{attr}' has non-finite value {val}.")
        elif attr == 'frequency' and val <= FLOAT_EPSILON:
             raise ValueError(f"Soul frequency ({val}) must be positive.")
        elif attr not in ['frequency', 'position', 'field_radius'] and not (0.0 <= val <= 1.0):
             logger.warning(f"Clamping soul attribute '{attr}' ({val}) to 0-1 range.")
             setattr(soul_spark, attr, max(0.0, min(1.0, val)))
    logger.debug("Soul properties ensured for Life Cord.")


def _establish_anchor_points(soul_spark: SoulSpark) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Establishes anchor points for the life cord using constants. Fails hard."""
    logger.info(f"Establishing anchor points for soul {soul_spark.spark_id}...")
    try:
        # Retrieve necessary attributes, raise error if missing
        _ensure_soul_properties(soul_spark) # Ensure props exist first
        soul_position = getattr(soul_spark, "position")
        soul_freq = getattr(soul_spark, "frequency")
        soul_stability = getattr(soul_spark, "stability")
        soul_coherence = getattr(soul_spark, "coherence")

        # Calculate soul anchor properties using constants
        soul_anchor_strength = soul_stability * ANCHOR_STRENGTH_MODIFIER + (1.0 - ANCHOR_STRENGTH_MODIFIER)
        soul_anchor_resonance = soul_coherence * ANCHOR_STRENGTH_MODIFIER + (1.0 - ANCHOR_STRENGTH_MODIFIER)

        soul_anchor = {
            "position": list(soul_position),
            "frequency": float(soul_freq),
            "strength": float(max(0.0, min(1.0, soul_anchor_strength))), # Clamp
            "resonance": float(max(0.0, min(1.0, soul_anchor_resonance))) # Clamp
        }
        logger.debug(f"  Soul Anchor: Pos={soul_anchor['position']}, Freq={soul_anchor['frequency']:.2f}, Str={soul_anchor['strength']:.3f}, Res={soul_anchor['resonance']:.3f}")

        # Use constants for Earth anchor
        earth_anchor_freq = EARTH_FREQUENCY
        if earth_anchor_freq <= FLOAT_EPSILON: raise ValueError("Earth anchor frequency is non-positive.")

        earth_anchor = {
            "position": [float(soul_position[0]), float(soul_position[1]), float(soul_position[2] - 100.0)], # Anchor 'below'
            "frequency": float(earth_anchor_freq),
            "strength": float(EARTH_ANCHOR_STRENGTH),
            "resonance": float(EARTH_ANCHOR_RESONANCE)
        }
        logger.debug(f"  Earth Anchor: Pos={earth_anchor['position']}, Freq={earth_anchor['frequency']:.2f}, Str={earth_anchor['strength']:.3f}, Res={earth_anchor['resonance']:.3f}")

        # Calculate connection viability using constants
        freq_ratio = min(soul_anchor["frequency"] / earth_anchor["frequency"],
                         earth_anchor["frequency"] / soul_anchor["frequency"])
        connection_strength = (
            soul_anchor["strength"] * 0.4 + earth_anchor["strength"] * 0.4 + freq_ratio * 0.2
        )
        connection_strength = max(0.0, min(1.0, connection_strength))
        logger.info(f"Anchor points established. Connection Strength: {connection_strength:.4f}")

        return soul_anchor, earth_anchor, float(connection_strength)

    except AttributeError as ae:
        logger.error(f"SoulSpark object missing attribute during anchor point establishment: {ae}", exc_info=True)
        raise AttributeError(f"Soul missing required attribute: {ae}") from ae
    except Exception as e:
        logger.error(f"Error establishing anchor points: {e}", exc_info=True)
        raise RuntimeError("Failed to establish anchor points.") from e


def _form_primary_channel(soul_anchor: Dict[str, Any], earth_anchor: Dict[str, Any],
                         connection_strength: float, complexity: float) -> Dict[str, Any]:
    """Forms the primary channel of the life cord using constants. Fails hard."""
    logger.info("Forming primary channel...")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity must be between 0.1 and 1.0.")
    if soul_anchor['frequency'] <= FLOAT_EPSILON or earth_anchor['frequency'] <= FLOAT_EPSILON:
        raise ValueError("Anchor frequencies must be positive.")

    try:
        # Use constants for factors
        channel_bandwidth = connection_strength * complexity * PRIMARY_CHANNEL_BANDWIDTH_FACTOR
        channel_stability = connection_strength * PRIMARY_CHANNEL_STABILITY_FACTOR_CONN + complexity * PRIMARY_CHANNEL_STABILITY_FACTOR_COMPLEX
        interference_resistance = connection_strength * PRIMARY_CHANNEL_INTERFERENCE_FACTOR_CONN + complexity * PRIMARY_CHANNEL_INTERFERENCE_FACTOR_COMPLEX
        primary_frequency = (soul_anchor['frequency'] * 0.8 + earth_anchor['frequency'] * 0.2) # Keep weighted average
        elasticity = PRIMARY_CHANNEL_ELASTICITY_BASE + complexity * PRIMARY_CHANNEL_ELASTICITY_FACTOR_COMPLEX

        # Clamp values
        channel_stability = max(0.0, min(1.0, channel_stability))
        interference_resistance = max(0.0, min(1.0, interference_resistance))
        elasticity = max(0.0, min(1.0, elasticity))

        primary_channel_data = {
            "bandwidth": float(max(0.0, channel_bandwidth)),
            "stability": float(channel_stability),
            "interference_resistance": float(interference_resistance),
            "primary_frequency": float(primary_frequency),
            "elasticity": float(elasticity),
            "channel_count": 1
        }
        logger.info(f"Primary channel formed: Freq={primary_frequency:.2f} Hz, BW={channel_bandwidth:.2f} Hz, Stab={channel_stability:.3f}")
        return primary_channel_data

    except Exception as e:
        logger.error(f"Error forming primary channel: {e}", exc_info=True)
        raise RuntimeError("Failed to form primary channel.") from e


def _create_harmonic_nodes(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """Creates harmonic nodes using constants. Modifies cord_structure. Fails hard."""
    logger.info("Creating harmonic nodes...")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity must be between 0.1 and 1.0.")
    primary_freq = cord_structure.get("primary_frequency")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON:
        raise ValueError("Cannot create nodes: Invalid or missing primary_frequency in cord_structure.")

    try:
        # Use constants for node count
        num_nodes = HARMONIC_NODE_COUNT_BASE + int(complexity * HARMONIC_NODE_COUNT_FACTOR)
        nodes = []
        logger.debug(f"  Targeting {num_nodes} nodes.")

        for i in range(num_nodes):
            position = (i + 1.0) / (num_nodes + 1.0) # Relative distance 0..1
            freq = 0.0; harmonic_type = "unknown"
            type_idx = i % 3
            if type_idx == 0: # Phi
                phi_exp = (i // 3) % 5 + 1
                freq = primary_freq * (GOLDEN_RATIO ** phi_exp)
                harmonic_type = f"phi^{phi_exp}"
            elif type_idx == 1: # Integer
                int_mult = (i // 3) % 7 + 2
                freq = primary_freq * int_mult
                harmonic_type = f"integer*{int_mult}"
            else: # Silver
                silver_exp = (i // 3) % 3 + 1
                freq = primary_freq * (SILVER_RATIO ** silver_exp)
                harmonic_type = f"silver^{silver_exp}"

            # Use constants for amplitude calculation
            amplitude = (HARMONIC_NODE_AMP_BASE + complexity * HARMONIC_NODE_AMP_FACTOR_COMPLEX) * (1.0 - position * HARMONIC_NODE_AMP_FALLOFF)

            if freq > FLOAT_EPSILON and amplitude > FLOAT_EPSILON:
                 nodes.append({"position": float(position), "frequency": float(freq), "harmonic_type": harmonic_type, "amplitude": float(max(0.0, min(1.0, amplitude)))})
            else: logger.warning(f"Skipping node {i+1} due to invalid frequency ({freq:.2f}) or amplitude ({amplitude:.2f}).")


        # Update cord structure
        cord_structure["harmonic_nodes"] = nodes
        # Use constant for bandwidth increase
        bandwidth_increase = len(nodes) * complexity * HARMONIC_NODE_BW_INCREASE_FACTOR
        cord_structure["bandwidth"] = cord_structure.get("bandwidth", 0.0) + bandwidth_increase
        logger.debug(f"  Created {len(nodes)} nodes. Bandwidth increased by {bandwidth_increase:.2f} Hz.")

        # Calculate Metrics
        phase_metrics = {
            "num_nodes": len(nodes), "bandwidth_increase": float(bandwidth_increase), "final_bandwidth": float(cord_structure["bandwidth"]),
            "node_types_summary": {ht: sum(1 for n in nodes if n['harmonic_type'].startswith(ht)) for ht in ['phi', 'integer', 'silver']},
            "timestamp": datetime.now().isoformat() }
        try: metrics.record_metrics('life_cord_nodes', phase_metrics)
        except Exception as e: logger.error(f"Failed to record harmonic node metrics: {e}")

        logger.info(f"Created {len(nodes)} harmonic nodes. Total Bandwidth: {cord_structure['bandwidth']:.2f} Hz.")
        return phase_metrics

    except Exception as e: logger.error(f"Error creating harmonic nodes: {e}", exc_info=True); raise RuntimeError("Failed to create harmonic nodes.") from e


def _add_secondary_channels(cord_structure: Dict[str, Any], complexity: float) -> Dict[str, Any]:
    """Adds secondary channels using constants. Modifies cord_structure. Fails hard."""
    logger.info("Adding secondary channels...")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity must be between 0.1 and 1.0.")
    primary_freq = cord_structure.get("primary_frequency")
    if primary_freq is None or primary_freq <= FLOAT_EPSILON: raise ValueError("Invalid or missing primary_frequency.")

    try:
        # Use constants
        num_channels = min(MAX_CORD_CHANNELS, int(complexity * SECONDARY_CHANNEL_COUNT_FACTOR))
        secondary_channels = []
        total_secondary_bw = 0.0
        logger.debug(f"  Targeting {num_channels} secondary channels.")

        channel_types = ["emotional", "mental", "spiritual"]
        # Use constants for channel configs
        bw_configs = {"emotional": SECONDARY_CHANNEL_BW_EMOTIONAL, "mental": SECONDARY_CHANNEL_BW_MENTAL, "spiritual": SECONDARY_CHANNEL_BW_SPIRITUAL}
        resist_configs = {"emotional": SECONDARY_CHANNEL_RESIST_EMOTIONAL, "mental": SECONDARY_CHANNEL_RESIST_MENTAL, "spiritual": SECONDARY_CHANNEL_RESIST_SPIRITUAL}

        for i in range(num_channels):
            channel_type = channel_types[i % len(channel_types)]
            bw_base, bw_factor = bw_configs[channel_type]
            resist_base, resist_factor = resist_configs[channel_type]

            bandwidth = bw_base + complexity * bw_factor
            resistance = resist_base + complexity * resist_factor
            # Use constant for frequency factor
            frequency = primary_freq * (1.0 + (i + 1) * SECONDARY_CHANNEL_FREQ_FACTOR)

            if bandwidth > FLOAT_EPSILON and resistance >= 0 and frequency > FLOAT_EPSILON:
                 secondary_channels.append({ "type": channel_type, "bandwidth": float(bandwidth),
                     "interference_resistance": float(max(0.0, min(1.0, resistance))), "frequency": float(frequency) })
                 total_secondary_bw += bandwidth
            else: logger.warning(f"Skipping secondary channel {i+1} ('{channel_type}') due to invalid calculated properties.")

        # Update cord structure
        cord_structure["secondary_channels"] = secondary_channels
        cord_structure["channel_count"] = 1 + len(secondary_channels)
        cord_structure["bandwidth"] += total_secondary_bw
        logger.debug(f"  Added {len(secondary_channels)} channels. Total Bandwidth: {cord_structure['bandwidth']:.2f} Hz.")

        # Calculate Metrics
        phase_metrics = {
            "num_secondary_channels": len(secondary_channels), "secondary_bandwidth_added": float(total_secondary_bw),
            "final_bandwidth": float(cord_structure["bandwidth"]), "final_channel_count": cord_structure["channel_count"],
            "channel_types_summary": {ctype: sum(1 for c in secondary_channels if c['type'] == ctype) for ctype in channel_types},
            "timestamp": datetime.now().isoformat() }
        try: metrics.record_metrics('life_cord_secondary', phase_metrics)
        except Exception as e: logger.error(f"Failed to record secondary channel metrics: {e}")

        logger.info(f"Added {len(secondary_channels)} secondary channels. Total Bandwidth: {cord_structure['bandwidth']:.2f} Hz.")
        return phase_metrics

    except Exception as e: logger.error(f"Error adding secondary channels: {e}", exc_info=True); raise RuntimeError("Failed to add secondary channels.") from e


def _integrate_with_soul_field(soul_spark: SoulSpark, cord_structure: Dict[str, Any], connection_strength: float) -> Dict[str, Any]:
    """Integrates cord with soul field using constants. Modifies SoulSpark. Fails hard."""
    logger.info(f"Integrating cord with soul field for {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure field props exist

    try:
        current_radius = getattr(soul_spark, "field_radius"); current_strength = getattr(soul_spark, "field_strength")
        logger.debug(f"  Initial Field Radius: {current_radius:.4f}, Strength: {current_strength:.4f}")

        # Calculate integration strength using constants
        integration_strength = (current_strength * FIELD_INTEGRATION_FACTOR_FIELD_STR +
                                connection_strength * FIELD_INTEGRATION_FACTOR_CONN_STR)
        integration_strength = max(0.0, min(1.0, integration_strength))
        logger.debug(f"  Calculated Integration Strength: {integration_strength:.4f}")

        # Update soul field using constants
        new_radius = current_radius * FIELD_EXPANSION_FACTOR
        new_strength = min(1.0, current_strength + 0.05 * integration_strength) # Keep minor strength increase
        logger.debug(f"  New Field Radius: {new_radius:.4f}, New Strength: {new_strength:.4f}")

        # --- Update SoulSpark ---
        setattr(soul_spark, "field_radius", float(new_radius))
        setattr(soul_spark, "field_strength", float(new_strength))
        setattr(soul_spark, "cord_integration", float(integration_strength))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "integration_strength": float(integration_strength),
            "field_radius_before": float(current_radius), "field_radius_after": float(new_radius),
            "field_strength_before": float(current_strength), "field_strength_after": float(new_strength),
            "timestamp": getattr(soul_spark, 'last_modified') }
        try: metrics.record_metrics('life_cord_integration', phase_metrics)
        except Exception as e: logger.error(f"Failed to record field integration metrics: {e}")

        logger.info(f"Integrated cord with soul field. Integration Strength: {integration_strength:.4f}")
        return phase_metrics

    except Exception as e: logger.error(f"Error integrating cord with soul field: {e}", exc_info=True); raise RuntimeError("Failed to integrate cord with soul field.") from e


def _establish_earth_connection(cord_structure: Dict[str, Any], connection_strength: float) -> Tuple[float, float, Dict[str, Any]]:
    """Establishes Earth connection using constants. Modifies cord_structure. Fails hard."""
    logger.info("Establishing Earth connection...")
    try:
        # Use get with validation
        elasticity = cord_structure.get("elasticity")
        primary_stability = cord_structure.get("primary_channel", {}).get("stability")
        if elasticity is None or not isinstance(elasticity, (int, float)) or not (0.0 <= elasticity <= 1.0):
            raise ValueError(f"Invalid or missing 'elasticity' in cord_structure: {elasticity}")
        if primary_stability is None or not isinstance(primary_stability, (int, float)) or not (0.0 <= primary_stability <= 1.0):
            raise ValueError(f"Invalid or missing 'primary_channel.stability' in cord_structure: {primary_stability}")

        # Use constants for factors
        earth_connection = (connection_strength * EARTH_CONN_FACTOR_CONN_STR +
                            elasticity * EARTH_CONN_FACTOR_ELASTICITY +
                            EARTH_CONN_BASE_FACTOR)
        earth_connection = max(0.1, min(0.95, earth_connection)) # Clamp
        logger.debug(f"  Calculated Earth Connection Strength: {earth_connection:.4f}")

        cord_structure["earth_connection"] = float(earth_connection)

        # Use constants for integrity calculation
        final_cord_integrity = (connection_strength * CORD_INTEGRITY_FACTOR_CONN_STR +
                                primary_stability * CORD_INTEGRITY_FACTOR_STABILITY +
                                earth_connection * CORD_INTEGRITY_FACTOR_EARTH_CONN)
        final_cord_integrity = max(0.0, min(1.0, final_cord_integrity))
        logger.debug(f"  Calculated Final Cord Integrity: {final_cord_integrity:.4f}")

        # --- Calculate & Record Metrics ---
        phase_metrics = {
            "earth_connection_strength": float(earth_connection),
            "final_cord_integrity": float(final_cord_integrity),
            "timestamp": datetime.now().isoformat() }
        try: metrics.record_metrics('life_cord_earth_connection', phase_metrics)
        except Exception as e: logger.error(f"Failed to record earth connection metrics: {e}")

        logger.info(f"Established Earth connection. Strength: {earth_connection:.4f}, Final Cord Integrity: {final_cord_integrity:.4f}")
        return float(earth_connection), float(final_cord_integrity), phase_metrics

    except KeyError as ke: logger.error(f"Missing key during Earth connection: {ke}", exc_info=True); raise KeyError(f"Cord structure missing key '{ke}' for Earth connection.") from ke
    except ValueError as ve: logger.error(f"Value error during Earth connection: {ve}", exc_info=True); raise
    except Exception as e: logger.error(f"Error establishing Earth connection: {e}", exc_info=True); raise RuntimeError("Failed to establish Earth connection.") from e

# --- Orchestration Function ---

def form_life_cord(soul_spark: SoulSpark, complexity: float = 0.7) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Forms the complete life cord. Modifies SoulSpark object. Fails hard. Uses constants.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        complexity (float): Complexity of the cord structure (0.1-1.0).

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the entire cord formation process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If parameters invalid or prerequisites not met.
        RuntimeError: If any phase fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not (0.1 <= complexity <= 1.0): raise ValueError("Complexity must be between 0.1 and 1.0")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Life Cord Formation for Soul {spark_id} (Complexity={complexity:.2f}) ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}
    final_cord_structure = {} # Initialize dict to build cord structure

    try:
        # --- Check Prerequisites (raises ValueError if failed) ---
        if not _check_prerequisites(soul_spark):
             # This path should not be reached if _check raises ValueError, but included for clarity
            raise ValueError("Soul prerequisites for life cord formation not met.")
        logger.info("Prerequisites checked successfully.")

        # --- Store Initial State ---
        initial_state = {
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0), 'field_strength': getattr(soul_spark, 'field_strength', 0.0), }
        logger.info(f"Initial State: Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}, FieldR={initial_state['field_radius']:.2f}, FieldS={initial_state['field_strength']:.2f}")

        # --- Run Phases (Fail hard within each) ---
        logger.info("Step 1: Establish Anchor Points...")
        soul_anchor, earth_anchor, connection_strength = _establish_anchor_points(soul_spark)
        process_metrics_summary['steps']['anchors'] = {'connection_strength': connection_strength, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 2: Form Primary Channel...")
        final_cord_structure = _form_primary_channel(soul_anchor, earth_anchor, connection_strength, complexity)
        process_metrics_summary['steps']['primary_channel'] = final_cord_structure.copy(); process_metrics_summary['steps']['primary_channel']['timestamp'] = datetime.now().isoformat()

        logger.info("Step 3: Create Harmonic Nodes...")
        metrics3 = _create_harmonic_nodes(final_cord_structure, complexity)
        process_metrics_summary['steps']['harmonic_nodes'] = metrics3

        logger.info("Step 4: Add Secondary Channels...")
        metrics4 = _add_secondary_channels(final_cord_structure, complexity)
        process_metrics_summary['steps']['secondary_channels'] = metrics4

        logger.info("Step 5: Integrate with Soul Field...")
        metrics5 = _integrate_with_soul_field(soul_spark, final_cord_structure, connection_strength)
        process_metrics_summary['steps']['integration'] = metrics5

        logger.info("Step 6: Establish Earth Connection...")
        earth_conn_strength, final_integrity, metrics6 = _establish_earth_connection(final_cord_structure, connection_strength)
        process_metrics_summary['steps']['earth_connection'] = metrics6

        # --- Update SoulSpark Object ---
        logger.info(f"Updating SoulSpark {soul_spark.spark_id} with final cord properties...")
        setattr(soul_spark, "life_cord", final_cord_structure.copy())
        setattr(soul_spark, "cord_integrity", final_integrity)
        setattr(soul_spark, "cord_formation_complete", True)
        # Use constant for stability bonus
        stability_bonus = final_integrity * FINAL_STABILITY_BONUS_FACTOR
        setattr(soul_spark, "stability", min(1.0, initial_state['stability'] + stability_bonus))
        setattr(soul_spark, "ready_for_earth", True)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"SoulSpark updated. Cord Integrity: {final_integrity:.4f}, Final Stability: {soul_spark.stability:.4f}")
        # Log memory echo
        logger.info(f"Memory echo created: Life cord formed for soul {spark_id}.")
        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
            soul_spark.memory_echoes.append(f"Life cord formed @ {getattr(soul_spark, 'last_modified')}")

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
            'stability': getattr(soul_spark, 'stability', 0.0), 'coherence': getattr(soul_spark, 'coherence', 0.0),
            'field_radius': getattr(soul_spark, 'field_radius', 0.0), 'field_strength': getattr(soul_spark, 'field_strength', 0.0),
            'cord_integrity': final_integrity, 'cord_bandwidth': final_cord_structure.get('bandwidth', 0.0),
            'cord_formation_complete': getattr(soul_spark, 'cord_formation_complete', False)}

        overall_metrics = {
            'action': 'form_life_cord', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(), 'complexity_setting': complexity,
            'initial_state': initial_state, 'final_state': final_state, 'final_cord_integrity': final_integrity,
            'final_bandwidth': final_cord_structure.get('bandwidth', 0.0), 'success': True, }
        try: metrics.record_metrics('life_cord_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for life cord formation: {e}")

        logger.info(f"--- Life Cord Formation Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final Cord Integrity: {final_integrity:.4f}, Bandwidth: {overall_metrics['final_bandwidth']:.2f} Hz")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Life cord formation process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"; steps_completed = list(process_metrics_summary['steps'].keys())
        if steps_completed: failed_step = steps_completed[-1]
        # Mark soul as failed this stage
        setattr(soul_spark, "cord_formation_complete", False); setattr(soul_spark, "cord_integrity", 0.0)
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('life_cord_summary', {
                  'action': 'form_life_cord', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'complexity_setting': complexity, 'success': False, 'error': str(e), 'failed_step': failed_step })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Life cord formation process failed at step '{failed_step}'.") from e


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Life Cord Formation Module Example (with Constants)...")
    # Assume SoulSpark class is importable and works as expected
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        test_soul = SoulSpark() # Create instance
        test_soul.spark_id="test_lifecord_const_002"
        # Set state *after* harmonic strengthening
        test_soul.stability = 0.81
        test_soul.coherence = 0.86
        test_soul.frequency = 432.0
        test_soul.formation_complete = True
        test_soul.harmonically_strengthened = True # Set prerequisite flag
        test_soul.field_radius = 4.8
        test_soul.field_strength = 0.78
        test_soul.position = [5.0, 10.0, 15.0]
        test_soul.memory_echoes = [] # Initialize memory

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Harmonically Strengthened: {getattr(test_soul, 'harmonically_strengthened', False)}")

        try:
            print("\n--- Running Life Cord Formation Process ---")
            modified_soul, summary_metrics_result = form_life_cord(
                soul_spark=test_soul,
                complexity=0.8 # Example complexity
            )

            print("\n--- Formation Complete ---")
            print(f"  Cord Formation Complete: {getattr(modified_soul, 'cord_formation_complete', False)}")
            print(f"  Cord Integrity: {getattr(modified_soul, 'cord_integrity', 'N/A'):.4f}")
            print(f"  Ready for Earth Harmony: {getattr(modified_soul, 'ready_for_earth', False)}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            if hasattr(modified_soul, 'life_cord'): print(f"  Cord Bandwidth: {modified_soul.life_cord.get('bandwidth', 'N/A'):.2f}")
            print(f"  Memory Echoes: {getattr(modified_soul, 'memory_echoes', [])}")

        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Life Cord Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Life Cord Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nLife Cord Formation Module Example Finished.")


# --- END OF FILE life_cord.py ---
