# --- START OF FILE src/stage_1/soul_formation/creator_entanglement.py ---

"""
Creator Entanglement Functions (Refactored V4.1 - SEU/SU/CU Units)

Establishes connection post-Sephiroth journey using updated prerequisites (SU/CU),
aspect transfer logic, and stability/coherence updates in absolute units.
Operates on SoulSpark object, using KetherField properties conceptually.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import uuid
import json
from typing import Dict, List, Any, Tuple, Optional
from constants.constants import *

# Set up logger
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
    # KetherField is needed conceptually for properties, passed as argument
    from stage_1.fields.kether_field import KetherField
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

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

def _check_prerequisites(soul_spark: SoulSpark, kether_field: KetherField) -> bool:
    """ Checks prerequisites using SU/CU thresholds. """
    logger.debug(f"Checking Creator Entanglement prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): return False
    if not isinstance(kether_field, KetherField): return False # Check type

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False):
        logger.error(f"Prerequisite failed: {FLAG_SEPHIROTH_JOURNEY_COMPLETE} is False.")
        return False
    if getattr(soul_spark, 'creator_channel_id', None) is not None:
        logger.warning(f"Soul {soul_spark.spark_id} already has creator channel. Re-running.")

    # 2. Soul State Check (Absolute SU/CU Thresholds)
    stability_su = getattr(soul_spark, 'stability', 0.0)
    coherence_cu = getattr(soul_spark, 'coherence', 0.0)
    if stability_su < ENTANGLEMENT_PREREQ_STABILITY_MIN_SU:
        logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {ENTANGLEMENT_PREREQ_STABILITY_MIN_SU} SU.")
        return False
    if coherence_cu < ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU:
        logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU} CU.")
        return False

    # 3. Field Check (Conceptual Location)
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    if current_field != 'kether':
        logger.error(f"Prerequisite failed: Soul must be conceptually in Kether field (current: {current_field}).")
        # Note: We don't strictly need the FieldController position check here anymore
        # if the interaction logic doesn't rely on sampling the void grid directly.
        # Assume if current_field_key is 'kether', it's eligible.
        # We might want to ensure it's *not* in Guff though.
        if kether_field.is_in_guff(kether_field._validate_coordinates(getattr(soul_spark,'position',(0,0,0)))): # Need helper? Requires coords conversion.
             # Simplification: Assume soul is released from Guff before this stage is called by controller.
             # if kether_field.is_in_guff(field_controller._coords_to_int_tuple(soul_spark.position)): # Requires FieldController
             #     logger.error("Prerequisite failed: Soul should be released from Guff before entanglement.")
             #     return False
             pass # Assume released from Guff by controller logic


    logger.debug("Creator Entanglement prerequisites met.")
    return True

# --- Resonance Calculation (Duplicated from Journey for now, consider shared utility) ---
def calculate_resonance(freq1: float, freq2: float) -> float:
    """ Calculate resonance strength (0-1) between two frequencies. """
    if min(freq1, freq2) <= FLOAT_EPSILON: return 0.0
    ratio = max(freq1, freq2) / min(freq1, freq2)
    int_res = 0.0
    for i in range(1, 5):
        for j in range(1, 5):
            target = float(max(i,j))/float(min(i,j))
            dev = abs(ratio-target)
            if dev < RESONANCE_INTEGER_RATIO_TOLERANCE * target:
                int_res = max(int_res, (1.0 - dev / (RESONANCE_INTEGER_RATIO_TOLERANCE * target))**2)
    phi_res = 0.0
    for i in [1, 2]:
        phi_pow = GOLDEN_RATIO ** i
        dev_phi = abs(ratio - phi_pow)
        dev_inv = abs(ratio - (1.0/phi_pow))
        if dev_phi < RESONANCE_PHI_RATIO_TOLERANCE * phi_pow:
            phi_res = max(phi_res, (1.0 - dev_phi / (RESONANCE_PHI_RATIO_TOLERANCE * phi_pow))**2)
        if dev_inv < RESONANCE_PHI_RATIO_TOLERANCE * (1.0/phi_pow):
             phi_res = max(phi_res, (1.0 - dev_inv / (RESONANCE_PHI_RATIO_TOLERANCE * (1.0/phi_pow)))**2)
    return max(int_res, phi_res)

# --- Frequency Signature Calculation ---
def _calculate_frequency_signature(soul_spark: SoulSpark, kether_field: KetherField) -> Dict[str, Any]:
    """ Calculates frequency signature based on Soul and Kether aspect frequencies. """
    logger.debug(f"Calculating frequency signature for soul {soul_spark.spark_id} with Kether...")
    try:
        # Get Kether Frequencies (from aspect data)
        kether_aspects = kether_field.get_aspects()
        detailed_kether = kether_aspects.get('detailed_aspects', {})
        kether_base_freq = kether_aspects.get('base_frequency', KETHER_FREQ)
        creator_freqs_dict = {name: details.get('frequency', kether_base_freq)
                              for name, details in detailed_kether.items()
                              if details.get('frequency', 0.0) > FLOAT_EPSILON}
        if not creator_freqs_dict and kether_base_freq > FLOAT_EPSILON:
            creator_freqs_dict = {'kether_base': kether_base_freq}
        elif not creator_freqs_dict:
            raise ValueError("Kether field lacks valid frequencies.")
        creator_freqs_list = list(creator_freqs_dict.values())

        # Get Soul Frequencies
        soul_base_freq = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        if soul_base_freq <= FLOAT_EPSILON: raise ValueError("Soul has invalid base frequency.")
        soul_harmonics = getattr(soul_spark, 'harmonics', [])
        if not soul_harmonics or abs(soul_harmonics[0] - soul_base_freq) > FLOAT_EPSILON * soul_base_freq:
             soul_spark._generate_basic_harmonics() # Ensure harmonics list is present and based on current freq
             soul_harmonics = soul_spark.harmonics
        soul_freqs_list = [f for f in soul_harmonics if f > FLOAT_EPSILON]
        if not soul_freqs_list: soul_freqs_list = [soul_base_freq]
        soul_freqs_dict = {f'soul_h{i+1}': freq for i, freq in enumerate(soul_freqs_list)}

        # Find Resonance Points
        resonance_points_hz = []
        for f_creator in creator_freqs_list:
             for f_soul in soul_freqs_list:
                  res_score = calculate_resonance(f_creator, f_soul)
                  if res_score > 0.7: # Threshold for considering it a strong resonance point
                       resonance_points_hz.append((f_creator + f_soul) / 2.0) # Add midpoint as resonant freq

        signature = {
            'creator_frequencies': creator_freqs_dict,
            'soul_frequencies': soul_freqs_dict,
            'resonance_points_hz': sorted(list(set(resonance_points_hz))), # Unique sorted points
            'primary_resonance_hz': max(resonance_points_hz) if resonance_points_hz else 0.0,
            'calculation_time': datetime.now().isoformat()
        }
        # Don't directly modify soul_spark.frequency_signature here, return it
        # Let establish_creator_connection store it in the channel data
        return signature

    except Exception as e:
        logger.error(f"Error calculating frequency signature: {e}", exc_info=True)
        raise RuntimeError("Frequency signature calculation failed.") from e


# --- Core Entanglement Functions ---

def establish_creator_connection(soul_spark: SoulSpark,
                                 kether_field: KetherField,
                                 base_creator_resonance: float, # Factor 0-1
                                 edge_of_chaos_ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Establishes connection. Uses SU/CU. Modifies SoulSpark. """
    logger.info(f"Establishing creator connection for soul {soul_spark.spark_id}...")
    if not (0.0 <= base_creator_resonance <= 1.0): raise ValueError("base_creator_resonance out of range.")
    if not (0.0 <= edge_of_chaos_ratio <= 1.0): raise ValueError("edge_of_chaos_ratio out of range.")

    try:
        # Use current soul state (SU/CU)
        stability_norm = soul_spark.stability / MAX_STABILITY_SU # Normalize 0-1
        coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU # Normalize 0-1

        # Connection Quality (combine normalized state with base potential)
        connection_quality = (0.4 * stability_norm + 0.4 * coherence_norm + 0.2 * base_creator_resonance)
        chaos_factor = max(0.0, 1.0 - abs(connection_quality - edge_of_chaos_ratio) / max(FLOAT_EPSILON, edge_of_chaos_ratio))
        effective_strength = base_creator_resonance * chaos_factor * connection_quality
        effective_strength = min(1.0, max(0.0, effective_strength)) # Clamp 0-1 factor

        freq_signature = _calculate_frequency_signature(soul_spark, kether_field)

        channel_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        quantum_channel_data = {
            'channel_id': channel_id, 'spark_id': soul_spark.spark_id, 'creation_time': creation_time,
            'connection_strength': float(effective_strength), 'connection_quality': float(connection_quality),
            'initial_soul_stability_su': float(soul_spark.stability), # Record initial absolute SU
            'initial_soul_coherence_cu': float(soul_spark.coherence), # Record initial absolute CU
            'chaos_factor': float(chaos_factor), 'active': True, 'frequency_signature': freq_signature
        }

        # Update SoulSpark
        initial_alignment = soul_spark.creator_alignment # 0-1 factor
        soul_spark.creator_channel_id = channel_id
        soul_spark.creator_connection_strength = float(effective_strength) # 0-1 factor
        soul_spark.creator_alignment = float(min(1.0, initial_alignment + ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR * effective_strength))
        # Store frequency signature used for connection
        soul_spark.frequency_signature = freq_signature
        soul_spark.last_modified = creation_time

        logger.info(f"Creator connection established: Strength={effective_strength:.4f}, Alignment={soul_spark.creator_alignment:.4f}")

        # Record Metrics
        metrics_data = {
            'action': 'establish_connection', 'soul_id': soul_spark.spark_id, 'channel_id': channel_id,
            'strength': effective_strength, 'quality': connection_quality, 'chaos_factor': chaos_factor,
            'initial_stability_su': quantum_channel_data['initial_soul_stability_su'],
            'initial_coherence_cu': quantum_channel_data['initial_soul_coherence_cu'],
            'final_alignment': soul_spark.creator_alignment,
            'alignment_change': soul_spark.creator_alignment - initial_alignment,
            'success': True, 'timestamp': creation_time
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return quantum_channel_data, metrics_data

    except Exception as e:
        logger.error(f"Error establishing creator connection: {e}", exc_info=True)
        raise RuntimeError("Creator connection establishment failed.") from e


def form_resonance_patterns(soul_spark: SoulSpark,
                            quantum_channel_data: Dict[str, Any],
                            kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Forms resonance patterns based on Kether aspects and connection strength. """
    logger.info(f"Forming resonance patterns for soul {soul_spark.spark_id}...")
    # --- Validation ---
    if not isinstance(soul_spark, SoulSpark) or not isinstance(quantum_channel_data, dict) or not isinstance(kether_field, KetherField):
         raise TypeError("Invalid input types for form_resonance_patterns.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_strength = quantum_channel_data.get('connection_strength') # 0-1 factor
    freq_signature = quantum_channel_data.get('frequency_signature')
    if not channel_id or not spark_id or connection_strength is None or not freq_signature:
        raise ValueError("Quantum channel data missing essential keys.")

    try:
        kether_aspects = kether_field.get_aspects()
        detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
        if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

        patterns = {}
        # Add primary/secondary aspects from Kether data, strength scaled by connection_strength
        for name in kether_aspects.get('primary_aspects', []):
             if name in detailed_kether_aspects:
                  details = detailed_kether_aspects[name]
                  patterns[name] = { 'frequency': details.get('frequency', KETHER_FREQ), 'strength': float(details.get('strength', 0.0) * connection_strength), 'pattern_type': 'primary', **details } # Merge details
        for name in kether_aspects.get('secondary_aspects', []):
             if name in detailed_kether_aspects:
                  details = detailed_kether_aspects[name]
                  patterns[f"secondary_{name}"] = { 'frequency': details.get('frequency', KETHER_FREQ), 'strength': float(details.get('strength', 0.0) * connection_strength * 0.7), 'pattern_type': 'secondary', **details } # Merge details

        # Add harmonic patterns based on calculated resonance points
        resonance_points_hz = freq_signature.get('resonance_points_hz', [])
        for i, point_hz in enumerate(resonance_points_hz):
             if point_hz <= FLOAT_EPSILON: continue
             # Strength scales with connection strength
             strength = float(0.6 * connection_strength)
             patterns[f'harmonic_{i}'] = { 'frequency': float(point_hz), 'strength': strength, 'pattern_type': 'harmonic', 'description': f'Resonance at {point_hz:.2f} Hz'}

        # Calculate pattern coherence (average strength of formed patterns, 0-1 scale)
        pattern_coherence = 0.0
        if patterns:
            valid_strengths = [p.get('strength', 0.0) for p in patterns.values() if isinstance(p.get('strength'), (int, float))]
            if valid_strengths: pattern_coherence = sum(valid_strengths) / len(valid_strengths)

        formation_time = datetime.now().isoformat()
        resonance_pattern_data = {
            'channel_id': channel_id, 'spark_id': spark_id, 'patterns': patterns,
            'pattern_count': len(patterns), 'pattern_coherence': float(pattern_coherence),
            'formation_time': formation_time
        }

        # Update SoulSpark
        initial_resonance_factor = soul_spark.resonance # 0-1 factor
        soul_spark.resonance_patterns = patterns.copy() # Store formed patterns
        soul_spark.pattern_coherence = float(pattern_coherence) # Update 0-1 factor
        # Boost general resonance factor based on pattern coherence
        soul_spark.resonance = float(min(1.0, initial_resonance_factor + ENTANGLEMENT_RESONANCE_BOOST_FACTOR * pattern_coherence))
        soul_spark.last_modified = formation_time
        logger.info(f"Patterns formed: {len(patterns)}. New Resonance Factor: {soul_spark.resonance:.4f}, P.Coh: {pattern_coherence:.4f}")

        # Record Metrics
        metrics_data = {
            'action': 'form_patterns', 'soul_id': spark_id, 'channel_id': channel_id,
            'pattern_count': len(patterns), 'pattern_coherence': pattern_coherence,
            'final_soul_resonance_factor': soul_spark.resonance,
            'resonance_factor_change': soul_spark.resonance - initial_resonance_factor,
            'success': True, 'timestamp': formation_time
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return resonance_pattern_data, metrics_data

    except Exception as e:
        logger.error(f"Error forming resonance patterns: {e}", exc_info=True)
        raise RuntimeError("Pattern formation failed.") from e


def transfer_creator_aspects(soul_spark: SoulSpark,
                             quantum_channel_data: Dict[str, Any],
                             resonance_pattern_data: Dict[str, Any],
                             kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Transfers Kether aspects scaled by connection quality/pattern resonance. No threshold. """
    logger.info(f"Transferring creator aspects to soul {soul_spark.spark_id}...")
    # --- Validation ---
    if not isinstance(soul_spark, SoulSpark) or not isinstance(quantum_channel_data, dict) or \
       not isinstance(resonance_pattern_data, dict) or not isinstance(kether_field, KetherField):
         raise TypeError("Invalid input types for transfer_creator_aspects.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_quality = quantum_channel_data.get('connection_quality') # 0-1 factor
    patterns = resonance_pattern_data.get('patterns')
    if not channel_id or not spark_id or connection_quality is None or patterns is None:
        raise ValueError("Channel/pattern data missing essential keys.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
        setattr(soul_spark, 'aspects', {}) # Initialize if missing

    try:
        kether_aspects = kether_field.get_aspects()
        detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
        if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

        transferable_aspects = {} # Tracks potential transfers {name: {details}}
        gained_count = 0; strengthened_count = 0
        transfer_time = datetime.now().isoformat()
        total_original_strength = 0.0; total_imparted_strength = 0.0

        for name, kether_details in detailed_kether_aspects.items():
            kether_strength = float(kether_details.get('strength', 0.0)) # 0-1 factor
            if kether_strength <= FLOAT_EPSILON: continue
            total_original_strength += kether_strength

            # Efficiency depends on connection quality AND resonance pattern strength for this aspect
            pattern_strength = patterns.get(name, {}).get('strength', 0.0) # Check primary pattern
            if pattern_strength <= FLOAT_EPSILON: pattern_strength = patterns.get(f"secondary_{name}", {}).get('strength', 0.0)

            efficiency = (connection_quality * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE +
                          pattern_strength * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE)
            efficiency = min(1.0, max(0.0, efficiency))
            imparted_strength = kether_strength * efficiency # 0-1 strength scale
            total_imparted_strength += imparted_strength

            # --- Record Potential Transfer ---
            transfer_data = {
                'original_strength': kether_strength, 'transfer_efficiency': efficiency,
                'imparted_strength': imparted_strength, 'aspect_type': 'Kether',
                'details': kether_details # Store Kether's details
            }
            transferable_aspects[name] = transfer_data

            # --- Update SoulSpark Aspects (No Threshold) ---
            current_soul_strength = float(soul_spark.aspects.get(name, {}).get('strength', 0.0))
            # Blend/Add based on imparted strength
            new_strength = min(MAX_ASPECT_STRENGTH, current_soul_strength + imparted_strength * 0.8) # Weighted add? Or just add? Let's try weighted add.
            actual_gain = new_strength - current_soul_strength

            if actual_gain > FLOAT_EPSILON:
                details_copy = kether_details.copy()
                details_copy['last_transfer_efficiency'] = efficiency
                details_copy['last_imparted_strength'] = imparted_strength

                if name not in soul_spark.aspects or current_soul_strength <= FLOAT_EPSILON:
                    soul_spark.aspects[name] = {'strength': new_strength, 'source': 'Kether', 'time_acquired': transfer_time, 'details': details_copy }
                    gained_count += 1
                    if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Gained Kether aspect '{name}' Strength: {new_strength:.3f}")
                else:
                    soul_spark.aspects[name]['strength'] = new_strength
                    soul_spark.aspects[name]['last_updated'] = transfer_time
                    soul_spark.aspects[name]['details'].update(details_copy) # Update details
                    strengthened_count += 1

        average_efficiency = (total_imparted_strength / total_original_strength) if total_original_strength > FLOAT_EPSILON else 0.0

        # Apply direct delta boost to Stability (SU) based on transfer efficiency
        initial_stability_su = soul_spark.stability
        stability_boost_su = ENTANGLEMENT_STABILITY_BOOST_FACTOR * average_efficiency * MAX_STABILITY_SU # Scale boost by max SU
        soul_spark.stability = min(MAX_STABILITY_SU, initial_stability_su + stability_boost_su)
        stability_change_su = soul_spark.stability - initial_stability_su

        # Update alignment factor
        initial_alignment = soul_spark.creator_alignment
        soul_spark.creator_alignment = min(1.0, initial_alignment + 0.15 * average_efficiency) # Stronger boost

        soul_spark.last_modified = transfer_time
        logger.info(f"Aspect transfer: Gained={gained_count}, Strengthened={strengthened_count}, AvgEff={average_efficiency:.3f}, StabDelta=+{stability_change_su:.2f}SU")

        # Store Transfer Metrics
        aspect_transfer_metrics = { # Summary of what happened this step
            'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
            'average_efficiency': average_efficiency, 'transfers': transferable_aspects }
        # Record Core Metrics
        metrics_data = { # Core metrics for tracking
            'action': 'transfer_aspects', 'soul_id': spark_id, 'channel_id': channel_id,
            'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
            'avg_efficiency': average_efficiency,
            'final_soul_alignment': soul_spark.creator_alignment,
            'final_soul_stability_su': soul_spark.stability,
            'stability_change_su': stability_change_su,
            'success': True, 'timestamp': transfer_time }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return aspect_transfer_metrics, metrics_data

    except Exception as e:
        logger.error(f"Error transferring creator aspects: {e}", exc_info=True)
        raise RuntimeError("Aspect transfer failed.") from e


def stabilize_creator_connection(soul_spark: SoulSpark,
                                 quantum_channel_data: Dict[str, Any],
                                 resonance_pattern_data: Dict[str, Any],
                                 iterations: int = ENTANGLEMENT_STABILIZATION_ITERATIONS) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Stabilizes connection, applying SU/CU deltas. """
    logger.info(f"Stabilizing creator connection for soul {soul_spark.spark_id} ({iterations} iterations)...")
    # --- Validation ---
    if not isinstance(soul_spark, SoulSpark) or not isinstance(quantum_channel_data, dict) or \
       not isinstance(resonance_pattern_data, dict) or not isinstance(iterations, int) or iterations <= 0:
         raise TypeError("Invalid input types or iterations for stabilize_creator_connection.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    if not channel_id or not spark_id : raise ValueError("Channel data missing keys.")

    try:
        # Store Initial State (Absolute SU/CU and Factors)
        initial_metrics = {
            'connection_strength': soul_spark.creator_connection_strength, # 0-1
            'stability_su': soul_spark.stability, # SU
            'coherence_cu': soul_spark.coherence, # CU
            'creator_alignment': soul_spark.creator_alignment, # 0-1
            'resonance_factor': soul_spark.resonance, # 0-1
            'pattern_coherence': soul_spark.pattern_coherence # 0-1
        }
        current_strength = initial_metrics['connection_strength']
        current_pattern_coherence = initial_metrics['pattern_coherence']
        current_patterns = soul_spark.resonance_patterns.copy() # Get current patterns

        logger.info(f"Initial state for stabilization: Str={current_strength:.4f}, Stab={initial_metrics['stability_su']:.1f}SU, Coh={initial_metrics['coherence_cu']:.1f}CU, P.Coh={current_pattern_coherence:.3f}")

        # Stabilization Iterations
        for i in range(iterations):
            iter_start_time = datetime.now().isoformat()
            # Improvement factors (based on current state and iteration)
            strength_increase_factor = ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH * (1.0 - current_strength) * (1.0 / (i + 1)) # Diminishing returns
            pattern_increase_factor = 1.003 * (1.0 / (i + 1)) # Diminishing boost

            # Calculate change in strength factor
            strength_gain = current_strength * strength_increase_factor
            current_strength = min(1.0, current_strength + strength_gain)

            # Strengthen patterns and recalculate pattern coherence factor
            if current_patterns:
                 pattern_strengths = []
                 for name in current_patterns:
                      pat_str = current_patterns[name].get('strength', 0.0)
                      current_patterns[name]['strength'] = min(1.0, pat_str * pattern_increase_factor)
                      pattern_strengths.append(current_patterns[name]['strength'])
                 if pattern_strengths: current_pattern_coherence = sum(pattern_strengths) / len(pattern_strengths)

            # --- Update SoulSpark State ---
            soul_spark.creator_connection_strength = float(current_strength)
            soul_spark.pattern_coherence = float(current_pattern_coherence)
            soul_spark.resonance_patterns = current_patterns.copy()

            # Apply direct delta boost to SU/CU based on connection strength gain
            # Improvement scales with how much strength increased this iteration
            improvement_factor = strength_gain # Use the gain in the 0-1 factor as the driver
            stability_boost_su = improvement_factor * 0.5 * MAX_STABILITY_SU # Boost SU (e.g., 50% of gain factor applied to max SU)
            coherence_boost_cu = improvement_factor * 0.3 * MAX_COHERENCE_CU # Boost CU

            soul_spark.stability = min(MAX_STABILITY_SU, soul_spark.stability + stability_boost_su)
            soul_spark.coherence = min(MAX_COHERENCE_CU, soul_spark.coherence + coherence_boost_cu)

            # Boost alignment factor slightly too
            soul_spark.creator_alignment = min(1.0, soul_spark.creator_alignment + improvement_factor * 0.05)

            soul_spark.last_modified = iter_start_time
            logger.debug(f"  Stabilization iter {i+1}: Str={current_strength:.4f}, P.Coh={current_pattern_coherence:.4f}, Stab={soul_spark.stability:.1f}, Coh={soul_spark.coherence:.1f}")

        # Store Final Metrics (Absolute SU/CU and Factors)
        final_metrics = {
            'connection_strength': float(current_strength),
            'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
            'creator_alignment': soul_spark.creator_alignment, 'resonance_factor': soul_spark.resonance,
            'pattern_coherence': float(current_pattern_coherence)
        }
        improvements_su_cu = { # Calculate SU/CU change
            'stability_change_su': final_metrics['stability_su'] - initial_metrics['stability_su'],
            'coherence_change_cu': final_metrics['coherence_cu'] - initial_metrics['coherence_cu'],
        }
        stabilization_result = { # Summary of the process
            'iterations': iterations, 'initial_metrics': initial_metrics, 'final_metrics': final_metrics,
            'improvements_su_cu': improvements_su_cu, 'stabilization_time': getattr(soul_spark, 'last_modified') }

        logger.info(f"Creator connection stabilized. Final Str={current_strength:.4f}, Stab={soul_spark.stability:.1f}SU, Coh={soul_spark.coherence:.1f}CU")
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Creator connection stabilized. Strength:{current_strength:.3f}")

        # Record Core Metrics
        metrics_data = {
            'action': 'stabilize_connection', 'soul_id': spark_id, 'channel_id': channel_id, 'iterations': iterations,
            'initial_strength': initial_metrics['connection_strength'], 'final_strength': final_metrics['connection_strength'],
            'initial_stability_su': initial_metrics['stability_su'], 'final_stability_su': final_metrics['stability_su'],
            'initial_coherence_cu': initial_metrics['coherence_cu'], 'final_coherence_cu': final_metrics['coherence_cu'],
            'success': True, 'timestamp': stabilization_result['stabilization_time']
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return stabilization_result, metrics_data

    except Exception as e:
        logger.error(f"Error stabilizing creator connection: {e}", exc_info=True)
        raise RuntimeError("Stabilization failed.") from e

# --- Orchestration Function ---
def run_full_entanglement_process(soul_spark: SoulSpark,
                                  kether_field: KetherField,
                                  creator_resonance: float = 0.8, # Base potential factor 0-1
                                  edge_of_chaos_ratio: float = EDGE_OF_CHAOS_RATIO, # Target 0-1 factor
                                  stabilization_iterations: int = ENTANGLEMENT_STABILIZATION_ITERATIONS
                                  ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Runs complete entanglement process. Uses SU/CU prerequisites. Modifies SoulSpark. """
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    process_metrics_summary = {'steps': {}}

    logger.info(f"--- Starting Full Entanglement Process for Soul {spark_id} ---")

    try:
        if not _check_prerequisites(soul_spark, kether_field): # Uses SU/CU thresholds
            raise ValueError("Soul prerequisites for Creator Entanglement not met.")

        initial_state = { # Record state in correct units
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
             'aspect_count': len(soul_spark.aspects) }

        # Run Stages
        logger.info("Step 1: Establishing Connection...")
        channel_data, metrics1 = establish_creator_connection(soul_spark, kether_field, creator_resonance, edge_of_chaos_ratio)
        process_metrics_summary['steps']['connection'] = metrics1

        logger.info("Step 2: Forming Resonance Patterns...")
        pattern_data, metrics2 = form_resonance_patterns(soul_spark, channel_data, kether_field)
        process_metrics_summary['steps']['patterns'] = metrics2

        logger.info("Step 3: Transferring Creator Aspects...")
        transfer_metrics, metrics3 = transfer_creator_aspects(soul_spark, channel_data, pattern_data, kether_field)
        process_metrics_summary['steps']['transfer'] = metrics3

        logger.info("Step 4: Stabilizing Connection...")
        stabilization_result, metrics4 = stabilize_creator_connection(soul_spark, channel_data, pattern_data, stabilization_iterations)
        process_metrics_summary['steps']['stabilization'] = metrics4

        # Finalize
        setattr(soul_spark, FLAG_READY_FOR_COMPLETION, True)
        setattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, True)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Final update of S/C scores

        # Compile Overall Metrics
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report state in correct units
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
             'aspect_count': len(soul_spark.aspects) }
        overall_metrics = {
            'action': 'full_entanglement', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'final_connection_strength': stabilization_result['final_metrics']['connection_strength'], # 0-1 factor
            'aspects_transferred_count': transfer_metrics.get('aspects_gained_count', 0) + transfer_metrics.get('aspects_strengthened_count', 0),
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement_summary', overall_metrics)

        logger.info(f"--- Full Entanglement Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Entanglement failed for soul {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
        failed_step = process_metrics_summary['steps'].keys()[-1] if process_metrics_summary['steps'] else 'prerequisites'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_val))
        raise # Re-raise validation errors
    except RuntimeError as e_rt:
        logger.critical(f"Entanglement failed critically for soul {spark_id}: {e_rt}", exc_info=True)
        failed_step = process_metrics_summary['steps'].keys()[-1] if process_metrics_summary['steps'] else 'runtime'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        raise # Re-raise runtime errors
    except Exception as e:
        logger.critical(f"Unexpected error during entanglement for {spark_id}: {e}", exc_info=True)
        failed_step = process_metrics_summary['steps'].keys()[-1] if process_metrics_summary['steps'] else 'unexpected'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e))
        raise RuntimeError(f"Unexpected entanglement failure: {e}") from e

# --- Failure Metric Helper ---
def record_entanglement_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('creator_entanglement_summary', {
                'action': 'full_entanglement', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record entanglement failure metrics for {spark_id}: {metric_e}")


# --- END OF FILE src/stage_1/soul_formation/creator_entanglement.py ---




