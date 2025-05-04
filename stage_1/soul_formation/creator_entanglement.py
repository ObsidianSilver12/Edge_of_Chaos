# --- START OF FILE src/stage_1/soul_formation/creator_entanglement.py ---

"""
Creator Entanglement Functions (Refactored V4.3.8 - Principle-Driven)

Establishes connection post-Sephiroth journey using SU/CU prerequisites.
Connection strength/quality and aspect transfer are modulated by frequency resonance.
Stability/Coherence emerge via update_state after factor changes. No artificial stabilization loop.
Operates on SoulSpark object, using KetherField properties conceptually. Hard fails on critical errors.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import uuid
import json
from math import pi as PI, sqrt, atan2, exp
from typing import Dict, List, Any, Tuple, Optional
from constants.constants import *

# Set up logger
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Creator Entanglement cannot function.")

    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    from stage_1.fields.kether_field import KetherField
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
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

def _check_prerequisites(soul_spark: SoulSpark, kether_field: KetherField) -> bool:
    """ Checks prerequisites using SU/CU thresholds. Raises ValueError on failure. """
    logger.debug(f"Checking Creator Entanglement prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("Invalid SoulSpark object.")
    if not isinstance(kether_field, KetherField):
        raise TypeError("Invalid KetherField object.")

    # 1. Stage Completion Check
    if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False):
        msg = f"Prerequisite failed: {FLAG_SEPHIROTH_JOURNEY_COMPLETE} is False."
        logger.error(msg)
        raise ValueError(msg)
    if getattr(soul_spark, 'creator_channel_id', None) is not None:
        logger.warning(f"Soul {soul_spark.spark_id} already has creator channel. Overwriting.")

    # 2. Soul State Check (Absolute SU/CU Thresholds)
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    if stability_su < ENTANGLEMENT_PREREQ_STABILITY_MIN_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {ENTANGLEMENT_PREREQ_STABILITY_MIN_SU} SU."
        logger.error(msg)
        raise ValueError(msg)
    if coherence_cu < ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU} CU."
        logger.error(msg)
        raise ValueError(msg)

    # 3. Field Check (Conceptual Location)
    current_field = getattr(soul_spark, 'current_field_key', 'unknown')
    if current_field != 'kether':
        msg = f"Prerequisite failed: Soul must be conceptually in Kether field (current: {current_field})."
        logger.error(msg)
        raise ValueError(msg)
    # Assume Guff release handled by controller

    logger.debug("Creator Entanglement prerequisites met.")
    return True

# --- Resonance Calculation (Using detailed version) ---
def calculate_resonance(freq1: float, freq2: float) -> float:
    """ Calculate resonance strength (0-1) between two frequencies. """
    if min(freq1, freq2) <= FLOAT_EPSILON: return 0.0
    ratio = max(freq1, freq2) / min(freq1, freq2)
    int_res = 0.0
    for i in range(1, 6): # Check up to 5th harmonic
        for j in range(1, 6):
            target = float(max(i, j)) / float(min(i, j))
            tolerance = RESONANCE_INTEGER_RATIO_TOLERANCE * target * 1.5 # Wider tolerance
            deviation = abs(ratio - target)
            if deviation < tolerance:
                score = max(0.0, 1.0 - (deviation / tolerance))**2 # Sharper peak
                complexity_penalty = 1.0 / (1.0 + (i - 1) * 0.1 + (j - 1) * 0.1)
                int_res = max(int_res, score * complexity_penalty)
    phi_res = 0.0
    for i in [1, 2]: # Check Phi and Phi^2
        phi_pow = GOLDEN_RATIO ** i
        phi_tolerance = RESONANCE_PHI_RATIO_TOLERANCE * phi_pow * 1.5
        dev_phi = abs(ratio - phi_pow)
        if dev_phi < phi_tolerance:
            score = max(0.0, 1.0 - (dev_phi / phi_tolerance))**2
            phi_res = max(phi_res, score)
        inv_phi_pow = 1.0 / phi_pow
        inv_phi_tolerance = RESONANCE_PHI_RATIO_TOLERANCE * inv_phi_pow * 1.5
        dev_inv_phi = abs(ratio - inv_phi_pow)
        if dev_inv_phi < inv_phi_tolerance:
            score = max(0.0, 1.0 - (dev_inv_phi / inv_phi_tolerance))**2
            phi_res = max(phi_res, score)

    final_resonance = max(int_res, phi_res) # Take the stronger match
    # logger.debug(f"  Calculate Resonance: F1={freq1:.2f}, F2={freq2:.2f}, Ratio={ratio:.3f} -> IntRes={int_res:.4f}, PhiRes={phi_res:.4f} -> Final={final_resonance:.4f}")
    return float(final_resonance)

def find_best_resonance_match(soul_sig: Dict, kether_sig: Dict) -> Tuple[float, float, float]:
    """ Finds the best resonant frequency pair between soul and Kether signatures. """
    best_score = -1.0
    # Get potential frequency lists/arrays
    soul_freqs_data = soul_sig.get('frequencies', [])
    kether_freqs_data = kether_sig.get('frequencies', [])

    # Ensure they are numpy arrays for size check, handle potential None
    soul_freqs = np.array(soul_freqs_data) if soul_freqs_data is not None else np.array([])
    kether_freqs = np.array(kether_freqs_data) if kether_freqs_data is not None else np.array([])

    best_soul_freq = soul_sig.get('base_frequency', 0.0)
    best_kether_freq = kether_sig.get('base_frequency', 0.0)

    # --- MODIFIED CHECK ---
    # Explicitly check if either array is empty using .size
    if soul_freqs.size == 0 or kether_freqs.size == 0:
    # --- END MODIFICATION ---
        logger.warning("Cannot find resonance match: empty frequency lists.")
        return 0.0, best_soul_freq, best_kether_freq # Return 0 score if no frequencies

    # Ensure frequencies are floats for calculation
    soul_freqs = soul_freqs.astype(float)
    kether_freqs = kether_freqs.astype(float)

    for f_soul in soul_freqs:
        for f_kether in kether_freqs:
            score = calculate_resonance(f_soul, f_kether)
            if score > best_score:
                best_score = score
                best_soul_freq = f_soul
                best_kether_freq = f_kether

    logger.debug(f"  Best Resonance Match: Score={best_score:.4f} (Soul={best_soul_freq:.1f}Hz, Kether={best_kether_freq:.1f}Hz)")
    return max(0.0, best_score), best_soul_freq, best_kether_freq

# --- Core Entanglement Functions ---

def establish_creator_connection(soul_spark: SoulSpark,
                                 kether_field: KetherField,
                                 base_creator_potential: float, # Innate potential factor 0-1
                                 edge_of_chaos_target: float
                                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Establishes connection based on resonance. Modifies SoulSpark factors. """
    logger.info(f"Establishing creator connection for soul {soul_spark.spark_id}...")
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    if not (0.0 <= base_creator_potential <= 1.0): raise ValueError("base_creator_potential out of range.")
    if not (0.0 <= edge_of_chaos_target <= 1.0): raise ValueError("edge_of_chaos_target out of range.")

    try:
        # 1. Get Current State & Signatures
        initial_stability_su = soul_spark.stability
        initial_coherence_cu = soul_spark.coherence
        coherence_norm = initial_coherence_cu / MAX_COHERENCE_CU
        soul_freq_sig = soul_spark.frequency_signature
        if not soul_freq_sig or 'frequencies' not in soul_freq_sig:
            logger.warning("Soul frequency signature missing or invalid, regenerating.")
            soul_spark._validate_or_init_frequency_structure()
            soul_freq_sig = soul_spark.frequency_signature

        kether_aspects = kether_field.get_aspects()
        kether_freq = kether_aspects.get('base_frequency', KETHER_FREQ)
        # Create a simplified Kether signature for resonance matching
        kether_sig = {'base_frequency': kether_freq, 'frequencies': [kether_freq] + [d.get('frequency', kether_freq) for d in kether_aspects.get('detailed_aspects', {}).values() if d.get('frequency', 0) > FLOAT_EPSILON]}
        logger.debug(f"  Connection: Soul Sig Base={soul_freq_sig.get('base_frequency'):.1f}Hz, Kether Sig Base={kether_sig.get('base_frequency'):.1f}Hz")

        # 2. Find Best Resonance Match
        frequency_resonance_score, _, primary_connection_freq_kether = find_best_resonance_match(soul_freq_sig, kether_sig)
        logger.debug(f"  Connection: Freq Resonance Score = {frequency_resonance_score:.4f}")

        # 3. Calculate Connection Strength (Emergent Factor 0-1)
        # Strength depends on resonance quality and soul's coherence
        connection_strength = (frequency_resonance_score * CE_CONNECTION_FREQ_WEIGHT +
                               coherence_norm * CE_CONNECTION_COHERENCE_WEIGHT)
        # Modulate slightly by innate potential
        connection_strength *= (0.8 + 0.4 * base_creator_potential) # Range 0.8 to 1.2 multiplier
        connection_strength = min(1.0, max(0.0, connection_strength)) # Clamp 0-1
        logger.debug(f"  Connection: Strength Calculated = {connection_strength:.4f} (FreqW={CE_CONNECTION_FREQ_WEIGHT}, CohW={CE_CONNECTION_COHERENCE_WEIGHT}, BasePot={base_creator_potential})")

        # 4. Calculate Connection Quality (Includes Stability & Chaos Factor)
        stability_norm = initial_stability_su / MAX_STABILITY_SU
        connection_quality_base = (0.3 * stability_norm + 0.7 * coherence_norm) # Weighted average
        chaos_deviation = abs(connection_quality_base - edge_of_chaos_target)
        chaos_factor = exp(-(chaos_deviation**2) / (2 * (0.15**2))) # Gaussian falloff
        connection_quality = connection_strength * chaos_factor # Overall quality depends on strength AND chaos alignment
        connection_quality = min(1.0, max(0.0, connection_quality))
        logger.debug(f"  Connection: Quality Base={connection_quality_base:.3f}, ChaosFactor={chaos_factor:.3f} -> Final Quality={connection_quality:.4f}")

        # 5. Generate Channel Data
        channel_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        quantum_channel_data = {
            'channel_id': channel_id, 'spark_id': soul_spark.spark_id, 'creation_time': creation_time,
            'connection_strength': float(connection_strength),
            'connection_quality': float(connection_quality), # Used for aspect transfer efficiency
            'frequency_resonance_score': float(frequency_resonance_score),
            'primary_connection_freq_kether_hz': float(primary_connection_freq_kether),
            'initial_soul_stability_su': float(initial_stability_su),
            'initial_soul_coherence_cu': float(initial_coherence_cu),
            'chaos_factor': float(chaos_factor),
            'active': True
            # Frequency signature stored on soul spark now
        }
        logger.debug(f"  Quantum Channel Data Created: {channel_id}")

        # 6. Update SoulSpark Factors
        initial_alignment = soul_spark.creator_alignment # 0-1 factor
        soul_spark.creator_channel_id = channel_id
        soul_spark.creator_connection_strength = float(connection_strength)
        # Alignment boost now depends on connection quality (more aligned connection = bigger boost)
        alignment_gain = ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR * connection_quality
        soul_spark.creator_alignment = float(min(1.0, initial_alignment + alignment_gain))
        # Store the frequency signature that *resulted* in this connection
        # soul_spark.frequency_signature = soul_freq_sig # Already validated/generated if needed
        soul_spark.last_modified = creation_time
        logger.info(f"Creator connection established: Strength={connection_strength:.4f}, Quality={connection_quality:.4f}, Alignment={soul_spark.creator_alignment:.4f} (+{alignment_gain:.4f})")

        # 7. Record Metrics
        metrics_data = {
            'action': 'establish_connection', 'soul_id': soul_spark.spark_id, 'channel_id': channel_id,
            'strength': connection_strength, 'quality': connection_quality, 'chaos_factor': chaos_factor,
            'frequency_resonance_score': frequency_resonance_score,
            'initial_stability_su': initial_stability_su,
            'initial_coherence_cu': initial_coherence_cu,
            'initial_alignment': initial_alignment,
            'final_alignment': soul_spark.creator_alignment,
            'alignment_change': alignment_gain,
            'success': True, 'timestamp': creation_time
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return quantum_channel_data, metrics_data

    except Exception as e:
        logger.error(f"Error establishing creator connection: {e}", exc_info=True)
        # Hard fail
        raise RuntimeError("Creator connection establishment failed critically.") from e

def form_resonance_patterns(soul_spark: SoulSpark,
                            quantum_channel_data: Dict[str, Any],
                            kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Forms resonance patterns based on Kether aspects and connection quality. """
    logger.info(f"Forming resonance patterns for soul {soul_spark.spark_id}...")
    # --- Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_strength = quantum_channel_data.get('connection_strength') # For strength scaling
    connection_quality = quantum_channel_data.get('connection_quality') # For coherence calc? maybe just strength
    freq_res_score = quantum_channel_data.get('frequency_resonance_score') # Use actual resonance score
    if not channel_id or not spark_id or connection_strength is None or freq_res_score is None:
        raise ValueError("Quantum channel data missing essential keys (strength, freq_res_score).")

    try:
        kether_aspects = kether_field.get_aspects()
        if not kether_aspects: raise ValueError("Kether field aspect data is empty.")
        detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
        if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

        patterns = {}
        logger.debug("  Forming patterns from Kether primary/secondary aspects:")
        # Pattern strength now depends more on overall connection quality/resonance
        base_pattern_strength = connection_strength * freq_res_score # Use product?

        for name in kether_aspects.get('primary_aspects', []):
             if name in detailed_kether_aspects:
                  details = detailed_kether_aspects[name]
                  # Scale Kether's base strength by the calculated base pattern strength
                  pat_strength = float(details.get('strength', 0.0) * base_pattern_strength)
                  patterns[name] = {
                      'frequency': details.get('frequency', KETHER_FREQ),
                      'strength': pat_strength, 'pattern_type': 'primary', **details
                  }
                  logger.debug(f"    Added Primary Pattern: {name} (Strength: {pat_strength:.4f})")
        for name in kether_aspects.get('secondary_aspects', []):
             if name in detailed_kether_aspects:
                  details = detailed_kether_aspects[name]
                  # Secondary patterns are weaker
                  pat_strength = float(details.get('strength', 0.0) * base_pattern_strength * 0.7)
                  patterns[f"secondary_{name}"] = {
                      'frequency': details.get('frequency', KETHER_FREQ),
                      'strength': pat_strength, 'pattern_type': 'secondary', **details
                  }
                  logger.debug(f"    Added Secondary Pattern: {name} (Strength: {pat_strength:.4f})")

        # Add harmonic patterns based on Kether's primary connection frequency
        logger.debug("  Forming patterns from Kether's primary connection frequency harmonics:")
        primary_kether_freq = quantum_channel_data.get('primary_connection_freq_kether_hz', KETHER_FREQ)
        # Generate a few harmonics based on Kether's freq
        kether_harmonics = [primary_kether_freq * r for r in [PHI, 2, 1/PHI, 3] if primary_kether_freq * r > FLOAT_EPSILON]
        for i, h_freq in enumerate(kether_harmonics):
            # Strength also depends on connection strength/resonance
            strength = float(base_pattern_strength * (0.6 / (i + 1.5))) # Diminishing strength
            patterns[f'kether_harmonic_{i}'] = {
                'frequency': float(h_freq), 'strength': strength,
                'pattern_type': 'kether_harmonic', 'description': f'Kether Harmonic ({h_freq:.1f} Hz)'
            }
            logger.debug(f"    Added Kether Harmonic Pattern: kether_harmonic_{i} (Freq: {h_freq:.1f}, Strength: {strength:.4f})")

        # Calculate pattern coherence (average strength of formed patterns)
        pattern_coherence = 0.0
        if patterns:
            valid_strengths = [p.get('strength', 0.0) for p in patterns.values() if isinstance(p.get('strength'), (int, float))]
            if valid_strengths: pattern_coherence = sum(valid_strengths) / len(valid_strengths)
        logger.debug(f"  Calculated Pattern Coherence factor: {pattern_coherence:.4f}")

        formation_time = datetime.now().isoformat()
        resonance_pattern_data = {
            'channel_id': channel_id, 'spark_id': spark_id, 'patterns': patterns,
            'pattern_count': len(patterns), 'pattern_coherence': float(pattern_coherence),
            'formation_time': formation_time
        }

        # Update SoulSpark Factors
        initial_resonance_factor = soul_spark.resonance # 0-1 factor
        soul_spark.resonance_patterns = patterns.copy()
        # Update pattern_coherence based on the calculated average strength
        soul_spark.pattern_coherence = float(pattern_coherence)
        # Boost general resonance factor based on pattern coherence improvement
        resonance_boost = ENTANGLEMENT_RESONANCE_BOOST_FACTOR * pattern_coherence
        soul_spark.resonance = float(min(1.0, initial_resonance_factor + resonance_boost))
        soul_spark.last_modified = formation_time
        logger.info(f"Patterns formed: {len(patterns)}. New Resonance Factor: {soul_spark.resonance:.4f} (+{resonance_boost:.4f}), P.Coh factor: {pattern_coherence:.4f}")

        # Record Metrics
        metrics_data = {
            'action': 'form_patterns', 'soul_id': spark_id, 'channel_id': channel_id,
            'pattern_count': len(patterns), 'pattern_coherence_factor': pattern_coherence, # Renamed metric key
            'initial_soul_resonance_factor': initial_resonance_factor, # Added initial
            'final_soul_resonance_factor': soul_spark.resonance,
            'resonance_factor_change': soul_spark.resonance - initial_resonance_factor,
            'success': True, 'timestamp': formation_time
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return resonance_pattern_data, metrics_data

    except Exception as e:
        logger.error(f"Error forming resonance patterns: {e}", exc_info=True)
        # Hard fail
        raise RuntimeError("Pattern formation failed critically.") from e


def transfer_creator_aspects(soul_spark: SoulSpark,
                             quantum_channel_data: Dict[str, Any],
                             resonance_pattern_data: Dict[str, Any],
                             kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Transfers Kether aspects scaled by connection quality and pattern resonance. No threshold. """
    logger.info(f"Transferring creator aspects to soul {soul_spark.spark_id}...")
    # --- Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data invalid.")
    if not isinstance(resonance_pattern_data, dict): raise TypeError("resonance_pattern_data invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_quality = quantum_channel_data.get('connection_quality') # Use quality (0-1) for efficiency
    patterns = resonance_pattern_data.get('patterns')
    if not channel_id or not spark_id or connection_quality is None or patterns is None:
        raise ValueError("Channel/pattern data missing essential keys.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
        setattr(soul_spark, 'aspects', {}) # Initialize if missing

    try:
        kether_aspects = kether_field.get_aspects()
        if not kether_aspects: raise ValueError("Kether field aspect data is empty.")
        detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
        if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

        transferable_aspects = {}
        gained_count = 0; strengthened_count = 0
        transfer_time = datetime.now().isoformat()
        total_original_strength = 0.0; total_imparted_strength = 0.0
        total_alignment_gain = 0.0 # Accumulate alignment gain from aspects

        logger.debug("  Calculating aspect transfer efficiencies:")
        for name, kether_details in detailed_kether_aspects.items():
            kether_strength = float(kether_details.get('strength', 0.0))
            if kether_strength <= FLOAT_EPSILON: continue
            total_original_strength += kether_strength

            # Efficiency depends on connection quality AND pattern resonance strength
            pattern_strength = patterns.get(name, {}).get('strength', 0.0)
            if pattern_strength <= FLOAT_EPSILON: pattern_strength = patterns.get(f"secondary_{name}", {}).get('strength', 0.0)
            logger.debug(f"    Aspect '{name}': KetherStr={kether_strength:.3f}, ConnQual={connection_quality:.3f}, PatternStr={pattern_strength:.3f}")

            # Recalculate efficiency using updated factors
            efficiency = (connection_quality * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE +
                          pattern_strength * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE)
            efficiency = min(1.0, max(0.0, efficiency))
            imparted_strength = kether_strength * efficiency
            total_imparted_strength += imparted_strength
            logger.debug(f"      -> Efficiency={efficiency:.4f}, ImpartedStr={imparted_strength:.4f}")

            # Record Potential Transfer
            transfer_data = {
                'original_strength': kether_strength, 'transfer_efficiency': efficiency,
                'imparted_strength': imparted_strength, 'aspect_type': 'Kether',
                'details': kether_details
            }
            transferable_aspects[name] = transfer_data

            # Update SoulSpark Aspects
            current_soul_strength = float(soul_spark.aspects.get(name, {}).get('strength', 0.0))
            # Use MAX_ASPECT_STRENGTH constant
            new_strength = min(MAX_ASPECT_STRENGTH, current_soul_strength + imparted_strength)
            actual_gain = new_strength - current_soul_strength
            logger.debug(f"      -> Soul Aspect '{name}': CurrentStr={current_soul_strength:.4f}, NewStr={new_strength:.4f}, Gain={actual_gain:.4f}")

            if actual_gain > FLOAT_EPSILON:
                details_copy = kether_details.copy()
                details_copy['last_transfer_efficiency'] = efficiency
                details_copy['last_imparted_strength'] = imparted_strength

                if name not in soul_spark.aspects or current_soul_strength <= FLOAT_EPSILON:
                    soul_spark.aspects[name] = {'strength': new_strength, 'source': 'Kether', 'time_acquired': transfer_time, 'details': details_copy }
                    gained_count += 1
                else:
                    soul_spark.aspects[name]['strength'] = new_strength
                    soul_spark.aspects[name]['last_updated'] = transfer_time
                    soul_spark.aspects[name]['details'].update(details_copy)
                    strengthened_count += 1

                # Accumulate alignment gain based on transferred aspect strength
                # Assume aspects contributing to alignment have a flag or higher base strength?
                # Simple model: alignment gain proportional to imparted strength
                total_alignment_gain += imparted_strength * 0.1 # Small alignment gain per aspect transfer

        average_efficiency = (total_imparted_strength / total_original_strength) if total_original_strength > FLOAT_EPSILON else 0.0
        logger.debug(f"  Average Transfer Efficiency: {average_efficiency:.4f}")

        # --- No Direct Stability Boost ---
        initial_stability_su = soul_spark.stability # Store for metrics reporting only
        stability_change_su = 0.0 # No direct change applied here

        # Update alignment factor based on accumulated gain from aspects
        initial_alignment = soul_spark.creator_alignment
        # Total alignment gain is now driven by aspect transfer + base boost from connection
        # Use the gain calculated during connection establishment + gain from aspects
        connection_alignment_gain = soul_spark.creator_alignment - initial_alignment # Gain from establish_connection step
        final_alignment_gain = connection_alignment_gain + total_alignment_gain
        soul_spark.creator_alignment = min(1.0, initial_alignment + final_alignment_gain)
        logger.debug(f"  Creator Alignment Updated: {initial_alignment:.4f} -> {soul_spark.creator_alignment:.4f} (+{final_alignment_gain:.4f})")

        soul_spark.last_modified = transfer_time
        logger.info(f"Aspect transfer: Gained={gained_count}, Strengthened={strengthened_count}, AvgEff={average_efficiency:.3f}")

        # Store Transfer Metrics
        aspect_transfer_metrics = {
            'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
            'average_efficiency': average_efficiency, 'transfers': transferable_aspects
        }
        # Record Core Metrics
        metrics_data = {
            'action': 'transfer_aspects', 'soul_id': spark_id, 'channel_id': channel_id,
            'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
            'avg_efficiency': average_efficiency,
            'initial_soul_alignment': initial_alignment,
            'final_soul_alignment': soul_spark.creator_alignment,
            'alignment_gain_total': final_alignment_gain, # Record total gain
            'initial_soul_stability_su': initial_stability_su, # Report initial SU for context
            'stability_change_su': stability_change_su, # Report 0 direct change
            'success': True, 'timestamp': transfer_time
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
        return aspect_transfer_metrics, metrics_data

    except Exception as e:
        logger.error(f"Error transferring creator aspects: {e}", exc_info=True)
        # Hard fail
        raise RuntimeError("Aspect transfer failed critically.") from e

# --- Orchestration Function ---
def run_full_entanglement_process(soul_spark: SoulSpark,
                                  kether_field: KetherField,
                                  creator_resonance: float = 0.8, # Base potential factor 0-1
                                  edge_of_chaos_ratio: float = EDGE_OF_CHAOS_RATIO, # Target 0-1 factor
                                  # stabilization_iterations removed
                                  ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Runs complete entanglement process based on resonance. Modifies SoulSpark factors. S/C emerge via update_state. """
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
    if not (0.0 <= creator_resonance <= 1.0): raise ValueError("creator_resonance invalid.")
    if not (0.0 <= edge_of_chaos_ratio <= 1.0): raise ValueError("edge_of_chaos_ratio invalid.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    process_metrics_summary = {'steps': {}}

    logger.info(f"--- Starting Full Entanglement Process [Resonance Driven] for Soul {spark_id} ---")

    try:
        # --- Prerequisites Check (Raises ValueError if failed) ---
        _check_prerequisites(soul_spark, kether_field)

        # Store Initial State
        initial_state = {
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
             'pattern_coherence': soul_spark.pattern_coherence, # Added
             'aspect_count': len(soul_spark.aspects) }
        logger.debug(f"  Entanglement Initial State: {initial_state}")

        # --- Run Stages ---
        logger.info("Step 1: Establishing Connection...")
        channel_data, metrics1 = establish_creator_connection(soul_spark, kether_field, creator_resonance, edge_of_chaos_ratio)
        process_metrics_summary['steps']['connection'] = metrics1

        logger.info("Step 2: Forming Resonance Patterns...")
        pattern_data, metrics2 = form_resonance_patterns(soul_spark, channel_data, kether_field)
        process_metrics_summary['steps']['patterns'] = metrics2

        logger.info("Step 3: Transferring Creator Aspects...")
        transfer_metrics, metrics3 = transfer_creator_aspects(soul_spark, channel_data, pattern_data, kether_field)
        process_metrics_summary['steps']['transfer'] = metrics3

        # --- REMOVED Stabilization Step ---
        # logger.info("Step 4: Stabilizing Connection...")
        # stabilization_result, metrics4 = stabilize_creator_connection(soul_spark, channel_data, pattern_data, stabilization_iterations)
        # process_metrics_summary['steps']['stabilization'] = metrics4

        # --- Final State Update ---
        logger.info("Step 4: Updating internal state scores...") # Renumbered step
        # Capture S/C *before* final update_state for peak reporting relative to start
        pre_update_stability_su = soul_spark.stability
        pre_update_coherence_cu = soul_spark.coherence
        logger.debug(f"  Before final update_state in entanglement: S={pre_update_stability_su:.1f}, C={pre_update_coherence_cu:.1f}")

        setattr(soul_spark, FLAG_READY_FOR_COMPLETION, True)
        setattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, True)
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)

        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state()
            logger.debug(f"  After final update_state in entanglement: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else:
             logger.error("SoulSpark object missing 'update_state' method!")
             raise AttributeError("SoulSpark needs 'update_state' method for entanglement.")

        # Compile Overall Metrics
        end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report state in correct units after update_state
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
             'pattern_coherence': soul_spark.pattern_coherence, # Added final factor
             'aspect_count': len(soul_spark.aspects) }
        overall_metrics = {
            'action': 'full_entanglement', 'soul_id': spark_id,
            'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state,
            # Report the state *before* the final update_state was called
            'pre_update_stability_su': float(pre_update_stability_su),
            'pre_update_coherence_cu': float(pre_update_coherence_cu),
            'final_state': final_state, # State AFTER update_state
            'final_connection_strength': channel_data['connection_strength'], # From connection step
            'connection_quality': channel_data['connection_quality'], # Added quality metric
            'frequency_resonance_score': channel_data['frequency_resonance_score'], # Added resonance score
            'aspects_transferred_count': transfer_metrics.get('aspects_gained_count', 0) + transfer_metrics.get('aspects_strengthened_count', 0),
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement_summary', overall_metrics)

        logger.info(f"--- Full Entanglement Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    # --- Error Handling (Ensure Hard Fail) ---
    except (ValueError, TypeError, AttributeError) as e_val:
        logger.error(f"Entanglement failed for soul {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_val))
        # Hard fail
        raise ValueError(f"Entanglement validation/attribute error: {e_val}") from e_val
    except RuntimeError as e_rt:
        logger.critical(f"Entanglement failed critically for soul {spark_id}: {e_rt}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'runtime'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_rt))
        # Hard fail
        raise e_rt
    except Exception as e:
        logger.critical(f"Unexpected error during entanglement for {spark_id}: {e}", exc_info=True)
        failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
        record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e))
        # Hard fail
        raise RuntimeError(f"Unexpected entanglement failure: {e}") from e

# --- Failure Metric Helper ---
def record_entanglement_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('creator_entanglement_summary', {
                'action': 'full_entanglement', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time,
                'duration_seconds': duration,
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record entanglement failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/creator_entanglement.py ---















# # --- START OF FILE src/stage_1/soul_formation/creator_entanglement.py ---

# """
# Creator Entanglement Functions (Refactored V4.1 - SEU/SU/CU Units)

# Establishes connection post-Sephiroth journey using updated prerequisites (SU/CU),
# aspect transfer logic. Stability/Coherence emerge via update_state after factor changes.
# Operates on SoulSpark object, using KetherField properties conceptually.
# """

# import logging
# import numpy as np
# import os
# import sys
# from datetime import datetime
# import uuid
# import json
# from math import pi as PI, sqrt, atan2, exp # Added missing imports
# from typing import Dict, List, Any, Tuple, Optional

# # Set up logger
# logger = logging.getLogger(__name__) # Ensure logger is named correctly

# # --- Constants ---
# try:
#     from constants.constants import *
# except ImportError as e:
#     # Fallback logging setup if constants fail
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Creator Entanglement cannot function.")
#     # # Define minimal necessary constants as fallbacks ONLY to allow script parsing,
#     # # but raise the error to prevent execution without proper constants.
#     # MAX_STABILITY_SU = 100.0; MAX_COHERENCE_CU = 100.0; FLOAT_EPSILON = 1e-9;
#     # ENTANGLEMENT_PREREQ_STABILITY_MIN_SU = 75.0; ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU = 75.0
#     # FLAG_SEPHIROTH_JOURNEY_COMPLETE = "sephiroth_journey_complete"; KETHER_FREQ = 963.0
#     # ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR = 0.15; GOLDEN_RATIO = 1.618
#     # RESONANCE_INTEGER_RATIO_TOLERANCE = 0.02; RESONANCE_PHI_RATIO_TOLERANCE = 0.03
#     # ENTANGLEMENT_RESONANCE_BOOST_FACTOR = 0.05
#     # ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE = 0.6; ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE = 0.4
#     # MAX_ASPECT_STRENGTH = 1.0; ENTANGLEMENT_STABILIZATION_ITERATIONS = 5
#     # ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH = 1.005 # Small multiplier > 1
#     # EDGE_OF_CHAOS_RATIO = 1.0 / GOLDEN_RATIO
#     # FLAG_READY_FOR_COMPLETION="ready_for_completion"
#     # FLAG_READY_FOR_STRENGTHENING="ready_for_strengthening"
#     raise ImportError(f"Essential constants missing: {e}") from e

# # --- Dependency Imports ---
# try:
#     from stage_1.soul_spark.soul_spark import SoulSpark
#     # KetherField is needed conceptually for properties, passed as argument
#     from stage_1.fields.kether_field import KetherField
# except ImportError as e:
#     logger.critical(f"CRITICAL ERROR: Failed to import dependencies: {e}")
#     # Re-raise to ensure hard fail
#     raise ImportError(f"Core dependencies missing: {e}") from e

# # --- Metrics Tracking ---
# try:
#     import metrics_tracking as metrics
#     METRICS_AVAILABLE = True
# except ImportError:
#     logger.error("Metrics tracking module not found. Metrics will not be recorded.")
#     METRICS_AVAILABLE = False
#     class MetricsPlaceholder:
#         def record_metrics(self, *args, **kwargs): pass
#     metrics = MetricsPlaceholder()

# # --- Helper Functions ---

# def _check_prerequisites(soul_spark: SoulSpark, kether_field: KetherField) -> bool:
#     """ Checks prerequisites using SU/CU thresholds. """
#     logger.debug(f"Checking Creator Entanglement prerequisites for soul {soul_spark.spark_id}...")
#     if not isinstance(soul_spark, SoulSpark):
#         logger.error("Prerequisite failed: Invalid SoulSpark object.")
#         return False
#     if not isinstance(kether_field, KetherField):
#         logger.error("Prerequisite failed: Invalid KetherField object.")
#         return False

#     # 1. Stage Completion Check
#     if not getattr(soul_spark, FLAG_SEPHIROTH_JOURNEY_COMPLETE, False):
#         logger.error(f"Prerequisite failed: {FLAG_SEPHIROTH_JOURNEY_COMPLETE} is False.")
#         return False
#     if getattr(soul_spark, 'creator_channel_id', None) is not None:
#         logger.warning(f"Soul {soul_spark.spark_id} already has creator channel. Re-running.")

#     # 2. Soul State Check (Absolute SU/CU Thresholds)
#     stability_su = getattr(soul_spark, 'stability', 0.0)
#     coherence_cu = getattr(soul_spark, 'coherence', 0.0)
#     if stability_su < ENTANGLEMENT_PREREQ_STABILITY_MIN_SU:
#         logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {ENTANGLEMENT_PREREQ_STABILITY_MIN_SU} SU.")
#         return False
#     if coherence_cu < ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU:
#         logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {ENTANGLEMENT_PREREQ_COHERENCE_MIN_CU} CU.")
#         return False

#     # 3. Field Check (Conceptual Location)
#     current_field = getattr(soul_spark, 'current_field_key', 'unknown')
#     if current_field != 'kether':
#         logger.error(f"Prerequisite failed: Soul must be conceptually in Kether field (current: {current_field}).")
#         return False
#     # Ensure not in Guff (using placeholder check - assumes controller handles placement)
#     # Add actual check if needed:
#     # if kether_field.is_in_guff(field_controller._coords_to_int_tuple(soul_spark.position)):
#     #    logger.error("Prerequisite failed: Soul should be released from Guff before entanglement.")
#     #    return False

#     logger.debug("Creator Entanglement prerequisites met.")
#     return True

# # --- Resonance Calculation (Using Journey version for consistency) ---
# def calculate_resonance(freq1: float, freq2: float) -> float:
#     """ Calculate resonance strength (0-1) between two frequencies. """
#     if min(freq1, freq2) <= FLOAT_EPSILON: return 0.0
#     ratio = max(freq1, freq2) / min(freq1, freq2)
#     int_res = 0.0
#     # Check integer ratios up to 5:1
#     for i in range(1, 6):
#         for j in range(1, 6):
#             target = float(max(i, j)) / float(min(i, j))
#             # Use a slightly wider tolerance for better matching chances
#             tolerance = RESONANCE_INTEGER_RATIO_TOLERANCE * target * 1.5
#             deviation = abs(ratio - target)
#             if deviation < tolerance:
#                 # Score decays linearly within tolerance window
#                 score = max(0.0, 1.0 - (deviation / tolerance))
#                 # Optional: Weight simpler ratios higher (e.g., penalize complexity)
#                 complexity_penalty = 1.0 / (1.0 + (i - 1) * 0.1 + (j - 1) * 0.1)
#                 int_res = max(int_res, score * complexity_penalty)

#     phi_res = 0.0
#     # Check Phi ratios up to Phi^2
#     for i in [1, 2]:
#         phi_pow = GOLDEN_RATIO ** i
#         # Check direct Phi power
#         phi_tolerance = RESONANCE_PHI_RATIO_TOLERANCE * phi_pow * 1.5
#         dev_phi = abs(ratio - phi_pow)
#         if dev_phi < phi_tolerance:
#             score = max(0.0, 1.0 - (dev_phi / phi_tolerance))
#             phi_res = max(phi_res, score)
#         # Check inverse Phi power
#         inv_phi_pow = 1.0 / phi_pow
#         inv_phi_tolerance = RESONANCE_PHI_RATIO_TOLERANCE * inv_phi_pow * 1.5
#         dev_inv_phi = abs(ratio - inv_phi_pow)
#         if dev_inv_phi < inv_phi_tolerance:
#             score = max(0.0, 1.0 - (dev_inv_phi / inv_phi_tolerance))
#             phi_res = max(phi_res, score)

#     # Combine integer and Phi resonance, giving Phi slightly more weight?
#     final_resonance = max(int_res * 0.9, phi_res) # Prioritize Phi slightly if both match
#     logger.debug(f"  Calculate Resonance: F1={freq1:.2f}, F2={freq2:.2f}, Ratio={ratio:.3f} -> IntRes={int_res:.4f}, PhiRes={phi_res:.4f} -> Final={final_resonance:.4f}")
#     return float(final_resonance)


# # --- Frequency Signature Calculation ---
# def _calculate_frequency_signature(soul_spark: SoulSpark, kether_field: KetherField) -> Dict[str, Any]:
#     """ Calculates frequency signature based on Soul and Kether aspect frequencies. """
#     logger.debug(f"Calculating frequency signature for soul {soul_spark.spark_id} with Kether...")
#     try:
#         # Get Kether Frequencies (from aspect data)
#         kether_aspects = kether_field.get_aspects()
#         if not kether_aspects: raise ValueError("Kether field aspect data is empty.")
#         detailed_kether = kether_aspects.get('detailed_aspects', {})
#         kether_base_freq = kether_aspects.get('base_frequency', KETHER_FREQ)
#         creator_freqs_dict = {
#             name: details.get('frequency', kether_base_freq)
#             for name, details in detailed_kether.items()
#             if details.get('frequency', 0.0) > FLOAT_EPSILON
#         }
#         if not creator_freqs_dict and kether_base_freq > FLOAT_EPSILON:
#             creator_freqs_dict = {'kether_base': kether_base_freq}
#         elif not creator_freqs_dict:
#             logger.error("Kether field lacks valid frequencies in aspect data.")
#             raise ValueError("Kether field lacks valid frequencies.")
#         creator_freqs_list = list(creator_freqs_dict.values())
#         logger.debug(f"  Kether Frequencies: {creator_freqs_dict}")

#         # Get Soul Frequencies
#         soul_base_freq = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
#         if soul_base_freq <= FLOAT_EPSILON: raise ValueError("Soul has invalid base frequency.")
#         soul_harmonics = getattr(soul_spark, 'harmonics', [])
#         # Ensure harmonics are valid and relate to base frequency
#         if not soul_harmonics or not isinstance(soul_harmonics, list) or \
#            abs(soul_harmonics[0] - soul_base_freq) > FLOAT_EPSILON * soul_base_freq:
#              logger.debug("  Regenerating soul harmonics for signature calc.")
#              if hasattr(soul_spark, '_generate_phi_integer_harmonics'):
#                  soul_spark._generate_phi_integer_harmonics()
#                  soul_harmonics = soul_spark.harmonics
#              else:
#                  raise AttributeError("SoulSpark missing harmonic generation method.")

#         soul_freqs_list = [f for f in soul_harmonics if isinstance(f, (int, float)) and f > FLOAT_EPSILON]
#         if not soul_freqs_list: soul_freqs_list = [soul_base_freq]
#         soul_freqs_dict = {f'soul_h{i+1}': freq for i, freq in enumerate(soul_freqs_list)}
#         logger.debug(f"  Soul Frequencies: {soul_freqs_dict}")

#         # Find Resonance Points
#         resonance_points_hz = []
#         resonance_details = {} # Store details for debugging
#         for creator_name, f_creator in creator_freqs_dict.items():
#              for soul_name, f_soul in soul_freqs_dict.items():
#                   res_score = calculate_resonance(f_creator, f_soul)
#                   resonance_details[f"{creator_name}<->{soul_name}"] = res_score
#                   # Increased threshold for strong resonance
#                   if res_score > 0.8:
#                        # Midpoint as resonance frequency
#                        resonance_point = (f_creator + f_soul) / 2.0
#                        resonance_points_hz.append(resonance_point)
#                        logger.debug(f"    Strong Resonance Found: {creator_name}({f_creator:.1f}) - {soul_name}({f_soul:.1f}), Score={res_score:.3f}, Point={resonance_point:.1f}")

#         logger.debug(f"  Resonance Detail Scores: {resonance_details}")

#         # Ensure at least one resonance point (fallback if none strong enough)
#         if not resonance_points_hz:
#             logger.warning("  No strong resonance points found, using fallback midpoint.")
#             resonance_points_hz.append((creator_freqs_list[0] + soul_freqs_list[0]) / 2.0)

#         # Generate phases structure - Aim for slightly more coherence
#         num_points = len(resonance_points_hz)
#         phases = []
#         if num_points > 0:
#             # Introduce a bias towards phase alignment based on soul's current coherence
#             coherence_norm = soul_spark.coherence / MAX_COHERENCE_CU
#             mean_phase_target = PI * coherence_norm # Bias towards 0 or pi based on coherence
#             phase_spread = PI * (1.0 - coherence_norm) * 0.8 # Reduced spread for higher coherence
#             phases = np.random.normal(loc=mean_phase_target, scale=phase_spread, size=num_points)
#             phases = (phases + PI) % (2 * PI) - PI # Wrap to -pi to pi around the target

#         # Create final signature components
#         frequencies = np.array(sorted(list(set(resonance_points_hz)))) # Unique sorted
#         num_frequencies = len(frequencies)
#         # Regenerate phases based on the unique, sorted frequencies
#         phases = np.random.normal(loc=mean_phase_target, scale=phase_spread, size=num_frequencies)
#         phases = (phases + PI) % (2 * PI) - PI # Wrap to -pi to pi

#         # Amplitudes based on frequency (e.g., lower freqs slightly stronger) - simple model
#         min_freq = min(frequencies) if num_frequencies > 0 else 1.0
#         max_freq = max(frequencies) if num_frequencies > 0 else 1.0
#         freq_range = max(FLOAT_EPSILON, max_freq - min_freq)
#         # Basic amplitude: higher for lower frequencies within the resonant set
#         amplitudes = 1.0 - np.clip((frequencies - min_freq) / freq_range, 0, 1) * 0.5 if freq_range > FLOAT_EPSILON else np.ones(num_frequencies)
#         # Normalize amplitudes
#         amp_sum = np.sum(amplitudes)
#         amplitudes = amplitudes / max(FLOAT_EPSILON, amp_sum) if amp_sum > FLOAT_EPSILON else np.ones(num_frequencies) / max(1, num_frequencies)

#         signature = {
#             'creator_frequencies': creator_freqs_dict,
#             'soul_frequencies': soul_freqs_dict,
#             'resonance_points_hz': frequencies.tolist(),
#             'primary_resonance_hz': max(frequencies) if num_frequencies > 0 else 0.0,
#             'calculation_time': datetime.now().isoformat(),
#             'base_frequency': float(soul_base_freq),
#             'frequencies': frequencies.tolist(),
#             'phases': phases.tolist(),
#             'amplitudes': amplitudes.tolist(),
#             'num_frequencies': num_frequencies
#         }

#         logger.debug(f"Created frequency signature with {num_frequencies} points. Keys: {list(signature.keys())}")
#         if 'phases' not in signature or signature['phases'] is None:
#             # This should not happen with the new logic, but keep as safeguard
#             logger.error("CRITICAL: Failed to generate valid phases in frequency signature!")
#             raise ValueError("Failed to generate valid phases in frequency signature")

#         return signature

#     except Exception as e:
#         logger.error(f"Error calculating frequency signature: {e}", exc_info=True)
#         # Hard fail - signature is critical
#         raise RuntimeError("Frequency signature calculation failed critically.") from e

# # --- Core Entanglement Functions ---

# def establish_creator_connection(soul_spark: SoulSpark,
#                                  kether_field: KetherField,
#                                  base_creator_resonance: float, # Factor 0-1
#                                  edge_of_chaos_ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """ Establishes connection. Uses SU/CU. Modifies SoulSpark. """
#     logger.info(f"Establishing creator connection for soul {soul_spark.spark_id}...")
#     # --- Input Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
#     if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
#     if not (0.0 <= base_creator_resonance <= 1.0): raise ValueError("base_creator_resonance out of range.")
#     if not (0.0 <= edge_of_chaos_ratio <= 1.0): raise ValueError("edge_of_chaos_ratio out of range.")

#     try:
#         # Use current soul state (SU/CU)
#         stability_su = soul_spark.stability
#         coherence_cu = soul_spark.coherence
#         if stability_su < 0 or coherence_cu < 0:
#             raise ValueError(f"Invalid S/C values for connection: S={stability_su}, C={coherence_cu}")
#         stability_norm = stability_su / MAX_STABILITY_SU # Normalize 0-1
#         coherence_norm = coherence_cu / MAX_COHERENCE_CU # Normalize 0-1
#         logger.debug(f"  Connection Calc: StabNorm={stability_norm:.3f}, CohNorm={coherence_norm:.3f}, BaseRes={base_creator_resonance:.3f}")

#         # Connection Quality (combine normalized state with base potential)
#         # Increased weight for coherence
#         connection_quality = (0.3 * stability_norm + 0.5 * coherence_norm + 0.2 * base_creator_resonance)
#         # Chaos Factor: How close quality is to the edge of chaos target
#         # Using a smoother Gaussian-like function instead of sharp cutoff
#         chaos_deviation = abs(connection_quality - edge_of_chaos_ratio)
#         chaos_factor = exp(-(chaos_deviation**2) / (2 * (0.15**2))) # Sigma=0.15 controls width
#         logger.debug(f"  Connection Calc: ConnQual={connection_quality:.3f}, EdgeTarget={edge_of_chaos_ratio:.3f}, ChaosFactor={chaos_factor:.3f}")

#         # Effective Strength: Modulated by quality and chaos factor
#         effective_strength = base_creator_resonance * chaos_factor * connection_quality
#         effective_strength = min(1.0, max(0.0, effective_strength)) # Clamp 0-1 factor

#         # Calculate frequency signature *before* storing channel data
#         freq_signature = _calculate_frequency_signature(soul_spark, kether_field)
#         logger.debug(f"  Frequency signature generated: {json.dumps(freq_signature, default=str)}") # Log generated signature

#         channel_id = str(uuid.uuid4())
#         creation_time = datetime.now().isoformat()
#         quantum_channel_data = {
#             'channel_id': channel_id, 'spark_id': soul_spark.spark_id, 'creation_time': creation_time,
#             'connection_strength': float(effective_strength),
#             'connection_quality': float(connection_quality),
#             'initial_soul_stability_su': float(stability_su), # Record initial absolute SU
#             'initial_soul_coherence_cu': float(coherence_cu), # Record initial absolute CU
#             'chaos_factor': float(chaos_factor), 'active': True,
#             'frequency_signature': freq_signature # Store the calculated signature
#         }
#         logger.debug(f"  Quantum Channel Data: {quantum_channel_data}")

#         # Update SoulSpark
#         initial_alignment = soul_spark.creator_alignment # 0-1 factor
#         soul_spark.creator_channel_id = channel_id
#         soul_spark.creator_connection_strength = float(effective_strength) # 0-1 factor
#         # Boost alignment based on strength
#         soul_spark.creator_alignment = float(min(1.0, initial_alignment + ENTANGLEMENT_ALIGNMENT_BOOST_FACTOR * effective_strength))
#         # Store frequency signature used for connection
#         soul_spark.frequency_signature = freq_signature
#         soul_spark.last_modified = creation_time
#         logger.info(f"Creator connection established: Strength={effective_strength:.4f}, Alignment={soul_spark.creator_alignment:.4f}")

#         # Record Metrics
#         metrics_data = {
#             'action': 'establish_connection', 'soul_id': soul_spark.spark_id, 'channel_id': channel_id,
#             'strength': effective_strength, 'quality': connection_quality, 'chaos_factor': chaos_factor,
#             'initial_stability_su': quantum_channel_data['initial_soul_stability_su'],
#             'initial_coherence_cu': quantum_channel_data['initial_soul_coherence_cu'],
#             'final_alignment': soul_spark.creator_alignment,
#             'alignment_change': soul_spark.creator_alignment - initial_alignment,
#             'success': True, 'timestamp': creation_time
#         }
#         if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
#         return quantum_channel_data, metrics_data

#     except Exception as e:
#         logger.error(f"Error establishing creator connection: {e}", exc_info=True)
#         # Hard fail
#         raise RuntimeError("Creator connection establishment failed critically.") from e

# def form_resonance_patterns(soul_spark: SoulSpark,
#                             quantum_channel_data: Dict[str, Any],
#                             kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """ Forms resonance patterns based on Kether aspects and connection strength. """
#     logger.info(f"Forming resonance patterns for soul {soul_spark.spark_id}...")
#     # --- Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
#     if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data invalid.")
#     if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
#     channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
#     connection_strength = quantum_channel_data.get('connection_strength') # 0-1 factor
#     freq_signature = quantum_channel_data.get('frequency_signature')
#     if not channel_id or not spark_id or connection_strength is None or not isinstance(freq_signature, dict):
#         raise ValueError("Quantum channel data missing essential keys or invalid freq_signature.")

#     try:
#         kether_aspects = kether_field.get_aspects()
#         if not kether_aspects: raise ValueError("Kether field aspect data is empty.")
#         detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
#         if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

#         patterns = {}
#         # Add primary/secondary aspects from Kether data, strength scaled by connection_strength
#         logger.debug("  Forming patterns from Kether primary/secondary aspects:")
#         for name in kether_aspects.get('primary_aspects', []):
#              if name in detailed_kether_aspects:
#                   details = detailed_kether_aspects[name]
#                   pat_strength = float(details.get('strength', 0.0) * connection_strength)
#                   patterns[name] = {
#                       'frequency': details.get('frequency', KETHER_FREQ),
#                       'strength': pat_strength, 'pattern_type': 'primary', **details
#                   }
#                   logger.debug(f"    Added Primary Pattern: {name} (Strength: {pat_strength:.4f})")
#         for name in kether_aspects.get('secondary_aspects', []):
#              if name in detailed_kether_aspects:
#                   details = detailed_kether_aspects[name]
#                   pat_strength = float(details.get('strength', 0.0) * connection_strength * 0.7)
#                   patterns[f"secondary_{name}"] = {
#                       'frequency': details.get('frequency', KETHER_FREQ),
#                       'strength': pat_strength, 'pattern_type': 'secondary', **details
#                   }
#                   logger.debug(f"    Added Secondary Pattern: {name} (Strength: {pat_strength:.4f})")

#         # Add harmonic patterns based on calculated resonance points
#         logger.debug("  Forming patterns from calculated resonance points:")
#         resonance_points_hz = freq_signature.get('resonance_points_hz', [])
#         if not resonance_points_hz: logger.warning("  No resonance points found in frequency signature!")
#         for i, point_hz in enumerate(resonance_points_hz):
#              if point_hz <= FLOAT_EPSILON: continue
#              # Strength scales with connection strength
#              strength = float(0.6 * connection_strength)
#              patterns[f'harmonic_{i}'] = {
#                  'frequency': float(point_hz), 'strength': strength,
#                  'pattern_type': 'harmonic', 'description': f'Resonance at {point_hz:.2f} Hz'
#              }
#              logger.debug(f"    Added Harmonic Pattern: harmonic_{i} (Freq: {point_hz:.1f}, Strength: {strength:.4f})")

#         # Calculate pattern coherence (average strength of formed patterns, 0-1 scale)
#         pattern_coherence = 0.0
#         if patterns:
#             valid_strengths = [p.get('strength', 0.0) for p in patterns.values() if isinstance(p.get('strength'), (int, float))]
#             if valid_strengths: pattern_coherence = sum(valid_strengths) / len(valid_strengths)
#         logger.debug(f"  Calculated Pattern Coherence: {pattern_coherence:.4f}")

#         formation_time = datetime.now().isoformat()
#         resonance_pattern_data = {
#             'channel_id': channel_id, 'spark_id': spark_id, 'patterns': patterns,
#             'pattern_count': len(patterns), 'pattern_coherence': float(pattern_coherence),
#             'formation_time': formation_time
#         }

#         # Update SoulSpark
#         initial_resonance_factor = soul_spark.resonance # 0-1 factor
#         soul_spark.resonance_patterns = patterns.copy() # Store formed patterns
#         soul_spark.pattern_coherence = float(pattern_coherence) # Update 0-1 factor
#         # Boost general resonance factor based on pattern coherence
#         resonance_boost = ENTANGLEMENT_RESONANCE_BOOST_FACTOR * pattern_coherence
#         soul_spark.resonance = float(min(1.0, initial_resonance_factor + resonance_boost))
#         soul_spark.last_modified = formation_time
#         logger.info(f"Patterns formed: {len(patterns)}. New Resonance Factor: {soul_spark.resonance:.4f} (+{resonance_boost:.4f}), P.Coh: {pattern_coherence:.4f}")

#         # Record Metrics
#         metrics_data = {
#             'action': 'form_patterns', 'soul_id': spark_id, 'channel_id': channel_id,
#             'pattern_count': len(patterns), 'pattern_coherence': pattern_coherence,
#             'final_soul_resonance_factor': soul_spark.resonance,
#             'resonance_factor_change': soul_spark.resonance - initial_resonance_factor,
#             'success': True, 'timestamp': formation_time
#         }
#         if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
#         return resonance_pattern_data, metrics_data

#     except Exception as e:
#         logger.error(f"Error forming resonance patterns: {e}", exc_info=True)
#         # Hard fail
#         raise RuntimeError("Pattern formation failed critically.") from e


# def transfer_creator_aspects(soul_spark: SoulSpark,
#                              quantum_channel_data: Dict[str, Any],
#                              resonance_pattern_data: Dict[str, Any],
#                              kether_field: KetherField) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """ Transfers Kether aspects scaled by connection quality/pattern resonance. No threshold. """
#     logger.info(f"Transferring creator aspects to soul {soul_spark.spark_id}...")
#     # --- Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
#     if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data invalid.")
#     if not isinstance(resonance_pattern_data, dict): raise TypeError("resonance_pattern_data invalid.")
#     if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
#     channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
#     connection_quality = quantum_channel_data.get('connection_quality') # 0-1 factor
#     patterns = resonance_pattern_data.get('patterns')
#     if not channel_id or not spark_id or connection_quality is None or patterns is None:
#         raise ValueError("Channel/pattern data missing essential keys.")
#     if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict):
#         setattr(soul_spark, 'aspects', {}) # Initialize if missing

#     try:
#         kether_aspects = kether_field.get_aspects()
#         if not kether_aspects: raise ValueError("Kether field aspect data is empty.")
#         detailed_kether_aspects = kether_aspects.get('detailed_aspects', {})
#         if not detailed_kether_aspects: raise ValueError("Kether detailed aspects missing.")

#         transferable_aspects = {} # Tracks potential transfers {name: {details}}
#         gained_count = 0; strengthened_count = 0
#         transfer_time = datetime.now().isoformat()
#         total_original_strength = 0.0; total_imparted_strength = 0.0

#         logger.debug("  Calculating aspect transfer efficiencies:")
#         for name, kether_details in detailed_kether_aspects.items():
#             kether_strength = float(kether_details.get('strength', 0.0)) # 0-1 factor
#             if kether_strength <= FLOAT_EPSILON: continue
#             total_original_strength += kether_strength

#             # Efficiency depends on connection quality AND resonance pattern strength for this aspect
#             pattern_strength = patterns.get(name, {}).get('strength', 0.0) # Check primary pattern
#             if pattern_strength <= FLOAT_EPSILON: pattern_strength = patterns.get(f"secondary_{name}", {}).get('strength', 0.0)
#             logger.debug(f"    Aspect '{name}': KetherStr={kether_strength:.3f}, ConnQual={connection_quality:.3f}, PatternStr={pattern_strength:.3f}")

#             efficiency = (connection_quality * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_BASE +
#                           pattern_strength * ENTANGLEMENT_ASPECT_TRANSFER_EFFICIENCY_RESONANCE_SCALE)
#             efficiency = min(1.0, max(0.0, efficiency))
#             imparted_strength = kether_strength * efficiency # 0-1 strength scale
#             total_imparted_strength += imparted_strength
#             logger.debug(f"      -> Efficiency={efficiency:.4f}, ImpartedStr={imparted_strength:.4f}")

#             # --- Record Potential Transfer ---
#             transfer_data = {
#                 'original_strength': kether_strength, 'transfer_efficiency': efficiency,
#                 'imparted_strength': imparted_strength, 'aspect_type': 'Kether',
#                 'details': kether_details # Store Kether's details
#             }
#             transferable_aspects[name] = transfer_data

#             # --- Update SoulSpark Aspects (No Threshold) ---
#             current_soul_strength = float(soul_spark.aspects.get(name, {}).get('strength', 0.0))
#             # Blend/Add based on imparted strength - More aggressive addition
#             new_strength = min(MAX_ASPECT_STRENGTH, current_soul_strength + imparted_strength)
#             actual_gain = new_strength - current_soul_strength
#             logger.debug(f"      -> Soul Aspect '{name}': CurrentStr={current_soul_strength:.4f}, NewStr={new_strength:.4f}, Gain={actual_gain:.4f}")

#             if actual_gain > FLOAT_EPSILON:
#                 details_copy = kether_details.copy()
#                 details_copy['last_transfer_efficiency'] = efficiency
#                 details_copy['last_imparted_strength'] = imparted_strength

#                 if name not in soul_spark.aspects or current_soul_strength <= FLOAT_EPSILON:
#                     soul_spark.aspects[name] = {'strength': new_strength, 'source': 'Kether', 'time_acquired': transfer_time, 'details': details_copy }
#                     gained_count += 1
#                     if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Gained Kether aspect '{name}' Strength: {new_strength:.3f}")
#                 else:
#                     soul_spark.aspects[name]['strength'] = new_strength
#                     soul_spark.aspects[name]['last_updated'] = transfer_time
#                     soul_spark.aspects[name]['details'].update(details_copy) # Update details
#                     strengthened_count += 1

#         average_efficiency = (total_imparted_strength / total_original_strength) if total_original_strength > FLOAT_EPSILON else 0.0
#         logger.debug(f"  Average Transfer Efficiency: {average_efficiency:.4f}")

#         # --- No Direct Stability Boost ---
#         initial_stability_su = soul_spark.stability # Store for metrics reporting
#         stability_change_su = 0.0 # No direct change applied here
#         logger.debug(f"  Direct stability boost REMOVED. Current SU: {initial_stability_su:.1f}")

#         # Update alignment factor (Keep this factor boost)
#         initial_alignment = soul_spark.creator_alignment
#         # Slightly increase alignment boost as compensation for removing direct stability boost
#         alignment_boost_factor = 0.20 # Increased from 0.15
#         alignment_gain = alignment_boost_factor * average_efficiency
#         soul_spark.creator_alignment = min(1.0, initial_alignment + alignment_gain)
#         logger.debug(f"  Creator Alignment Updated: {initial_alignment:.4f} -> {soul_spark.creator_alignment:.4f} (+{alignment_gain:.4f})")

#         soul_spark.last_modified = transfer_time
#         logger.info(f"Aspect transfer: Gained={gained_count}, Strengthened={strengthened_count}, AvgEff={average_efficiency:.3f}")

#         # Store Transfer Metrics
#         aspect_transfer_metrics = { # Summary of what happened this step
#             'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
#             'average_efficiency': average_efficiency, 'transfers': transferable_aspects }
#         # Record Core Metrics
#         metrics_data = { # Core metrics for tracking
#             'action': 'transfer_aspects', 'soul_id': spark_id, 'channel_id': channel_id,
#             'aspects_gained_count': gained_count, 'aspects_strengthened_count': strengthened_count,
#             'avg_efficiency': average_efficiency,
#             'initial_soul_alignment': initial_alignment, # Added initial for context
#             'final_soul_alignment': soul_spark.creator_alignment,
#             'initial_soul_stability_su': initial_stability_su, # Report initial SU for context
#             'final_soul_stability_su': soul_spark.stability, # Report current SU (before update_state)
#             'stability_change_su': stability_change_su, # Report 0 direct change
#             'success': True, 'timestamp': transfer_time }
#         if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
#         return aspect_transfer_metrics, metrics_data

#     except Exception as e:
#         logger.error(f"Error transferring creator aspects: {e}", exc_info=True)
#         # Hard fail
#         raise RuntimeError("Aspect transfer failed critically.") from e

# def stabilize_creator_connection(soul_spark: SoulSpark,
#                                  quantum_channel_data: Dict[str, Any],
#                                  resonance_pattern_data: Dict[str, Any],
#                                  iterations: int = ENTANGLEMENT_STABILIZATION_ITERATIONS) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """ Stabilizes connection, applying factor improvements. S/C emerges via update_state. """
#     logger.info(f"Stabilizing creator connection for soul {soul_spark.spark_id} ({iterations} iterations)...")
#     # --- Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
#     if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data invalid.")
#     if not isinstance(resonance_pattern_data, dict): raise TypeError("resonance_pattern_data invalid.")
#     if not isinstance(iterations, int) or iterations <= 0: raise ValueError("Iterations must be positive.")
#     channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
#     if not channel_id or not spark_id : raise ValueError("Channel data missing keys.")

#     try:
#         # Store Initial State (Factors and current S/C for logging comparison)
#         initial_metrics = {
#             'connection_strength': soul_spark.creator_connection_strength, # 0-1
#             'stability_su': soul_spark.stability, # SU
#             'coherence_cu': soul_spark.coherence, # CU
#             'creator_alignment': soul_spark.creator_alignment, # 0-1
#             'resonance_factor': soul_spark.resonance, # 0-1
#             'pattern_coherence': soul_spark.pattern_coherence # 0-1
#         }
#         current_strength = initial_metrics['connection_strength']
#         current_pattern_coherence = initial_metrics['pattern_coherence']
#         current_alignment = initial_metrics['creator_alignment']
#         current_patterns = soul_spark.resonance_patterns.copy() # Get current patterns

#         logger.info(f"Initial state for stabilization: Str={current_strength:.4f}, Stab={initial_metrics['stability_su']:.1f}SU, Coh={initial_metrics['coherence_cu']:.1f}CU, P.Coh={current_pattern_coherence:.3f}, Align={current_alignment:.3f}")

#         # Stabilization Iterations
#         for i in range(iterations):
#             iter_start_time = datetime.now().isoformat()
#             # Improvement factors (based on current state and iteration)
#             # Use multiplier > 1 for strength increase
#             strength_multiplier = ENTANGLEMENT_STABILIZATION_FACTOR_STRENGTH # e.g., 1.005
#             strength_gain = current_strength * (strength_multiplier - 1.0) * (1.0 / (i + 1)) # Gain diminishes
#             current_strength = min(1.0, current_strength + strength_gain)

#             # Strengthen patterns and recalculate pattern coherence factor
#             if current_patterns:
#                 pattern_strengths = []
#                 # Small additive boost, diminishing over iterations
#                 pattern_boost_amount = 0.005 * (1.0 / (i + 1))

#                 for name in current_patterns:
#                     pat_str = current_patterns[name].get('strength', 0.0)
#                     current_patterns[name]['strength'] = min(1.0, pat_str + pattern_boost_amount)
#                     pattern_strengths.append(current_patterns[name]['strength'])
#                 if pattern_strengths: current_pattern_coherence = sum(pattern_strengths) / len(pattern_strengths)

#             # --- Update SoulSpark State (Factors Only) ---
#             soul_spark.creator_connection_strength = float(current_strength)
#             soul_spark.pattern_coherence = float(current_pattern_coherence)
#             soul_spark.resonance_patterns = current_patterns.copy()

#             # --- REMOVED Direct S/C Boosts ---

#             # Boost alignment factor slightly too, based on strength gain
#             alignment_gain = strength_gain * 0.05 # Small boost relative to strength gain
#             current_alignment = min(1.0, current_alignment + alignment_gain)
#             soul_spark.creator_alignment = float(current_alignment)

#             soul_spark.last_modified = iter_start_time
#             # Log factor changes, S/C will be updated later by update_state
#             logger.debug(f"  Stabilization iter {i+1}: Str={current_strength:.4f}(+{strength_gain:.5f}), P.Coh={current_pattern_coherence:.4f}, Align={current_alignment:.4f}(+{alignment_gain:.5f})")

#         # Store Final Metrics (Factors Only)
#         final_factors = {
#             'connection_strength': float(current_strength),
#             'creator_alignment': float(current_alignment),
#             'resonance_factor': soul_spark.resonance, # May have changed if patterns changed
#             'pattern_coherence': float(current_pattern_coherence)
#         }
#         # Report current S/C before update_state for peak info in summary
#         final_stabilization_metrics_before_update = {
#              'stability_su': soul_spark.stability,
#              'coherence_cu': soul_spark.coherence
#         }

#         stabilization_result = { # Summary of the process
#             'iterations': iterations,
#             'initial_metrics': initial_metrics, # Factors + initial S/C
#             'final_factors': final_factors, # Factors only
#             'final_metrics_before_update': final_stabilization_metrics_before_update, # S/C before recalc
#             'stabilization_time': getattr(soul_spark, 'last_modified')
#         }

#         logger.info(f"Creator connection stabilization complete (Factors updated). Final Str={current_strength:.4f}")
#         if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Creator connection factors stabilized. Strength:{current_strength:.3f}")

#         # Record Core Metrics (Record factors, S/C will be recorded in summary after update_state)
#         metrics_data = {
#             'action': 'stabilize_connection', 'soul_id': spark_id, 'channel_id': channel_id, 'iterations': iterations,
#             'initial_strength': initial_metrics['connection_strength'],
#             'final_strength': final_factors['connection_strength'],
#             'initial_pattern_coherence': initial_metrics['pattern_coherence'],
#             'final_pattern_coherence': final_factors['pattern_coherence'],
#             'initial_alignment': initial_metrics['creator_alignment'],
#             'final_alignment': final_factors['creator_alignment'],
#             'success': True, 'timestamp': stabilization_result['stabilization_time']
#         }
#         if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement', metrics_data)
#         return stabilization_result, metrics_data

#     except Exception as e:
#         logger.error(f"Error stabilizing creator connection: {e}", exc_info=True)
#         # Hard fail
#         raise RuntimeError("Stabilization failed critically.") from e

# # --- Orchestration Function ---
# def run_full_entanglement_process(soul_spark: SoulSpark,
#                                   kether_field: KetherField,
#                                   creator_resonance: float = 0.8, # Base potential factor 0-1
#                                   edge_of_chaos_ratio: float = EDGE_OF_CHAOS_RATIO, # Target 0-1 factor
#                                   stabilization_iterations: int = ENTANGLEMENT_STABILIZATION_ITERATIONS
#                                   ) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """ Runs complete entanglement process. Uses SU/CU prerequisites. Modifies SoulSpark. """
#     start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
#     # --- Input Validation ---
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
#     if not isinstance(kether_field, KetherField): raise TypeError("kether_field invalid.")
#     if not (0.0 <= creator_resonance <= 1.0): raise ValueError("creator_resonance invalid.")
#     if not (0.0 <= edge_of_chaos_ratio <= 1.0): raise ValueError("edge_of_chaos_ratio invalid.")
#     if not isinstance(stabilization_iterations, int) or stabilization_iterations < 0: raise ValueError("stabilization_iterations invalid.")

#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     process_metrics_summary = {'steps': {}}

#     logger.info(f"--- Starting Full Entanglement Process for Soul {spark_id} ---")

#     try:
#         # --- Prerequisites Check ---
#         if not _check_prerequisites(soul_spark, kether_field):
#             # Raise error if prerequisites fail - Hard Fail
#             raise ValueError("Soul prerequisites for Creator Entanglement not met.")

#         # Store Initial State
#         initial_state = { # Record state in correct units
#              'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
#              'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
#              'aspect_count': len(soul_spark.aspects) }
#         logger.debug(f"  Entanglement Initial State: {initial_state}")

#         # --- Run Stages ---
#         logger.info("Step 1: Establishing Connection...")
#         channel_data, metrics1 = establish_creator_connection(soul_spark, kether_field, creator_resonance, edge_of_chaos_ratio)
#         process_metrics_summary['steps']['connection'] = metrics1

#         logger.info("Step 2: Forming Resonance Patterns...")
#         pattern_data, metrics2 = form_resonance_patterns(soul_spark, channel_data, kether_field)
#         process_metrics_summary['steps']['patterns'] = metrics2

#         logger.info("Step 3: Transferring Creator Aspects...")
#         transfer_metrics, metrics3 = transfer_creator_aspects(soul_spark, channel_data, pattern_data, kether_field)
#         process_metrics_summary['steps']['transfer'] = metrics3

#         logger.info("Step 4: Stabilizing Connection...")
#         stabilization_result, metrics4 = stabilize_creator_connection(soul_spark, channel_data, pattern_data, stabilization_iterations)
#         process_metrics_summary['steps']['stabilization'] = metrics4

#         # --- Final State Update ---
#         # Capture S/C *before* final update_state for peak reporting
#         peak_stability_su = stabilization_result['final_metrics_before_update']['stability_su']
#         peak_coherence_cu = stabilization_result['final_metrics_before_update']['coherence_cu']
#         logger.debug(f"  Before final update_state in entanglement: S={peak_stability_su:.1f}, C={peak_coherence_cu:.1f}")

#         setattr(soul_spark, FLAG_READY_FOR_COMPLETION, True)
#         setattr(soul_spark, FLAG_READY_FOR_STRENGTHENING, True)
#         last_mod_time = datetime.now().isoformat()
#         setattr(soul_spark, 'last_modified', last_mod_time)

#         if hasattr(soul_spark, 'update_state'):
#             soul_spark.update_state()
#             logger.debug(f"  After final update_state in entanglement: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
#         else:
#              logger.error("SoulSpark object missing 'update_state' method!")
#              # Hard fail - critical method missing
#              raise AttributeError("SoulSpark needs 'update_state' method for entanglement.")

#         # Compile Overall Metrics
#         end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
#         final_state = { # Report state in correct units after update_state
#              'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
#              'resonance_factor': soul_spark.resonance, 'creator_alignment': soul_spark.creator_alignment,
#              'aspect_count': len(soul_spark.aspects) }
#         overall_metrics = {
#             'action': 'full_entanglement', 'soul_id': spark_id,
#             'start_time': start_time_iso, 'end_time': end_time_iso,
#             'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
#             'initial_state': initial_state,
#             'peak_stability_su_during_stabilization': float(peak_stability_su), # ADDED PEAK
#             'peak_coherence_cu_during_stabilization': float(peak_coherence_cu), # ADDED PEAK
#             'final_state': final_state, # State AFTER update_state
#             'final_connection_strength': stabilization_result['final_factors']['connection_strength'], # Get factor
#             'aspects_transferred_count': transfer_metrics.get('aspects_gained_count', 0) + transfer_metrics.get('aspects_strengthened_count', 0),
#             'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
#             'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
#             'success': True,
#         }
#         if METRICS_AVAILABLE: metrics.record_metrics('creator_entanglement_summary', overall_metrics)

#         logger.info(f"--- Full Entanglement Completed Successfully for Soul {spark_id} ---")
#         return soul_spark, overall_metrics

#     # --- Error Handling (Ensure Hard Fail) ---
#     except (ValueError, TypeError, AttributeError) as e_val:
#         logger.error(f"Entanglement failed for soul {spark_id} due to validation/attribute error: {e_val}", exc_info=True)
#         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'prerequisites'
#         record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_val))
#         # Hard fail - re-raise critical setup errors
#         raise ValueError(f"Entanglement validation failed: {e_val}") from e_val
#     except RuntimeError as e_rt:
#         logger.critical(f"Entanglement failed critically for soul {spark_id}: {e_rt}", exc_info=True)
#         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'runtime'
#         record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e_rt))
#         # Hard fail - re-raise critical runtime errors
#         raise e_rt
#     except Exception as e:
#         logger.critical(f"Unexpected error during entanglement for {spark_id}: {e}", exc_info=True)
#         failed_step = list(process_metrics_summary['steps'].keys())[-1] if process_metrics_summary['steps'] else 'unexpected'
#         record_entanglement_failure(spark_id, start_time_iso, failed_step, str(e))
#         # Hard fail - wrap unexpected errors
#         raise RuntimeError(f"Unexpected entanglement failure: {e}") from e

# # --- Failure Metric Helper ---
# def record_entanglement_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
#     """ Helper to record failure metrics consistently. """
#     if METRICS_AVAILABLE:
#         try:
#             end_time = datetime.now().isoformat()
#             duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
#             metrics.record_metrics('creator_entanglement_summary', {
#                 'action': 'full_entanglement', 'soul_id': spark_id,
#                 'start_time': start_time_iso, 'end_time': end_time,
#                 'duration_seconds': duration,
#                 'success': False, 'error': error_msg, 'failed_step': failed_step
#             })
#         except Exception as metric_e:
#             logger.error(f"Failed to record entanglement failure metrics for {spark_id}: {metric_e}")

# # --- END OF FILE src/stage_1/soul_formation/creator_entanglement.py ---




