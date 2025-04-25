# --- START OF FILE creator_entanglement.py ---

"""
Creator Entanglement Functions (Refactored - Operates on SoulSpark Object)

Provides functions for establishing and managing the quantum connection between
a soul spark object and the creator (Kether aspects). Enables resonance matching,
aspect transfer, and stabilization by directly modifying the SoulSpark instance.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import uuid
import json
from typing import Dict, List, Any, Tuple, Optional
from src.constants.constants import *

# Extract specific Earth freq
EARTH_BREATH_FREQUENCY = EARTH_FREQUENCIES.get("breath", 0.2)  # Default ~12 breaths/min

# --- Logging ---
# Logging should be configured by the main application entry point.
# Using __name__ ensures logs from this module are identifiable.
logger = logging.getLogger(__name__)


try:
    from src.stage_1.soul_formation.soul_spark import SoulSpark
    from stage_1.sephiroth.kether_aspects import KetherAspects, AspectType
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    DEPENDENCIES_AVAILABLE = True
    if aspect_dictionary is None:
        raise ImportError("Aspect Dictionary failed to initialize.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import core dependencies (SoulSpark, KetherAspects, aspect_dictionary): {e}")
    raise ImportError(f"Core dependencies missing: {e}") from e

try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import metrics_tracking: {e}. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    # Define dummy metrics functions if needed for code structure to work
    class MetricsPlaceholder:
        def record_metrics(*args, **kwargs): pass
    metrics = MetricsPlaceholder()


# --- Helper Functions ---

def _get_kether_aspects_instance() -> KetherAspects:
    """Gets a KetherAspects instance, failing hard if unavailable."""
    if not DEPENDENCIES_AVAILABLE or aspect_dictionary is None:
        # aspect_dictionary check is redundant given the top-level check, but safe.
        raise RuntimeError("Cannot get Kether aspects: Core dependencies unavailable.")
    try:
        # Assuming aspect_dictionary is the singleton instance and preloaded/validated
        instance = aspect_dictionary.load_aspect_instance("kether")
        if not isinstance(instance, KetherAspects):
             raise TypeError("Loaded instance is not of type KetherAspects.")
        return instance
    except (ValueError, AttributeError, RuntimeError, TypeError) as e:
        logger.critical(f"CRITICAL: Failed to get KetherAspects instance: {e}", exc_info=True)
        raise RuntimeError("Could not load Kether aspects.") from e

def _calculate_frequency_signature(soul_spark: SoulSpark, kether_aspects: KetherAspects) -> Dict[str, Any]:
    """
    Calculates the frequency signature. Modifies soul_spark.frequency_signature. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        kether_aspects (KetherAspects): Instance of Kether aspects.

    Returns:
        Dict[str, Any]: The calculated frequency signature dictionary.

    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If soul spark is missing valid frequency information.
        RuntimeError: If resonance point calculation fails.
    """
    logger.debug(f"Calculating frequency signature for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(kether_aspects, KetherAspects): raise TypeError("kether_aspects must be a KetherAspects instance.")

    try:
        # --- Get Frequencies ---
        creator_freqs_dict = kether_aspects.get_frequencies()
        creator_freqs_list = list(creator_freqs_dict.values()) if isinstance(creator_freqs_dict, dict) else []

        soul_freqs_list = []
        fsig_attr = getattr(soul_spark, 'frequency_signature', None)
        if isinstance(fsig_attr, dict) and 'frequencies' in fsig_attr and isinstance(fsig_attr['frequencies'], list):
            soul_freqs_list = [f for f in fsig_attr['frequencies'] if isinstance(f, (int, float)) and f > FLOAT_EPSILON]
        # If no valid list, check for single frequency attribute
        if not soul_freqs_list:
            base_freq_attr = getattr(soul_spark, 'frequency', None)
            if isinstance(base_freq_attr, (int, float)) and base_freq_attr > FLOAT_EPSILON:
                soul_freqs_list = [base_freq_attr]
                logger.debug(f"Using soul_spark.frequency ({base_freq_attr}Hz) as base for signature.")
            else: # No valid frequency source found
                raise ValueError(f"SoulSpark {soul_spark.spark_id} missing valid 'frequency' or 'frequency_signature.frequencies'.")

        soul_freqs_dict = {f'soul_{i}': freq for i, freq in enumerate(soul_freqs_list)}
        logger.debug(f"Soul frequencies identified: {soul_freqs_list}")
        logger.debug(f"Creator frequencies identified: {creator_freqs_list}")

        # --- Combine & Calculate Resonance ---
        combined_freqs = {}
        creator_resonance = getattr(soul_spark, 'creator_resonance', 0.7) # Use current soul's resonance
        for name, freq in creator_freqs_dict.items():
            combined_freqs[f'creator_{name}'] = {'frequency': freq, 'weight': 0.7 * creator_resonance}
        for name, freq in soul_freqs_dict.items():
            combined_freqs[name] = {'frequency': freq, 'weight': 0.3}

        resonance_points = _find_resonance_points(creator_freqs_list, soul_freqs_list) # Can raise RuntimeError
        logger.debug(f"Resonance points calculated: {resonance_points}")

        # --- Create Signature & Update SoulSpark ---
        signature = {
            'creator_frequencies': creator_freqs_dict,
            'soul_frequencies': soul_freqs_dict,
            'combined_frequencies': combined_freqs,
            'resonance_points': resonance_points,
            'primary_resonance': max(resonance_points) if resonance_points else 0.0
        }
        # Store signature directly on the soul spark object
        setattr(soul_spark, 'frequency_signature', signature)
        logger.debug(f"Frequency signature updated on soul {soul_spark.spark_id}.")

        return signature
    except ValueError as ve:
         logger.error(f"Value error calculating frequency signature for {soul_spark.spark_id}: {ve}")
         raise # Re-raise specific error
    except Exception as e:
        logger.error(f"Unexpected error calculating frequency signature for soul {soul_spark.spark_id}: {e}", exc_info=True)
        raise RuntimeError("Frequency signature calculation failed.") from e

def _find_resonance_points(creator_freqs: List[float], soul_freqs: List[float]) -> List[float]:
    """Find frequency points where creator and soul resonance aligns. Fails hard."""
    # (Implementation from previous step - unchanged, includes error handling)
    if not creator_freqs or not soul_freqs: return []
    resonance_points_set = set()
    try:
        for c_freq in creator_freqs:
            if c_freq <= FLOAT_EPSILON: continue
            for s_freq in soul_freqs:
                if s_freq <= FLOAT_EPSILON: continue
                if abs(c_freq - s_freq) < FLOAT_EPSILON * c_freq: resonance_points_set.add(c_freq); continue
                ratio = c_freq / s_freq; inverse_ratio = s_freq / c_freq
                for i in range(1, 6):
                    if abs(ratio - i) < 0.05 or abs(inverse_ratio - i) < 0.05: resonance_points_set.add(min(c_freq, s_freq)); break
                if abs(ratio - GOLDEN_RATIO) < 0.05 or abs(inverse_ratio - GOLDEN_RATIO) < 0.05: resonance_points_set.add(min(c_freq, s_freq) * GOLDEN_RATIO)
        return sorted(list(resonance_points_set))
    except Exception as e: logger.error(f"Error finding resonance points: {e}", exc_info=True); raise RuntimeError("Resonance point calculation failed.") from e

# --- Core Entanglement Functions ---

def establish_creator_connection(soul_spark: SoulSpark,
                                 kether_aspects: KetherAspects,
                                 creator_resonance: float,
                                 edge_of_chaos_ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Establishes the initial quantum connection. Modifies SoulSpark object. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        kether_aspects (KetherAspects): Instance of Kether aspects.
        creator_resonance (float): Base creator resonance strength (0-1).
        edge_of_chaos_ratio (float): Ideal ratio for entanglement (0-1).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - quantum_channel_data (Dict): Information about the established channel.
            - metrics_data (Dict): Metrics recorded for this action.

    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If input values are invalid or soul lacks required attributes.
        RuntimeError: If frequency signature calculation fails.
    """
    logger.info(f"Establishing creator quantum connection for soul {soul_spark.spark_id}...")
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(kether_aspects, KetherAspects): raise TypeError("kether_aspects must be a KetherAspects instance.")
    if not isinstance(creator_resonance, (int, float)) or not (0.0 <= creator_resonance <= 1.0): raise ValueError("creator_resonance out of range.")
    if not isinstance(edge_of_chaos_ratio, (int, float)) or not (0.0 <= edge_of_chaos_ratio <= 1.0): raise ValueError("edge_of_chaos_ratio out of range.")

    # --- Calculate Connection Properties ---
    stability = getattr(soul_spark, 'stability', 0.7); resonance = getattr(soul_spark, 'resonance', 0.7)
    if not (0.0 <= stability <= 1.0): raise ValueError(f"Invalid soul stability: {stability}")
    if not (0.0 <= resonance <= 1.0): raise ValueError(f"Invalid soul resonance: {resonance}")
    logger.debug(f"Soul properties for connection: Stability={stability:.4f}, Resonance={resonance:.4f}")

    connection_quality = (0.4 * stability + 0.4 * resonance + 0.2 * creator_resonance)
    chaos_factor = 1.0 - abs(connection_quality - edge_of_chaos_ratio) / max(FLOAT_EPSILON, edge_of_chaos_ratio)
    effective_strength = creator_resonance * max(0.0, chaos_factor)
    logger.debug(f"Calculated connection: Quality={connection_quality:.4f}, ChaosFactor={chaos_factor:.4f}, EffectiveStrength={effective_strength:.4f}")

    # --- Calculate Frequency Signature (raises RuntimeError on failure, updates soul_spark) ---
    freq_signature = _calculate_frequency_signature(soul_spark, kether_aspects)

    # --- Create Channel Data ---
    channel_id = str(uuid.uuid4())
    quantum_channel_data = {
        'channel_id': channel_id, 'spark_id': soul_spark.spark_id, 'creation_time': datetime.now().isoformat(),
        'connection_strength': float(effective_strength), 'connection_quality': float(connection_quality),
        'initial_soul_stability': float(stability), 'initial_soul_resonance': float(resonance),
        'chaos_factor': float(chaos_factor), 'active': True, 'frequency_signature': freq_signature
    }

    # --- Update SoulSpark Object ---
    logger.info(f"Updating SoulSpark {soul_spark.spark_id} with connection data...")
    initial_alignment = getattr(soul_spark, 'creator_alignment', 0.0)
    setattr(soul_spark, 'creator_channel_id', channel_id)
    setattr(soul_spark, 'creator_connection_strength', float(effective_strength))
    setattr(soul_spark, 'creator_alignment', float(min(1.0, initial_alignment + 0.1 * effective_strength)))
    setattr(soul_spark, 'last_modified', quantum_channel_data['creation_time'])
    logger.info(f"SoulSpark updated: Channel={channel_id}, Strength={effective_strength:.4f}, Alignment={soul_spark.creator_alignment:.4f} (from {initial_alignment:.4f})")

    # --- Record Metrics ---
    metrics_data = {
        'action': 'establish_connection', 'soul_id': soul_spark.spark_id, 'channel_id': channel_id,
        'strength': effective_strength, 'quality': connection_quality, 'chaos_factor': chaos_factor,
        'final_alignment': soul_spark.creator_alignment, 'alignment_change': soul_spark.creator_alignment - initial_alignment,
        'success': True, 'timestamp': quantum_channel_data['creation_time']
    }
    try: metrics.record_metrics('creator_entanglement', metrics_data)
    except Exception as e: logger.error(f"Failed to record metrics for connection: {e}")

    return quantum_channel_data, metrics_data


def form_resonance_patterns(soul_spark: SoulSpark,
                            quantum_channel_data: Dict[str, Any],
                            kether_aspects: KetherAspects) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Forms resonance patterns. Modifies SoulSpark object. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        quantum_channel_data (Dict[str, Any]): Data from establish_creator_connection.
        kether_aspects (KetherAspects): Instance of Kether aspects.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - resonance_pattern_data (Dict): Information about the formed patterns.
            - metrics_data (Dict): Metrics recorded for this action.

    Raises:
        TypeError, ValueError, AttributeError, RuntimeError as appropriate.
    """
    logger.info(f"Forming resonance patterns for soul {soul_spark.spark_id}...")
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data must be a dictionary.")
    if not isinstance(kether_aspects, KetherAspects): raise TypeError("kether_aspects must be a KetherAspects instance.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_strength = quantum_channel_data.get('connection_strength')
    freq_signature = quantum_channel_data.get('frequency_signature')
    if not channel_id or not spark_id or connection_strength is None or not freq_signature:
        raise ValueError("Quantum channel data missing essential keys.")
    if channel_id != getattr(soul_spark, 'creator_channel_id', None):
         logger.warning(f"Channel ID mismatch during pattern formation.")

    try:
        # --- Design Resonance Patterns ---
        if not hasattr(kether_aspects, 'get_primary_aspects') or not hasattr(kether_aspects, 'get_creator_aspects'):
             raise AttributeError("KetherAspects instance missing required methods.")
        primary_aspects = kether_aspects.get_primary_aspects(); creator_aspects = kether_aspects.get_creator_aspects()
        patterns = {}

        # Add primary aspects
        for name, aspect in primary_aspects.items():
            patterns[name] = {
                'frequency': aspect.get('frequency', KETHER_FREQUENCY), # Use Kether Freq as default
                'strength': float(aspect.get('strength', 0.0) * connection_strength),
                'pattern_type': 'primary', 'description': aspect.get('description', ''),
                'color': aspect.get('color', 'white'), 'element': aspect.get('element', 'aether'),
                'keywords': aspect.get('keywords', [])
            }
            logger.debug(f"  Pattern (Primary) '{name}': Strength={patterns[name]['strength']:.4f}")

        # Add creator aspects
        for name, aspect in creator_aspects.items():
            patterns[name] = {
                'frequency': aspect.get('frequency', KETHER_FREQUENCY),
                'strength': float(aspect.get('strength', 0.0) * connection_strength * 0.8), # Slightly lower weight
                'pattern_type': 'creator', 'description': aspect.get('description', ''),
                'color': aspect.get('color', 'white'), 'element': aspect.get('element', 'aether'),
                'keywords': aspect.get('keywords', [])
            }
            logger.debug(f"  Pattern (Creator) '{name}': Strength={patterns[name]['strength']:.4f}")

        # Add harmonic patterns
        resonance_points = freq_signature.get('resonance_points', [])
        logger.debug(f"  Adding patterns for {len(resonance_points)} resonance points: {resonance_points}")
        for i, point in enumerate(resonance_points):
            if point <= FLOAT_EPSILON: continue
            pattern_name = f'harmonic_{i}'
            strength = float(0.7 * connection_strength)
            patterns[pattern_name] = {
                'frequency': float(point), 'strength': strength, 'pattern_type': 'harmonic',
                'description': f'Harmonic resonance at {point:.2f} Hz', 'color': 'white',
                'element': 'harmony', 'keywords': ['harmonic', 'resonance', 'alignment']
            }
            logger.debug(f"  Pattern (Harmonic) '{pattern_name}': Freq={point:.2f} Hz, Strength={strength:.4f}")


        # --- Calculate Coherence ---
        pattern_coherence = 0.0
        if patterns:
            valid_strengths = [p.get('strength', 0.0) for p in patterns.values() if isinstance(p.get('strength'), (int, float))]
            if valid_strengths: pattern_coherence = sum(valid_strengths) / len(valid_strengths)

        # --- Store Pattern Data ---
        resonance_pattern_data = {
            'channel_id': channel_id, 'spark_id': spark_id, 'patterns': patterns,
            'pattern_count': len(patterns), 'pattern_coherence': float(pattern_coherence),
            'formation_time': datetime.now().isoformat()
        }

        # --- Update SoulSpark Object ---
        logger.info(f"Updating SoulSpark {soul_spark.spark_id} with {len(patterns)} resonance patterns...")
        initial_resonance = getattr(soul_spark, 'resonance', 0.7)
        setattr(soul_spark, 'resonance_patterns', patterns)
        setattr(soul_spark, 'pattern_coherence', float(pattern_coherence))
        setattr(soul_spark, 'resonance', float(min(1.0, initial_resonance + 0.05 * pattern_coherence)))
        setattr(soul_spark, 'last_modified', resonance_pattern_data['formation_time'])
        logger.info(f"SoulSpark updated. New Resonance: {soul_spark.resonance:.4f} (from {initial_resonance:.4f}), Coherence: {pattern_coherence:.4f}")

        # --- Record Metrics ---
        metrics_data = {
            'action': 'form_patterns', 'soul_id': spark_id, 'channel_id': channel_id,
            'pattern_count': len(patterns), 'coherence': pattern_coherence,
            'final_soul_resonance': soul_spark.resonance, 'resonance_change': soul_spark.resonance - initial_resonance,
            'success': True, 'timestamp': resonance_pattern_data['formation_time']
        }
        try: metrics.record_metrics('creator_entanglement', metrics_data)
        except Exception as e: logger.error(f"Failed to record metrics for pattern formation: {e}")

        return resonance_pattern_data, metrics_data

    except Exception as e:
        logger.error(f"Error forming resonance patterns for soul {spark_id}: {e}", exc_info=True)
        raise RuntimeError("Resonance pattern formation failed.") from e


def transfer_creator_aspects(soul_spark: SoulSpark,
                             quantum_channel_data: Dict[str, Any],
                             resonance_pattern_data: Dict[str, Any],
                             kether_aspects: KetherAspects,
                             aspect_types: Optional[List[AspectType]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Transfers creator aspects. Modifies SoulSpark object. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        quantum_channel_data (Dict[str, Any]): Data from establish_creator_connection.
        resonance_pattern_data (Dict[str, Any]): Data from form_resonance_patterns.
        kether_aspects (KetherAspects): Instance of Kether aspects.
        aspect_types (Optional[List[AspectType]]): Specific AspectType enums to transfer.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - aspect_transfer_metrics (Dict): Information about the transferred aspects.
            - metrics_data (Dict): Metrics recorded for this action.

    Raises:
        TypeError, ValueError, AttributeError, RuntimeError as appropriate.
    """
    logger.info(f"Transferring creator aspects to soul {soul_spark.spark_id}...")
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data must be a dictionary.")
    if not isinstance(resonance_pattern_data, dict): raise TypeError("resonance_pattern_data must be a dictionary.")
    if not isinstance(kether_aspects, KetherAspects): raise TypeError("kether_aspects must be a KetherAspects instance.")
    if not hasattr(soul_spark, 'aspects') or not isinstance(soul_spark.aspects, dict): raise AttributeError("SoulSpark requires a valid 'aspects' dictionary attribute.")

    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    connection_strength = quantum_channel_data.get('connection_strength')
    connection_quality = quantum_channel_data.get('connection_quality')
    patterns = resonance_pattern_data.get('patterns')
    if not channel_id or not spark_id or connection_strength is None or connection_quality is None or patterns is None: raise ValueError("Channel/pattern data missing keys.")
    if channel_id != getattr(soul_spark, 'creator_channel_id', None): logger.warning(f"Channel ID mismatch during aspect transfer.")

    if aspect_types is None: types_to_transfer = [t for t in AspectType if t != AspectType.DIMENSIONAL]
    elif isinstance(aspect_types, list) and all(isinstance(t, AspectType) for t in aspect_types): types_to_transfer = aspect_types
    else: raise ValueError("aspect_types must be a list of AspectType enums or None.")

    try:
        # --- Calculate Transfer ---
        transferable_aspects = {}; gained_aspect_names = []; strengthened_aspect_names = []
        if not hasattr(kether_aspects, 'get_aspects_by_type'): raise AttributeError("KetherAspects missing 'get_aspects_by_type'.")

        for aspect_type in types_to_transfer:
            aspects_of_type = kether_aspects.get_aspects_by_type(aspect_type)
            if not isinstance(aspects_of_type, dict): continue

            for name, aspect_details in aspects_of_type.items():
                if not isinstance(aspect_details, dict): continue

                original_strength = aspect_details.get('strength', 0.0)
                pattern_efficiency = patterns.get(name, {}).get('strength', 0.0)
                efficiency = (connection_quality * 0.6) + (pattern_efficiency * 0.4)
                transferred_strength = original_strength * max(0.0, efficiency)

                if transferred_strength > 0.05:
                    transfer_data = {
                        'original_strength': float(original_strength), 'transfer_efficiency': float(efficiency),
                        'transferred_strength': float(transferred_strength), 'aspect_type': aspect_type.name,
                        'frequency': aspect_details.get('frequency', KETHER_FREQUENCY), 'color': aspect_details.get('color', 'white'),
                        'element': aspect_details.get('element', 'aether'), 'description': aspect_details.get('description', ''),
                        'keywords': aspect_details.get('keywords', []), 'transfer_timestamp': datetime.now().isoformat()
                    }
                    transferable_aspects[name] = transfer_data

                    # --- Update SoulSpark Aspects ---
                    if name in soul_spark.aspects:
                        existing_data = soul_spark.aspects[name]
                        existing_strength = existing_data.get('strength', 0.0)
                        new_strength = float((existing_strength * 0.3) + (transfer_data['transferred_strength'] * 0.7))
                        soul_spark.aspects[name]['strength'] = min(1.0, new_strength)
                        soul_spark.aspects[name]['last_updated'] = transfer_data['transfer_timestamp']
                        logger.debug(f"  Strengthened aspect '{name}' to {new_strength:.4f}")
                        strengthened_aspect_names.append(name)
                    else:
                        soul_spark.aspects[name] = {
                            'strength': transfer_data['transferred_strength'],
                            'source': 'Kether',
                            'time_acquired': transfer_data['transfer_timestamp'],
                            'details': transfer_data # Store full details of transferred aspect
                        }
                        logger.info(f"  Gained aspect '{name}' with strength {transfer_data['transferred_strength']:.4f}")
                        # Log potential memory echo
                        logger.info(f"Memory echo created: Gained Kether aspect '{name}' via entanglement.")
                        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
                           soul_spark.memory_echoes.append(f"Gained Kether aspect '{name}' @ {transfer_data['transfer_timestamp']}")
                        gained_aspect_names.append(name)

        # --- Calculate Metrics ---
        total_original = sum(a['original_strength'] for a in transferable_aspects.values())
        total_transferred = sum(a['transferred_strength'] for a in transferable_aspects.values())
        average_efficiency = (total_transferred / total_original) if total_original > FLOAT_EPSILON else 0.0

        # --- Update SoulSpark Object Overall Stats ---
        initial_alignment = getattr(soul_spark, 'creator_alignment', 0.0)
        initial_stability = getattr(soul_spark, 'stability', 0.7)
        setattr(soul_spark, 'creator_alignment', float(min(1.0, initial_alignment + 0.1 * average_efficiency)))
        setattr(soul_spark, 'stability', float(min(1.0, initial_stability + 0.05 * average_efficiency)))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        logger.info(f"Aspect transfer complete for soul {spark_id}. Gained={len(gained_aspect_names)}, Strengthened={len(strengthened_aspect_names)}")
        logger.info(f"  Avg Efficiency: {average_efficiency:.4f}, New Alignment: {soul_spark.creator_alignment:.4f} (from {initial_alignment:.4f}), New Stability: {soul_spark.stability:.4f} (from {initial_stability:.4f})")

        # --- Store Transfer Metrics ---
        aspect_transfer_metrics = {
            'channel_id': channel_id, 'spark_id': spark_id,
            'aspects_gained_count': len(gained_aspect_names), 'aspects_strengthened_count': len(strengthened_aspect_names),
            'total_original_strength': float(total_original), 'total_transferred_strength': float(total_transferred),
            'average_efficiency': float(average_efficiency), 'transfer_time': getattr(soul_spark, 'last_modified'),
            'gained_aspect_names': gained_aspect_names, 'strengthened_aspect_names': strengthened_aspect_names
        }

        # --- Record Metrics ---
        metrics_data = {
            'action': 'transfer_aspects', 'soul_id': spark_id, 'channel_id': channel_id,
            'aspects_gained_count': len(gained_aspect_names), 'aspects_strengthened_count': len(strengthened_aspect_names),
            'avg_efficiency': average_efficiency, 'final_soul_alignment': soul_spark.creator_alignment,
            'final_soul_stability': soul_spark.stability, 'success': True, 'timestamp': aspect_transfer_metrics['transfer_time']
        }
        try: metrics.record_metrics('creator_entanglement', metrics_data)
        except Exception as e: logger.error(f"Failed to record metrics for aspect transfer: {e}")

        return aspect_transfer_metrics, metrics_data

    except Exception as e:
        logger.error(f"Error transferring creator aspects for soul {spark_id}: {e}", exc_info=True)
        raise RuntimeError("Creator aspect transfer failed.") from e


def stabilize_creator_connection(soul_spark: SoulSpark,
                                 quantum_channel_data: Dict[str, Any],
                                 resonance_pattern_data: Dict[str, Any],
                                 iterations: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Stabilizes the connection iteratively. Modifies SoulSpark object. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        quantum_channel_data (Dict[str, Any]): Data from establish_creator_connection.
        resonance_pattern_data (Dict[str, Any]): Data from form_resonance_patterns.
        iterations (int): Number of stabilization iterations (must be positive).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - stabilization_result (Dict): Information about the stabilization process.
            - metrics_data (Dict): Metrics recorded for this action.

    Raises:
        TypeError, ValueError, RuntimeError as appropriate.
    """
    logger.info(f"Stabilizing creator connection for soul {soul_spark.spark_id} ({iterations} iterations)...")
    # --- Input Validation & Setup ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(quantum_channel_data, dict): raise TypeError("quantum_channel_data must be a dictionary.")
    if not isinstance(resonance_pattern_data, dict): raise TypeError("resonance_pattern_data must be a dictionary.")
    if not isinstance(iterations, int) or iterations <= 0: raise ValueError("Iterations must be positive.")
    channel_id = quantum_channel_data.get('channel_id'); spark_id = quantum_channel_data.get('spark_id')
    patterns = resonance_pattern_data.get('patterns')
    if not channel_id or not spark_id or patterns is None: raise ValueError("Channel/pattern data missing keys.")
    if channel_id != getattr(soul_spark, 'creator_channel_id', None): logger.warning(f"Channel ID mismatch during stabilization.")


    try:
        # --- Store Initial State from SoulSpark ---
        initial_metrics = {
            'connection_strength': getattr(soul_spark, 'creator_connection_strength', 0.0),
            'connection_quality': quantum_channel_data.get('connection_quality', 0.0), # Quality is property of channel setup
            'stability': getattr(soul_spark, 'stability', 0.0),
            'creator_alignment': getattr(soul_spark, 'creator_alignment', 0.0),
            'resonance': getattr(soul_spark, 'resonance', 0.0),
            'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0) # Use coherence from soul
        }
        current_patterns = getattr(soul_spark, 'resonance_patterns', {}).copy() # Operate on a copy of soul's patterns

        logger.info(f"Initial state for stabilization: Str={initial_metrics['connection_strength']:.4f}, Qual={initial_metrics['connection_quality']:.4f}, Align={initial_metrics['creator_alignment']:.4f}, Stab={initial_metrics['stability']:.4f}, Res={initial_metrics['resonance']:.4f}, P.Coh={initial_metrics['pattern_coherence']:.4f}")

        # --- Perform Stabilization Iterations ---
        current_strength = initial_metrics['connection_strength']
        current_quality = initial_metrics['connection_quality'] # Quality likely doesn't change much here, but strength does
        current_pattern_coherence = initial_metrics['pattern_coherence']

        for i in range(iterations):
            iter_start_time = datetime.now().isoformat()
            # Factors for diminishing returns
            strength_increase_factor = 1.005 + 0.02 / (i + 2) # Start denom at 2
            quality_increase_factor = 1.002 + 0.01 / (i + 2)
            pattern_increase_factor = 1.003 + 0.01 / (i + 2)

            current_strength = min(1.0, current_strength * strength_increase_factor)
            current_quality = min(1.0, current_quality * quality_increase_factor) # Quality might slightly increase

            # Strengthen resonance patterns (modify the copied dictionary)
            if current_patterns:
                 pattern_strengths = []
                 for name in current_patterns:
                      current_patterns[name]['strength'] = min(1.0, current_patterns[name].get('strength', 0.0) * pattern_increase_factor)
                      pattern_strengths.append(current_patterns[name]['strength'])
                 if pattern_strengths: current_pattern_coherence = sum(pattern_strengths) / len(pattern_strengths)

            # Calculate boosts based on *changes* from initial state for this process
            alignment_boost = 0.01 * (current_strength - initial_metrics['connection_strength'])
            stability_boost = 0.01 * (current_quality - initial_metrics['connection_quality'])
            resonance_boost = 0.01 * (current_pattern_coherence - initial_metrics['pattern_coherence'])

            # --- Update SoulSpark Object Attributes ---
            setattr(soul_spark, 'creator_connection_strength', float(current_strength))
            setattr(soul_spark, 'creator_alignment', float(min(1.0, initial_metrics['creator_alignment'] + alignment_boost))) # Apply boost to original alignment
            setattr(soul_spark, 'stability', float(min(1.0, initial_metrics['stability'] + stability_boost)))
            setattr(soul_spark, 'resonance', float(min(1.0, initial_metrics['resonance'] + resonance_boost)))
            setattr(soul_spark, 'pattern_coherence', float(current_pattern_coherence))
            setattr(soul_spark, 'resonance_patterns', current_patterns.copy()) # Update soul with stabilized patterns
            setattr(soul_spark, 'last_modified', iter_start_time)

            logger.info(f"  Stabilization iter {i+1}: Str={current_strength:.4f}, Qual={current_quality:.4f}, Align={soul_spark.creator_alignment:.4f}, Stab={soul_spark.stability:.4f}, Res={soul_spark.resonance:.4f}, P.Coh={current_pattern_coherence:.4f}")

        # --- Store Final Metrics & Results ---
        final_metrics = {
            'connection_strength': float(current_strength), 'connection_quality': float(current_quality),
            'stability': soul_spark.stability, 'creator_alignment': soul_spark.creator_alignment,
            'resonance': soul_spark.resonance, 'pattern_coherence': float(current_pattern_coherence)
        }
        improvements = {key: final_metrics[key] - initial_metrics.get(key, 0.0) for key in final_metrics}

        stabilization_result = {
            'channel_id': channel_id, 'spark_id': spark_id, 'iterations': iterations,
            'initial_metrics': initial_metrics, # Record initial state for comparison
            'final_metrics': final_metrics,     # Record final state
            'improvements': improvements,       # Record changes
            'stabilization_time': datetime.now().isoformat()
        }

        # Log potential memory echo
        logger.info(f"Memory echo created: Creator connection stabilized for soul {spark_id}.")
        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
            soul_spark.memory_echoes.append(f"Creator connection stabilized @ {stabilization_result['stabilization_time']}")

        logger.info(f"Completed creator connection stabilization for soul {spark_id}")
        logger.info(f"  Overall Improvement: Str={improvements.get('connection_strength', 0):.4f}, Align={improvements.get('creator_alignment', 0):.4f}, Stab={improvements.get('stability', 0):.4f}")

        # --- Record Metrics ---
        metrics_data = {
            'action': 'stabilize_connection', 'soul_id': spark_id, 'channel_id': channel_id, 'iterations': iterations,
            'initial_strength': initial_metrics['connection_strength'], 'final_strength': final_metrics['connection_strength'],
            'initial_alignment': initial_metrics['creator_alignment'], 'final_alignment': final_metrics['creator_alignment'],
            'initial_stability': initial_metrics['stability'], 'final_stability': final_metrics['stability'],
            'success': True, 'timestamp': stabilization_result['stabilization_time']
        }
        try: metrics.record_metrics('creator_entanglement', metrics_data)
        except Exception as e: logger.error(f"Failed to record metrics for connection stabilization: {e}")

        return stabilization_result, metrics_data

    except Exception as e:
        logger.error(f"Error stabilizing creator connection for soul {spark_id}: {e}", exc_info=True)
        raise RuntimeError("Creator connection stabilization failed.") from e


def run_full_entanglement_process(soul_spark: SoulSpark,
                                  creator_resonance: float = 0.8,
                                  edge_of_chaos_ratio: float = DEFAULT_EDGE_OF_CHAOS_RATIO,
                                  stabilization_iterations: int = 5) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Runs the complete creator entanglement process. Modifies SoulSpark object. Fails hard.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        creator_resonance (float): Base creator resonance strength (0-1).
        edge_of_chaos_ratio (float): Ideal ratio for entanglement (0-1).
        stabilization_iterations (int): Number of stabilization iterations.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object.
            - overall_metrics (Dict): Summary metrics for the entire entanglement process.

    Raises:
        RuntimeError: If any stage of the entanglement process fails critically.
        TypeError: If soul_spark is not a SoulSpark instance.
        ValueError: If input parameters are invalid.
    """
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    process_metrics_summary = {'steps': {}} # To store summary metrics from each step

    logger.info(f"--- Starting Full Entanglement Process for Soul {spark_id} ---")

    try:
        # --- Get Kether Aspects ---
        kether_aspects = _get_kether_aspects_instance() # Fails hard if unavailable

        # --- Store Initial State ---
        initial_state = { # Capture key state metrics before process
            'stability': getattr(soul_spark, 'stability', 0.0),
            'coherence': getattr(soul_spark, 'coherence', 0.0),
            'resonance': getattr(soul_spark, 'resonance', 0.0),
            'creator_alignment': getattr(soul_spark, 'creator_alignment', 0.0),
            'aspect_count': len(getattr(soul_spark, 'aspects', {}))
        }
        logger.info(f"Initial State: Stab={initial_state['stability']:.4f}, Coh={initial_state['coherence']:.4f}, Res={initial_state['resonance']:.4f}, Align={initial_state['creator_alignment']:.4f}, Aspects={initial_state['aspect_count']}")

        # --- 1. Establish Connection ---
        logger.info("Step 1: Establishing Connection...")
        channel_data, metrics1 = establish_creator_connection(soul_spark, kether_aspects, creator_resonance, edge_of_chaos_ratio)
        process_metrics_summary['steps']['connection'] = metrics1
        logger.info("Step 1 Complete.")

        # --- 2. Form Resonance Patterns ---
        logger.info("Step 2: Forming Resonance Patterns...")
        pattern_data, metrics2 = form_resonance_patterns(soul_spark, channel_data, kether_aspects)
        process_metrics_summary['steps']['patterns'] = metrics2
        logger.info("Step 2 Complete.")

        # --- 3. Transfer Creator Aspects ---
        logger.info("Step 3: Transferring Creator Aspects...")
        transfer_metrics, metrics3 = transfer_creator_aspects(soul_spark, channel_data, pattern_data, kether_aspects)
        process_metrics_summary['steps']['transfer'] = metrics3
        logger.info("Step 3 Complete.")

        # --- 4. Stabilize Connection ---
        logger.info("Step 4: Stabilizing Connection...")
        stabilization_result, metrics4 = stabilize_creator_connection(soul_spark, channel_data, pattern_data, stabilization_iterations)
        process_metrics_summary['steps']['stabilization'] = metrics4
        logger.info("Step 4 Complete.")

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat()
        end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Capture final state metrics
            'stability': getattr(soul_spark, 'stability', 0.0),
            'coherence': getattr(soul_spark, 'coherence', 0.0),
            'resonance': getattr(soul_spark, 'resonance', 0.0),
            'creator_alignment': getattr(soul_spark, 'creator_alignment', 0.0),
            'aspect_count': len(getattr(soul_spark, 'aspects', {}))
        }

        overall_metrics = {
            'action': 'full_entanglement', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, # Include initial state snapshot
            'final_state': final_state,     # Include final state snapshot
            'final_connection_strength': stabilization_result['final_metrics']['connection_strength'],
            'aspects_transferred_count': transfer_metrics['aspect_count'],
            'success': True,
            # 'steps_metrics': process_metrics_summary['steps'] # Keep optional for brevity
        }
        try: metrics.record_metrics('creator_entanglement_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for entanglement: {e}")

        logger.info(f"--- Full Entanglement Process Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final State: Stab={final_state['stability']:.4f}, Coh={final_state['coherence']:.4f}, Res={final_state['resonance']:.4f}, Align={final_state['creator_alignment']:.4f}, Aspects={final_state['aspect_count']}")

        return soul_spark, overall_metrics # Return modified soul object and summary metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Full entanglement process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"
        if 'stabilization' in process_metrics_summary['steps']: failed_step = 'stabilization'
        elif 'transfer' in process_metrics_summary['steps']: failed_step = 'transfer'
        elif 'patterns' in process_metrics_summary['steps']: failed_step = 'patterns'
        elif 'connection' in process_metrics_summary['steps']: failed_step = 'connection'

        if METRICS_AVAILABLE:
             try: metrics.record_metrics('creator_entanglement_summary', {
                  'action': 'full_entanglement', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'success': False, 'error': str(e), 'failed_step': failed_step
             })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Full entanglement process failed at step '{failed_step}'.") from e


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Creator Entanglement Module Example...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        # Create a SoulSpark instance using its constructor
        test_soul = SoulSpark(creator_resonance=0.7) # Pass resonance to constructor if needed
        test_soul.spark_id="test_entangle_002"
        # Set/override initial values if necessary AFTER construction
        test_soul.stability = 0.50
        test_soul.resonance = 0.55
        test_soul.creator_alignment = 0.03
        test_soul.frequency = 410.0 # Example frequency
        # Initialize frequency signature if not done by constructor
        if not hasattr(test_soul, 'frequency_signature') or not test_soul.frequency_signature:
             test_soul.generate_harmonic_structure() # Call the method if it exists
        test_soul.aspects = {} # Ensure aspects dict exists
        test_soul.memory_echoes = [] # Add memory echoes list

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Resonance: {test_soul.resonance:.4f}")
        print(f"  Alignment: {test_soul.creator_alignment:.4f}")
        print(f"  Aspects: {len(test_soul.aspects)}")
        print(f"  Base Freq: {getattr(test_soul, 'frequency', 'N/A')}")

        try:
            print("\n--- Running Full Entanglement Process ---")
            # Run the process, modifying the test_soul object
            modified_soul, summary_metrics_result = run_full_entanglement_process(
                soul_spark=test_soul, # Pass the object
                creator_resonance=0.88,
                edge_of_chaos_ratio=DEFAULT_EDGE_OF_CHAOS_RATIO, # Use constant
                stabilization_iterations=5
            )

            print("\n--- Entanglement Complete ---")
            print("Final Soul State Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Creator Connection ID: {getattr(modified_soul, 'creator_channel_id', 'N/A')}")
            print(f"  Creator Connection Strength: {getattr(modified_soul, 'creator_connection_strength', 'N/A'):.4f}")
            print(f"  Creator Alignment: {getattr(modified_soul, 'creator_alignment', 'N/A'):.4f}")
            print(f"  Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Resonance: {getattr(modified_soul, 'resonance', 'N/A'):.4f}")
            print(f"  Pattern Coherence: {getattr(modified_soul, 'pattern_coherence', 'N/A'):.4f}")
            print(f"  Creator Aspects Added: {len(getattr(modified_soul, 'aspects', {}))}")
            print(f"  Memory Echoes: {len(getattr(modified_soul, 'memory_echoes', []))}")

            # print("\nOverall Process Metrics:")
            # print(json.dumps(summary_metrics_result, indent=2, default=str))


        except (ValueError, TypeError, RuntimeError, ImportError) as e:
            print(f"\n--- ERROR during Creator Entanglement Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Creator Entanglement Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\nCreator Entanglement Module Example Finished.")


# --- END OF FILE creator_entanglement.py ---
