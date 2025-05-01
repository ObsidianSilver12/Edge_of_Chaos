# --- START OF FILE src/stage_1/soul_formation/identity_crystallization.py ---

"""
Identity Crystallization Functions (Refactored V4.1 - SEU/SU/CU Units)

Handles identity crystallization using SU/CU prerequisites. Frequency in Hz.
Most internal calculations use 0-1 factors/scores. Geometry application influences SU/CU.
Modifies SoulSpark directly. Uses constants. Requires user input for name if unspecified.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
import random
import uuid
from typing import Dict, List, Any, Tuple, Optional

from stage_1.soul_formation.creator_entanglement import calculate_resonance

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
    # Aspect dictionary needed for mappings
    try: from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
    except ImportError: aspect_dictionary = None
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
    # Edge of Chaos (Optional - Placeholder)
    try: from shared.edge_of_chaos import calculate_edge_of_chaos_metric; EDGE_OF_CHAOS_AVAILABLE = True
    except ImportError: EDGE_OF_CHAOS_AVAILABLE = False; calculate_edge_of_chaos_metric = lambda x: 0.5 # Dummy

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


# --- Helper Functions (Check Prerequisites, Ensure Properties, Calculations) ---

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds and 0-1 factor threshold. """
    logger.debug(f"Checking identity prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): return False

    # 1. Stage Flag
    if not getattr(soul_spark, FLAG_READY_FOR_IDENTITY, False): # Set by Earth Harmony
        logger.error(f"Prerequisite failed: Soul not marked {FLAG_READY_FOR_IDENTITY}.")
        return False

    # 2. State Thresholds (Absolute SU/CU, Factor 0-1)
    stability_su = getattr(soul_spark, 'stability', 0.0)
    coherence_cu = getattr(soul_spark, 'coherence', 0.0)
    earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0) # 0-1 score

    if stability_su < IDENTITY_STABILITY_THRESHOLD_SU:
        logger.error(f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {IDENTITY_STABILITY_THRESHOLD_SU} SU."); return False
    if coherence_cu < IDENTITY_COHERENCE_THRESHOLD_CU:
        logger.error(f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {IDENTITY_COHERENCE_THRESHOLD_CU} CU."); return False
    if earth_resonance < IDENTITY_EARTH_RESONANCE_THRESHOLD:
        logger.error(f"Prerequisite failed: Earth Resonance ({earth_resonance:.3f}) < {IDENTITY_EARTH_RESONANCE_THRESHOLD}."); return False

    # 3. Essential Attributes for Calculation
    if getattr(soul_spark, 'soul_color', None) is None:
        logger.error("Prerequisite failed: SoulSpark missing 'soul_color'.")
        return False
    if getattr(soul_spark, 'soul_frequency', 0.0) <= FLOAT_EPSILON:
        logger.error(f"Prerequisite failed: SoulSpark missing valid 'soul_frequency' ({getattr(soul_spark, 'soul_frequency', 0.0)}).")
        return False

    if getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_IDENTITY_CRYSTALLIZED}. Re-running.")

    logger.debug("Identity prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. """
    logger.debug(f"Ensuring properties for identity process (Soul {soul_spark.spark_id})...")
    # Check critical attributes (should exist after previous stages)
    required = ['stability', 'coherence', 'earth_resonance', 'soul_color', 'soul_frequency', 'frequency']
    if not all(hasattr(soul_spark, attr) for attr in required):
        missing = [attr for attr in required if not hasattr(soul_spark, attr)]
        raise AttributeError(f"SoulSpark missing essential attributes for Identity: {missing}")
    if soul_spark.soul_color is None: raise ValueError("SoulColor cannot be None.")
    if soul_spark.soul_frequency <= FLOAT_EPSILON: raise ValueError("SoulFrequency must be positive.")
    if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Base Frequency must be positive.")

    # Initialize attributes set during this stage if missing
    defaults = {
        "name": None, "gematria_value": 0, "name_resonance": 0.0, "voice_frequency": 0.0,
        "response_level": 0.0, "heartbeat_entrainment": 0.0, "color_frequency": 0.0,
        "sephiroth_aspect": None, "elemental_affinity": None, "platonic_symbol": None,
        "yin_yang_balance": 0.5, "emotional_resonance": {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0},
        "crystallization_level": 0.0, "attribute_coherence": 0.0,
        "identity_metrics": None, "sacred_geometry_imprint": None
    }
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
             # Special handling for emotional_resonance dict
             if attr == 'emotional_resonance': setattr(soul_spark, attr, default.copy())
             else: setattr(soul_spark, attr, default)

    # Final validation call (might be redundant but safe)
    if hasattr(soul_spark, '_validate_attributes'): soul_spark._validate_attributes()
    logger.debug("Soul properties ensured for Identity Crystallization.")

# --- Calculation Helpers (_calculate_gematria, _calculate_name_resonance, _calculate_frequency_resonance - unchanged) ---
def _calculate_gematria(name: str) -> int:
    if not isinstance(name, str): raise TypeError("Name must be a string.")
    return sum(ord(char) - ord('a') + 1 for char in name.lower() if 'a' <= char <= 'z')

def _calculate_name_resonance(name: str, gematria: int) -> float:
    # ... (implementation unchanged from previous version) ...
    if not name: return 0.0
    try:
        vowels = sum(1 for c in name.lower() if c in 'aeiouy'); consonants = sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz')
        vowel_ratio = vowels / max(1, vowels + consonants); phi_inv = 1.0 / GOLDEN_RATIO
        vowel_factor = max(0.0, 1.0 - abs(vowel_ratio - phi_inv) * 2.0)
        unique_letters = len(set(c for c in name.lower() if 'a' <= c <= 'z')); letter_factor = unique_letters / max(1, len(name))
        gematria_factor = 0.5
        if gematria > 0: # Check > 0
            for num in NAME_GEMATRIA_RESONANT_NUMBERS:
                 if gematria % num == 0: gematria_factor = 0.9; break
        total_weight = (NAME_RESONANCE_BASE + NAME_RESONANCE_WEIGHT_VOWEL + NAME_RESONANCE_WEIGHT_LETTER + NAME_RESONANCE_WEIGHT_GEMATRIA)
        resonance = (NAME_RESONANCE_BASE + NAME_RESONANCE_WEIGHT_VOWEL * vowel_factor + NAME_RESONANCE_WEIGHT_LETTER * letter_factor + NAME_RESONANCE_WEIGHT_GEMATRIA * gematria_factor)
        resonance = max(0.0, min(1.0, resonance / max(FLOAT_EPSILON, total_weight)))
        return float(resonance)
    except Exception as e: logger.error(f"Error calculating name resonance for '{name}': {e}"); return 0.0

def _calculate_frequency_resonance(freq1: float, freq2: float) -> float:
    # Simple resonance calculation if helper not available
    if abs(freq1) <= FLOAT_EPSILON or abs(freq2) <= FLOAT_EPSILON:
        return 0.0
    ratio = min(freq1, freq2) / max(freq1, freq2)
    return max(0.0, min(1.0, ratio))

# --- Core Crystallization Functions (Most logic unchanged, ensure correct attributes/units used) ---

def assign_name(soul_spark: SoulSpark, specified_name: Optional[str] = None) -> None:
    """ Assigns name via user input if needed, calculates gematria/resonance(0-1). """
    # ... (User input logic unchanged) ...
    logger.info(f"Assigning name stage for soul {soul_spark.spark_id}...")
    current_name = getattr(soul_spark, 'name', None)
    name_to_use = None
    if isinstance(current_name, str) and current_name: name_to_use = current_name
    elif isinstance(specified_name, str) and specified_name: name_to_use = specified_name
    else: # Prompt user
        print("-" * 30+"\nIDENTITY CRYSTALLIZATION: SOUL NAMING"+"\nSoul ID: "+soul_spark.spark_id)
        print(f"  Color: {getattr(soul_spark, 'soul_color', 'N/A')}, Freq: {getattr(soul_spark, 'soul_frequency', 'N/A'):.1f}Hz")
        print(f"  S/C: {soul_spark.stability:.1f}SU / {soul_spark.coherence:.1f}CU")
        print("-" * 30)
        while not name_to_use:
            try:
                user_name = input(f"*** Please enter the name for this soul: ").strip()
                if not user_name: print("Name cannot be empty.")
                else: name_to_use = user_name
            except EOFError: raise RuntimeError("Failed to get soul name from user input (EOF).")
            except Exception as e: raise RuntimeError(f"Failed to get soul name: {e}")
        print("-" * 30)

    try:
        gematria = _calculate_gematria(name_to_use)
        name_resonance = _calculate_name_resonance(name_to_use, gematria) # 0-1 score

        setattr(soul_spark, 'name', name_to_use)
        setattr(soul_spark, 'gematria_value', gematria)
        setattr(soul_spark, 'name_resonance', name_resonance)
        timestamp = datetime.now().isoformat(); setattr(soul_spark, 'last_modified', timestamp)
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Name assigned: {name_to_use} (G:{gematria}, NR:{name_resonance:.3f})")
        logger.info(f"Name assignment complete: {name_to_use} (Res: {name_resonance:.4f})")
    except Exception as e: logger.error(f"Error processing assigned name: {e}", exc_info=True); raise

def assign_voice_frequency(soul_spark: SoulSpark) -> None:
    """ Assigns voice frequency (Hz) based on name/attributes. """
    # ... (Calculation logic unchanged, uses 0-1 factors like name_resonance, yin_yang) ...
    logger.info(f"Assigning voice frequency for soul {soul_spark.spark_id}...")
    name = getattr(soul_spark, 'name'); gematria = getattr(soul_spark, 'gematria_value')
    name_resonance = getattr(soul_spark, 'name_resonance'); yin_yang = getattr(soul_spark, 'yin_yang_balance')
    if not name: raise ValueError("Cannot assign voice frequency without a name.")
    try:
        length_factor = len(name) / 10.0
        vowels = sum(1 for c in name.lower() if c in 'aeiouy'); vowel_ratio = vowels / max(1, len(name))
        gematria_factor = (gematria % 100) / 100.0 # Simple mod 100 scaling
        resonance_factor = name_resonance
        # Calculate base voice freq using factors
        voice_frequency = (VOICE_FREQ_BASE +
                          VOICE_FREQ_ADJ_LENGTH_FACTOR * (length_factor - 0.5) +
                          VOICE_FREQ_ADJ_VOWEL_FACTOR * (vowel_ratio - 0.5) +
                          VOICE_FREQ_ADJ_GEMATRIA_FACTOR * (gematria_factor - 0.5) +
                          VOICE_FREQ_ADJ_RESONANCE_FACTOR * (resonance_factor - 0.5) +
                          VOICE_FREQ_ADJ_YINYANG_FACTOR * (yin_yang - 0.5))
        # Snap to nearest Solfeggio if close
        solfeggio_values = list(SOLFEGGIO_FREQUENCIES.values()); closest_solfeggio = min(solfeggio_values, key=lambda x: abs(x - voice_frequency))
        if abs(closest_solfeggio - voice_frequency) < VOICE_FREQ_SOLFEGGIO_SNAP_HZ: voice_frequency = closest_solfeggio
        # Clamp within defined Hz range
        voice_frequency = min(VOICE_FREQ_MAX_HZ, max(VOICE_FREQ_MIN_HZ, voice_frequency))

        setattr(soul_spark, 'voice_frequency', float(voice_frequency)) # Store Hz
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Voice frequency assigned: {voice_frequency:.2f} Hz")
    except Exception as e: logger.error(f"Error assigning voice frequency: {e}", exc_info=True); raise RuntimeError("Voice frequency assignment failed.") from e

def process_soul_color(soul_spark: SoulSpark) -> None:
    """ Verifies inherent soul color, calculates color frequency (Hz). """
    # ... (Logic unchanged, finds freq from COLOR_SPECTRUM) ...
    logger.info(f"Processing inherent soul color for soul {soul_spark.spark_id}...")
    soul_color = getattr(soul_spark, 'soul_color');
    if soul_color is None or not isinstance(soul_color, str) or not soul_color: raise ValueError("SoulSpark missing valid 'soul_color'.")
    soul_color_lower = soul_color.lower()
    color_data = COLOR_SPECTRUM.get(soul_color_lower)
    if color_data is None: raise ValueError(f"Soul color '{soul_color}' not in COLOR_SPECTRUM.")
    try:
        freq_range = color_data.get('frequency', (COLOR_FREQ_DEFAULT, COLOR_FREQ_DEFAULT))
        if isinstance(freq_range, (list, tuple)) and len(freq_range) == 2: color_frequency = float((freq_range[0] + freq_range[1]) / 2.0)
        elif isinstance(freq_range, (int, float)): color_frequency = float(freq_range)
        else: color_frequency = float(COLOR_FREQ_DEFAULT)
        color_frequency = max(0.0, color_frequency) # Allow 0?

        setattr(soul_spark, 'color_frequency', color_frequency) # Store Hz
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Soul color processed: {soul_color} (Freq: {color_frequency:.2f} Hz)")
    except Exception as e: logger.error(f"Error processing soul color: {e}", exc_info=True); raise RuntimeError("Soul color processing failed.") from e

def apply_heartbeat_entrainment(soul_spark: SoulSpark, bpm: float = 72.0, duration: float = 120.0) -> None:
    """ Applies heartbeat entrainment (updates 0-1 factor). """
    # ... (Logic unchanged, uses frequency_resonance helper, updates 0-1 factor) ...
    logger.info(f"Applying heartbeat entrainment ({bpm} BPM, {duration}s)...")
    if bpm <= 0 or duration <= 0: raise ValueError("BPM and duration must be positive.")
    try:
        current_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0) # 0-1 factor
        voice_frequency = getattr(soul_spark, 'voice_frequency', 0.0) # Hz
        beat_freq = bpm / 60.0 # Hz
        if beat_freq <= FLOAT_EPSILON or voice_frequency <= FLOAT_EPSILON: beat_resonance = 0.0
        else: beat_resonance = _calculate_frequency_resonance(beat_freq, voice_frequency) # 0-1 score

        duration_factor = min(1.0, duration / HEARTBEAT_ENTRAINMENT_DURATION_CAP)
        entrainment_increase = beat_resonance * duration_factor * HEARTBEAT_ENTRAINMENT_INC_FACTOR # Delta for 0-1 factor
        new_entrainment = min(1.0, current_entrainment + entrainment_increase)

        setattr(soul_spark, 'heartbeat_entrainment', float(new_entrainment))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Heartbeat entrainment applied. New level: {new_entrainment:.4f}")
    except Exception as e: logger.error(f"Error heartbeat entrainment: {e}", exc_info=True); raise RuntimeError("Heartbeat entrainment failed.") from e

def train_name_response(soul_spark: SoulSpark, cycles: int = 5) -> None:
    """ Trains name response (updates 0-1 factor). """
    # ... (Logic unchanged, uses various 0-1 factors) ...
    logger.info(f"Training name response ({cycles} cycles)...")
    if not isinstance(cycles, int) or cycles <= 0: raise ValueError("Cycles must be positive.")
    try:
        name_resonance = getattr(soul_spark, 'name_resonance'); consciousness_state = getattr(soul_spark, 'consciousness_state')
        heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment')
        current_response = getattr(soul_spark, 'response_level', 0.0) # 0-1 factor

        for cycle in range(cycles):
            base_increase = NAME_RESPONSE_TRAIN_BASE_INC + NAME_RESPONSE_TRAIN_CYCLE_INC * cycle
            name_factor = NAME_RESPONSE_TRAIN_NAME_FACTOR + (1.0 - NAME_RESPONSE_TRAIN_NAME_FACTOR) * name_resonance
            state_factor = NAME_RESPONSE_STATE_FACTORS.get(consciousness_state, NAME_RESPONSE_STATE_FACTORS['default'])
            heartbeat_factor = NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR + NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT * heartbeat_entrainment
            # eoc_boost = calculate_edge_of_chaos_metric(soul_spark.get_state_vector()) if EDGE_OF_CHAOS_AVAILABLE else 1.0 # Placeholder
            cycle_increase = base_increase * name_factor * state_factor * heartbeat_factor # Delta for 0-1 factor
            current_response = min(1.0, current_response + cycle_increase)

        setattr(soul_spark, 'response_level', float(current_response))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Name response trained. Final level: {current_response:.4f}")
    except Exception as e: logger.error(f"Error training name response: {e}", exc_info=True); raise RuntimeError("Name response training failed.") from e

def identify_primary_sephiroth(soul_spark: SoulSpark) -> None:
    """ Identifies primary Sephiroth aspect based on soul state (Hz, factors). """
    # ... (Logic unchanged, uses soul_frequency Hz and 0-1 factors) ...
    logger.info(f"Identifying primary Sephiroth aspect for soul {soul_spark.spark_id}...")
    soul_frequency = getattr(soul_spark, 'soul_frequency'); soul_color = getattr(soul_spark, 'soul_color')
    consciousness_state = getattr(soul_spark, 'consciousness_state'); gematria = getattr(soul_spark, 'gematria_value')
    yin_yang = getattr(soul_spark, 'yin_yang_balance')
    if soul_frequency <= FLOAT_EPSILON or soul_color is None: raise ValueError("Missing required soul_frequency or soul_color.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        sephiroth_affinities = {}
        # Gematria
        for gem_range, sephirah in SEPHIROTH_AFFINITY_GEMATRIA_RANGES.items():
            if gematria in gem_range: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT; break
        # Soul Color
        color_match = SEPHIROTH_AFFINITY_COLOR_MAP.get(soul_color.lower())
        if color_match: sephiroth_affinities[color_match] = sephiroth_affinities.get(color_match, 0.0) + SEPHIROTH_AFFINITY_COLOR_WEIGHT
        # Consciousness State
        state_match = SEPHIROTH_AFFINITY_STATE_MAP.get(consciousness_state)
        if state_match: sephiroth_affinities[state_match] = sephiroth_affinities.get(state_match, 0.0) + SEPHIROTH_AFFINITY_STATE_WEIGHT
        # Frequency Resonance
        for sephirah_name in aspect_dictionary.sephiroth_names:
            sephirah_freq = aspect_dictionary.get_aspects(sephirah_name).get('base_frequency', 0.0)
            if sephirah_freq > FLOAT_EPSILON:
                resonance = _calculate_frequency_resonance(soul_frequency, sephirah_freq) # 0-1 score
                if resonance > SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD:
                     sephiroth_affinities[sephirah_name] = sephiroth_affinities.get(sephirah_name, 0.0) + resonance * 0.5 # Apply resonance score
        # Yin-Yang
        if yin_yang < SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD: # Yin
            for sephirah in SEPHIROTH_AFFINITY_YIN_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * (1.0 - yin_yang)
        elif yin_yang > SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD: # Yang
            for sephirah in SEPHIROTH_AFFINITY_YANG_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * yin_yang
        else: # Balanced
            balance_factor = 1.0 - abs(yin_yang - 0.5) * 2
            for sephirah in SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_BALANCE_WEIGHT * balance_factor

        sephiroth_aspect = SEPHIROTH_ASPECT_DEFAULT # Default
        if sephiroth_affinities: sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda item: item[1])[0]

        setattr(soul_spark, 'sephiroth_aspect', sephiroth_aspect)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Primary Sephiroth aspect identified: {sephiroth_aspect}")
    except Exception as e: logger.error(f"Error identifying Sephiroth aspect: {e}", exc_info=True); setattr(soul_spark, 'sephiroth_aspect', SEPHIROTH_ASPECT_DEFAULT)

def determine_elemental_affinity(soul_spark: SoulSpark) -> None:
    """ Determines elemental affinity based on soul state (Hz, factors). """
    # ... (Logic unchanged, uses soul_frequency Hz and 0-1 factors) ...
    logger.info(f"Determining elemental affinity for soul {soul_spark.spark_id}...")
    # Retrieve necessary attributes safely
    name=getattr(soul_spark,'name'); seph_aspect=getattr(soul_spark,'sephiroth_aspect')
    color=getattr(soul_spark,'soul_color'); state=getattr(soul_spark,'consciousness_state')
    freq=getattr(soul_spark,'soul_frequency')
    if not all([name, seph_aspect, color, state]) or freq <= FLOAT_EPSILON: raise ValueError("Missing attributes for elemental affinity.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")

    try:
        elemental_affinities = {}
        # Name (Vowel/Consonant ratio)
        vowels=sum(1 for c in name.lower() if c in 'aeiouy'); consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz'); total=vowels+consonants
        if total > 0:
             v_ratio=vowels/total; c_ratio=consonants/total
             if v_ratio > ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: elem = 'air'
             elif c_ratio > ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: elem = 'earth'
             elif 0.4 <= v_ratio <= 0.6: elem = 'water'
             else: elem = 'fire'
             elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP.get(elem, 0.1) # Use map value
        # Sephiroth Element
        seph_element = aspect_dictionary.get_aspects(seph_aspect).get('element', '').lower()
        if '/' in seph_element:
             elements=seph_element.split('/'); weight = ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT / len(elements)
             for elem in elements: elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + weight
        elif seph_element: elemental_affinities[seph_element] = elemental_affinities.get(seph_element, 0.0) + ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT
        # Soul Color
        color_element = ELEMENTAL_AFFINITY_COLOR_MAP.get(color.lower())
        if color_element:
             if '/' in color_element: # Handle mixed elements like earth/water
                  elements=color_element.split('/'); weight = ELEMENTAL_AFFINITY_COLOR_WEIGHT / len(elements)
                  for elem in elements: elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + weight
             else: elemental_affinities[color_element] = elemental_affinities.get(color_element, 0.0) + ELEMENTAL_AFFINITY_COLOR_WEIGHT
        # Consciousness State
        state_element = ELEMENTAL_AFFINITY_STATE_MAP.get(state)
        if state_element: elemental_affinities[state_element] = elemental_affinities.get(state_element, 0.0) + ELEMENTAL_AFFINITY_STATE_WEIGHT
        # Frequency
        assigned_element = ELEMENTAL_AFFINITY_DEFAULT
        for upper_bound, element in ELEMENTAL_AFFINITY_FREQ_RANGES:
             if freq < upper_bound: assigned_element = element; break
        elemental_affinities[assigned_element] = elemental_affinities.get(assigned_element, 0.0) + ELEMENTAL_AFFINITY_FREQ_WEIGHT

        elemental_affinity = ELEMENTAL_AFFINITY_DEFAULT
        if elemental_affinities: elemental_affinity = max(elemental_affinities.items(), key=lambda item: item[1])[0]

        setattr(soul_spark, 'elemental_affinity', elemental_affinity)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Elemental affinity determined: {elemental_affinity}")
    except Exception as e: logger.error(f"Error determining elemental affinity: {e}", exc_info=True); setattr(soul_spark, 'elemental_affinity', ELEMENTAL_AFFINITY_DEFAULT)

def assign_platonic_symbol(soul_spark: SoulSpark) -> None:
    """ Assigns Platonic symbol based on affinities. """
    # ... (Logic unchanged, uses elemental_affinity, sephiroth_aspect, gematria) ...
    logger.info(f"Assigning Platonic symbol for soul {soul_spark.spark_id}...")
    elem_affinity = getattr(soul_spark, 'elemental_affinity'); seph_aspect = getattr(soul_spark, 'sephiroth_aspect')
    gematria = getattr(soul_spark, 'gematria_value')
    if not elem_affinity or not seph_aspect: raise ValueError("Missing affinities for Platonic symbol.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        symbol = PLATONIC_ELEMENT_MAP.get(elem_affinity)
        if not symbol:
             seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
             if seph_geom in PLATONIC_SOLIDS: symbol = seph_geom
             elif seph_geom == 'cube': symbol = 'hexahedron'
        if not symbol:
             unique_symbols = sorted(list(set(PLATONIC_ELEMENT_MAP.values()) - {None})); # Get unique valid symbols
             if unique_symbols:
                 symbol_idx = (gematria // PLATONIC_DEFAULT_GEMATRIA_RANGE) % len(unique_symbols); symbol = unique_symbols[symbol_idx]
             else: symbol = 'sphere' # Absolute fallback

        setattr(soul_spark, 'platonic_symbol', symbol)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Platonic symbol assigned: {symbol}")
    except Exception as e: logger.error(f"Error assigning Platonic symbol: {e}", exc_info=True); setattr(soul_spark, 'platonic_symbol', 'sphere')

def activate_love_resonance(soul_spark: SoulSpark, cycles: int = 7) -> None:
    """ Activates love resonance (updates 0-1 factors in emotional_resonance). """
    # ... (Logic unchanged, uses soul_frequency Hz and 0-1 factors) ...
    logger.info(f"Activating love resonance ({cycles} cycles)...")
    if not isinstance(cycles, int) or cycles <= 0: raise ValueError("Cycles must be positive.")
    try:
        soul_frequency = getattr(soul_spark, 'soul_frequency'); state = getattr(soul_spark, 'consciousness_state')
        heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment')
        emotional_resonance = getattr(soul_spark, 'emotional_resonance', {})
        current_love = float(emotional_resonance.get('love', 0.0)) # 0-1 factor
        love_freq = LOVE_RESONANCE_FREQ # Hz
        if love_freq <= FLOAT_EPSILON or soul_frequency <= FLOAT_EPSILON: raise ValueError("Frequencies invalid.")

        for cycle in range(cycles):
            cycle_factor = 1.0 - LOVE_RESONANCE_CYCLE_FACTOR_DECAY * (cycle / max(1, cycles))
            base_increase = LOVE_RESONANCE_BASE_INC * cycle_factor
            state_factor = LOVE_RESONANCE_STATE_WEIGHT.get(state, LOVE_RESONANCE_STATE_WEIGHT['default'])
            freq_resonance = _calculate_frequency_resonance(love_freq, soul_frequency) # 0-1 score
            heartbeat_factor = LOVE_RESONANCE_HEARTBEAT_WEIGHT + LOVE_RESONANCE_HEARTBEAT_SCALE * heartbeat_entrainment
            increase = base_increase * state_factor * (LOVE_RESONANCE_FREQ_RES_WEIGHT * freq_resonance) * heartbeat_factor # Delta for 0-1 factor
            current_love = min(1.0, current_love + increase)

        emotional_resonance['love'] = float(current_love)
        # Boost related emotions
        for emotion in ['joy', 'peace', 'harmony', 'compassion']:
            current = float(emotional_resonance.get(emotion, 0.0))
            emotional_resonance[emotion] = float(min(1.0, current + LOVE_RESONANCE_EMOTION_BOOST_FACTOR * current_love))

        setattr(soul_spark, 'emotional_resonance', emotional_resonance) # Update the dict
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Love resonance activated. Final level: {current_love:.4f}")
    except Exception as e: logger.error(f"Error activating love resonance: {e}", exc_info=True); raise RuntimeError("Love resonance activation failed.") from e

def apply_sacred_geometry(soul_spark: SoulSpark, stages: int = 5) -> None:
    """ Applies sacred geometry. Influences crystallization_level (0-1 score) and stability/coherence (SU/CU indirectly). """
    logger.info(f"Applying sacred geometry ({stages} stages)...")
    if not isinstance(stages, int) or stages <= 0: raise ValueError("Stages positive.")
    try:
        seph_aspect = getattr(soul_spark, 'sephiroth_aspect'); elem_affinity = getattr(soul_spark, 'elemental_affinity')
        name_resonance = getattr(soul_spark, 'name_resonance') # 0-1 factor
        if not seph_aspect or not elem_affinity: raise ValueError("Missing affinities for geometry.")
        if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")

        geometries = SACRED_GEOMETRY_STAGES; actual_stages = min(stages, len(geometries))
        current_cryst_level = getattr(soul_spark, 'crystallization_level', 0.0) # 0-1 score
        dominant_geometry = None; max_increase = -1.0
        total_stability_influence = 0.0 # Track influence on SU/CU factors
        total_coherence_influence = 0.0

        for i in range(actual_stages):
            geometry = geometries[i]
            stage_factor = SACRED_GEOMETRY_STAGE_FACTOR_BASE + SACRED_GEOMETRY_STAGE_FACTOR_SCALE * (i / max(1, actual_stages - 1))
            base_increase = SACRED_GEOMETRY_BASE_INC_BASE + SACRED_GEOMETRY_BASE_INC_SCALE * i # Base delta for 0-1 score

            seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
            seph_weight = SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT.get(seph_geom, SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT['default'])
            sephiroth_factor = seph_weight

            elem_weight = SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT.get(elem_affinity, SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT['default'])
            elemental_factor = elem_weight

            fib_idx = min(i, len(FIBONACCI_SEQUENCE) - 1); fib_val = FIBONACCI_SEQUENCE[fib_idx]
            fib_norm_idx = min(SACRED_GEOMETRY_FIB_MAX_IDX, len(FIBONACCI_SEQUENCE) - 1); fib_norm = FIBONACCI_SEQUENCE[fib_norm_idx]
            name_factor = SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE + SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE * name_resonance * (fib_val / max(1, fib_norm))

            # Calculate increase for crystallization level (0-1 score)
            increase = base_increase * stage_factor * sephiroth_factor * elemental_factor * name_factor
            current_cryst_level = min(1.0, current_cryst_level + increase)
            if increase > max_increase: dominant_geometry = geometry; max_increase = increase

            # Calculate influence on Stability/Coherence factors based on GEOMETRY_EFFECTS
            geom_effects = GEOMETRY_EFFECTS.get(geometry, DEFAULT_GEOMETRY_EFFECT)
            total_stability_influence += geom_effects.get('stability_factor_boost', 0.0) * stage_factor * increase # Influence scaled by stage progress
            total_coherence_influence += geom_effects.get('coherence_factor_boost', 0.0) * stage_factor * increase

        # Apply crystallization level score
        setattr(soul_spark, 'crystallization_level', float(current_cryst_level))
        setattr(soul_spark, 'sacred_geometry_imprint', dominant_geometry)

        # Apply Influence to SU/CU (Modify directly for simplicity in this stage)
        stab_boost_su = total_stability_influence * 0.5 * MAX_STABILITY_SU # Scale total influence to SU
        coh_boost_cu = total_coherence_influence * 0.5 * MAX_COHERENCE_CU # Scale total influence to CU
        soul_spark.stability = min(MAX_STABILITY_SU, soul_spark.stability + stab_boost_su)
        soul_spark.coherence = min(MAX_COHERENCE_CU, soul_spark.coherence + coh_boost_cu)

        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Sacred geometry applied. Dominant: {dominant_geometry}, Cryst Level: {current_cryst_level:.4f}, Stab+={stab_boost_su:.1f}, Coh+={coh_boost_cu:.1f}")

    except Exception as e: logger.error(f"Error applying sacred geometry: {e}", exc_info=True); raise RuntimeError("Sacred geometry application failed.") from e

def calculate_attribute_coherence(soul_spark: SoulSpark) -> None:
    """ Calculates attribute coherence score (0-1) based on other 0-1 factors. """
    # ... (Logic unchanged, calculates 0-1 score from other 0-1 factors) ...
    logger.info(f"Calculating attribute coherence for soul {soul_spark.spark_id}...")
    try:
        attributes = { # Collect relevant 0-1 factors/scores
            'name_resonance': getattr(soul_spark,'name_resonance', 0.0),
            'response_level': getattr(soul_spark,'response_level', 0.0),
            'state_stability': getattr(soul_spark,'state_stability', 0.0), # Stability of consciousness state
            'crystallization_level': getattr(soul_spark,'crystallization_level', 0.0),
            'heartbeat_entrainment': getattr(soul_spark,'heartbeat_entrainment', 0.0),
            'emotional_resonance_avg': np.mean(list(getattr(soul_spark,'emotional_resonance', {}).values())) if getattr(soul_spark,'emotional_resonance') else 0.0,
            'creator_connection': getattr(soul_spark,'creator_connection_strength', 0.0),
            'earth_resonance': getattr(soul_spark,'earth_resonance', 0.0),
            'elemental_alignment': getattr(soul_spark, 'elemental_alignment', 0.0),
            'cycle_synchronization': getattr(soul_spark, 'cycle_synchronization', 0.0),
            'harmony': getattr(soul_spark, 'harmony', 0.0)
        }
        attr_values = [v for v in attributes.values() if isinstance(v, (int, float)) and np.isfinite(v) and 0.0 <= v <= 1.0]
        if len(attr_values) < 3: coherence_score = 0.5 # Default if not enough data
        else:
             std_dev = np.std(attr_values) # Standard deviation of the 0-1 scores
             # Inverse relationship: lower std dev = higher coherence score (0-1)
             coherence_score = max(0.0, 1.0 - min(1.0, std_dev * ATTRIBUTE_COHERENCE_STD_DEV_SCALE))

        setattr(soul_spark, 'attribute_coherence', float(coherence_score)) # Store 0-1 score
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Attribute coherence calculated: {coherence_score:.4f}")
    except Exception as e: logger.error(f"Error calculating attribute coherence: {e}", exc_info=True); raise RuntimeError("Attribute coherence calc failed.") from e

def verify_identity_crystallization(soul_spark: SoulSpark, threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD) -> Tuple[bool, Dict[str, Any]]:
    """ Verifies identity crystallization score (0-1) against threshold. """
    # ... (Logic unchanged, uses 0-1 scores/factors) ...
    logger.info(f"Verifying identity crystallization (Threshold: {threshold})...")
    if not (0.0 < threshold <= 1.0): raise ValueError("Threshold invalid.")
    try:
        # Check presence of required attributes for the score calculation itself
        required_attrs_for_score = CRYSTALLIZATION_REQUIRED_ATTRIBUTES
        missing_attributes = [attr for attr in required_attrs_for_score if getattr(soul_spark, attr, None) is None]
        attr_presence_score = (len(required_attrs_for_score) - len(missing_attributes)) / max(1, len(required_attrs_for_score)) # 0-1 score

        # Gather components for final score (all should be 0-1 factors/scores)
        component_metrics = {
            'name_resonance': getattr(soul_spark,'name_resonance', 0.0),
            'response_level': getattr(soul_spark,'response_level', 0.0),
            'state_stability': getattr(soul_spark,'state_stability', 0.0),
            'crystallization_level': getattr(soul_spark,'crystallization_level', 0.0),
            'attribute_coherence': getattr(soul_spark,'attribute_coherence', 0.0),
            'attribute_presence': attr_presence_score,
            'emotional_resonance': np.mean(list(getattr(soul_spark,'emotional_resonance', {}).values())) if getattr(soul_spark,'emotional_resonance') else 0.0
        }

        # Calculate weighted final score (0-1)
        total_crystallization_score = 0.0; total_weight = 0.0
        for component, weight in CRYSTALLIZATION_COMPONENT_WEIGHTS.items():
            value = component_metrics.get(component, 0.0)
            if isinstance(value, (int, float)) and np.isfinite(value):
                total_crystallization_score += value * weight; total_weight += weight
        if total_weight > FLOAT_EPSILON: total_crystallization_score /= total_weight
        total_crystallization_score = max(0.0, min(1.0, total_crystallization_score))

        is_crystallized = (total_crystallization_score >= threshold and
                           attr_presence_score >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD)

        setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, is_crystallized)
        setattr(soul_spark, 'crystallization_level', float(total_crystallization_score)) # Store final 0-1 score
        timestamp = datetime.now().isoformat(); setattr(soul_spark, 'last_modified', timestamp)

        verification_result = { # Store details
            'total_crystallization_score': float(total_crystallization_score), 'threshold': threshold,
            'is_crystallized': is_crystallized, 'components': component_metrics, 'missing_attributes': missing_attributes }
        setattr(soul_spark, 'identity_metrics', verification_result)

        logger.info(f"Identity check: Score={total_crystallization_score:.4f}, Passed={is_crystallized}")
        if missing_attributes: logger.warning(f"  Missing attributes for check: {', '.join(missing_attributes)}")
        return is_crystallized, verification_result

    except Exception as e: logger.error(f"Error verifying identity: {e}", exc_info=True); raise RuntimeError("Identity verification failed.") from e


# --- Orchestration Function ---
def perform_identity_crystallization(soul_spark: SoulSpark,
                                    # Remove life_cord_data argument - not needed by internal functions
                                    specified_name: Optional[str] = None,
                                    train_cycles: int = 5,
                                    entrainment_bpm: float = 72.0,
                                    entrainment_duration: float = 120.0,
                                    love_cycles: int = 7,
                                    geometry_stages: int = 5,
                                    crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD
                                    ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete identity crystallization. Modifies SoulSpark. Uses SU/CU prerequisites. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    # ... (validate other params as before) ...
    if not isinstance(train_cycles,int) or train_cycles<0: raise ValueError("train_cycles>=0")
    if not isinstance(entrainment_bpm,(int,float)) or entrainment_bpm<=0: raise ValueError("bpm>0")
    # ... etc for love_cycles, geometry_stages, threshold

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Identity Crystallization for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps_completed': []} # Track completed steps

    try:
        _ensure_soul_properties(soul_spark)
        if not _check_prerequisites(soul_spark): # Uses SU/CU thresholds
            raise ValueError("Soul prerequisites for identity crystallization not met.")

        initial_state = { # Record state in correct units/scores
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level, 'attribute_coherence': soul_spark.attribute_coherence }

        # --- Run Sequence ---
        assign_name(soul_spark, specified_name); process_metrics_summary['steps_completed'].append('name')
        assign_voice_frequency(soul_spark); process_metrics_summary['steps_completed'].append('voice')
        process_soul_color(soul_spark); process_metrics_summary['steps_completed'].append('color')
        apply_heartbeat_entrainment(soul_spark, entrainment_bpm, entrainment_duration); process_metrics_summary['steps_completed'].append('heartbeat')
        train_name_response(soul_spark, train_cycles); process_metrics_summary['steps_completed'].append('response')
        identify_primary_sephiroth(soul_spark); process_metrics_summary['steps_completed'].append('sephiroth_id')
        determine_elemental_affinity(soul_spark); process_metrics_summary['steps_completed'].append('elemental_id')
        assign_platonic_symbol(soul_spark); process_metrics_summary['steps_completed'].append('platonic_id')
        activate_love_resonance(soul_spark, love_cycles); process_metrics_summary['steps_completed'].append('love')
        apply_sacred_geometry(soul_spark, geometry_stages); process_metrics_summary['steps_completed'].append('geometry') # This influences SU/CU
        calculate_attribute_coherence(soul_spark); process_metrics_summary['steps_completed'].append('attr_coherence')
        if hasattr(soul_spark, 'update_state'): soul_spark.update_state() # Update SU/CU scores after geometry influence
        is_crystallized, verification_metrics = verify_identity_crystallization(soul_spark, crystallization_threshold); process_metrics_summary['steps_completed'].append('verify')

        if not is_crystallized:
             error_msg = f"Identity crystallization failed: Score {verification_metrics['total_crystallization_score']:.4f} < Threshold {crystallization_threshold}."
             if verification_metrics['missing_attributes']: error_msg += f" Missing: {verification_metrics['missing_attributes']}"
             logger.error(error_msg)
             if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(error_msg)
             raise RuntimeError(error_msg) # Fail the process explicitly

        # --- Final Update & Metrics ---
        setattr(soul_spark, FLAG_READY_FOR_BIRTH, True)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Identity crystallized as '{soul_spark.name}'. Level: {soul_spark.crystallization_level:.3f}")

        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = { # Report final state in correct units/scores
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level, 'attribute_coherence': soul_spark.attribute_coherence,
             'name': soul_spark.name, FLAG_IDENTITY_CRYSTALLIZED: getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED) }
        overall_metrics = {
            'action': 'identity_crystallization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'final_crystallization_score': verification_metrics['total_crystallization_score'],
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('identity_crystallization_summary', overall_metrics)

        logger.info(f"--- Identity Crystallization Completed Successfully for Soul {spark_id} ---")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Identity Crystallization failed for {spark_id}: {e_val}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'prerequisites'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e_val))
         raise
    except RuntimeError as e_rt: # Catch critical internal failures (incl. verification failure)
         logger.critical(f"Identity Crystallization failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'runtime'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False); setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         raise
    except Exception as e:
         logger.critical(f"Unexpected error during Identity Crystallization for {spark_id}: {e}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'unexpected'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e))
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False); setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         raise RuntimeError(f"Unexpected Identity Crystallization failure: {e}") from e

# --- Failure Metric Helper ---
def record_id_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            metrics.record_metrics('identity_crystallization_summary', {
                'action': 'identity_crystallization', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - datetime.fromisoformat(start_time_iso)).total_seconds(),
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record ID failure metrics for {spark_id}: {metric_e}")


# --- END OF FILE src/stage_1/soul_formation/identity_crystallization.py ---