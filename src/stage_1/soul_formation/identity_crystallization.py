# --- START OF FILE identity_crystallization.py ---

"""
Identity Crystallization Functions (Refactored - Operates on SoulSpark Object)

Handles the crystallization of a soul's unique identity after earth harmonization.
Assigns name, determines core properties (color, frequency, affinities), trains
response, and verifies crystallization level. Modifies the SoulSpark object
instance directly. Uses constants.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime
import time
import random
import string # For name generation
import uuid
from typing import Dict, List, Any, Tuple, Optional

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from src.constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants from src.constants: {e}. Identity Crystallization cannot function correctly.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
try:
    from src.stage_1.soul_formation.soul_spark import SoulSpark
    from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary
    # Import Edge of Chaos functions/classes if available
    try:
        from src.stage_1.soul_formation.edge_of_chaos import * # Import all edge_of_chaos functionality for potential future use
        EDGE_OF_CHAOS_AVAILABLE = True
    except ImportError:
        logger.warning("edge_of_chaos module not found. Related boost features disabled.")
        EDGE_OF_CHAOS_AVAILABLE = False

    DEPENDENCIES_AVAILABLE = True
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import dependencies (SoulSpark, aspect_dictionary): {e}")
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

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """Checks if the soul meets prerequisites for identity crystallization."""
    logger.debug(f"Checking identity prerequisites for soul {soul_spark.spark_id}...")
    # Requires earth harmonization completion
    if not getattr(soul_spark, "earth_harmonized", False):
        logger.error("Prerequisite failed: Soul earth harmonization not complete.")
        return False
    # Requires minimum stability/coherence/earth resonance after harmony
    if getattr(soul_spark, "stability", 0.0) < 0.80: logger.error(f"Prerequisite failed: Stability too low ({getattr(soul_spark, 'stability', 0.0):.3f} < 0.80)"); return False
    if getattr(soul_spark, "coherence", 0.0) < 0.80: logger.error(f"Prerequisite failed: Coherence too low ({getattr(soul_spark, 'coherence', 0.0):.3f} < 0.80)"); return False
    if getattr(soul_spark, "earth_resonance", 0.0) < 0.75: logger.error(f"Prerequisite failed: Earth Resonance too low ({getattr(soul_spark, 'earth_resonance', 0.0):.3f} < 0.75)"); return False

    logger.debug("Identity prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """Ensure soul has necessary numeric properties for identity. Fails hard."""
    logger.debug(f"Ensuring properties exist for identity process (Soul {soul_spark.spark_id})...")
    attributes_to_check = [
        # Attributes potentially modified or read by this module
        ("name", None), ("gematria_value", 0), ("name_resonance", 0.0),
        ("voice_frequency", 0.0), ("response_level", 0.0),
        ("heartbeat_entrainment", 0.0), ("soul_color", None),
        ("color_frequency", 0.0), ("soul_frequency", 0.0),
        ("sephiroth_aspect", None), ("elemental_affinity", None),
        ("platonic_symbol", None), ("yin_yang_balance", 0.5),
        ("emotional_resonance", {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0}),
        ("crystallization_level", 0.0), ("attribute_coherence", 0.0),
        # Attributes read from soul state
        ("stability", SOUL_SPARK_DEFAULT_STABILITY),
        ("coherence", SOUL_SPARK_DEFAULT_COHERENCE),
        ("frequency", SOUL_SPARK_DEFAULT_FREQ),
        ("consciousness_state", "aware"), # Assume 'aware' for crystallization
        ("aspects", {}),
        ("creator_connection_strength", 0.0) # Needed for name gen example
    ]
    for attr, default in attributes_to_check:
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
            logger.warning(f"Soul {soul_spark.spark_id} missing '{attr}' for identity. Initializing to default: {default}")
            setattr(soul_spark, attr, default)

    # Validate numerical types and ranges
    for attr, _ in attributes_to_check:
        val = getattr(soul_spark, attr)
        if attr in ['name', 'soul_color', 'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol', 'consciousness_state']:
             if val is not None and not isinstance(val, str): raise TypeError(f"Attribute '{attr}' should be str or None, found {type(val)}.")
        elif attr in ['aspects', 'emotional_resonance']:
             if not isinstance(val, dict): raise TypeError(f"Attribute '{attr}' should be dict, found {type(val)}.")
        elif attr == 'gematria_value':
             if not isinstance(val, int): raise TypeError(f"Attribute '{attr}' should be int, found {type(val)}.")
        elif isinstance(val, (int, float)): # Numeric checks
             if not np.isfinite(val): raise ValueError(f"Attribute '{attr}' has non-finite value {val}.")
             # Add specific range checks if needed (e.g., frequencies > 0)
             if 'frequency' in attr and val <= FLOAT_EPSILON and attr != 'color_frequency': raise ValueError(f"Frequency attribute '{attr}' ({val}) must be positive.")
             if attr not in ['frequency', 'gematria_value', 'voice_frequency', 'color_frequency', 'soul_frequency'] and not (0.0 <= val <= 1.0):
                  logger.warning(f"Clamping identity attribute '{attr}' ({val}) to 0-1 range.")
                  setattr(soul_spark, attr, max(0.0, min(1.0, val)))
        # else: raise TypeError(f"Attribute '{attr}' has unexpected type {type(val)}.") # Fail on unexpected types
    logger.debug("Soul properties ensured for Identity Crystallization.")

def _calculate_frequency_resonance(freq1: float, freq2: float) -> float:
    """Calculate resonance between two frequencies. Fails hard."""
    # (Copied from harmonic_strengthening - consider moving to shared utils)
    if freq1 <= FLOAT_EPSILON or freq2 <= FLOAT_EPSILON: return 0.0
    if abs(freq1 - freq2) < FLOAT_EPSILON: return 1.0
    try:
        ratio = max(freq1, freq2) / min(freq1, freq2)
        harmonic_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, GOLDEN_RATIO, 1.0/GOLDEN_RATIO, 1.0/1.5, 1.0/2.0, 1.0/3.0, 1.0/4.0]
        min_distance = min(abs(np.log(ratio) - np.log(hr)) for hr in harmonic_ratios)
        resonance = max(0.0, 1.0 - min_distance * 0.5)
        return resonance
    except Exception as e: logger.error(f"Freq resonance error: {e}"); raise RuntimeError("Freq resonance failed.") from e

def _generate_soul_name(soul_spark: SoulSpark) -> str:
    """Generates a name based on soul properties using constants. Fails hard."""
    logger.debug(f"Generating name for soul {soul_spark.spark_id}...")
    try:
        # Use constants for name generation logic
        connection_strength = getattr(soul_spark, 'creator_connection_strength', 0.7)
        yin_yang = getattr(soul_spark, 'yin_yang_balance', 0.5)
        primary_element = getattr(soul_spark, 'elemental_affinity', 'aether')

        name_length = NAME_LENGTH_BASE + int(connection_strength * NAME_LENGTH_FACTOR)
        vowel_ratio = NAME_VOWEL_RATIO_BASE + NAME_VOWEL_RATIO_FACTOR * (1.0 - yin_yang)

        seed_val = int(uuid.UUID(soul_spark.spark_id).int >> 64)
        random.seed(seed_val)

        vowels = 'aeiouy'; consonants = 'bcdfghjklmnpqrstvwxz'
        if primary_element == 'earth': consonants *= 2
        elif primary_element == 'air': vowels *= 2

        name = ""; start_char_set = vowels.replace('y','') + consonants
        name += random.choice(start_char_set).upper()
        for _ in range(name_length - 1):
             last_char_is_vowel = name[-1].lower() in vowels
             next_char = random.choice(vowels if (last_char_is_vowel and random.random() < (1.0-vowel_ratio)) or (not last_char_is_vowel and random.random() < vowel_ratio) else consonants)
             name += next_char

        logger.debug(f"Generated name: {name}")
        return name
    except Exception as e: logger.error(f"Error generating soul name: {e}", exc_info=True); raise RuntimeError("Soul name generation failed.") from e

def _calculate_gematria(name: str) -> int:
    """Calculates simple gematria value."""
    if not name: return 0
    return sum(ord(char) - ord('a') + 1 for char in name.lower() if 'a' <= char <= 'z')

def _calculate_name_resonance(name: str, gematria: int) -> float:
    """Calculates how well name resonates using constants. Fails hard."""
    logger.debug(f"Calculating resonance for name '{name}' (Gematria: {gematria})...")
    if not name: return 0.0
    try:
        vowels = sum(1 for c in name.lower() if c in 'aeiouy'); consonants = sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz')
        vowel_ratio = vowels / max(1, vowels + consonants); phi_inv = 1.0 / GOLDEN_RATIO
        vowel_factor = max(0.0, 1.0 - abs(vowel_ratio - phi_inv) * 2.0)
        unique_letters = len(set(c for c in name.lower() if 'a' <= c <= 'z')); letter_factor = unique_letters / max(1, len(name))
        gematria_factor = 0.5
        for num in NAME_GEMATRIA_RESONANT_NUMBERS:
            if gematria % num == 0: gematria_factor = 0.9; break
        # Use constant weights
        total_weight = (NAME_RESONANCE_BASE + NAME_RESONANCE_WEIGHT_VOWEL + NAME_RESONANCE_WEIGHT_LETTER + NAME_RESONANCE_WEIGHT_GEMATRIA)
        resonance = (NAME_RESONANCE_BASE + NAME_RESONANCE_WEIGHT_VOWEL * vowel_factor +
                     NAME_RESONANCE_WEIGHT_LETTER * letter_factor + NAME_RESONANCE_WEIGHT_GEMATRIA * gematria_factor)
        resonance = max(0.0, min(1.0, resonance / max(FLOAT_EPSILON, total_weight))) # Normalize
        logger.debug(f"  Name Resonance Factors -> Resonance={resonance:.4f}")
        return float(resonance)
    except Exception as e: logger.error(f"Error calculating name resonance: {e}", exc_info=True); raise RuntimeError("Name resonance calculation failed.") from e

# --- Core Crystallization Functions ---

def assign_name(soul_spark: SoulSpark, specified_name: Optional[str] = None) -> None:
    """Assigns name and calculates related properties. Modifies SoulSpark."""
    logger.info(f"Assigning name (Specified: {specified_name}) to soul {soul_spark.spark_id}...")
    if specified_name and isinstance(specified_name, str): name = specified_name
    else: name = _generate_soul_name(soul_spark) # Fails hard if soul_spark invalid

    gematria = _calculate_gematria(name)
    name_resonance = _calculate_name_resonance(name, gematria) # Fails hard

    # --- Update SoulSpark ---
    setattr(soul_spark, 'name', name)
    setattr(soul_spark, 'gematria_value', gematria)
    setattr(soul_spark, 'name_resonance', name_resonance)
    setattr(soul_spark, 'last_modified', datetime.now().isoformat())

    # Log memory echo
    echo_msg = f"Name assigned: {name} (Gematria: {gematria}, Resonance: {name_resonance:.3f})"
    logger.info(f"Memory echo created: {echo_msg}")
    if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
        soul_spark.memory_echoes.append(f"{getattr(soul_spark, 'last_modified')}: {echo_msg}")

    logger.info(f"Name assigned: {name} (Gematria: {gematria}, Resonance: {name_resonance:.4f})")


def assign_voice_frequency(soul_spark: SoulSpark) -> None:
    """Assigns voice frequency using constants. Modifies SoulSpark."""
    logger.info(f"Assigning voice frequency for soul {soul_spark.spark_id}...")
    name = getattr(soul_spark, 'name', None)
    gematria = getattr(soul_spark, 'gematria_value', 0)
    name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
    yin_yang = getattr(soul_spark, 'yin_yang_balance', 0.5)
    if not name: raise ValueError("Cannot assign voice frequency without a name.")

    try:
        length_factor = len(name) / 10.0
        vowels = sum(1 for c in name.lower() if c in 'aeiouy'); vowel_ratio = vowels / max(1, len(name))
        gematria_factor = (gematria % 100) / 100.0
        resonance_factor = name_resonance

        # Use constants
        voice_frequency = (VOICE_FREQ_BASE +
                          VOICE_FREQ_ADJ_LENGTH_FACTOR * (length_factor - 0.5) +
                          VOICE_FREQ_ADJ_VOWEL_FACTOR * (vowel_ratio - 0.5) +
                          VOICE_FREQ_ADJ_GEMATRIA_FACTOR * (gematria_factor - 0.5) +
                          VOICE_FREQ_ADJ_RESONANCE_FACTOR * (resonance_factor - 0.5) +
                          VOICE_FREQ_ADJ_YINYANG_FACTOR * (yin_yang - 0.5))

        # Snap to Solfeggio using constant
        solfeggio_values = list(SOLFEGGIO_FREQUENCIES.values()); closest_freq = min(solfeggio_values, key=lambda x: abs(x - voice_frequency))
        if abs(closest_freq - voice_frequency) < VOICE_FREQ_SOLFEGGIO_SNAP_HZ: voice_frequency = closest_freq; logger.debug("Snapped voice frequency to Solfeggio.")

        # Clamp to range using constants
        voice_frequency = min(VOICE_FREQ_MAX_HZ, max(VOICE_FREQ_MIN_HZ, voice_frequency))

        # --- Update SoulSpark ---
        setattr(soul_spark, 'voice_frequency', float(voice_frequency))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Voice frequency assigned: {voice_frequency:.2f} Hz")

    except Exception as e: logger.error(f"Error assigning voice frequency: {e}", exc_info=True); raise RuntimeError("Voice frequency assignment failed.") from e


def determine_soul_color(soul_spark: SoulSpark) -> None:
    """Determines soul color using constants. Modifies SoulSpark."""
    logger.info(f"Determining soul color for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs like name, voice_freq exist
    name = getattr(soul_spark, 'name'); gematria = getattr(soul_spark, 'gematria_value')
    consciousness_state = getattr(soul_spark, 'consciousness_state'); yin_yang = getattr(soul_spark, 'yin_yang_balance')
    voice_frequency = getattr(soul_spark, 'voice_frequency')

    try:
        colors = list(COLOR_SPECTRUM.keys()); color_affinities = {}

        # 1. Gematria (Use constant weight)
        gematria_factor = (gematria % 100) / 100.0; gematria_color_idx = int(gematria_factor * len(colors)) % len(colors)
        gematria_color = colors[gematria_color_idx]; color_affinities[gematria_color] = COLOR_AFFINITY_GEMATRIA_WEIGHT

        # 2. Vowel (Use constant weight)
        vowel_counts = {v: name.lower().count(v) for v in 'aeiouy'}; total_vowels = sum(vowel_counts.values())
        vowel_colors = {'a': 'red', 'e': 'green', 'i': 'blue', 'o': 'gold', 'u': 'violet', 'y': 'indigo'}
        if total_vowels > 0:
             for vowel, count in vowel_counts.items():
                 if vowel in vowel_colors and vowel_colors[vowel] in colors:
                     color = vowel_colors[vowel]; color_affinities[color] = color_affinities.get(color, 0.0) + COLOR_AFFINITY_VOWEL_WEIGHT_FACTOR * (count / total_vowels)

        # 3. Consciousness State (Use constant weight)
        state_color = SEPHIROTH_AFFINITY_STATE_MAP.get(consciousness_state) # Reuse map
        if state_color and state_color in colors: color_affinities[state_color] = color_affinities.get(state_color, 0.0) + COLOR_AFFINITY_STATE_WEIGHT

        # 4. Yin-Yang (Use constant weights/thresholds)
        if yin_yang < SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD:
            for color in SEPHIROTH_AFFINITY_YIN_SEPHIROTH:
                if color in colors: color_affinities[color] = color_affinities.get(color, 0.0) + COLOR_AFFINITY_YINYANG_WEIGHT * (1.0 - yin_yang)
        elif yin_yang > SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD:
            for color in SEPHIROTH_AFFINITY_YANG_SEPHIROTH:
                if color in colors: color_affinities[color] = color_affinities.get(color, 0.0) + COLOR_AFFINITY_YINYANG_WEIGHT * yin_yang
        else:
            balance_factor = 1.0 - abs(yin_yang - 0.5) * 2
            for color in SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH:
                if color in colors: color_affinities[color] = color_affinities.get(color, 0.0) + COLOR_AFFINITY_BALANCE_WEIGHT * balance_factor

        # Determine strongest affinity or fallback
        if color_affinities: soul_color = max(color_affinities.items(), key=lambda item: item[1])[0]
        else: # Fallback based on voice frequency
            freq_factor = (voice_frequency - VOICE_FREQ_MIN_HZ) / max(FLOAT_EPSILON, VOICE_FREQ_MAX_HZ - VOICE_FREQ_MIN_HZ)
            color_idx = int(freq_factor * len(colors)); soul_color = colors[max(0, min(len(colors) - 1, color_idx))]

        # Get color frequency using constant default
        color_data = COLOR_SPECTRUM.get(soul_color, {}); freq_range = color_data.get('frequency', (COLOR_FREQ_DEFAULT, COLOR_FREQ_DEFAULT))
        color_frequency = float((freq_range[0] + freq_range[1]) / 2.0)

        # --- Update SoulSpark ---
        setattr(soul_spark, 'soul_color', soul_color)
        setattr(soul_spark, 'color_frequency', color_frequency)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Soul color determined: {soul_color} (Freq: {color_frequency:.2f} Hz)")
        logger.debug(f"  Color Affinities: {color_affinities}")

    except Exception as e: logger.error(f"Error determining soul color: {e}", exc_info=True); raise RuntimeError("Soul color determination failed.") from e


def determine_soul_frequency(soul_spark: SoulSpark) -> None:
    """Determines soul resonant frequency using constants. Modifies SoulSpark."""
    logger.info(f"Determining soul frequency for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure voice/color freq exist
    voice_frequency = getattr(soul_spark, 'voice_frequency')
    color_frequency = getattr(soul_spark, 'color_frequency')
    consciousness_frequency = getattr(soul_spark, 'consciousness_frequency')

    try:
        base_freqs = [voice_frequency, color_frequency, consciousness_frequency]
        solfeggio_freqs = list(SOLFEGGIO_FREQUENCIES.values())
        weighted_resonances = {}

        for base_freq in base_freqs:
             if base_freq <= FLOAT_EPSILON: continue
             best_resonance = -1.0; best_freq = solfeggio_freqs[0]
             for solf_freq in solfeggio_freqs:
                  resonance = _calculate_frequency_resonance(base_freq, solf_freq)
                  if resonance > best_resonance: best_resonance = resonance; best_freq = solf_freq
             # Weight contribution (e.g., voice more than color/consciousness)
             weight = 0.8 if base_freq == voice_frequency else (0.6 if base_freq == color_frequency else 0.4)
             weighted_resonances[best_freq] = weighted_resonances.get(best_freq, 0.0) + best_resonance * weight

        if weighted_resonances: soul_frequency = float(max(weighted_resonances.items(), key=lambda item: item[1])[0])
        else: soul_frequency = SOUL_FREQ_DEFAULT # Use constant fallback

        # --- Update SoulSpark ---
        setattr(soul_spark, 'soul_frequency', soul_frequency)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Soul frequency determined: {soul_frequency:.2f} Hz")
        logger.debug(f"  Weighted Solfeggio Resonances: {weighted_resonances}")

    except Exception as e: logger.error(f"Error determining soul frequency: {e}", exc_info=True); setattr(soul_spark, 'soul_frequency', SOUL_FREQ_DEFAULT); raise RuntimeError("Soul frequency determination failed.") from e


def identify_primary_sephiroth(soul_spark: SoulSpark) -> None:
    """Identifies primary Sephiroth aspect using constants. Modifies SoulSpark."""
    logger.info(f"Identifying primary Sephiroth aspect for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    name = getattr(soul_spark, 'name'); gematria = getattr(soul_spark, 'gematria_value')
    soul_color = getattr(soul_spark, 'soul_color'); consciousness_state = getattr(soul_spark, 'consciousness_state')
    soul_frequency = getattr(soul_spark, 'soul_frequency'); yin_yang = getattr(soul_spark, 'yin_yang_balance')

    if not DEPENDENCIES_AVAILABLE or aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        sephiroth_affinities = {}

        # 1. Gematria (Use constant map & weight)
        for gem_range, sephirah in SEPHIROTH_AFFINITY_GEMATRIA_RANGES.items():
            if gematria in gem_range: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT; break

        # 2. Soul Color (Use constant map & weight)
        color_match = SEPHIROTH_AFFINITY_COLOR_MAP.get(soul_color)
        if color_match: sephiroth_affinities[color_match] = sephiroth_affinities.get(color_match, 0.0) + SEPHIROTH_AFFINITY_COLOR_WEIGHT

        # 3. Consciousness State (Use constant map & weight)
        state_match = SEPHIROTH_AFFINITY_STATE_MAP.get(consciousness_state)
        if state_match: sephiroth_affinities[state_match] = sephiroth_affinities.get(state_match, 0.0) + SEPHIROTH_AFFINITY_STATE_WEIGHT

        # 4. Frequency Resonance (Use constant threshold)
        for sephirah_name in aspect_dictionary.sephiroth_names:
            try:
                aspect_inst = aspect_dictionary.load_aspect_instance(sephirah_name)
                sephirah_freq = getattr(aspect_inst, 'base_frequency', 0.0)
                if sephirah_freq > FLOAT_EPSILON:
                    resonance = _calculate_frequency_resonance(soul_frequency, sephirah_freq)
                    if resonance > SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD:
                        sephiroth_affinities[sephirah_name] = sephiroth_affinities.get(sephirah_name, 0.0) + resonance # Weight by resonance value itself
            except Exception as e: logger.warning(f"Could not process frequency for {sephirah_name}: {e}")

        # 5. Yin-Yang (Use constant thresholds, lists, weights)
        if yin_yang < SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD:
            for sephirah in SEPHIROTH_AFFINITY_YIN_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * (1.0 - yin_yang)
        elif yin_yang > SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD:
            for sephirah in SEPHIROTH_AFFINITY_YANG_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * yin_yang
        else:
            balance_factor = 1.0 - abs(yin_yang - 0.5) * 2
            for sephirah in SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_BALANCE_WEIGHT * balance_factor

        # Determine strongest affinity or use constant default
        if sephiroth_affinities: sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda item: item[1])[0]
        else: sephiroth_aspect = SEPHIROTH_ASPECT_DEFAULT

        # --- Update SoulSpark ---
        setattr(soul_spark, 'sephiroth_aspect', sephiroth_aspect)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Primary Sephiroth aspect identified: {sephiroth_aspect}")
        logger.debug(f"  Sephiroth Affinities: {sephiroth_affinities}")

    except Exception as e: logger.error(f"Error identifying Sephiroth aspect: {e}", exc_info=True); setattr(soul_spark, 'sephiroth_aspect', SEPHIROTH_ASPECT_DEFAULT); raise RuntimeError("Sephiroth aspect identification failed.") from e


def determine_elemental_affinity(soul_spark: SoulSpark) -> None:
    """Determines elemental affinity using constants. Modifies SoulSpark."""
    logger.info(f"Determining elemental affinity for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    name=getattr(soul_spark,'name'); sephiroth_aspect=getattr(soul_spark,'sephiroth_aspect')
    soul_color=getattr(soul_spark,'soul_color'); consciousness_state=getattr(soul_spark,'consciousness_state')
    soul_frequency=getattr(soul_spark,'soul_frequency')

    try:
        elemental_affinities = {}

        # 1. Name (Vowel/Consonant Ratio - Use constant thresholds/map)
        vowels=sum(1 for c in name.lower() if c in 'aeiouy'); consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz'); total_letters=vowels+consonants
        if total_letters > 0:
            v_ratio=vowels/total_letters; c_ratio=consonants/total_letters
            if v_ratio > ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: elemental_affinities['air'] = elemental_affinities.get('air', 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP['air']
            elif c_ratio > ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: elemental_affinities['earth'] = elemental_affinities.get('earth', 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP['earth']
            elif 0.4 <= v_ratio <= 0.6: elemental_affinities['water'] = elemental_affinities.get('water', 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP['water']
            else: elemental_affinities['fire'] = elemental_affinities.get('fire', 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP['fire']

        # 2. Sephiroth Element (Use constant weight)
        seph_element = aspect_dictionary.get_aspects(sephiroth_aspect).get('element', '').lower()
        if '/' in seph_element:
            elements=seph_element.split('/'); weight = ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT / len(elements)
            for elem in elements: elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + weight
        elif seph_element: elemental_affinities[seph_element] = elemental_affinities.get(seph_element, 0.0) + ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT

        # 3. Soul Color (Use constant map & weight)
        color_element = ELEMENTAL_AFFINITY_COLOR_MAP.get(soul_color)
        if color_element: elemental_affinities[color_element] = elemental_affinities.get(color_element, 0.0) + ELEMENTAL_AFFINITY_COLOR_WEIGHT

        # 4. Consciousness State (Use constant map & weight)
        state_element = ELEMENTAL_AFFINITY_STATE_MAP.get(consciousness_state)
        if state_element: elemental_affinities[state_element] = elemental_affinities.get(state_element, 0.0) + ELEMENTAL_AFFINITY_STATE_WEIGHT

        # 5. Frequency (Use constant ranges & weight)
        assigned_element = 'aether' # Default high element
        for upper_bound, element in ELEMENTAL_AFFINITY_FREQ_RANGES:
             if soul_frequency < upper_bound: assigned_element = element; break
        elemental_affinities[assigned_element] = elemental_affinities.get(assigned_element, 0.0) + ELEMENTAL_AFFINITY_FREQ_WEIGHT

        # Determine strongest affinity or use constant default
        if elemental_affinities: elemental_affinity = max(elemental_affinities.items(), key=lambda item: item[1])[0]
        else: elemental_affinity = ELEMENTAL_AFFINITY_DEFAULT

        # --- Update SoulSpark ---
        setattr(soul_spark, 'elemental_affinity', elemental_affinity)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Elemental affinity determined: {elemental_affinity}")
        logger.debug(f"  Elemental Affinities: {elemental_affinities}")

    except Exception as e: logger.error(f"Error determining elemental affinity: {e}", exc_info=True); setattr(soul_spark, 'elemental_affinity', ELEMENTAL_AFFINITY_DEFAULT); raise RuntimeError("Elemental affinity determination failed.") from e


def assign_platonic_symbol(soul_spark: SoulSpark) -> None:
    """Assigns Platonic symbol using constants. Modifies SoulSpark."""
    logger.info(f"Assigning Platonic symbol for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    elemental_affinity = getattr(soul_spark, 'elemental_affinity'); sephiroth_aspect = getattr(soul_spark, 'sephiroth_aspect')
    gematria = getattr(soul_spark, 'gematria_value')

    try:
        symbol = None
        # 1. Primary: Element Map (Use constant map)
        symbol = PLATONIC_ELEMENT_MAP.get(elemental_affinity)

        # 2. Secondary: Sephiroth geometry (Use constant list for validation)
        if not symbol:
             seph_geom = aspect_dictionary.get_aspects(sephiroth_aspect).get('geometric_correspondence')
             seph_geom_lower = seph_geom.lower() if seph_geom else None
             if seph_geom_lower in PLATONIC_SOLIDS: symbol = seph_geom_lower
             elif seph_geom_lower == 'cube': symbol = 'hexahedron'

        # 3. Tertiary: Gematria (Use constant range/map)
        if not symbol:
             unique_symbols = sorted(list(set(PLATONIC_ELEMENT_MAP.values()))) # Get unique symbols
             symbol_idx = (gematria // PLATONIC_DEFAULT_GEMATRIA_RANGE) % len(unique_symbols)
             symbol = unique_symbols[symbol_idx]

        # --- Update SoulSpark ---
        setattr(soul_spark, 'platonic_symbol', symbol)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Platonic symbol assigned: {symbol}")

    except Exception as e: logger.error(f"Error assigning Platonic symbol: {e}", exc_info=True); setattr(soul_spark, 'platonic_symbol', 'hexahedron'); raise RuntimeError("Platonic symbol assignment failed.") from e


def train_name_response(soul_spark: SoulSpark, cycles: int = 5) -> None:
    """Trains soul response using constants. Modifies SoulSpark."""
    logger.info(f"Training name response for soul {soul_spark.spark_id} ({cycles} cycles)...")
    if not isinstance(cycles, int) or cycles <= 0: raise ValueError("Cycles must be positive.")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    name_resonance = getattr(soul_spark, 'name_resonance'); consciousness_state = getattr(soul_spark, 'consciousness_state')
    heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment')

    try:
        initial_response = getattr(soul_spark, 'response_level', 0.0)
        current_response = initial_response
        logger.debug(f"  Initial response level: {current_response:.4f}")

        for cycle in range(cycles):
            # Use constants
            base_increase = NAME_RESPONSE_TRAIN_BASE_INC + NAME_RESPONSE_TRAIN_CYCLE_INC * cycle
            name_factor = NAME_RESPONSE_TRAIN_NAME_FACTOR + (1.0 - NAME_RESPONSE_TRAIN_NAME_FACTOR) * name_resonance
            state_factor = NAME_RESPONSE_STATE_FACTORS.get(consciousness_state, NAME_RESPONSE_STATE_FACTORS['default'])
            heartbeat_factor = NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR + NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT * heartbeat_entrainment

            # Edge of Chaos Boost (placeholder)
            eoc_boost = 1.0
            # if EDGE_OF_CHAOS_AVAILABLE:
            #    # Call edge of chaos function if condition met
            #    # e.g., if state_factor > 0.8: eoc_boost = apply_eoc_boost(soul_spark, 1.1)
            #    pass

            cycle_increase = base_increase * name_factor * state_factor * heartbeat_factor * eoc_boost
            current_response = min(1.0, current_response + cycle_increase)
            logger.debug(f"  Cycle {cycle+1}: Response level now {current_response:.4f} (+{cycle_increase:.4f})")

        # --- Update SoulSpark ---
        setattr(soul_spark, 'response_level', float(current_response))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Name response trained. Final level: {current_response:.4f}")

    except Exception as e: logger.error(f"Error training name response: {e}", exc_info=True); raise RuntimeError("Name response training failed.") from e


def apply_heartbeat_entrainment(soul_spark: SoulSpark, bpm: float = 60.0, duration: float = 60.0) -> None:
    """Applies heartbeat entrainment using constants. Modifies SoulSpark."""
    logger.info(f"Applying heartbeat entrainment ({bpm} BPM, {duration}s) for soul {soul_spark.spark_id}...")
    if bpm <= 0 or duration <= 0: raise ValueError("BPM and duration must be positive.")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    voice_frequency = getattr(soul_spark, 'voice_frequency')

    try:
        current_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0)
        beat_freq = bpm / 60.0

        beat_resonance = _calculate_frequency_resonance(beat_freq, voice_frequency) # Fails hard
        # Use constants
        duration_factor = min(1.0, duration / HEARTBEAT_ENTRAINMENT_DURATION_CAP)
        entrainment_increase = beat_resonance * duration_factor * HEARTBEAT_ENTRAINMENT_INC_FACTOR
        new_entrainment = min(1.0, current_entrainment + entrainment_increase)

        # --- Update SoulSpark ---
        setattr(soul_spark, 'heartbeat_entrainment', float(new_entrainment))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Heartbeat entrainment applied. New level: {new_entrainment:.4f}")

    except Exception as e: logger.error(f"Error applying heartbeat entrainment: {e}", exc_info=True); raise RuntimeError("Heartbeat entrainment failed.") from e


def activate_love_resonance(soul_spark: SoulSpark, cycles: int = 7) -> None:
    """Activates love resonance using constants. Modifies SoulSpark."""
    logger.info(f"Activating love resonance for soul {soul_spark.spark_id} ({cycles} cycles)...")
    if not isinstance(cycles, int) or cycles <= 0: raise ValueError("Cycles must be positive.")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    soul_frequency = getattr(soul_spark, 'soul_frequency'); consciousness_state = getattr(soul_spark, 'consciousness_state')
    heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment')

    try:
        emotional_resonance = getattr(soul_spark, 'emotional_resonance') # Assumed dict exists
        current_love = emotional_resonance.get('love', 0.0)
        love_freq = LOVE_RESONANCE_FREQ # Use constant
        if love_freq <= FLOAT_EPSILON: raise ValueError("Love frequency constant is invalid.")

        for cycle in range(cycles):
            # Use constants
            cycle_factor = 1.0 - LOVE_RESONANCE_CYCLE_FACTOR_DECAY * (cycle / max(1, cycles))
            base_increase = LOVE_RESONANCE_BASE_INC * cycle_factor
            state_factor = LOVE_RESONANCE_STATE_WEIGHT.get(consciousness_state, LOVE_RESONANCE_STATE_WEIGHT['default'])
            freq_resonance = _calculate_frequency_resonance(love_freq, soul_frequency)
            heartbeat_factor = LOVE_RESONANCE_HEARTBEAT_WEIGHT + LOVE_RESONANCE_HEARTBEAT_SCALE * heartbeat_entrainment
            increase = base_increase * state_factor * (LOVE_RESONANCE_FREQ_RES_WEIGHT * freq_resonance) * heartbeat_factor
            current_love = min(1.0, current_love + increase)
            logger.debug(f"  Love cycle {cycle+1}: resonance now {current_love:.4f} (+{increase:.4f})")

        # --- Update SoulSpark ---
        emotional_resonance['love'] = float(current_love)
        for emotion in ['joy', 'peace', 'harmony', 'compassion']: # Boost others based on love using constant
            current = emotional_resonance.get(emotion, 0.0)
            emotional_resonance[emotion] = float(min(1.0, current + LOVE_RESONANCE_EMOTION_BOOST_FACTOR * current_love))
        # setattr(soul_spark, 'emotional_resonance', emotional_resonance) # Dict modified in place
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Love resonance activated. Final level: {current_love:.4f}")

    except Exception as e: logger.error(f"Error activating love resonance: {e}", exc_info=True); raise RuntimeError("Love resonance activation failed.") from e


def apply_sacred_geometry(soul_spark: SoulSpark, stages: int = 5) -> None:
    """Applies sacred geometry using constants. Modifies SoulSpark."""
    logger.info(f"Applying sacred geometry to crystallize identity ({stages} stages)...")
    if not isinstance(stages, int) or stages <= 0: raise ValueError("Stages must be positive.")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    sephiroth_aspect = getattr(soul_spark, 'sephiroth_aspect'); elemental_affinity = getattr(soul_spark, 'elemental_affinity')
    name_resonance = getattr(soul_spark, 'name_resonance')

    try:
        geometries = SACRED_GEOMETRY_STAGES # Use constant
        actual_stages = min(stages, len(geometries))
        current_level = getattr(soul_spark, 'crystallization_level', 0.0)
        dominant_geometry = None; max_increase = -1.0
        logger.debug(f"  Initial crystallization level: {current_level:.4f}")

        for i in range(actual_stages):
            geometry = geometries[i]
            # Use constants
            stage_factor = SACRED_GEOMETRY_STAGE_FACTOR_BASE + SACRED_GEOMETRY_STAGE_FACTOR_SCALE * (i / max(1, actual_stages - 1))
            base_increase = SACRED_GEOMETRY_BASE_INC_BASE + SACRED_GEOMETRY_BASE_INC_SCALE * i

            # Sephiroth resonance factor (using constant map)
            seph_geom = aspect_dictionary.get_aspects(sephiroth_aspect).get('geometric_correspondence')
            seph_weight = SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT.get(seph_geom, SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT['default'])
            sephiroth_factor = seph_weight # Simplified approach

            # Elemental resonance factor (using constant map)
            elem_weight = SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT.get(elemental_affinity, SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT['default'])
            elemental_factor = elem_weight # Simplified approach

            # Name resonance factor (using constants)
            fib_idx = min(i, len(FIBONACCI_SEQUENCE) - 1); fib_val = FIBONACCI_SEQUENCE[fib_idx]
            fib_norm_idx = min(SACRED_GEOMETRY_FIB_MAX_IDX, len(FIBONACCI_SEQUENCE) - 1); fib_norm = FIBONACCI_SEQUENCE[fib_norm_idx]
            name_factor = SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE + SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE * name_resonance * (fib_val / max(1, fib_norm))

            increase = base_increase * stage_factor * sephiroth_factor * elemental_factor * name_factor
            current_level = min(1.0, current_level + increase)
            logger.debug(f"  Stage {i+1} ({geometry}): crystallization level {current_level:.4f} (+{increase:.4f})")
            if increase > max_increase: dominant_geometry = geometry; max_increase = increase

        # --- Update SoulSpark ---
        setattr(soul_spark, 'crystallization_level', float(current_level))
        setattr(soul_spark, 'sacred_geometry_imprint', dominant_geometry)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Sacred geometry applied. Dominant imprint: {dominant_geometry}, Final Crystallization: {current_level:.4f}")

    except Exception as e: logger.error(f"Error applying sacred geometry: {e}", exc_info=True); raise RuntimeError("Sacred geometry application failed.") from e


def calculate_attribute_coherence(soul_spark: SoulSpark) -> None:
    """Calculates attribute coherence using constants. Modifies SoulSpark."""
    logger.info(f"Calculating attribute coherence for soul {soul_spark.spark_id}...")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist
    try:
        attributes = { # Gather key attributes from soul_spark
            'name_resonance': getattr(soul_spark,'name_resonance'), 'response_level': getattr(soul_spark,'response_level'),
            'state_stability': getattr(soul_spark,'state_stability'), 'crystallization_level': getattr(soul_spark,'crystallization_level'),
            'heartbeat_entrainment': getattr(soul_spark,'heartbeat_entrainment'),
            'emotional_resonance_avg': np.mean(list(getattr(soul_spark,'emotional_resonance').values())) if getattr(soul_spark,'emotional_resonance') else 0.0,
            'creator_connection': getattr(soul_spark,'creator_connection_strength', 0.0) }

        attr_values = [v for v in attributes.values() if isinstance(v, (int, float)) and np.isfinite(v)]
        if len(attr_values) < 2: coherence = 0.5 # Default if not enough data
        else:
            avg_level = np.mean(attr_values); std_dev = np.std(attr_values)
            # Use constant for scaling
            coherence = max(0.0, 1.0 - min(1.0, std_dev * ATTRIBUTE_COHERENCE_STD_DEV_SCALE))

        # --- Update SoulSpark ---
        setattr(soul_spark, 'attribute_coherence', float(coherence))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Attribute coherence calculated: {coherence:.4f}")
        logger.debug(f"  Attributes used: {attributes}")

    except Exception as e: logger.error(f"Error calculating attribute coherence: {e}", exc_info=True); raise RuntimeError("Attribute coherence calculation failed.") from e


def verify_identity_crystallization(soul_spark: SoulSpark, threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD) -> Tuple[bool, Dict[str, Any]]:
    """Verifies identity crystallization using constants. Modifies SoulSpark."""
    logger.info(f"Verifying identity crystallization for soul {soul_spark.spark_id} (Threshold: {threshold})...")
    if not isinstance(threshold, (int, float)) or not (0.0 < threshold <= 1.0): raise ValueError("Threshold must be between 0 and 1.")
    _ensure_soul_properties(soul_spark) # Ensure needed attrs exist

    try:
        # Check required attributes using constant list
        missing_attributes = [attr for attr in CRYSTALLIZATION_REQUIRED_ATTRIBUTES if getattr(soul_spark, attr, None) is None]
        attr_presence = (len(CRYSTALLIZATION_REQUIRED_ATTRIBUTES) - len(missing_attributes)) / len(CRYSTALLIZATION_REQUIRED_ATTRIBUTES)

        # Gather components from soul_spark using constant weights
        component_metrics = { # Gather components directly from soul_spark
            'name_resonance': getattr(soul_spark,'name_resonance'), 'response_level': getattr(soul_spark,'response_level'),
            'state_stability': getattr(soul_spark,'state_stability'), 'crystallization_level': getattr(soul_spark,'crystallization_level'),
            'attribute_coherence': getattr(soul_spark,'attribute_coherence'), 'attribute_presence': attr_presence,
            'emotional_resonance': np.mean(list(getattr(soul_spark,'emotional_resonance').values())) if getattr(soul_spark,'emotional_resonance') else 0.0 }

        total_crystallization = 0.0; total_weight = 0.0
        for component, weight in CRYSTALLIZATION_COMPONENT_WEIGHTS.items():
            value = component_metrics.get(component, 0.0)
            if isinstance(value, (int, float)) and np.isfinite(value): # Check value validity
                total_crystallization += value * weight
                total_weight += weight
        if total_weight > FLOAT_EPSILON: total_crystallization /= total_weight # Normalize
        else: total_crystallization = 0.0

        # Check against thresholds using constants
        is_crystallized = total_crystallization >= threshold and attr_presence >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD

        # --- Update SoulSpark ---
        setattr(soul_spark, 'identity_crystallized', is_crystallized)
        setattr(soul_spark, 'crystallization_level', float(total_crystallization)) # Store the calculated level
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # Prepare metrics dictionary
        crystallization_metrics_result = {
            'total_crystallization': float(total_crystallization), 'threshold': threshold, 'is_crystallized': is_crystallized,
            'components': component_metrics, 'weights': CRYSTALLIZATION_COMPONENT_WEIGHTS, 'missing_attributes': missing_attributes }
        # Store detailed verification metrics on soul? Optional.
        # setattr(soul_spark, 'identity_metrics', crystallization_metrics_result)

        logger.info(f"Identity crystallization check: Level={total_crystallization:.4f}, Threshold={threshold}, Passed={is_crystallized}")
        if missing_attributes: logger.warning(f"  Missing attributes for check: {', '.join(missing_attributes)}")

        return is_crystallized, crystallization_metrics_result

    except Exception as e: logger.error(f"Error verifying identity crystallization: {e}", exc_info=True); raise RuntimeError("Identity verification failed.") from e


# --- Orchestration Function ---

def perform_identity_crystallization(soul_spark: SoulSpark, life_cord_data: Dict[str, Any],
                                    specified_name: Optional[str] = None,
                                    train_cycles: int = 5, entrainment_bpm: float = 72.0,
                                    entrainment_duration: float = 120.0, love_cycles: int = 7,
                                    geometry_stages: int = 5,
                                    crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD
                                    ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs the complete identity crystallization process. Modifies SoulSpark. Fails hard. Uses constants.

    Args:
        soul_spark (SoulSpark): The soul spark object (will be modified).
        life_cord_data (Dict[str, Any]): Data dictionary for the formed life cord (may be needed for context).
        specified_name (Optional[str]): Optional specific name for the soul.
        train_cycles (int): Number of name response training cycles.
        entrainment_bpm (float): BPM for heartbeat entrainment.
        entrainment_duration (float): Duration for heartbeat entrainment.
        love_cycles (int): Number of love resonance activation cycles.
        geometry_stages (int): Number of sacred geometry stages to apply.
        crystallization_threshold (float): Threshold for successful crystallization.

    Returns:
        Tuple[SoulSpark, Dict[str, Any]]: A tuple containing:
            - The modified SoulSpark object with crystallized identity.
            - overall_metrics (Dict): Summary metrics for the crystallization process.

    Raises:
        TypeError: If soul_spark is not a SoulSpark instance or life_cord_data invalid.
        ValueError: If parameters are invalid or prerequisites not met.
        RuntimeError: If any phase fails critically.
    """
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark must be a SoulSpark instance.")
    if not isinstance(life_cord_data, dict): raise TypeError("life_cord_data must be a dictionary.")
    # Add more validation for parameters
    if not isinstance(train_cycles, int) or train_cycles < 0: raise ValueError("train_cycles must be non-negative.")
    if not isinstance(love_cycles, int) or love_cycles < 0: raise ValueError("love_cycles must be non-negative.")
    if not isinstance(geometry_stages, int) or geometry_stages < 0: raise ValueError("geometry_stages must be non-negative.")
    if not (0.0 < crystallization_threshold <= 1.0): raise ValueError("crystallization_threshold must be between 0 and 1.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown')
    logger.info(f"--- Starting Identity Crystallization for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps': {}}

    try:
        # --- Ensure Soul Properties ---
        # Ensure soul has necessary attributes before starting
        _ensure_soul_properties(soul_spark)
        logger.info("Soul properties ensured.")

        # --- Store Initial State ---
        initial_state = soul_spark.get_spark_metrics()['identity'] # Get initial identity metrics
        logger.info(f"Initial Identity State: Name={initial_state.get('name')}, CrystLevel={initial_state.get('crystallization_level', 0.0):.4f}")


        # --- Run Phases (Fail hard within each) ---
        logger.info("Step 1: Assign Name...")
        assign_name(soul_spark, specified_name)
        process_metrics_summary['steps']['name'] = {'name': soul_spark.name, 'gematria': soul_spark.gematria_value, 'resonance': soul_spark.name_resonance, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 2: Assign Voice Frequency...")
        assign_voice_frequency(soul_spark)
        process_metrics_summary['steps']['voice'] = {'frequency': soul_spark.voice_frequency, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 3: Apply Heartbeat Entrainment...")
        apply_heartbeat_entrainment(soul_spark, entrainment_bpm, entrainment_duration)
        process_metrics_summary['steps']['heartbeat'] = {'level': soul_spark.heartbeat_entrainment, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 4: Train Name Response...")
        train_name_response(soul_spark, train_cycles)
        process_metrics_summary['steps']['response_train'] = {'level': soul_spark.response_level, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 5: Determine Soul Color...")
        determine_soul_color(soul_spark)
        process_metrics_summary['steps']['color'] = {'color': soul_spark.soul_color, 'frequency': soul_spark.color_frequency, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 6: Determine Soul Frequency...")
        determine_soul_frequency(soul_spark)
        process_metrics_summary['steps']['soul_freq'] = {'frequency': soul_spark.soul_frequency, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 7: Identify Primary Sephiroth...")
        identify_primary_sephiroth(soul_spark)
        process_metrics_summary['steps']['sephiroth'] = {'aspect': soul_spark.sephiroth_aspect, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 8: Determine Elemental Affinity...")
        determine_elemental_affinity(soul_spark)
        process_metrics_summary['steps']['element'] = {'affinity': soul_spark.elemental_affinity, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 9: Assign Platonic Symbol...")
        assign_platonic_symbol(soul_spark)
        process_metrics_summary['steps']['platonic'] = {'symbol': soul_spark.platonic_symbol, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 10: Activate Love Resonance...")
        activate_love_resonance(soul_spark, love_cycles)
        process_metrics_summary['steps']['love'] = {'level': soul_spark.emotional_resonance.get('love',0.0), 'timestamp': datetime.now().isoformat()}

        logger.info("Step 11: Apply Sacred Geometry...")
        apply_sacred_geometry(soul_spark, geometry_stages)
        process_metrics_summary['steps']['geometry'] = {'level': soul_spark.crystallization_level, 'dominant_geo': getattr(soul_spark, 'sacred_geometry_imprint', None), 'timestamp': datetime.now().isoformat()}

        logger.info("Step 12: Calculate Attribute Coherence...")
        calculate_attribute_coherence(soul_spark) # Modifies soul_spark
        process_metrics_summary['steps']['coherence'] = {'level': soul_spark.attribute_coherence, 'timestamp': datetime.now().isoformat()}

        logger.info("Step 13: Verify Crystallization...")
        is_crystallized, verification_metrics = verify_identity_crystallization(soul_spark, crystallization_threshold)
        process_metrics_summary['steps']['verification'] = verification_metrics # Store full verification details

        if not is_crystallized:
             # Log memory echo for failure
             logger.info(f"Memory echo created: Identity crystallization failed for soul {spark_id}.")
             if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
                 soul_spark.memory_echoes.append(f"Identity crystallization failed @ {datetime.now().isoformat()}")
             raise RuntimeError(f"Identity crystallization failed: Level {verification_metrics['total_crystallization']:.4f} did not meet threshold {crystallization_threshold}.")

        # --- Final Update ---
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        # Log memory echo for success
        logger.info(f"Memory echo created: Identity crystallized for soul {spark_id}.")
        if hasattr(soul_spark, 'memory_echoes') and isinstance(soul_spark.memory_echoes, list):
            soul_spark.memory_echoes.append(f"Identity crystallized as '{soul_spark.name}' @ {getattr(soul_spark, 'last_modified')}")

        # --- Compile Overall Metrics ---
        end_time_iso = datetime.now().isoformat(); end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = soul_spark.get_spark_metrics()['identity'] # Get final identity metrics

        overall_metrics = {
            'action': 'identity_crystallization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'final_crystallization_level': verification_metrics['total_crystallization'],
            'success': True,
            # 'steps_metrics': process_metrics_summary['steps'] # Optional detail
        }
        try: metrics.record_metrics('identity_crystallization_summary', overall_metrics)
        except Exception as e: logger.error(f"Failed to record summary metrics for identity crystallization: {e}")

        logger.info(f"--- Identity Crystallization Completed Successfully for Soul {spark_id} ---")
        logger.info(f"Duration: {overall_metrics['duration_seconds']:.2f}s")
        logger.info(f"Final Name: {final_state.get('name')}")
        logger.info(f"Final Crystallization Level: {final_state.get('crystallization_level'):.4f}")

        return soul_spark, overall_metrics

    except Exception as e:
        end_time_iso = datetime.now().isoformat()
        logger.critical(f"Identity crystallization process failed critically for soul {spark_id}: {e}", exc_info=True)
        failed_step = "unknown"; steps_completed = list(process_metrics_summary['steps'].keys())
        if steps_completed: failed_step = steps_completed[-1]
        # Mark soul as failed this stage
        setattr(soul_spark, "identity_crystallized", False)
        if METRICS_AVAILABLE:
             try: metrics.record_metrics('identity_crystallization_summary', {
                  'action': 'identity_crystallization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
                  'duration_seconds': (datetime.fromisoformat(end_time_iso) - datetime.fromisoformat(start_time_iso)).total_seconds(),
                  'success': False, 'error': str(e), 'failed_step': failed_step })
             except Exception as metric_e: logger.error(f"Failed to record failure metrics: {metric_e}")
        raise RuntimeError(f"Identity crystallization process failed at step '{failed_step}'.") from e


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Identity Crystallization Module Example (with Constants)...")
    if not DEPENDENCIES_AVAILABLE:
         print("ERROR: Core dependencies not available. Cannot run example.")
    else:
        test_soul = SoulSpark()
        test_soul.spark_id="test_identity_const_002"
        # Set state *after* earth harmonization
        test_soul.stability = 0.92
        test_soul.coherence = 0.91
        test_soul.frequency = 432.0
        test_soul.formation_complete = True
        test_soul.harmonically_strengthened = True
        test_soul.cord_formation_complete = True
        test_soul.earth_harmonized = True # Prerequisite
        test_soul.cord_integrity = 0.95
        test_soul.earth_resonance = 0.90
        test_soul.aspects = {'unity': {'strength': 0.9}, 'love': {'strength': 0.8}, 'wisdom': {'strength': 0.7}}
        test_soul.consciousness_state = 'aware'
        test_soul.creator_connection_strength = 0.9
        test_soul.last_modified = datetime.now().isoformat()
        test_soul.memory_echoes = []

        # Dummy life cord data
        dummy_life_cord_data = { "cord_integrity": 0.95, "bandwidth": 600.0 }

        print(f"\nInitial Soul State ({test_soul.spark_id}):")
        print(f"  Stability: {test_soul.stability:.4f}")
        print(f"  Coherence: {test_soul.coherence:.4f}")
        print(f"  Earth Harmonized: {getattr(test_soul, 'earth_harmonized', False)}")
        print(f"  Aspects Count: {len(getattr(test_soul, 'aspects', {}))}")

        try:
            print("\n--- Running Identity Crystallization Process ---")
            modified_soul, summary_metrics_result = perform_identity_crystallization(
                soul_spark=test_soul,
                life_cord_data=dummy_life_cord_data,
                train_cycles=6,
                love_cycles=9,
                geometry_stages=5,
                crystallization_threshold=IDENTITY_CRYSTALLIZATION_THRESHOLD # Use constant
            )

            print("\n--- Crystallization Complete ---")
            print("Final Soul Identity Summary:")
            print(f"  ID: {modified_soul.spark_id}")
            print(f"  Name: {getattr(modified_soul, 'name', 'N/A')}")
            print(f"  Crystallization Level: {getattr(modified_soul, 'crystallization_level', 'N/A'):.4f}")
            print(f"  Identity Crystallized Flag: {getattr(modified_soul, 'identity_crystallized', False)}")
            print(f"  Final Stability: {getattr(modified_soul, 'stability', 'N/A'):.4f}")
            print(f"  Memory Echoes: {getattr(modified_soul, 'memory_echoes', [])}")

            print("\nOverall Process Metrics:")
            print(f"  Duration: {summary_metrics_result.get('duration_seconds', 'N/A'):.2f}s")
            print(f"  Success: {summary_metrics_result.get('success')}")

        except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as e:
            print(f"\n--- ERROR during Identity Crystallization Example ---")
            print(f"An error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        except Exception as e:
            print(f"\n--- UNEXPECTED ERROR during Identity Crystallization Example ---")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\nIdentity Crystallization Module Example Finished.")

# --- END OF FILE identity_crystallization.py ---




