# --- START OF FILE src/stage_1/soul_formation/identity_crystallization.py ---

"""
Identity Crystallization Functions (Refactored V4.3.8 - Principle-Driven)

Crystallizes identity after Earth attunement. Includes Name (user input),
Voice, Color, Affinities (Seph, Elem, Platonic), Astrological Signature (NEW).
Heartbeat/Love cycles boost Harmony factor. Sacred Geometry imprint reinforces
internal factors (P.Coh, Phi, Harmony, Torus) leading to emergent S/C increase.
Modifies SoulSpark directly. Uses SU/CU prerequisites. Hard fails.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime, timedelta # Added timedelta for random date
import time
import random
import uuid
from typing import Dict, List, Any, Tuple, Optional
from constants.constants import *
# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
try:
    from constants.constants import *
    # # Add NEW Constants for Astrology if not already present
    # if 'PLANETARY_FREQUENCIES' not in globals(): raise NameError("PLANETARY_FREQUENCIES not found")
    # if 'ZODIAC_SIGNS' not in globals(): raise NameError("ZODIAC_SIGNS not found")
    # if 'ZODIAC_TRAITS' not in globals(): raise NameError("ZODIAC_TRAITS not found")
    # ASTROLOGY_MAX_POSITIVE_TRAITS = 5
    # ASTROLOGY_MAX_NEGATIVE_TRAITS = 2

except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"CRITICAL ERROR: Could not import constants: {e}. Identity Crystallization cannot function.")
    # Define minimal fallbacks for parsing - CRITICAL: ADD NEW ASTROLOGY CONSTANTS HERE
    # FLAG_READY_FOR_IDENTITY="ready_for_identity"; IDENTITY_STABILITY_THRESHOLD_SU=85.0; IDENTITY_COHERENCE_THRESHOLD_CU=85.0; IDENTITY_EARTH_RESONANCE_THRESHOLD=0.75; FLAG_IDENTITY_CRYSTALLIZED="identity_crystallized"; FLOAT_EPSILON=1e-9; MAX_STABILITY_SU=100.0; MAX_COHERENCE_CU=100.0; GOLDEN_RATIO = 1.618; NAME_GEMATRIA_RESONANT_NUMBERS=[3,7,9,11,13,22]; NAME_RESONANCE_BASE=0.1; NAME_RESONANCE_WEIGHT_VOWEL=0.3; NAME_RESONANCE_WEIGHT_LETTER=0.2; NAME_RESONANCE_WEIGHT_GEMATRIA=0.4; VOICE_FREQ_BASE=220.0; VOICE_FREQ_ADJ_LENGTH_FACTOR=-50.0; VOICE_FREQ_ADJ_VOWEL_FACTOR=80.0; VOICE_FREQ_ADJ_GEMATRIA_FACTOR=40.0; VOICE_FREQ_ADJ_RESONANCE_FACTOR=60.0; VOICE_FREQ_ADJ_YINYANG_FACTOR=-70.0; VOICE_FREQ_MIN_HZ=80.0; VOICE_FREQ_MAX_HZ=600.0; VOICE_FREQ_SOLFEGGIO_SNAP_HZ=5.0; SOLFEGGIO_FREQUENCIES={}; COLOR_SPECTRUM={}; COLOR_FREQ_DEFAULT=500.0; HEARTBEAT_ENTRAINMENT_DURATION_CAP=300.0; HEARTBEAT_ENTRAINMENT_INC_FACTOR=0.05; NAME_RESPONSE_TRAIN_BASE_INC=0.02; NAME_RESPONSE_TRAIN_CYCLE_INC=0.005; NAME_RESPONSE_TRAIN_NAME_FACTOR=0.5; NAME_RESPONSE_STATE_FACTORS={'default':0.8}; NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR=0.8; NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT=0.4; SEPHIROTH_ASPECT_DEFAULT="tiphareth"; SEPHIROTH_AFFINITY_GEMATRIA_RANGES={}; SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT=0.2; SEPHIROTH_AFFINITY_COLOR_MAP={}; SEPHIROTH_AFFINITY_COLOR_WEIGHT=0.25; SEPHIROTH_AFFINITY_STATE_MAP={}; SEPHIROTH_AFFINITY_STATE_WEIGHT=0.15; RESONANCE_INTEGER_RATIO_TOLERANCE=0.02; RESONANCE_PHI_RATIO_TOLERANCE=0.03; SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD=0.8; SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD=0.35; SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD=0.65; SEPHIROTH_AFFINITY_YIN_SEPHIROTH=[]; SEPHIROTH_AFFINITY_YANG_SEPHIROTH=[]; SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH=[]; SEPHIROTH_AFFINITY_YINYANG_WEIGHT=0.1; SEPHIROTH_AFFINITY_BALANCE_WEIGHT=0.1; ELEMENTAL_AFFINITY_DEFAULT="aether"; ELEMENTAL_AFFINITY_VOWEL_THRESHOLD=0.55; ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD=0.7; ELEMENTAL_AFFINITY_VOWEL_MAP={'air':0.2}; ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT=0.3; ELEMENTAL_AFFINITY_COLOR_MAP={}; ELEMENTAL_AFFINITY_COLOR_WEIGHT=0.2; ELEMENTAL_AFFINITY_STATE_MAP={}; ELEMENTAL_AFFINITY_STATE_WEIGHT=0.1; ELEMENTAL_AFFINITY_FREQ_RANGES=[]; ELEMENTAL_AFFINITY_FREQ_WEIGHT=0.1; LOVE_RESONANCE_FREQ=528.0; PLATONIC_ELEMENT_MAP={}; PLATONIC_SOLIDS=[]; PLATONIC_DEFAULT_GEMATRIA_RANGE=50; NAME_RESPONSE_TRAIN_BASE_INC=0.02; NAME_RESPONSE_TRAIN_CYCLE_INC=0.005; NAME_RESPONSE_TRAIN_NAME_FACTOR=0.5; NAME_RESPONSE_STATE_FACTORS={'default':0.8}; NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR=0.8; NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT=0.4; HEARTBEAT_ENTRAINMENT_INC_FACTOR=0.05; HEARTBEAT_ENTRAINMENT_DURATION_CAP=300.0; LOVE_RESONANCE_BASE_INC=0.03; LOVE_RESONANCE_CYCLE_FACTOR_DECAY=0.3; LOVE_RESONANCE_STATE_WEIGHT={'default':0.7}; LOVE_RESONANCE_FREQ_RES_WEIGHT=0.5; LOVE_RESONANCE_HEARTBEAT_WEIGHT=0.3; LOVE_RESONANCE_HEARTBEAT_SCALE=0.4; LOVE_RESONANCE_EMOTION_BOOST_FACTOR=0.1; SACRED_GEOMETRY_STAGES=["seed_of_life"]; SACRED_GEOMETRY_STAGE_FACTOR_BASE=1.0; SACRED_GEOMETRY_STAGE_FACTOR_SCALE=0.5; SACRED_GEOMETRY_BASE_INC_BASE=0.01; SACRED_GEOMETRY_BASE_INC_SCALE=0.005; SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT={'default':1.0}; SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT={'default':1.0}; SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE=0.8; SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE=0.4; FIBONACCI_SEQUENCE=[1,1,2]; SACRED_GEOMETRY_FIB_MAX_IDX=1; GEOMETRY_EFFECTS={}; DEFAULT_GEOMETRY_EFFECT={'pattern_coherence_boost':0.01,'phi_resonance_boost':0.01,'harmony_boost':0.01,'toroidal_flow_strength_boost':0.01}; ATTRIBUTE_COHERENCE_STD_DEV_SCALE=2.0; IDENTITY_CRYSTALLIZATION_THRESHOLD=0.85; CRYSTALLIZATION_REQUIRED_ATTRIBUTES=[]; CRYSTALLIZATION_COMPONENT_WEIGHTS={}; CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD=0.9; FLAG_READY_FOR_BIRTH="ready_for_birth"
    # # Add NEW Astrology constants here
    # PLANETARY_FREQUENCIES = {'Sun': 126.22} # Example
    # ZODIAC_SIGNS = [{"name": "Aries", "symbol": "♈", "start_date": "April 19", "end_date": "May 13"}] # Example
    # ZODIAC_TRAITS = {"Aries": {"positive": ["Courageous"], "negative": ["Impulsive"]}} # Example
    # ASTROLOGY_MAX_POSITIVE_TRAITS = 5
    # ASTROLOGY_MAX_NEGATIVE_TRAITS = 2
    raise ImportError(f"Essential constants missing or incomplete: {e}") from e

# --- Dependency Imports ---
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    try: from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
    except ImportError: aspect_dictionary = None
    if aspect_dictionary is None: raise ImportError("Aspect Dictionary failed to initialize.")
    from .creator_entanglement import calculate_resonance
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

def _check_prerequisites(soul_spark: SoulSpark) -> bool:
    """ Checks prerequisites using SU/CU thresholds and 0-1 factor threshold. Raises ValueError on failure. """
    logger.debug(f"Checking identity prerequisites for soul {soul_spark.spark_id}...")
    if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")

    # 1. Stage Flag Check (Use FLAG_EARTH_ATTUNED)
    if not getattr(soul_spark, FLAG_EARTH_ATTUNED, False):
        msg = f"Prerequisite failed: Soul not marked {FLAG_EARTH_ATTUNED}."
        logger.error(msg); raise ValueError(msg)

    # 2. State Thresholds
    stability_su = getattr(soul_spark, 'stability', -1.0)
    coherence_cu = getattr(soul_spark, 'coherence', -1.0)
    earth_resonance = getattr(soul_spark, 'earth_resonance', -1.0)
    if stability_su < 0 or coherence_cu < 0 or earth_resonance < 0:
        msg = "Prerequisite failed: Soul missing stability, coherence, or earth_resonance."; logger.error(msg); raise AttributeError(msg)

    if stability_su < IDENTITY_STABILITY_THRESHOLD_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {IDENTITY_STABILITY_THRESHOLD_SU} SU."; logger.error(msg); raise ValueError(msg)
    if coherence_cu < IDENTITY_COHERENCE_THRESHOLD_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {IDENTITY_COHERENCE_THRESHOLD_CU} CU."; logger.error(msg); raise ValueError(msg)
    if earth_resonance < IDENTITY_EARTH_RESONANCE_THRESHOLD:
        msg = f"Prerequisite failed: Earth Resonance ({earth_resonance:.3f}) < {IDENTITY_EARTH_RESONANCE_THRESHOLD}."; logger.error(msg); raise ValueError(msg)

    # 3. Essential Attributes for Calculation
    if getattr(soul_spark, 'soul_color', None) is None:
        msg = "Prerequisite failed: SoulSpark missing 'soul_color'."; logger.error(msg); raise AttributeError(msg)
    if getattr(soul_spark, 'soul_frequency', 0.0) <= FLOAT_EPSILON:
        msg = f"Prerequisite failed: SoulSpark missing valid 'soul_frequency' ({getattr(soul_spark, 'soul_frequency', 0.0)})."; logger.error(msg); raise ValueError(msg)
    if getattr(soul_spark, 'frequency', 0.0) <= FLOAT_EPSILON:
        msg = f"Prerequisite failed: SoulSpark missing valid base 'frequency'."; logger.error(msg); raise ValueError(msg)

    if getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_IDENTITY_CRYSTALLIZED}. Re-running.")

    logger.debug("Identity prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Raises error if missing/invalid. """
    logger.debug(f"Ensuring properties for identity process (Soul {soul_spark.spark_id})...")
    required = ['stability', 'coherence', 'earth_resonance', 'soul_color', 'soul_frequency', 'frequency',
                'yin_yang_balance', 'aspects', 'consciousness_state', 'phi_resonance',
                'pattern_coherence', 'harmony', 'toroidal_flow_strength'] # Added torus
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
        "emotional_resonance": {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0},
        "crystallization_level": 0.0, "attribute_coherence": 0.0,
        "identity_metrics": None, "sacred_geometry_imprint": None,
        "conceptual_birth_datetime": None, "zodiac_sign": None, "astrological_traits": None # Added Astrology
    }
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
             if attr == 'emotional_resonance': setattr(soul_spark, attr, default.copy())
             elif attr == 'astrological_traits': setattr(soul_spark, attr, {}) # Init as empty dict
             else: setattr(soul_spark, attr, default)

    if hasattr(soul_spark, '_validate_attributes'): soul_spark._validate_attributes()
    logger.debug("Soul properties ensured for Identity Crystallization.")

# --- Calculation Helpers ---
def _calculate_gematria(name: str) -> int:
    # ... (as before) ...
    if not isinstance(name, str): raise TypeError("Name must be a string.")
    return sum(ord(char) - ord('a') + 1 for char in name.lower() if 'a' <= char <= 'z')

def _calculate_name_resonance(name: str, gematria: int) -> float:
    # ... (as before) ...
    if not name: return 0.0
    try:
        vowels=sum(1 for c in name.lower() if c in 'aeiouy'); consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz'); total_letters=vowels+consonants
        if total_letters == 0: return 0.0
        vowel_ratio=vowels/total_letters; phi_inv = 1.0 / GOLDEN_RATIO; vowel_factor = max(0.0, 1.0 - abs(vowel_ratio - phi_inv) * 2.5)
        unique_letters = len(set(c for c in name.lower() if 'a' <= c <= 'z')); letter_factor = unique_letters / max(1, len(name))
        gematria_factor = 0.5
        if gematria > 0:
            for num in NAME_GEMATRIA_RESONANT_NUMBERS:
                 if gematria % num == 0: gematria_factor = 0.9; break
        total_weight = (NAME_RESONANCE_WEIGHT_VOWEL + NAME_RESONANCE_WEIGHT_LETTER + NAME_RESONANCE_WEIGHT_GEMATRIA)
        if total_weight <= FLOAT_EPSILON: return NAME_RESONANCE_BASE
        resonance_contrib = (NAME_RESONANCE_WEIGHT_VOWEL * vowel_factor + NAME_RESONANCE_WEIGHT_LETTER * letter_factor + NAME_RESONANCE_WEIGHT_GEMATRIA * gematria_factor)
        resonance = NAME_RESONANCE_BASE + (1.0 - NAME_RESONANCE_BASE) * (resonance_contrib / total_weight)
        return float(max(0.0, min(1.0, resonance)))
    except Exception as e: logger.error(f"Error calculating name resonance for '{name}': {e}"); raise RuntimeError("Name resonance calc failed.") from e


# --- Core Crystallization Functions ---

def assign_name(soul_spark: SoulSpark) -> None:
    """ Assigns name via user input, calculates gematria/resonance(0-1). """
    # ... (User input logic unchanged, fails hard) ...
    logger.info("Identity Step: Assign Name...")
    name_to_use = None
    print("-" * 30+"\nIDENTITY CRYSTALLIZATION: SOUL NAMING"+"\nSoul ID: "+soul_spark.spark_id)
    print(f"  Color: {getattr(soul_spark, 'soul_color', 'N/A')}, Freq: {getattr(soul_spark, 'soul_frequency', 0.0):.1f}Hz")
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
        logger.debug(f"  Name assigned: '{name_to_use}', Gematria: {gematria}, Resonance Factor: {name_resonance:.4f}")
        setattr(soul_spark, 'name', name_to_use)
        setattr(soul_spark, 'gematria_value', gematria)
        setattr(soul_spark, 'name_resonance', name_resonance)
        timestamp = datetime.now().isoformat(); setattr(soul_spark, 'last_modified', timestamp)
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Name assigned: {name_to_use} (G:{gematria}, NR:{name_resonance:.3f})")
        logger.info(f"Name assignment complete: {name_to_use} (ResFactor: {name_resonance:.4f})")
    except Exception as e: logger.error(f"Error processing assigned name: {e}", exc_info=True); raise RuntimeError("Name processing failed.") from e

def assign_voice_frequency(soul_spark: SoulSpark) -> None:
    """ Assigns voice frequency (Hz) based on name/attributes. """
    # ... (Calculation logic unchanged, fails hard) ...
    logger.info("Identity Step: Assign Voice Frequency...")
    name = getattr(soul_spark, 'name'); gematria = getattr(soul_spark, 'gematria_value')
    name_resonance = getattr(soul_spark, 'name_resonance'); yin_yang = getattr(soul_spark, 'yin_yang_balance')
    if not name: raise ValueError("Cannot assign voice frequency without a name.")
    try:
        length_factor = len(name) / 10.0; vowels = sum(1 for c in name.lower() if c in 'aeiouy'); total_letters = len(name); vowel_ratio = vowels / max(1, total_letters); gematria_factor = (gematria % 100) / 100.0; resonance_factor = name_resonance
        voice_frequency = (VOICE_FREQ_BASE + VOICE_FREQ_ADJ_LENGTH_FACTOR * (length_factor - 0.5) + VOICE_FREQ_ADJ_VOWEL_FACTOR * (vowel_ratio - 0.5) + VOICE_FREQ_ADJ_GEMATRIA_FACTOR * (gematria_factor - 0.5) + VOICE_FREQ_ADJ_RESONANCE_FACTOR * (resonance_factor - 0.5) + VOICE_FREQ_ADJ_YINYANG_FACTOR * (yin_yang - 0.5))
        logger.debug(f"  Voice Freq Calc -> Raw={voice_frequency:.2f}")
        solfeggio_values = list(SOLFEGGIO_FREQUENCIES.values()); closest_solfeggio = min(solfeggio_values, key=lambda x: abs(x - voice_frequency)) if solfeggio_values else voice_frequency
        if abs(closest_solfeggio - voice_frequency) < VOICE_FREQ_SOLFEGGIO_SNAP_HZ: voice_frequency = closest_solfeggio; logger.debug(f"  Voice Freq Snapped to Solfeggio: {voice_frequency:.2f} Hz")
        voice_frequency = min(VOICE_FREQ_MAX_HZ, max(VOICE_FREQ_MIN_HZ, voice_frequency))
        setattr(soul_spark, 'voice_frequency', float(voice_frequency)); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Voice frequency assigned: {voice_frequency:.2f} Hz")
    except Exception as e: logger.error(f"Error assigning voice frequency: {e}", exc_info=True); raise RuntimeError("Voice frequency assignment failed.") from e


def process_soul_color(soul_spark: SoulSpark) -> None:
    """ Verifies inherent soul color, calculates color frequency (Hz). """
    # ... (Logic unchanged, fails hard) ...
    logger.info("Identity Step: Process Soul Color...")
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
        color_frequency = max(0.0, color_frequency)
        setattr(soul_spark, 'color_frequency', color_frequency); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Soul color processed: {soul_color} (Freq: {color_frequency:.2f} Hz)")
    except Exception as e: logger.error(f"Error processing soul color: {e}", exc_info=True); raise RuntimeError("Soul color processing failed.") from e

def apply_heartbeat_entrainment(soul_spark: SoulSpark, bpm: float = 72.0, duration: float = 120.0) -> None:
    """ Applies heartbeat entrainment (updates harmony factor 0-1). """
    # ... (Logic unchanged - updates harmony factor, fails hard) ...
    logger.info("Identity Step: Apply Heartbeat Entrainment...")
    if bpm <= 0 or duration < 0: raise ValueError("BPM must be positive, duration non-negative.")
    try:
        current_harmony = getattr(soul_spark, 'harmony', 0.0); voice_frequency = getattr(soul_spark, 'voice_frequency', 0.0); beat_freq = bpm / 60.0
        if beat_freq <= FLOAT_EPSILON or voice_frequency <= FLOAT_EPSILON: beat_resonance = 0.0
        else: beat_resonance = calculate_resonance(beat_freq, voice_frequency)
        duration_factor = min(1.0, duration / HEARTBEAT_ENTRAINMENT_DURATION_CAP)
        harmony_increase = beat_resonance * duration_factor * HEARTBEAT_ENTRAINMENT_INC_FACTOR
        new_harmony = min(1.0, current_harmony + harmony_increase)
        logger.debug(f"  Heartbeat Entrainment: BeatRes={beat_resonance:.3f}, DurFactor={duration_factor:.2f} -> HarmonyIncrease={harmony_increase:.4f}")
        setattr(soul_spark, 'harmony', float(new_harmony))
        setattr(soul_spark, 'heartbeat_entrainment', beat_resonance * duration_factor) # Store sync factor
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Heartbeat entrainment applied. Harmony -> {new_harmony:.4f}")
    except Exception as e: logger.error(f"Error heartbeat entrainment: {e}", exc_info=True); raise RuntimeError("Heartbeat entrainment failed.") from e

def train_name_response(soul_spark: SoulSpark, cycles: int = 5) -> None:
    """ Trains name response (updates 0-1 factor). """
    # ... (Logic unchanged, fails hard) ...
    logger.info("Identity Step: Train Name Response...")
    if not isinstance(cycles, int) or cycles < 0: raise ValueError("Cycles must be non-negative.")
    if cycles == 0: logger.info("Skipping name response training (0 cycles)."); return
    try:
        name_resonance = getattr(soul_spark, 'name_resonance'); consciousness_state = getattr(soul_spark, 'consciousness_state')
        heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment')
        current_response = getattr(soul_spark, 'response_level', 0.0)
        logger.debug(f"  Name Response Train Init: NR={name_resonance:.3f}, State={consciousness_state}, HBEnt={heartbeat_entrainment:.3f}, CurrentResp={current_response:.3f}")
        for cycle in range(cycles):
            base_increase = NAME_RESPONSE_TRAIN_BASE_INC + NAME_RESPONSE_TRAIN_CYCLE_INC * cycle
            name_factor = NAME_RESPONSE_TRAIN_NAME_FACTOR + (1.0 - NAME_RESPONSE_TRAIN_NAME_FACTOR) * name_resonance
            state_factor = NAME_RESPONSE_STATE_FACTORS.get(consciousness_state, NAME_RESPONSE_STATE_FACTORS['default'])
            heartbeat_factor = NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR + NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT * heartbeat_entrainment
            cycle_increase = base_increase * name_factor * state_factor * heartbeat_factor
            current_response = min(1.0, current_response + cycle_increase)
            logger.debug(f"    Cycle {cycle+1}: CycleInc={cycle_increase:.5f}, NewResp={current_response:.4f}")
        setattr(soul_spark, 'response_level', float(current_response))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Name response trained. Final level: {current_response:.4f}")
    except Exception as e: logger.error(f"Error training name response: {e}", exc_info=True); raise RuntimeError("Name response training failed.") from e

def identify_primary_sephiroth(soul_spark: SoulSpark) -> None:
    """ Identifies primary Sephiroth aspect based on soul state (Hz, factors). """
    # ... (Logic unchanged, hard fails on missing dict/attrs) ...
    logger.info("Identity Step: Identify Primary Sephiroth...")
    soul_frequency = getattr(soul_spark, 'soul_frequency'); soul_color = getattr(soul_spark, 'soul_color')
    consciousness_state = getattr(soul_spark, 'consciousness_state'); gematria = getattr(soul_spark, 'gematria_value')
    yin_yang = getattr(soul_spark, 'yin_yang_balance')
    if soul_frequency <= FLOAT_EPSILON or soul_color is None: raise ValueError("Missing required soul_frequency or soul_color.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        sephiroth_affinities = {}
        for gem_range, sephirah in SEPHIROTH_AFFINITY_GEMATRIA_RANGES.items():
            if gematria in gem_range: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT; break
        color_match = SEPHIROTH_AFFINITY_COLOR_MAP.get(soul_color.lower())
        if color_match: sephiroth_affinities[color_match] = sephiroth_affinities.get(color_match, 0.0) + SEPHIROTH_AFFINITY_COLOR_WEIGHT
        state_match = SEPHIROTH_AFFINITY_STATE_MAP.get(consciousness_state)
        if state_match: sephiroth_affinities[state_match] = sephiroth_affinities.get(state_match, 0.0) + SEPHIROTH_AFFINITY_STATE_WEIGHT
        for sephirah_name in aspect_dictionary.sephiroth_names:
            sephirah_freq = aspect_dictionary.get_aspects(sephirah_name).get('base_frequency', 0.0)
            if sephirah_freq > FLOAT_EPSILON:
                resonance = calculate_resonance(soul_frequency, sephirah_freq)
                if resonance > SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD: sephiroth_affinities[sephirah_name] = sephiroth_affinities.get(sephirah_name, 0.0) + resonance * 0.5
        if yin_yang < SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD:
            for sephirah in SEPHIROTH_AFFINITY_YIN_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * (1.0 - yin_yang)
        elif yin_yang > SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD:
            for sephirah in SEPHIROTH_AFFINITY_YANG_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_YINYANG_WEIGHT * yin_yang
        else:
            balance_factor = 1.0 - abs(yin_yang - 0.5) * 2
            for sephirah in SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH: sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + SEPHIROTH_AFFINITY_BALANCE_WEIGHT * balance_factor
        sephiroth_aspect = SEPHIROTH_ASPECT_DEFAULT
        if sephiroth_affinities: sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda item: item[1])[0]
        logger.debug(f"  Sephirah Affinities: {sephiroth_affinities} -> Identified: {sephiroth_aspect}")
        setattr(soul_spark, 'sephiroth_aspect', sephiroth_aspect)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Primary Sephiroth aspect identified: {sephiroth_aspect}")
    except Exception as e: logger.error(f"Error identifying Sephiroth aspect: {e}", exc_info=True); raise RuntimeError("Sephirah aspect ID failed.") from e

def determine_elemental_affinity(soul_spark: SoulSpark) -> None:
    """ Determines elemental affinity based on soul state (Hz, factors). """
    # ... (Logic unchanged, hard fails) ...
    logger.info("Identity Step: Determine Elemental Affinity...")
    name=getattr(soul_spark,'name'); seph_aspect=getattr(soul_spark,'sephiroth_aspect'); color=getattr(soul_spark,'soul_color'); state=getattr(soul_spark,'consciousness_state'); freq=getattr(soul_spark,'soul_frequency')
    if not all([name, seph_aspect, color, state]) or freq <= FLOAT_EPSILON: raise ValueError("Missing attributes for elemental affinity determination.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        elemental_affinities = {}
        vowels=sum(1 for c in name.lower() if c in 'aeiouy'); consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz'); total=vowels+consonants
        if total > 0:
             v_ratio=vowels/total; c_ratio=consonants/total; elem = 'fire' # default
             if v_ratio > ELEMENTAL_AFFINITY_VOWEL_THRESHOLD: elem = 'air'
             elif c_ratio > ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD: elem = 'earth'
             elif 0.4 <= v_ratio <= 0.6: elem = 'water'
             elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP.get(elem, 0.1)
        seph_element = aspect_dictionary.get_aspects(seph_aspect).get('element', '').lower()
        if '/' in seph_element: elements=seph_element.split('/'); weight=ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT/len(elements); [elemental_affinities.update({e: elemental_affinities.get(e,0.0)+weight}) for e in elements]
        elif seph_element: elemental_affinities[seph_element] = elemental_affinities.get(seph_element, 0.0) + ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT
        color_element = ELEMENTAL_AFFINITY_COLOR_MAP.get(color.lower())
        if color_element:
             if '/' in color_element: elements=color_element.split('/'); weight=ELEMENTAL_AFFINITY_COLOR_WEIGHT/len(elements); [elemental_affinities.update({e: elemental_affinities.get(e,0.0)+weight}) for e in elements]
             else: elemental_affinities[color_element] = elemental_affinities.get(color_element, 0.0) + ELEMENTAL_AFFINITY_COLOR_WEIGHT
        state_element = ELEMENTAL_AFFINITY_STATE_MAP.get(state)
        if state_element: elemental_affinities[state_element] = elemental_affinities.get(state_element, 0.0) + ELEMENTAL_AFFINITY_STATE_WEIGHT
        assigned_element = ELEMENTAL_AFFINITY_DEFAULT
        for upper_bound, element in ELEMENTAL_AFFINITY_FREQ_RANGES:
             if freq < upper_bound: assigned_element = element; break
        elemental_affinities[assigned_element] = elemental_affinities.get(assigned_element, 0.0) + ELEMENTAL_AFFINITY_FREQ_WEIGHT
        elemental_affinity = ELEMENTAL_AFFINITY_DEFAULT
        if elemental_affinities: elemental_affinity = max(elemental_affinities.items(), key=lambda item: item[1])[0]
        logger.debug(f"  Elemental Affinities: {elemental_affinities} -> Identified: {elemental_affinity}")
        setattr(soul_spark, 'elemental_affinity', elemental_affinity); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Elemental affinity determined: {elemental_affinity}")
    except Exception as e: logger.error(f"Error determining elemental affinity: {e}", exc_info=True); raise RuntimeError("Elemental affinity determination failed.") from e

def assign_platonic_symbol(soul_spark: SoulSpark) -> None:
    """ Assigns Platonic symbol based on affinities. """
    # ... (Logic unchanged, hard fails) ...
    logger.info("Identity Step: Assign Platonic Symbol...")
    elem_affinity = getattr(soul_spark, 'elemental_affinity'); seph_aspect = getattr(soul_spark, 'sephiroth_aspect'); gematria = getattr(soul_spark, 'gematria_value')
    if not elem_affinity or not seph_aspect: raise ValueError("Missing affinities for Platonic symbol assignment.")
    if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
    try:
        symbol = PLATONIC_ELEMENT_MAP.get(elem_affinity)
        if not symbol:
             seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
             if seph_geom in PLATONIC_SOLIDS: symbol = seph_geom
             elif seph_geom == 'cube': symbol = 'hexahedron'
        if not symbol:
             unique_symbols = sorted(list(set(PLATONIC_ELEMENT_MAP.values()) - {None}));
             if unique_symbols: symbol_idx = (gematria // PLATONIC_DEFAULT_GEMATRIA_RANGE) % len(unique_symbols); symbol = unique_symbols[symbol_idx]
             else: symbol = 'sphere'
        logger.debug(f"  Platonic Symbol Logic -> Final='{symbol}'")
        setattr(soul_spark, 'platonic_symbol', symbol); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Platonic symbol assigned: {symbol}")
    except Exception as e: logger.error(f"Error assigning Platonic symbol: {e}", exc_info=True); raise RuntimeError("Platonic symbol assignment failed.") from e

def _determine_astrological_signature(soul_spark: SoulSpark) -> None:
    """ Determines Zodiac sign, governing planet, and selects traits. """
    logger.info("Identity Step: Determine Astrological Signature...")
    try:
        # 1. Generate Conceptual Birth Datetime
        # Use simulation start + random offset up to ~9 months? Or just random date?
        # Let's use simulation start + random offset within a year for simplicity now
        sim_start_dt = datetime.fromisoformat(getattr(soul_spark, 'creation_time', datetime.now().isoformat()))
        random_offset_days = random.uniform(0, 365.25)
        birth_dt = sim_start_dt + timedelta(days=random_offset_days)
        setattr(soul_spark, 'conceptual_birth_datetime', birth_dt.isoformat())
        logger.debug(f"  Conceptual Birth Datetime: {birth_dt.strftime('%Y-%m-%d %H:%M')}")

        # 2. Determine Zodiac Sign (Using 13 signs)
        birth_month_day = birth_dt.strftime('%B %d')
        zodiac_sign = "Unknown"
        zodiac_symbol = "?"
        # Need to parse the date strings from constants properly
        for sign_info in ZODIAC_SIGNS:
            # This requires careful date parsing logic to handle year wrap-around
            # Simplified check (might be inaccurate at year boundaries):
             try:
                 # Create date objects for comparison (assume current year for simplicity)
                 start_str = f"{sign_info['start_date']} {birth_dt.year}"
                 end_str = f"{sign_info['end_date']} {birth_dt.year}"
                 dt_format = "%B %d %Y"
                 start_date = datetime.strptime(start_str, dt_format).date()
                 end_date = datetime.strptime(end_str, dt_format).date()
                 birth_date_only = birth_dt.date()

                 if start_date <= end_date: # Normal case (e.g., April 19 - May 13)
                     if start_date <= birth_date_only <= end_date:
                         zodiac_sign = sign_info['name']
                         zodiac_symbol = sign_info['symbol']
                         break
                 else: # Case where end date is in the next year (e.g., Dec 18 - Jan 18)
                      # Check if birth date is after start OR before end (in the next year conceptually)
                      if birth_date_only >= start_date or birth_date_only <= end_date:
                           zodiac_sign = sign_info['name']
                           zodiac_symbol = sign_info['symbol']
                           break
             except ValueError as date_err:
                  logger.warning(f"Could not parse date for sign {sign_info['name']}: {date_err}")
                  continue # Skip sign if date parsing fails

        setattr(soul_spark, 'zodiac_sign', zodiac_sign)
        logger.debug(f"  Determined Zodiac Sign: {zodiac_sign} ({zodiac_symbol})")

        # 3. Determine Governing Planet (Can be linked to sign or kept random?)
        # Let's keep the random assignment from Earth Harmony for now
        governing_planet = getattr(soul_spark, 'governing_planet', 'Unknown')
        if governing_planet == 'Unknown': logger.warning("Governing planet was not set in Earth Harmonization.")
        logger.debug(f"  Confirming Governing Planet: {governing_planet}")

        # 4. Select Traits
        traits = {"positive": {}, "negative": {}}
        if zodiac_sign != "Unknown" and zodiac_sign in ZODIAC_TRAITS:
            all_pos = ZODIAC_TRAITS[zodiac_sign].get("positive", [])
            all_neg = ZODIAC_TRAITS[zodiac_sign].get("negative", [])
            # Select random traits up to the max count
            num_pos = min(len(all_pos), ASTROLOGY_MAX_POSITIVE_TRAITS)
            num_neg = min(len(all_neg), ASTROLOGY_MAX_NEGATIVE_TRAITS)
            selected_pos = random.sample(all_pos, num_pos) if num_pos > 0 else []
            selected_neg = random.sample(all_neg, num_neg) if num_neg > 0 else []
            # Assign random strengths (0.1-0.9)
            for trait in selected_pos: traits["positive"][trait] = round(random.uniform(0.1, 0.9), 3)
            for trait in selected_neg: traits["negative"][trait] = round(random.uniform(0.1, 0.9), 3)
        else:
            logger.warning(f"Zodiac sign '{zodiac_sign}' not found in ZODIAC_TRAITS. No traits assigned.")

        setattr(soul_spark, 'astrological_traits', traits)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Astrological Signature determined: Sign={zodiac_sign}, Planet={governing_planet}, Traits={len(traits['positive'])} Pos / {len(traits['negative'])} Neg")

    except Exception as e:
        logger.error(f"Error determining astrological signature: {e}", exc_info=True)
        # Don't hard fail, but log error and leave attributes as None/default
        setattr(soul_spark, 'zodiac_sign', getattr(soul_spark, 'zodiac_sign', "Error")) # Indicate error
        setattr(soul_spark, 'astrological_traits', getattr(soul_spark, 'astrological_traits', {}))


def activate_love_resonance(soul_spark: SoulSpark, cycles: int = 7) -> None:
    """ Activates love resonance (updates 0-1 factors in emotional_resonance). """
    # ... (Logic unchanged - updates factors, fails hard) ...
    logger.info("Identity Step: Activate Love Resonance...")
    if not isinstance(cycles, int) or cycles < 0: raise ValueError("Cycles non-negative.")
    if cycles == 0: return
    try:
        soul_frequency = getattr(soul_spark, 'soul_frequency'); state = getattr(soul_spark, 'consciousness_state'); heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment'); emotional_resonance = getattr(soul_spark, 'emotional_resonance', {}); current_love = float(emotional_resonance.get('love', 0.0)); love_freq = LOVE_RESONANCE_FREQ
        if love_freq <= FLOAT_EPSILON or soul_frequency <= FLOAT_EPSILON: raise ValueError("Frequencies invalid for love resonance.")
        logger.debug(f"  Love Resonance Init: CurrentLove={current_love:.3f}")
        for cycle in range(cycles):
            cycle_factor = 1.0 - LOVE_RESONANCE_CYCLE_FACTOR_DECAY * (cycle / max(1, cycles)); base_increase = LOVE_RESONANCE_BASE_INC * cycle_factor; state_factor = LOVE_RESONANCE_STATE_WEIGHT.get(state, LOVE_RESONANCE_STATE_WEIGHT['default']); freq_resonance = calculate_resonance(love_freq, soul_frequency); heartbeat_factor = LOVE_RESONANCE_HEARTBEAT_WEIGHT + LOVE_RESONANCE_HEARTBEAT_SCALE * heartbeat_entrainment; increase = base_increase * state_factor * (LOVE_RESONANCE_FREQ_RES_WEIGHT * freq_resonance) * heartbeat_factor; current_love = min(1.0, current_love + increase)
            logger.debug(f"    Cycle {cycle+1}: Inc={increase:.5f}, NewLove={current_love:.4f}")
        emotional_resonance['love'] = float(current_love)
        for emotion in ['joy', 'peace', 'harmony', 'compassion']:
            current = float(emotional_resonance.get(emotion, 0.0)); boost = LOVE_RESONANCE_EMOTION_BOOST_FACTOR * current_love; emotional_resonance[emotion] = float(min(1.0, current + boost))
        setattr(soul_spark, 'emotional_resonance', emotional_resonance); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Love resonance activated. Final level: {current_love:.4f}")
    except Exception as e: logger.error(f"Error activating love resonance: {e}", exc_info=True); raise RuntimeError("Love resonance activation failed.") from e

def apply_sacred_geometry(soul_spark: SoulSpark, stages: int = 5) -> None:
    """ Applies sacred geometry imprint. Reinforces internal factors. S/C emerge via update_state. """
    # ... (Logic unchanged - modifies factors, fails hard) ...
    logger.info("Identity Step: Apply Sacred Geometry Imprint...")
    if not isinstance(stages, int) or stages < 0: raise ValueError("Stages non-negative.")
    if stages == 0: return
    try:
        seph_aspect = getattr(soul_spark, 'sephiroth_aspect'); elem_affinity = getattr(soul_spark, 'elemental_affinity'); name_resonance = getattr(soul_spark, 'name_resonance')
        if not seph_aspect or not elem_affinity: raise ValueError("Missing affinities for geometry application.")
        if aspect_dictionary is None: raise RuntimeError("Aspect Dictionary unavailable.")
        geometries = SACRED_GEOMETRY_STAGES; actual_stages = min(stages, len(geometries)); current_cryst_level = getattr(soul_spark, 'crystallization_level', 0.0); dominant_geometry = None; max_increase = -1.0
        total_pcoh_boost=0.0; total_phi_boost=0.0; total_harmony_boost=0.0; total_torus_boost=0.0
        logger.debug("  Applying Sacred Geometry Stages:")
        for i in range(actual_stages):
            geometry = geometries[i]; stage_factor = SACRED_GEOMETRY_STAGE_FACTOR_BASE + SACRED_GEOMETRY_STAGE_FACTOR_SCALE * (i / max(1, actual_stages - 1)); base_increase = SACRED_GEOMETRY_BASE_INC_BASE + SACRED_GEOMETRY_BASE_INC_SCALE * i
            seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower(); seph_weight = SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT.get(seph_geom, SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT['default']); sephiroth_factor = seph_weight
            elem_weight = SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT.get(elem_affinity, SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT['default']); elemental_factor = elem_weight
            fib_idx = min(i, len(FIBONACCI_SEQUENCE) - 1); fib_val = FIBONACCI_SEQUENCE[fib_idx]; fib_norm_idx = min(SACRED_GEOMETRY_FIB_MAX_IDX, len(FIBONACCI_SEQUENCE) - 1); fib_norm = FIBONACCI_SEQUENCE[fib_norm_idx]
            name_factor = SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE + SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE * name_resonance * (fib_val / max(1, fib_norm))
            increase = base_increase * stage_factor * sephiroth_factor * elemental_factor * name_factor; current_cryst_level = min(1.0, current_cryst_level + increase)
            if increase > max_increase: dominant_geometry = geometry; max_increase = increase
            logger.debug(f"    Stage {i+1} ({geometry}): CrystInc={increase:.5f}")
            geom_effects = GEOMETRY_EFFECTS.get(geometry, DEFAULT_GEOMETRY_EFFECT); stage_effect_scale = increase * 0.5
            total_pcoh_boost += geom_effects.get('pattern_coherence_boost', 0.0) * stage_effect_scale; total_phi_boost += geom_effects.get('phi_resonance_boost', 0.0) * stage_effect_scale; total_harmony_boost += geom_effects.get('harmony_boost', 0.0) * stage_effect_scale; total_torus_boost += geom_effects.get('toroidal_flow_strength_boost', 0.0) * stage_effect_scale
            logger.debug(f"      Factor Boosts (Cum): dPcoh={total_pcoh_boost:.5f}, dPhi={total_phi_boost:.5f}, dHarm={total_harmony_boost:.5f}, dTorus={total_torus_boost:.5f}")
        soul_spark.pattern_coherence = min(1.0, soul_spark.pattern_coherence + total_pcoh_boost); soul_spark.phi_resonance = min(1.0, soul_spark.phi_resonance + total_phi_boost); soul_spark.harmony = min(1.0, soul_spark.harmony + total_harmony_boost); soul_spark.toroidal_flow_strength = min(1.0, soul_spark.toroidal_flow_strength + total_torus_boost)
        setattr(soul_spark, 'crystallization_level', float(current_cryst_level)); setattr(soul_spark, 'sacred_geometry_imprint', dominant_geometry); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Sacred geometry applied. Dominant: {dominant_geometry}, Cryst Level: {current_cryst_level:.4f}")
        logger.info(f"  Resulting Factors: P.Coh={soul_spark.pattern_coherence:.4f}, Phi={soul_spark.phi_resonance:.4f}, Harm={soul_spark.harmony:.4f}, Torus={soul_spark.toroidal_flow_strength:.4f}")
    except Exception as e: logger.error(f"Error applying sacred geometry: {e}", exc_info=True); raise RuntimeError("Sacred geometry application failed critically.") from e

def calculate_attribute_coherence(soul_spark: SoulSpark) -> None:
    """ Calculates attribute coherence score (0-1) based on other 0-1 factors/scores. """
    # ... (Logic unchanged, fails hard) ...
    logger.info("Identity Step: Calculate Attribute Coherence...")
    try:
        attributes = {'name_resonance': getattr(soul_spark,'name_resonance',0.0),'response_level': getattr(soul_spark,'response_level',0.0),'state_stability': getattr(soul_spark,'state_stability',0.0),'crystallization_level': getattr(soul_spark,'crystallization_level',0.0),'heartbeat_entrainment': getattr(soul_spark,'heartbeat_entrainment',0.0),'emotional_resonance_avg': np.mean(list(getattr(soul_spark,'emotional_resonance',{}).values())) if getattr(soul_spark,'emotional_resonance') else 0.0,'creator_connection': getattr(soul_spark,'creator_connection_strength',0.0),'earth_resonance': getattr(soul_spark,'earth_resonance',0.0),'elemental_alignment': getattr(soul_spark, 'elemental_alignment',0.0),'cycle_synchronization': getattr(soul_spark, 'cycle_synchronization',0.0),'harmony': getattr(soul_spark, 'harmony',0.0),'phi_resonance': getattr(soul_spark, 'phi_resonance',0.0),'pattern_coherence': getattr(soul_spark, 'pattern_coherence',0.0),'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength',0.0)}
        attr_values = [v for v in attributes.values() if isinstance(v,(int,float)) and np.isfinite(v) and 0.0<=v<=1.0]
        if len(attr_values) < 5: coherence_score = 0.5; logger.warning(f"  Attribute Coherence: Not enough valid attributes ({len(attr_values)}). Using default {coherence_score}.")
        else: std_dev = np.std(attr_values); coherence_score = max(0.0, 1.0 - min(1.0, std_dev * ATTRIBUTE_COHERENCE_STD_DEV_SCALE)); logger.debug(f"  Attr Coh Calc: StdDev={std_dev:.4f} -> Score={coherence_score:.4f}")
        setattr(soul_spark, 'attribute_coherence', float(coherence_score)); setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        logger.info(f"Attribute coherence calculated: {coherence_score:.4f}")
    except Exception as e: logger.error(f"Error calculating attribute coherence: {e}", exc_info=True); raise RuntimeError("Attribute coherence calc failed.") from e

def verify_identity_crystallization(soul_spark: SoulSpark, threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD) -> Tuple[bool, Dict[str, Any]]:
    """ Verifies identity crystallization score (0-1) against threshold. Raises error if failed. """
    # ... (Logic unchanged, hard fails if below threshold) ...
    logger.info("Identity Step: Verify Crystallization...")
    if not (0.0 < threshold <= 1.0): raise ValueError("Threshold invalid.")
    try:
        required_attrs_for_score = CRYSTALLIZATION_REQUIRED_ATTRIBUTES; missing_attributes = [attr for attr in required_attrs_for_score if getattr(soul_spark, attr, None) is None]; attr_presence_score = (len(required_attrs_for_score) - len(missing_attributes)) / max(1, len(required_attrs_for_score))
        component_metrics = {'name_resonance': getattr(soul_spark,'name_resonance',0.0),'response_level': getattr(soul_spark,'response_level',0.0),'state_stability': getattr(soul_spark,'state_stability',0.0),'crystallization_level': getattr(soul_spark,'crystallization_level',0.0),'attribute_coherence': getattr(soul_spark,'attribute_coherence',0.0),'attribute_presence': attr_presence_score,'emotional_resonance': np.mean(list(getattr(soul_spark,'emotional_resonance',{}).values())) if getattr(soul_spark,'emotional_resonance') else 0.0,'harmony': getattr(soul_spark, 'harmony', 0.0)}
        logger.debug(f"  Identity Verification Components: {component_metrics}")
        total_crystallization_score = 0.0; total_weight = 0.0
        for component, weight in CRYSTALLIZATION_COMPONENT_WEIGHTS.items():
            value = component_metrics.get(component, 0.0)
            if isinstance(value,(int,float)) and np.isfinite(value): total_crystallization_score += value * weight; total_weight += weight
            else: logger.warning(f"    Invalid value for component '{component}': {value}")
        if total_weight > FLOAT_EPSILON: total_crystallization_score /= total_weight
        total_crystallization_score = max(0.0, min(1.0, total_crystallization_score))
        is_crystallized = (total_crystallization_score >= threshold and attr_presence_score >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD)
        logger.debug(f"  Identity Verification: Score={total_crystallization_score:.4f}, Threshold={threshold}, AttrPresence={attr_presence_score:.2f} -> Crystallized={is_crystallized}")
        setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, is_crystallized); setattr(soul_spark, 'crystallization_level', float(total_crystallization_score)); timestamp = datetime.now().isoformat(); setattr(soul_spark, 'last_modified', timestamp)
        verification_result = {'total_crystallization_score': float(total_crystallization_score), 'threshold': threshold,'is_crystallized': is_crystallized, 'components': component_metrics, 'missing_attributes': missing_attributes}
        setattr(soul_spark, 'identity_metrics', verification_result)
        if not is_crystallized:
             error_msg = f"Identity crystallization failed: Score {total_crystallization_score:.4f} < Threshold {threshold} or Attr Presence {attr_presence_score:.2f} < {CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}."; logger.error(error_msg); raise RuntimeError(error_msg)
        logger.info(f"Identity check PASSED: Score={total_crystallization_score:.4f}")
        return is_crystallized, verification_result
    except Exception as e: logger.error(f"Error verifying identity: {e}", exc_info=True); raise RuntimeError("Identity verification failed critically.") from e


# --- Orchestration Function ---
def perform_identity_crystallization(soul_spark: SoulSpark,
                                    # specified_name removed
                                    train_cycles: int = 5,
                                    entrainment_bpm: float = 72.0,
                                    entrainment_duration: float = 120.0,
                                    love_cycles: int = 7,
                                    geometry_stages: int = 5,
                                    crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD
                                    ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """ Performs complete identity crystallization including astrology. Modifies SoulSpark factors. S/C emerge. """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark): raise TypeError("soul_spark invalid.")
    if not isinstance(train_cycles,int) or train_cycles<0: raise ValueError("train_cycles must be >= 0")
    if not isinstance(entrainment_bpm,(int,float)) or entrainment_bpm<=0: raise ValueError("bpm must be > 0")
    if not isinstance(entrainment_duration,(int,float)) or entrainment_duration<0: raise ValueError("duration must be >= 0")
    if not isinstance(love_cycles,int) or love_cycles<0: raise ValueError("love_cycles must be >= 0")
    if not isinstance(geometry_stages,int) or geometry_stages<0: raise ValueError("geometry_stages must be >= 0")
    if not (0.0 < crystallization_threshold <= 1.0): raise ValueError("Threshold invalid.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Identity Crystallization for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat(); start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps_completed': []}

    try:
        _ensure_soul_properties(soul_spark) # Raises error if fails
        _check_prerequisites(soul_spark) # Raises error if fails

        initial_state = {
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level,
             'attribute_coherence': soul_spark.attribute_coherence,
             'harmony': soul_spark.harmony
        }
        logger.info(f"Identity Init State: S={initial_state['stability_su']:.1f}, C={initial_state['coherence_cu']:.1f}, CrystLvl={initial_state['crystallization_level']:.3f}, AttrCoh={initial_state['attribute_coherence']:.3f}, Harm={initial_state['harmony']:.3f}")

        # --- Run Sequence ---
        assign_name(soul_spark); process_metrics_summary['steps_completed'].append('name')
        assign_voice_frequency(soul_spark); process_metrics_summary['steps_completed'].append('voice')
        process_soul_color(soul_spark); process_metrics_summary['steps_completed'].append('color')
        apply_heartbeat_entrainment(soul_spark, entrainment_bpm, entrainment_duration); process_metrics_summary['steps_completed'].append('heartbeat')
        train_name_response(soul_spark, train_cycles); process_metrics_summary['steps_completed'].append('response')
        identify_primary_sephiroth(soul_spark); process_metrics_summary['steps_completed'].append('sephiroth_id')
        determine_elemental_affinity(soul_spark); process_metrics_summary['steps_completed'].append('elemental_id')
        assign_platonic_symbol(soul_spark); process_metrics_summary['steps_completed'].append('platonic_id')
        # --- NEW ASTROLOGY STEP ---
        _determine_astrological_signature(soul_spark); process_metrics_summary['steps_completed'].append('astrology')
        # --- END NEW STEP ---
        activate_love_resonance(soul_spark, love_cycles); process_metrics_summary['steps_completed'].append('love')
        apply_sacred_geometry(soul_spark, geometry_stages); process_metrics_summary['steps_completed'].append('geometry')
        calculate_attribute_coherence(soul_spark); process_metrics_summary['steps_completed'].append('attr_coherence')

        logger.info("Identity Step: Final State Update & Verification...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state() # Update SU/CU scores after geometry factor influence
            logger.debug(f"  Identity S/C after update: S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        else: raise AttributeError("SoulSpark missing 'update_state' method.")

        is_crystallized, verification_metrics = verify_identity_crystallization(soul_spark, crystallization_threshold); process_metrics_summary['steps_completed'].append('verify')
        # verify_identity_crystallization raises RuntimeError if it fails

        # --- Final Update & Metrics ---
        setattr(soul_spark, FLAG_READY_FOR_BIRTH, True)
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)
        if hasattr(soul_spark, 'add_memory_echo'): soul_spark.add_memory_echo(f"Identity crystallized as '{soul_spark.name}'. Level: {soul_spark.crystallization_level:.3f}, Sign: {soul_spark.zodiac_sign}")

        end_time_iso = last_mod_time; end_time_dt = datetime.fromisoformat(end_time_iso)
        final_state = {
             'stability_su': soul_spark.stability, 'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level,
             'attribute_coherence': soul_spark.attribute_coherence,
             'harmony': soul_spark.harmony,
             'name': soul_spark.name, 'zodiac_sign': soul_spark.zodiac_sign, # Added zodiac
             FLAG_IDENTITY_CRYSTALLIZED: getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED)
        }
        overall_metrics = {
            'action': 'identity_crystallization', 'soul_id': spark_id, 'start_time': start_time_iso, 'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state, 'final_state': final_state,
            'final_crystallization_score': verification_metrics['total_crystallization_score'],
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'harmony_change': final_state['harmony'] - initial_state['harmony'],
            'astrological_signature': { # Added astro summary
                'sign': soul_spark.zodiac_sign,
                'planet': soul_spark.governing_planet,
                'traits': soul_spark.astrological_traits
            },
            'success': True,
        }
        if METRICS_AVAILABLE: metrics.record_metrics('identity_crystallization_summary', overall_metrics)

        logger.info(f"--- Identity Crystallization Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: Name='{soul_spark.name}', Sign='{soul_spark.zodiac_sign}', CrystLvl={soul_spark.crystallization_level:.3f}, S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        return soul_spark, overall_metrics

    except (ValueError, TypeError, AttributeError) as e_val:
         logger.error(f"Identity Crystallization failed for {spark_id}: {e_val}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'prerequisites/validation'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e_val))
         # Hard fail
         raise e_val
    except RuntimeError as e_rt:
         logger.critical(f"Identity Crystallization failed critically for {spark_id}: {e_rt}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'verification/runtime'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False); setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         # Hard fail
         raise e_rt
    except Exception as e:
         logger.critical(f"Unexpected error during Identity Crystallization for {spark_id}: {e}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'unexpected'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e))
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False); setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         # Hard fail
         raise RuntimeError(f"Unexpected Identity Crystallization failure: {e}") from e

# --- Failure Metric Helper ---
def record_id_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
    """ Helper to record failure metrics consistently. """
    if METRICS_AVAILABLE:
        try:
            end_time = datetime.now().isoformat()
            duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
            metrics.record_metrics('identity_crystallization_summary', {
                'action': 'identity_crystallization', 'soul_id': spark_id,
                'start_time': start_time_iso, 'end_time': end_time,
                'duration_seconds': duration,
                'success': False, 'error': error_msg, 'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record ID failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE src/stage_1/soul_formation/identity_crystallization.py ---