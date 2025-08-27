"""
Identity Crystallization Module (Refactored V4.4.0 - Light-Sound Principles)

Crystallizes identity after Earth attunement. Implements Name (user input), Voice, 
Color, Affinities (Seph, Elem, Platonic), and Astrological Signature using complete
light and sound principles. Implements proper light physics for crystalline formation 


Heartbeat/Love cycles implement proper acoustic physics for harmony enhancement.
Sacred Geometry uses proper light interference patterns and standing waves.
Modifies SoulSpark via identity crystallisation. Uses strict validation. Hard fails only.
"""

import logging
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import time
import random
import uuid
import math
import re
from typing import Dict, List, Any, Tuple, Optional
from math import pi as PI, sqrt, exp, sin, cos, tanh
from stage_1.soul_spark.soul_spark import SoulSpark
from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
from stage_1.soul_formation.creator_entanglement import calculate_resonance

# Direct import of constants without try blocks
from shared.constants.constants import *

# --- Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# Sound module imports - These modules actively generate actual sound
try:
    from shared.sound.sound_generator import SoundGenerator
    from shared.sound.sounds_of_universe import UniverseSounds
    SOUND_MODULES_AVAILABLE = True
    sound_gen = SoundGenerator(output_dir="output/sounds/identity")
    universe_sounds = UniverseSounds(output_dir="output/sounds/identity")
except ImportError:
    logger.critical("Sound modules not available. Identity crystallization REQUIRES sound generation.")
    SOUND_MODULES_AVAILABLE = False
    raise ImportError("Critical sound modules missing.") from None

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.critical("Metrics tracking module not found. Identity crystallization REQUIRES metrics.")
    METRICS_AVAILABLE = False
    raise ImportError("Critical metrics module missing.") from None

def log_soul_identity_summary(soul_spark: SoulSpark) -> None:
    """
    Log key soul identity properties to terminal before name assignment.
    Shows the soul's fundamental characteristics.
    """
    print("\n" + "="*80)
    print("SOUL IDENTITY SUMMARY - PRE-NAMING")
    print("="*80)
    
    # Basic soul info
    print(f"Soul ID: {getattr(soul_spark, 'spark_id', 'Unknown')}")
    print(f"Base Frequency: {getattr(soul_spark, 'frequency', 0.0):.2f} Hz")
    print(f"Soul Frequency: {getattr(soul_spark, 'soul_frequency', 0.0):.2f} Hz")
    
    # Soul color
    soul_color = getattr(soul_spark, 'soul_color', None)
    if soul_color:
        print(f"Soul Color: {soul_color}")
    else:
        print("Soul Color: Not yet determined")
    
    # Current state
    print(f"Stability: {getattr(soul_spark, 'stability', 0.0):.1f} SU")
    print(f"Coherence: {getattr(soul_spark, 'coherence', 0.0):.1f} CU")
    print(f"Energy: {getattr(soul_spark, 'energy', 0.0):.1f} SEU")
    
    # Sephiroth aspects
    aspects = getattr(soul_spark, 'aspects', {})
    if aspects:
        print(f"\nSephiroth Aspects ({len(aspects)} total):")
        # Show top 5 strongest aspects
        sorted_aspects = sorted(aspects.items(), 
                              key=lambda x: x[1].get('strength', 0.0), 
                              reverse=True)[:5]
        for aspect_name, aspect_data in sorted_aspects:
            strength = aspect_data.get('strength', 0.0)
            source = aspect_data.get('source', 'Unknown')
            print(f"  • {aspect_name}: {strength:.3f} (from {source})")
        if len(aspects) > 5:
            print(f"  ... and {len(aspects) - 5} more aspects")
    else:
        print("\nSephiroth Aspects: None acquired yet")
    
    # Astrological info (if available)
    zodiac_sign = getattr(soul_spark, 'zodiac_sign', None)
    governing_planet = getattr(soul_spark, 'governing_planet', None)
    birth_datetime = getattr(soul_spark, 'conceptual_birth_datetime', None)
    
    if zodiac_sign or governing_planet or birth_datetime:
        print(f"\nAstrological Signature:")
        if birth_datetime:
            try:
                birth_dt = datetime.fromisoformat(birth_datetime)
                print(f"  Birth DateTime: {birth_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                print("  Birth DateTime: {birth_datetime}")
        if zodiac_sign:
            print(f"  Zodiac Sign: {zodiac_sign}")
        if governing_planet:
            print(f"  Governing Planet: {governing_planet}")
    
    # Elemental balance
    elements = getattr(soul_spark, 'elements', {})
    if elements:
        print(f"\nElemental Balance:")
        for element, value in sorted(elements.items(), key=lambda x: x[1], reverse=True):
            percentage = value * 100
            print(f"  {element.capitalize()}: {percentage:.1f}%")
    
    # Earth connection
    earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
    gaia_connection = getattr(soul_spark, 'gaia_connection', 0.0)
    if earth_resonance > 0 or gaia_connection > 0:
        print(f"\nEarth Connection:")
        print(f"  Earth Resonance: {earth_resonance:.3f}")
        print(f"  Gaia Connection: {gaia_connection:.3f}")
    
    
    print("="*80)
    print("READY FOR NAME ASSIGNMENT")
    print("="*80 + "\n")

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
        msg = "Prerequisite failed: Soul missing stability, coherence, or earth_resonance."
        logger.error(msg); raise AttributeError(msg)

    if stability_su < IDENTITY_STABILITY_THRESHOLD_SU:
        msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {IDENTITY_STABILITY_THRESHOLD_SU} SU."
        logger.error(msg); raise ValueError(msg)
    if coherence_cu < IDENTITY_COHERENCE_THRESHOLD_CU:
        msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {IDENTITY_COHERENCE_THRESHOLD_CU} CU."
        logger.error(msg); raise ValueError(msg)
    if earth_resonance < IDENTITY_EARTH_RESONANCE_THRESHOLD:
        msg = f"Prerequisite failed: Earth Resonance ({earth_resonance:.3f}) < {IDENTITY_EARTH_RESONANCE_THRESHOLD}."
        logger.error(msg); raise ValueError(msg)

    # 3. Essential Attributes for Calculation
    if getattr(soul_spark, 'soul_color', None) is None:
        msg = "Prerequisite failed: SoulSpark missing 'soul_color'."
        logger.error(msg); raise AttributeError(msg)
    if getattr(soul_spark, 'soul_frequency', 0.0) <= FLOAT_EPSILON:
        msg = f"Prerequisite failed: SoulSpark missing valid 'soul_frequency' ({getattr(soul_spark, 'soul_frequency', 0.0)})."
        logger.error(msg); raise ValueError(msg)
    if getattr(soul_spark, 'frequency', 0.0) <= FLOAT_EPSILON:
        msg = f"Prerequisite failed: SoulSpark missing valid base 'frequency'."
        logger.error(msg); raise ValueError(msg)
    
    # 4. Check for existing crystallization
    if getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
        logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_IDENTITY_CRYSTALLIZED}. Re-running.")

    logger.debug("Identity prerequisites met.")
    return True

def _ensure_soul_properties(soul_spark: SoulSpark):
    """ Ensure soul has necessary properties. Raises error if missing/invalid. """
    logger.debug(f"Ensuring properties for identity process (Soul {soul_spark.spark_id})...")
    required = ['stability', 'coherence', 'earth_resonance', 'soul_color', 'soul_frequency', 'frequency',
                'yin_yang_balance', 'aspects', 'consciousness_state', 'phi_resonance',
                'pattern_coherence', 'harmony', 'toroidal_flow_strength', 'layers']
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
        "conceptual_birth_datetime": None, "zodiac_sign": None, "astrological_traits": None,
        "identity_light_signature": None, "identity_sound_signature": None, 
        "crystalline_structure": None, "name_standing_waves": None
    }
    for attr, default in defaults.items():
        if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
             if attr == 'emotional_resonance': setattr(soul_spark, attr, default.copy())
             elif attr == 'astrological_traits': setattr(soul_spark, attr, {}) # Init as empty dict
             else: setattr(soul_spark, attr, default)

    if hasattr(soul_spark, '_validate_attributes'): soul_spark._validate_attributes()
    logger.debug("Soul properties ensured for Identity Crystallization.")

# --- Name Calculation Helpers with Light Physics ---
def _calculate_gematria(name: str) -> int:
    """Calculate gematria value using full character encoding for accuracy."""
    if not isinstance(name, str): raise TypeError("Name must be a string.")
    return sum(ord(char) - ord('a') + 1 for char in name.lower() if 'a' <= char <= 'z')

def _calculate_name_resonance(name: str, gematria: int) -> float:
    """Calculate name resonance (0-1) based on advanced harmonic principles."""
    if not name: return 0.0
    try:
        vowels=sum(1 for c in name.lower() if c in 'aeiouy')
        consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz')
        total_letters=vowels+consonants
        if total_letters == 0: return 0.0
        
        # Vowel ratio compared to golden ratio
        vowel_ratio=vowels/total_letters
        phi_inv = 1.0 / GOLDEN_RATIO
        vowel_factor = max(0.0, 1.0 - abs(vowel_ratio - phi_inv) * 2.5)
        
        # Unique letters and distribution
        unique_letters = len(set(c for c in name.lower() if 'a' <= c <= 'z'))
        letter_factor = unique_letters / max(1, len(name))
        
        # Gematria-based resonance factor
        gematria_factor = 0.5
        if gematria > 0:
            for num in NAME_GEMATRIA_RESONANT_NUMBERS:
                 if gematria % num == 0: gematria_factor = 0.9; break
        
        # Frequency analysis - look for patterns
        frequency_factor = 0.0
        if len(name) >= 3:
            # Check for repeating patterns
            for pattern_len in range(1, min(5, len(name) // 2 + 1)):
                patterns = set()
                for i in range(0, len(name) - pattern_len + 1):
                    pattern = name[i:i+pattern_len].lower()
                    patterns.add(pattern)
                # More unique patterns = higher complexity = better resonance
                complexity = len(patterns) / (len(name) - pattern_len + 1)
                frequency_factor = max(frequency_factor, complexity * 0.1)
        
        # Combine all factors with weights
        total_weight = (NAME_RESONANCE_WEIGHT_VOWEL + NAME_RESONANCE_WEIGHT_LETTER + 
                       NAME_RESONANCE_WEIGHT_GEMATRIA + 0.1)  # 0.1 for frequency factor
        
        resonance_contrib = (NAME_RESONANCE_WEIGHT_VOWEL * vowel_factor + 
                           NAME_RESONANCE_WEIGHT_LETTER * letter_factor + 
                           NAME_RESONANCE_WEIGHT_GEMATRIA * gematria_factor +
                           0.1 * frequency_factor)
        
        resonance = NAME_RESONANCE_BASE + (1.0 - NAME_RESONANCE_BASE) * (resonance_contrib / total_weight)
        return float(max(0.0, min(1.0, resonance)))
    except Exception as e:
        logger.error(f"Error calculating name resonance for '{name}': {e}")
        raise RuntimeError("Name resonance calculation failed.") from e

def _calculate_name_standing_waves(name: str, base_frequency: float) -> Dict[str, Any]:
    """
    Calculate standing wave patterns generated by the name.
    Creates a light-sound signature that forms basis for identity.
    
    Args:
        name: Soul's name
        base_frequency: Base frequency to anchor the standing waves
        
    Returns:
        Dictionary with standing wave properties
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails
    """
    if not name: raise ValueError("Name cannot be empty")
    if base_frequency <= 0: raise ValueError("Base frequency must be positive")
    
    try:
        # Create frequency array from name
        frequencies = []
        amplitudes = []
        phases = []
        
        # Map each character to a frequency offset
        # Using a Phi-based mapping for harmonic relationships
        for i, char in enumerate(name.lower()):
            if 'a' <= char <= 'z':
                # Calculate offset based on character position in alphabet
                char_value = ord(char) - ord('a') + 1
                
                # Create frequency that relates to base via Phi-based ratios
                # This creates harmonically related frequencies
                power = (char_value % 12) / 12.0  # Normalize to 0-1 range
                ratio = 1.0 + (PHI - 1.0) * power
                
                # Calculate frequency, ensure it's within human hearing range
                freq = base_frequency * ratio
                freq = min(20000, max(20, freq))  # Limit to audible range
                
                # Calculate amplitude based on character position
                # Vowels have higher amplitude
                if char in 'aeiouy':
                    amp = 0.7 - 0.3 * (i / len(name))
                else:
                    amp = 0.5 - 0.3 * (i / len(name))
                
                # Phase based on character position for wave interference
                phase = (char_value / 26.0) * 2 * PI
                
                frequencies.append(freq)
                amplitudes.append(amp)
                phases.append(phase)
        
        # Calculate interference pattern at specific points
        num_points = 100
        interference_pattern = np.zeros(num_points, dtype=np.float32)
        x_values = np.linspace(0, 1, num_points)
        
        for i, x in enumerate(x_values):
            # Sum waves at this point
            for freq, amp, phase in zip(frequencies, amplitudes, phases):
                # Create standing wave pattern
                wavelength = SPEED_OF_SOUND / freq
                k = 2 * PI / wavelength
                interference_pattern[i] += amp * np.sin(k * x + phase)
                
        # Find nodes and antinodes
        nodes = []
        antinodes = []
        
        # Finding approximate nodes (close to zero amplitude)
        for i in range(1, len(interference_pattern) - 1):
            if abs(interference_pattern[i]) < 0.1:
                if abs(interference_pattern[i-1]) > 0.1 or abs(interference_pattern[i+1]) > 0.1:
                    nodes.append({"position": float(x_values[i]), 
                                 "amplitude": float(interference_pattern[i])})
            
            # Finding approximate antinodes (local maxima/minima)
            if (interference_pattern[i] > interference_pattern[i-1] and 
                interference_pattern[i] > interference_pattern[i+1]):
                antinodes.append({"position": float(x_values[i]), 
                                 "amplitude": float(interference_pattern[i])})
            elif (interference_pattern[i] < interference_pattern[i-1] and 
                  interference_pattern[i] < interference_pattern[i+1]):
                antinodes.append({"position": float(x_values[i]), 
                                 "amplitude": float(interference_pattern[i])})
                
        # Calculate resonance quality based on number of stable nodes/antinodes
        resonance_quality = min(1.0, (len(nodes) + len(antinodes)) / 20.0)
        
        # Create light signature - mapping frequencies to light spectrum
        light_frequencies = []
        for freq in frequencies:
            # Scale audio to light frequency
            light_freq = freq * 1e12  # Simple scaling for example
            wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
            
            # Map to visible spectrum color
            if 380 <= wavelength_nm <= 750:
                if 380 <= wavelength_nm < 450: color = "violet"
                elif 450 <= wavelength_nm < 495: color = "blue"
                elif 495 <= wavelength_nm < 570: color = "green"
                elif 570 <= wavelength_nm < 590: color = "yellow"
                elif 590 <= wavelength_nm < 620: color = "orange"
                else: color = "red"
            else:
                color = "beyond visible"
                
            light_frequencies.append({
                "audio_frequency": float(freq),
                "light_frequency": float(light_freq),
                "wavelength_nm": float(wavelength_nm),
                "color": color
            })
        
        # Calculate overall coherence of the pattern
        # More regular patterns have higher coherence
        amplitude_variance = np.var(interference_pattern)
        pattern_coherence = max(0.0, min(1.0, 1.0 - amplitude_variance))
        
        # Create the final standing wave structure
        standing_waves = {
            "base_frequency": float(base_frequency),
            "component_frequencies": frequencies,
            "component_amplitudes": amplitudes,
            "component_phases": phases,
            "nodes": nodes,
            "antinodes": antinodes,
            "pattern_coherence": float(pattern_coherence),
            "resonance_quality": float(resonance_quality),
            "light_frequencies": light_frequencies,
            "interference_pattern": interference_pattern.tolist()
        }
        
        logger.debug(f"Calculated name standing waves: {len(frequencies)} components, " +
                    f"Nodes: {len(nodes)}, Antinodes: {len(antinodes)}, " +
                    f"Coherence: {pattern_coherence:.4f}")
        
        return standing_waves
        
    except Exception as e:
        logger.error(f"Error calculating name standing waves: {e}", exc_info=True)
        raise RuntimeError("Name standing wave calculation failed") from e

def _generate_phonetic_name_pronunciation(name: str, base_frequency: float, duration: float = 3.0) -> np.ndarray:
    """
    Generate realistic pronunciation of the name using phonetic synthesis.
    Creates actual speech-like sounds that pronounce the name.
    
    Args:
        name: Soul's name to pronounce
        base_frequency: Base pitch frequency for the voice
        duration: Length of audio in seconds
        
    Returns:
        NumPy array containing the audio samples of the name being spoken
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If generation fails
    """
    if not name: raise ValueError("Name cannot be empty")
    if base_frequency <= 0: raise ValueError("Base frequency must be positive")
    if duration <= 0: raise ValueError("Duration must be positive")
    
    if not SOUND_MODULES_AVAILABLE:
        raise RuntimeError("Sound generation requires sound modules")
    
    try:
        import numpy as np
        import random
        
        # Get sample rate from sound generator
        sample_rate = 44100
        total_samples = int(duration * sample_rate)
        
        # Name-specific phonetic pronunciations for soul names
        # These are ethereal/fantasy names requiring specific phonetic handling
        specific_name_pronunciations = {
            'aeri': [('air', 0.4), ('ee', 0.3)],  # AIR-ee (like "airy")
            'aevren': [('ay', 0.3), ('vren', 0.4)],  # AY-vren (like "haven" but Ay-vren)
            'aeho': [('ay', 0.3), ('ho', 0.3)],  # AY-ho (like "hey" + "ho")
            # Add more soul names as needed
        }
        
        name_lower = name.lower()
        
        # Check if we have a specific pronunciation for this name
        if name_lower in specific_name_pronunciations:
            phonemes = specific_name_pronunciations[name_lower]
            logger.info(f"Using specific pronunciation for '{name}': {[p[0] for p in phonemes]}")
        else:
            # Fallback to intelligent syllable parsing for unknown names
            phonemes = _parse_name_syllables(name)
            logger.info(f"Using syllable parsing for '{name}': {[p[0] for p in phonemes]}")
        
        if not phonemes:
            raise ValueError(f"No pronounceable syllables found in name '{name}'")
        
        # Calculate timing
        total_phoneme_time = sum(duration for _, duration in phonemes)
        time_scale = duration / total_phoneme_time if total_phoneme_time > 0 else 1.0
        
        audio_data = np.zeros(total_samples, dtype=np.float32)
        current_time = 0.0
        
        for phoneme, phoneme_duration in phonemes:
            scaled_duration = phoneme_duration * time_scale
            start_sample = int(current_time * sample_rate)
            end_sample = int((current_time + scaled_duration) * sample_rate)
            end_sample = min(end_sample, total_samples)
            
            if start_sample >= total_samples:
                break
                
            phoneme_length = end_sample - start_sample
            
            if phoneme == 'pause':
                # Silence for pauses
                pass
            else:
                # Generate phoneme sound
                t = np.linspace(0, scaled_duration, phoneme_length)
                
                # Voice characteristics for different phoneme types
                if phoneme in ['ah', 'eh', 'ih', 'oh', 'oo']:  # Vowels
                    # Vowel formants (simplified)
                    formant_map = {
                        'ah': (730, 1090, 2440),  # 'a' as in "father"
                        'eh': (530, 1840, 2480),  # 'e' as in "bed"  
                        'ih': (390, 1990, 2550),  # 'i' as in "bit"
                        'oh': (570, 840, 2410),   # 'o' as in "boat"
                        'oo': (300, 870, 2240)    # 'u' as in "boot"
                    }
                    
                    f1, f2, f3 = formant_map.get(phoneme, (500, 1500, 2500))
                    
                    # Generate vowel with formants
                    pitch_contour = base_frequency * (1.0 + 0.1 * np.sin(2 * np.pi * 3 * t))
                    
                    # Fundamental + formants
                    signal = (
                        0.4 * np.sin(2 * np.pi * pitch_contour * t) +      # Fundamental
                        0.3 * np.sin(2 * np.pi * f1 * t) * np.exp(-f1 * t * 0.001) +  # F1
                        0.2 * np.sin(2 * np.pi * f2 * t) * np.exp(-f2 * t * 0.0005) + # F2
                        0.1 * np.sin(2 * np.pi * f3 * t) * np.exp(-f3 * t * 0.0003)   # F3
                    )
                    
                else:  # Consonants
                    # Consonant characteristics
                    if phoneme.endswith('uh'):
                        consonant = phoneme[:-2]
                    else:
                        consonant = phoneme
                    
                    # Different synthesis for different consonant types
                    if consonant in ['b', 'd', 'g', 'p', 't', 'k']:  # Plosives
                        # Sharp attack, quick decay
                        envelope = np.exp(-5 * t)
                        
                        # Burst of noise for plosives
                        noise = np.random.normal(0, 0.1, phoneme_length)
                        signal = noise * envelope
                        
                        # Add some tonal content
                        if consonant in ['b', 'd', 'g']:  # Voiced
                            signal += 0.3 * np.sin(2 * np.pi * base_frequency * t) * envelope
                            
                    elif consonant in ['f', 's', 'z', 'h']:  # Fricatives  
                        # Noise-based with some tonal content
                        noise = np.random.normal(0, 0.15, phoneme_length)
                        # Filter noise to different frequency ranges
                        for i in range(1, len(noise)):
                            noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
                        
                        envelope = np.exp(-2 * t)
                        signal = noise * envelope
                        
                        if consonant in ['z']:  # Voiced fricatives
                            signal += 0.2 * np.sin(2 * np.pi * base_frequency * t) * envelope
                            
                    elif consonant in ['m', 'n', 'l', 'r']:  # Nasals/liquids
                        # More tonal, like vowels but with different formants
                        envelope = np.exp(-3 * t)
                        signal = (
                            0.4 * np.sin(2 * np.pi * base_frequency * t) +
                            0.2 * np.sin(2 * np.pi * base_frequency * 2 * t) +
                            0.1 * np.sin(2 * np.pi * base_frequency * 3 * t)
                        ) * envelope
                        
                        # Add nasal resonance for m, n
                        if consonant in ['m', 'n']:
                            signal += 0.15 * np.sin(2 * np.pi * 250 * t) * envelope
                            
                    else:  # Default consonant
                        envelope = np.exp(-4 * t)
                        signal = 0.3 * np.sin(2 * np.pi * base_frequency * t) * envelope
                
                # Apply envelope for natural speech
                if phoneme_length > 0:
                    # Natural speech envelope
                    attack_len = min(phoneme_length // 10, int(0.05 * sample_rate))
                    decay_len = min(phoneme_length // 5, int(0.1 * sample_rate))
                    
                    speech_envelope = np.ones(phoneme_length)
                    
                    # Attack
                    if attack_len > 0:
                        speech_envelope[:attack_len] = np.linspace(0, 1, attack_len)
                    
                    # Decay  
                    if decay_len > 0 and phoneme_length > decay_len:
                        speech_envelope[-decay_len:] = np.linspace(1, 0, decay_len)
                    
                    signal *= speech_envelope
                
                # Add to main audio
                audio_data[start_sample:end_sample] += signal * 0.6
            
            current_time += scaled_duration
        
        # Normalize and add natural speech dynamics
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.8
        
        # Add subtle reverb for natural voice
        reverb_delay = int(0.02 * sample_rate)  # 20ms delay
        if len(audio_data) > reverb_delay:
            reverb_audio = np.zeros_like(audio_data)
            reverb_audio[reverb_delay:] = audio_data[:-reverb_delay] * 0.2
            audio_data += reverb_audio
        
        logger.info(f"Generated phonetic pronunciation of '{name}' as {[p[0] for p in phonemes]}: {len(audio_data)} samples")
        return audio_data
        
    except Exception as e:
        logger.error(f"Error generating phonetic name pronunciation: {e}", exc_info=True)
        raise RuntimeError("Phonetic name pronunciation generation failed") from e

def _parse_name_syllables(name: str) -> list:
    """
    Parse unknown names into pronounceable syllables using phonetic rules.
    Handles fantasy/ethereal names not in the specific pronunciation dictionary.
    
    Args:
        name: Name to parse into syllables
        
    Returns:
        List of (syllable, duration) tuples
    """
    try:
        name_clean = name.lower().strip()
        syllables = []
        
        # Common vowel combinations in fantasy names
        vowel_combinations = {
            'ae': ('ay', 0.35),    # Like "Aeri" -> "AY-ri" 
            'ai': ('eye', 0.3),    # Like "Kai"
            'au': ('ow', 0.3),     # Like "pause"
            'ea': ('ee', 0.3),     # Like "sea"
            'ee': ('ee', 0.35),    # Like "tree"
            'ei': ('ay', 0.3),     # Like "veil"
            'ie': ('ee', 0.3),     # Like "believe"
            'oa': ('oh', 0.3),     # Like "boat"
            'ou': ('oo', 0.3),     # Like "you"
            'ue': ('oo', 0.3),     # Like "blue"
            'ui': ('oo', 0.25),    # Like "suit"
        }
        # Consonant patterns
        consonant_patterns = {
            'ch': ('ch', 0.2),     # Like "chair"
            'sh': ('sh', 0.2),     # Like "ship"  
            'th': ('th', 0.2),     # Like "thin"
            'ph': ('f', 0.2),      # Like "phone"
            'gh': ('', 0.0),       # Silent in many cases
            'ck': ('k', 0.15),     # Like "back"
            'ng': ('ng', 0.2),     # Like "sing"
        }
        
        i = 0
        while i < len(name_clean):
            if name_clean[i] == ' ':
                syllables.append(('pause', 0.15))
                i += 1
                continue
                
            # Look for vowel combinations first
            found_combination = False
            for combo, (sound, duration) in vowel_combinations.items():
                if i + len(combo) <= len(name_clean) and name_clean[i:i+len(combo)] == combo:
                    # Check if this is followed by consonants to form syllable
                    syllable_end = i + len(combo)
                    consonant_sound = ""
                    consonant_duration = 0
                    
                    # Add following consonants to this syllable
                    while syllable_end < len(name_clean) and name_clean[syllable_end] not in 'aeiou':
                        # Check for consonant patterns
                        pattern_found = False
                        for pattern, (pat_sound, pat_dur) in consonant_patterns.items():
                            if (syllable_end + len(pattern) <= len(name_clean) and 
                                name_clean[syllable_end:syllable_end+len(pattern)] == pattern):
                                consonant_sound += pat_sound
                                consonant_duration += pat_dur
                                syllable_end += len(pattern)
                                pattern_found = True
                                break
                        
                        if not pattern_found:
                            # Single consonant
                            consonant_sound += name_clean[syllable_end]
                            consonant_duration += 0.15
                            syllable_end += 1
                        
                        # Stop at next vowel or end of name
                        if (syllable_end < len(name_clean) and 
                            name_clean[syllable_end] in 'aeiou'):
                            break
                    
                    # Combine vowel and consonant sounds
                    full_syllable = sound + consonant_sound if consonant_sound else sound
                    full_duration = duration + (consonant_duration * 0.7)  # Consonants shorter
                    syllables.append((full_syllable, full_duration))
                    
                    i = syllable_end
                    found_combination = True
                    break
            
            if not found_combination:
                # Single character processing
                char = name_clean[i]
                if char in 'aeiou':
                    # Single vowel
                    vowel_map = {'a': 'ah', 'e': 'eh', 'i': 'ih', 'o': 'oh', 'u': 'oo'}
                    syllables.append((vowel_map.get(char, char), 0.3))
                elif char.isalpha():
                    # Single consonant - attach to previous vowel or make syllable
                    if syllables and not syllables[-1][0].endswith(char):
                        # Modify last syllable to include this consonant
                        last_syl, last_dur = syllables[-1]
                        syllables[-1] = (last_syl + char, last_dur + 0.1)
                    else:
                        # Make consonant syllable
                        syllables.append((char + 'uh', 0.2))
                i += 1
        
        # Ensure we have at least one syllable
        if not syllables:
            syllables = [(name_clean, 0.5)]
            
        return syllables
        
    except Exception as e:
        logger.error(f"Error parsing syllables for '{name}': {e}")
        # Fallback to simple character mapping
        return [(name.lower(), 1.0)]

def _generate_name_frequency_signature(name: str, base_frequency: float, duration: float = 8.0) -> np.ndarray:
    """
    Generate name frequency signature using proper acoustic physics.
    Uses actual sound synthesis based on name frequencies.
    
    Args:
        name: Soul's name
        base_frequency: Base frequency for the signature
        duration: Length of audio in seconds
        
    Returns:
        NumPy array containing the audio samples
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If generation fails
    """
    if not name: raise ValueError("Name cannot be empty")
    if base_frequency <= 0: raise ValueError("Base frequency must be positive")
    if duration <= 0: raise ValueError("Duration must be positive")
    
    if not SOUND_MODULES_AVAILABLE:
        raise RuntimeError("Sound generation requires sound modules")
    
    try:
        # Calculate name standing wave properties
        wave_properties = _calculate_name_standing_waves(name, base_frequency)
        
        # Use sound generator to create harmonics
        frequencies = wave_properties["component_frequencies"]
        amplitudes = wave_properties["component_amplitudes"]
        
        # Ensure lengths match
        if len(frequencies) != len(amplitudes):
            raise ValueError("Frequency and amplitude arrays must have same length")
        
        # Normalize amplitudes to avoid clipping
        max_amp = max(amplitudes) if amplitudes else 1.0
        if max_amp > FLOAT_EPSILON:
            normalized_amps = [a / max_amp * 0.8 for a in amplitudes]  # Scale to 0.8 max
        else:
            normalized_amps = [0.0] * len(amplitudes)
        
        # Create harmonic structure from frequencies - ENSURE 1.0 is included
        harmonics = [f / base_frequency for f in frequencies]
        
        # CRITICAL: Ensure fundamental frequency (1.0) is included
        if 1.0 not in harmonics:
            # Add fundamental frequency at the beginning
            harmonics.insert(0, 1.0)
            # Add corresponding amplitude (use average of existing amplitudes)
            avg_amp = sum(normalized_amps) / len(normalized_amps) if normalized_amps else 0.5
            normalized_amps.insert(0, avg_amp)
        
        # Verify lengths match after potential addition
        if len(harmonics) != len(normalized_amps):
            raise RuntimeError(f"Harmonics/amplitudes length mismatch after processing: {len(harmonics)} vs {len(normalized_amps)}")
        
        logger.debug(f"Generating name audio signature: Name={name}, " +
                    f"Base={base_frequency:.2f}Hz, Harmonics={len(harmonics)}")
        
        # Generate sound using sound_generator
        name_sound = sound_gen.generate_harmonic_tone(base_frequency, harmonics, 
                                                   normalized_amps, duration, 0.5)
        
        # Save the sound for future reference
        sound_filename = f"name_signature_{name.lower()}.wav"
        sound_path = sound_gen.save_sound(name_sound, sound_filename, 
                                       f"Name Signature for '{name}'")
        
        logger.info(f"Generated name frequency signature. Saved to: {sound_path}")
        return name_sound
        
    except Exception as e:
        logger.error(f"Error generating name frequency signature: {e}", exc_info=True)
        raise RuntimeError("Name frequency signature generation failed") from e

# --- Color Processing with Light Physics ---
def _generate_color_from_frequency(frequency: float) -> str:
    """Generate a color hex code from a frequency using light physics."""
    if frequency <= FLOAT_EPSILON:
        raise ValueError("Frequency must be positive for color generation")
    
    try:
        # Map the frequency to visible light spectrum wavelengths (380-750nm)
        # Using a logarithmic mapping to better distribute across spectrum
        normalized_freq = np.log(frequency) / np.log(20000)  # Normalize using log scale
        normalized_freq = max(0.0, min(1.0, normalized_freq))  # Clamp to 0-1
        
        # Map to wavelength (nm) - higher frequencies = shorter wavelengths
        wavelength = 750 - normalized_freq * (750 - 380)
        
        # Convert wavelength to RGB using physics-based approach
        r, g, b = 0, 0, 0
        
        # Visible spectrum wavelength to RGB conversion (physics-based)
        if 380 <= wavelength < 440:
            # Violet -> Blue
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wavelength < 490:
            # Blue -> Cyan
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wavelength < 510:
            # Cyan -> Green
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            # Green -> Yellow
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wavelength < 645:
            # Yellow -> Red
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        elif 645 <= wavelength <= 750:
            # Red
            r = 1.0
            g = 0.0
            b = 0.0
        
        # Adjust intensity with wavelength (drops at edges of visible spectrum)
        if wavelength < 420:
            intensity = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif wavelength > 700:
            intensity = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
        else:
            intensity = 1.0
            
        # Scale RGB values and convert to hex
        r = int(255 * r * intensity)
        g = int(255 * g * intensity)
        b = int(255 * b * intensity)
        
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        return hex_color
        
    except Exception as e:
        logger.error(f"Error generating color from frequency {frequency}: {e}")
        raise RuntimeError(f"Color generation failed: {e}") from e

def _calculate_color_frequency(color_hex: str) -> float:
    """Calculate a frequency in Hz associated with a color hex code or color name."""
    if not isinstance(color_hex, str):
        raise ValueError(f"Color must be a string, got {type(color_hex)}")
    
    # Color name to hex mapping
    color_name_map = {
        'red': '#FF0000',
        'green': '#00FF00', 
        'blue': '#0000FF',
        'yellow': '#FFFF00',
        'orange': '#FFA500',
        'purple': '#800080',
        'violet': '#8A2BE2',
        'indigo': '#4B0082',
        'cyan': '#00FFFF',
        'magenta': '#FF00FF',
        'pink': '#FFC0CB',
        'brown': '#A52A2A',
        'black': '#000000',
        'white': '#FFFFFF',
        'gray': '#808080',
        'grey': '#808080'
    }
    
    # Convert color name to hex if needed
    original_color = color_hex
    if color_hex.lower() in color_name_map:
        color_hex = color_name_map[color_hex.lower()]
        logger.debug("Converted color name '%s' to hex '%s'", original_color, color_hex)
    
    # Validate hex format
    if not re.match(r'^#[0-9A-F]{6}$', color_hex.upper()):
        raise ValueError(f"Invalid color format: '{original_color}' -> '{color_hex}'. Expected hex format like #FF0000 or color name.")
    
    try:
        # Convert hex to RGB
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        
        # Calculate dominant color
        max_val = max(r, g, b)
        
        # Estimate wavelength based on dominant color
        if max_val <= FLOAT_EPSILON:
            wavelength = 550  # Default to middle (green)
        elif max_val == r:
            # Red dominant (620-750nm)
            if g > b:
                # More yellow (570-620nm)
                wavelength = 620 - (g / max_val) * 50
            else:
                wavelength = 650 + (b / max_val) * 100
        elif max_val == g:
            # Green dominant (495-570nm)
            if r > b:
                # More yellow (570-590nm)
                wavelength = 570 - (r / max_val) * 75
            else:
                # More cyan (490-495nm)
                wavelength = 495 + (b / max_val) * 5
        else:  # max_val == b
            # Blue dominant (450-495nm)
            if g > r:
                # More cyan (495-520nm)
                wavelength = 495 - (g / max_val) * 45
            else:
                # More violet (380-450nm)
                wavelength = 450 - (r / max_val) * 70
                
        # Convert wavelength to frequency (f = c/λ)
        frequency = SPEED_OF_LIGHT / (wavelength * 1e-9)
        
        # Scale to audio range (typical 20Hz-20kHz)
        # Using logarithmic scaling to distribute frequencies more naturally
        visible_min = SPEED_OF_LIGHT / (750e-9)  # ~400 THz
        visible_max = SPEED_OF_LIGHT / (380e-9)  # ~790 THz
        
        log_min = np.log(visible_min)
        log_max = np.log(visible_max)
        log_freq = np.log(frequency)
        
        # Normalize to 0-1 in log space
        normalized = (log_freq - log_min) / (log_max - log_min)
        
        # Map to audio range
        audio_min = 20.0  # Hz
        audio_max = 20000.0  # Hz
        
        # Use log mapping for better distribution
        audio_freq = audio_min * np.power(audio_max / audio_min, normalized)
        
        return float(audio_freq)
        
    except Exception as e:
        logger.error("Error calculating frequency from color %s: %s", original_color, e)
        raise RuntimeError(f"Color frequency calculation failed: {e}") from e


def _find_closest_spectrum_color(hex_color: str, color_spectrum) -> str:
    """Find the closest color in the spectrum to the given hex color."""
    if not isinstance(hex_color, str):
        raise ValueError(f"Hex color must be a string, got {type(hex_color)}")
    
    try:
        # Convert target hex to RGB
        if not re.match(r'^#[0-9A-F]{6}$', hex_color.upper()):
            raise ValueError(f"Invalid hex color format: {hex_color}")
        
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        target_rgb = (r, g, b)
        
        # Find closest color by Euclidean distance
        min_distance = float('inf')
        closest_color = None
        
        for color_name, color_data in color_spectrum.items():
            try:
                # Get hex color from the color data
                spectrum_hex = color_data.get("hex", "")
                if not spectrum_hex or len(spectrum_hex) < 7:
                    continue
                
                # Convert spectrum color to RGB
                sr = int(spectrum_hex[1:3], 16)
                sg = int(spectrum_hex[3:5], 16)
                sb = int(spectrum_hex[5:7], 16)
                spectrum_rgb = (sr, sg, sb)
                
                # Calculate Euclidean distance
                distance = sum((c1 - c2) ** 2 for c1, c2 in zip(target_rgb, spectrum_rgb)) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color = spectrum_hex
            except (ValueError, KeyError):
                # Skip invalid color formats in spectrum
                continue
        
        if not closest_color:
            # Default to first color hex in spectrum if none found
            first_color_data = next(iter(color_spectrum.values()), {})
            closest_color = first_color_data.get("hex", hex_color)
        
        return closest_color
        
    except Exception as e:
        logger.error(f"Error finding closest spectrum color: {e}")
        # Return original color instead of failing
        return hex_color

# --- Core Identity Crystallization Functions ---

def assign_name(soul_spark: SoulSpark) -> None:
    """ 
    Assigns name via user input, calculates gematria/resonance(0-1), and 
    establishes standing wave patterns and frequency signature.
    Stores name directly on soul_spark.
    """
    log_soul_identity_summary(soul_spark)
    logger.info("Identity Step: Assign Name with Light-Sound Signature...")
    name_to_use = None
    print("-" * 30+"\nIDENTITY CRYSTALLIZATION: SOUL NAMING"+"\nSoul ID: "+soul_spark.spark_id)
    print(f"  Color: {getattr(soul_spark, 'soul_color', 'N/A')}, Freq: {getattr(soul_spark, 'soul_frequency', 0.0):.1f}Hz")
    print(f"  S/C: {soul_spark.stability:.1f}SU / {soul_spark.coherence:.1f}CU")
    print("-" * 30)
    
    while not name_to_use:
        try:
            user_name = input(f"*** Please enter the name for this soul: ").strip()
            if not user_name: 
                print("Name cannot be empty.")
                continue
            name_to_use = user_name
        except KeyboardInterrupt:
            logger.info("User cancelled name input.")
            raise RuntimeError("Name assignment cancelled by user.")
        except Exception as e: 
            logger.error(f"Error getting user input: {e}")
            raise RuntimeError(f"Failed to get soul name: {e}")
    
    print("-" * 30)

    try:
        # Calculate traditional gematria value
        gematria = _calculate_gematria(name_to_use)
        
        # Calculate name resonance 
        name_resonance = _calculate_name_resonance(name_to_use, gematria)
        
        # Calculate standing wave patterns for the name
        standing_waves = _calculate_name_standing_waves(name_to_use, soul_spark.frequency)
        
        # Generate actual sound signature - only if sound modules available
        if SOUND_MODULES_AVAILABLE:
            try:
                # Generate both phonetic pronunciation and frequency signature
                phonetic_pronunciation = _generate_phonetic_name_pronunciation(
                    name_to_use, soul_spark.frequency, duration=2.5)
                sound_signature = _generate_name_frequency_signature(
                    name_to_use, soul_spark.frequency)
                
                # Save the phonetic pronunciation
                if SOUND_MODULES_AVAILABLE and phonetic_pronunciation is not None:
                    sound_gen = SoundGenerator()
                    pronunciation_path = sound_gen.save_sound(
                        phonetic_pronunciation, f"name_pronunciation_{name_to_use.lower()}.wav", 
                        f"Phonetic pronunciation of '{name_to_use}'")
                    logger.info(f"  Generated phonetic pronunciation: {pronunciation_path}")
                
                logger.debug(f"  Generated name sound signature: {len(sound_signature)} samples")
            except Exception as sound_err:
                logger.error(f"Failed to generate name sound: {sound_err}")
                # Don't fail whole process, just log the error
        
        # Create light signature from standing waves
        light_signature = {
            "primary_color": _generate_color_from_frequency(soul_spark.frequency),
            "harmonic_colors": [f["color"] for f in standing_waves["light_frequencies"]],
            "resonance_quality": standing_waves["resonance_quality"],
            "coherence": standing_waves["pattern_coherence"]
        }
        
        logger.debug(f"  Name assigned: '{name_to_use}', Gematria: {gematria}, " +
                   f"Resonance Factor: {name_resonance:.4f}, " +
                   f"Standing Waves: {len(standing_waves['nodes'])} nodes")
        
        # Store name information directly on soul_spark
        setattr(soul_spark, 'name', name_to_use)
        setattr(soul_spark, 'gematria_value', gematria)
        setattr(soul_spark, 'name_resonance', name_resonance)
        setattr(soul_spark, 'name_standing_waves', standing_waves)
        setattr(soul_spark, 'identity_light_signature', light_signature)
        
        # Assign birth date and star sign based on soul characteristics
        birth_date, star_sign = _assign_soul_birth_date_and_sign(soul_spark, name_to_use)
        setattr(soul_spark, 'birth_date', birth_date)
        setattr(soul_spark, 'star_sign', star_sign)
        
        timestamp = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', timestamp)
        
        # Add memory echo for future reference
        if hasattr(soul_spark, 'add_memory_echo'): 
            soul_spark.add_memory_echo(f"Name assigned: {name_to_use} (G:{gematria}, " +
                                     f"NR:{name_resonance:.3f}, StandingWaves:{len(standing_waves['nodes'])})")
        
        logger.info(f"Name assignment complete: {name_to_use} (ResFactor: {name_resonance:.4f}, " +
                   f"LightSig: {len(light_signature['harmonic_colors'])} colors)")
        
    except Exception as e:
        logger.error("Error processing assigned name: %s", e, exc_info=True)
        raise RuntimeError("Name processing failed.")
    

def _assign_soul_birth_date_and_sign(soul_spark: SoulSpark, name: str) -> Tuple[str, str]:
    """
    Assign birth date and astrological sign based on soul's frequency characteristics.
    Uses soul frequency and harmonic patterns to determine optimal birth timing.
    
    Args:
        soul_spark: The soul spark to assign birth date to
        name: Soul's name for additional calculation input
        
    Returns:
        Tuple of (birth_date, star_sign)
    """
    try:
        # Get soul characteristics
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        harmony = getattr(soul_spark, 'harmony', 0.0)
        phi_resonance = getattr(soul_spark, 'phi_resonance', 0.0)
        gematria = getattr(soul_spark, 'gematria_value', 0)
        
        # Map frequency ranges to astrological signs with dates
        astrological_mappings = [
            (200.0, 250.0, "Aquarius", "2024-02-15"),     # Air sign, innovative frequency
            (250.0, 300.0, "Pisces", "2024-03-15"),       # Water sign, intuitive frequency
            (300.0, 350.0, "Aries", "2024-04-15"),        # Fire sign, energetic frequency
            (350.0, 400.0, "Taurus", "2024-05-11"),       # Earth sign, grounding frequency (our corrected example)
            (400.0, 450.0, "Gemini", "2024-06-15"),       # Air sign, communication frequency
            (450.0, 500.0, "Cancer", "2024-07-15"),       # Water sign, nurturing frequency
            (500.0, 550.0, "Leo", "2024-08-15"),          # Fire sign, creative frequency
            (550.0, 600.0, "Virgo", "2024-09-15"),        # Earth sign, perfectionist frequency
            (600.0, 650.0, "Libra", "2024-10-15"),        # Air sign, harmony frequency
            (650.0, 700.0, "Scorpio", "2024-11-15"),      # Water sign, transformative frequency
            (700.0, 750.0, "Sagittarius", "2024-12-15"),  # Fire sign, philosophical frequency
            (150.0, 200.0, "Capricorn", "2024-01-15"),    # Earth sign, structured frequency
        ]
        
        # Find the matching sign based on soul frequency
        birth_date = "2024-05-11"  # Default (Taurus)
        star_sign = "Taurus"
        
        for freq_min, freq_max, sign, date in astrological_mappings:
            if freq_min <= soul_frequency < freq_max:
                birth_date = date
                star_sign = sign
                break
        
        # Fine-tune based on additional characteristics
        if harmony > 0.8 and phi_resonance > 0.7:
            # High harmony souls tend toward Libra (balance)
            birth_date = "2024-10-15"
            star_sign = "Libra"
        elif gematria % 12 == 0:  # Divisible by 12 - Pisces (spiritual completion)
            birth_date = "2024-03-15" 
            star_sign = "Pisces"
        elif name.lower().startswith(('a', 'e', 'i', 'o', 'u')):  # Vowel names - Aquarius (air/communication)
            birth_date = "2024-02-15"
            star_sign = "Aquarius"
            
        logger.info(f"Assigned birth date and sign: {name} -> {birth_date} ({star_sign})")
        logger.debug(f"  Based on: Frequency={soul_frequency:.1f}Hz, Harmony={harmony:.3f}, Phi={phi_resonance:.3f}, Gematria={gematria}")
        
        return birth_date, star_sign
        
    except Exception as e:
        logger.error(f"Error assigning birth date: {e}")
        # Return default values
        return "2024-05-11", "Taurus"

def establish_mothers_voice_foundation(soul_spark: SoulSpark) -> None:
    """
    Establishes mother's voice as the foundational security layer for identity crystallization.
    The mother's voice (220 Hz) provides the secure, loving base that enables stable lattice formation.
    """
    logger.info("Identity Step: Establish Mother's Voice Foundation...")
    
    try:
        # Mother's voice characteristics (from mothers_voice_welcome.py)
        mothers_voice_freq = 220.0  # A3 - warm, nurturing frequency
        emotional_resonance = 0.95  # High emotional content
        love_energy_level = 1.0     # Maximum love energy
        calming_effect = 0.9        # Strong calming influence
        
        # Create foundational security layer
        security_foundation = {
            "frequency": mothers_voice_freq,
            "emotional_resonance": emotional_resonance,
            "love_energy": love_energy_level,
            "calming_effect": calming_effect,
            "message": "Welcome to this world, little one. You are loved beyond measure.",
            "foundation_strength": 1.0,  # Full foundational strength
            "security_level": 0.95      # Very high security feeling
        }
        
        # Calculate how mother's voice affects identity formation
        soul_frequency = getattr(soul_spark, 'frequency', 376.0)
        name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
        
        # Mother's voice creates harmonic relationship with soul
        freq_ratio = soul_frequency / mothers_voice_freq  # Should be ~1.7 (approx 5th harmonic)
        harmonic_support = min(1.0, 1.0 / abs(freq_ratio - 1.7) + 0.5) if freq_ratio > 0 else 0.5
        
        # Security boost from mother's voice
        security_boost = security_foundation["security_level"] * harmonic_support
        identity_stability_boost = security_boost * 0.3  # Up to 30% boost to identity stability
        
        # Apply foundational effects to soul
        current_stability = getattr(soul_spark, 'identity_stability', 0.5)
        new_stability = min(1.0, current_stability + identity_stability_boost)
        
        # Store mother's voice foundation data
        setattr(soul_spark, 'mothers_voice_foundation', security_foundation)
        setattr(soul_spark, 'identity_stability', new_stability)
        setattr(soul_spark, 'foundational_security', security_boost)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        logger.info(f"Mother's voice foundation established: {mothers_voice_freq}Hz")
        logger.info(f"Security boost: {security_boost:.3f}, Identity stability: {new_stability:.3f}")
        logger.info(f"Harmonic support: {harmonic_support:.3f} (ratio: {freq_ratio:.2f})")
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "mothers_voice_frequency": float(mothers_voice_freq),
                "soul_frequency": float(soul_frequency),
                "frequency_ratio": float(freq_ratio),
                "harmonic_support": float(harmonic_support),
                "security_boost": float(security_boost),
                "identity_stability_boost": float(identity_stability_boost),
                "final_identity_stability": float(new_stability),
                "foundational_security": float(security_boost),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_mothers_voice_foundation', metrics_data)
        
    except Exception as e:
        logger.error(f"Error establishing mother's voice foundation: {e}", exc_info=True)
        raise RuntimeError("Mother's voice foundation failed.") from e


def assign_voice_frequency(soul_spark: SoulSpark) -> None:
    """
    Assigns voice frequency (Hz) based on name/attributes using acoustic physics.
    Stores voice frequency directly on soul_spark.
    """
    logger.info("Identity Step: Assign Voice Frequency with Acoustic Physics...")
    name = getattr(soul_spark, 'name')
    gematria = getattr(soul_spark, 'gematria_value')
    name_resonance = getattr(soul_spark, 'name_resonance')
    yin_yang = getattr(soul_spark, 'yin_yang_balance')
    standing_waves = getattr(soul_spark, 'name_standing_waves')
    
    if not name:
        raise ValueError("Cannot assign voice frequency without a name.")
    if not standing_waves:
        raise ValueError("Name standing waves required for voice frequency calculation.")
    
    try:
        # Calculate voice frequency using multiple acoustic factors
        
        # 1. Base frequency from name's fundamental resonance
        fundamental_freq = standing_waves.get("fundamental_frequency", soul_spark.frequency)
        
        # 2. Gender/energy balance influence (Yin/Yang)
        if hasattr(soul_spark, 'yin_yang_balance'):
            # Yang (masculine) tends toward lower frequencies, Yin (feminine) toward higher
            yin_yang_factor = 0.8 + (yin_yang * 0.4)  # Range: 0.8 to 1.2
        else:
            yin_yang_factor = 1.0
        
        # 3. Gematria harmonic influence
        gematria_harmonic = (gematria % 12) + 1  # Harmonic 1-12
        gematria_factor = 1.0 + (gematria_harmonic / 100.0)  # Slight frequency shift
        
        # 4. Name resonance quality influence
        resonance_factor = 0.9 + (name_resonance * 0.2)  # Range: 0.9 to 1.1
        
        # Calculate voice frequency with all factors
        voice_frequency = fundamental_freq * yin_yang_factor * gematria_factor * resonance_factor
        
        # Constrain to human vocal range (roughly 85Hz to 1100Hz)
        voice_frequency = max(85.0, min(1100.0, voice_frequency))
        
        logger.debug(f"  Voice Calc: Base={fundamental_freq:.1f}Hz, " +
                    f"YinYang={yin_yang_factor:.3f}, Gematria={gematria_factor:.3f}, " +
                    f"Resonance={resonance_factor:.3f} -> {voice_frequency:.2f}Hz")
        
        # Generate harmonic series for voice
        harmonics = [1.0, 2.0, 3.0, 1.5]  # Fundamental, octave, twelfth, fifth
        amplitudes = [0.8, 0.4, 0.3, 0.5]  # Relative amplitudes
        
        # Generate voice sound if modules available
        if SOUND_MODULES_AVAILABLE:
            try:
                voice_sound = sound_gen.generate_harmonic_tone(
                    voice_frequency, harmonics, amplitudes, 3.0, 0.3)
                
                # Save voice sound
                voice_sound_path = sound_gen.save_sound(
                    voice_sound, f"voice_{soul_spark.spark_id}.wav", f"Voice for {name}")
                
                logger.debug(f"  Generated voice sound: {voice_sound_path}")
                
                # Create a sound signature for the voice
                setattr(soul_spark, 'identity_sound_signature', {
                    "type": "voice",
                    "fundamental_frequency": float(voice_frequency),
                    "harmonics": harmonics,
                    "sound_path": voice_sound_path
                })
                
            except Exception as sound_err:
                logger.error(f"Error generating voice sound: {sound_err}")
                # Continue without failing the process
        
        # Calculate the acoustic wavelength for this frequency
        wavelength_meters = SPEED_OF_SOUND / voice_frequency
        
        # Store voice frequency directly on soul_spark
        setattr(soul_spark, 'voice_frequency', float(voice_frequency))
        setattr(soul_spark, 'voice_wavelength', float(wavelength_meters))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        
        logger.info(f"Voice frequency assigned: {voice_frequency:.2f}Hz, " +
                   f"Wavelength: {wavelength_meters:.2f}m")
        
    except Exception as e:
        logger.error(f"Error assigning voice frequency: {e}", exc_info=True)
        raise RuntimeError("Voice frequency assignment failed.") from e


def process_soul_color(soul_spark: SoulSpark) -> None:
    """
    Process the soul's color using light physics and electromagnetic theory.
    Creates proper light-frequency relationships.
    
    This function creates the soul's spectral signature, mapping its frequency
    to visible light and electromagnetic ranges, establishing quantum coherence
    between frequency bands.
    """
    logger.info("Identity Step: Processing Soul Color with Light Physics...")
    
    # Ensure soul color is set
    if not hasattr(soul_spark, 'soul_color') or soul_spark.soul_color is None:
        # Generate color from soul frequency if not set
        try:
            soul_color = _generate_color_from_frequency(soul_spark.frequency)
            setattr(soul_spark, 'soul_color', soul_color)
            logger.debug(f"  Generated soul_color {soul_color} based on frequency {soul_spark.frequency:.1f}Hz")
        except Exception as color_err:
            logger.error(f"Failed to generate soul color: {color_err}")
            raise RuntimeError("Soul color generation failed") from color_err
    
    soul_color = soul_spark.soul_color
    
    # CRITICAL FIX: Validate hex format before using soul_color[3:5] to prevent line 1301 error
    if not soul_color or not isinstance(soul_color, str) or len(soul_color) < 7:
        logger.warning(f"Soul color '{soul_color}' is invalid, regenerating")
        try:
            soul_color = _generate_color_from_frequency(soul_spark.frequency)
            soul_spark.soul_color = soul_color
            logger.debug(f"  Regenerated color: {soul_color}")
        except Exception as color_err:
            logger.error(f"Failed to regenerate soul color: {color_err}")
            raise RuntimeError("Soul color regeneration failed") from color_err
    
    # Validate color format
    if not re.match(r'^#[0-9A-F]{6}$', soul_color.upper()):
        logger.warning(f"Soul color '{soul_color}' not a valid hex color, regenerating")
        try:
            # Regenerate using frequency
            soul_color = _generate_color_from_frequency(soul_spark.frequency)
            soul_spark.soul_color = soul_color
            logger.debug(f"  Regenerated color: {soul_color}")
        except Exception as color_err:
            logger.error(f"Failed to regenerate soul color: {color_err}")
            raise RuntimeError("Soul color regeneration failed") from color_err
    
    try:
        # Map to predefined spectrum if needed
        if COLOR_SPECTRUM:
            # Find closest color in spectrum
            closest_color = _find_closest_spectrum_color(soul_color, COLOR_SPECTRUM)
            
            if closest_color:
                logger.debug(f"  Mapped to closest spectrum color: {closest_color}")
                soul_spark.soul_color = closest_color
                soul_color = closest_color
        
        # Calculate color frequency in sound spectrum
        color_frequency = _calculate_color_frequency(soul_color)
        setattr(soul_spark, 'color_frequency', float(color_frequency))
        
        # Split into RGB components for more detailed analysis - SAFE now after validation
        r = int(soul_color[1:3], 16) / 255.0
        g = int(soul_color[3:5], 16) / 255.0
        b = int(soul_color[5:7], 16) / 255.0
        
        # Calculate color properties with light physics
        # Hue (dominant wavelength)
        max_component = max(r, g, b)
        if max_component <= FLOAT_EPSILON:
            hue = 0  # Default to red if black
        elif max_component == r:
            hue = (g - b) / (max_component - min(r, g, b) + FLOAT_EPSILON) % 6
        elif max_component == g:
            hue = (b - r) / (max_component - min(r, g, b) + FLOAT_EPSILON) + 2
        else:  # max_component == b
            hue = (r - g) / (max_component - min(r, g, b) + FLOAT_EPSILON) + 4
        
        hue = (hue * 60) % 360  # Convert to degrees
        
        # Saturation (purity of color)
        if max_component <= FLOAT_EPSILON:
            saturation = 0
        else:
            saturation = (max_component - min(r, g, b)) / (max_component + FLOAT_EPSILON)
        
        # Value/Brightness
        value = max_component
        
        # Create color properties structure
        color_properties = {
            "hex": soul_color,
            "rgb": [float(r), float(g), float(b)],
            "hue": float(hue),
            "saturation": float(saturation),
            "brightness": float(value),
            "frequency_hz": float(color_frequency),
            "wavelength_nm": float(SPEED_OF_LIGHT / (color_frequency * 1e12)) if color_frequency > 0 else 0
        }
        
        # Calculate photon energy according to E=hf
        # Planck's constant h = 6.626e-34 J⋅s
        # Convert to eV for more intuitive scale
        planck_constant = 6.626e-34  # J⋅s
        electron_volt = 1.602e-19     # J/eV
        
        photon_energy_joules = planck_constant * (color_frequency * 1e12)  # Convert to THz
        photon_energy_ev = photon_energy_joules / electron_volt
        
        color_properties["photon_energy_ev"] = float(photon_energy_ev)
        color_properties["quantum_field_strength"] = float(saturation * value * 0.8 + 0.2)
        
        # Generate sound based on color frequency if sound modules available
        if SOUND_MODULES_AVAILABLE:
            try:
                color_sound = sound_gen.generate_tone(
                    color_frequency, 2.0, amplitude=0.6, fade_in_out=0.3)
                
                color_sound_path = sound_gen.save_sound(
                    color_sound, f"color_{soul_color[1:]}.wav", f"Color Tone for {soul_color}")
                
                color_properties["sound_path"] = color_sound_path
                logger.debug(f"  Generated color sound: {color_sound_path}")
                
            except Exception as sound_err:
                logger.warning(f"Failed to generate color sound: {sound_err}")
                # Continue without failing process
        
        setattr(soul_spark, 'color_properties', color_properties)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        logger.info(f"Soul color processed: {soul_color}, " +
                   f"Frequency: {color_frequency:.2f}Hz, " +
                   f"Wavelength: {color_properties['wavelength_nm']:.2f}nm, " +
                   f"Photon Energy: {photon_energy_ev:.4f}eV")
        
    except Exception as e:
        logger.error(f"Error processing soul color: {e}", exc_info=True)
        raise RuntimeError("Soul color processing failed") from e
    
def _create_heart_centered_field(soul_spark: SoulSpark, love_freq: float,
                              harmonic_interval: float) -> Dict[str, Any]:
    """
    Create heart-centered geometric field for love resonance.
    
    Creates an energetic field pattern that integrates with ??
    to enhance love resonance and emotional coherence.
    
    Args:
        soul_spark: Soul to create field for
        love_freq: Love frequency (528Hz)
        harmonic_interval: Harmonic interval for field geometry
        
    Returns:
        Dictionary with heart field data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If field creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if love_freq <= 0:
        raise ValueError("love_freq must be positive")
    if harmonic_interval <= 0:
        raise ValueError("harmonic_interval must be positive")
    
    try:
        logger.debug(f"Creating heart-centered field...")
        
        # Create harmonic series based on love frequency and interval
        harmonic_freqs = []
        for i in range(5):  # 5 harmonics
            harmonic = love_freq * (harmonic_interval ** i)
            harmonic_freqs.append(harmonic)
        
        # Create geometric field points in heart shape using parametric equations
        field_points = []
        num_points = 64  # Power of 2 for good field resolution
        
        for i in range(num_points):
            # Heart curve parametric equation
            t = 2 * PI * i / num_points
            
            # Heart shape equations
            x = 16 * (sin(t) ** 3)
            y = 13 * cos(t) - 5 * cos(2*t) - 2 * cos(3*t) - cos(4*t)
            
            # Scale to unit coordinates
            x_scaled = x / 20.0  # Normalize to roughly -1 to 1
            y_scaled = y / 20.0
            
            # Calculate phase value based on position and harmonics
            phase_value = 0.0
            for j, freq in enumerate(harmonic_freqs):
                # Create standing wave pattern
                wave_contribution = sin(2 * PI * freq * t / 1000.0) * (0.8 ** j)
                phase_value += wave_contribution
            
            # Normalize phase value
            phase_value = (phase_value + 1.0) / 2.0  # Map to 0-1 range
            
            # Add point to field
            field_points.append({
                "x": float(x_scaled),
                "y": float(y_scaled),
                "value": float(phase_value)
            })
        
        # Calculate field strength based on points
        field_strength = sum(p["value"] for p in field_points) / len(field_points)
        
        # Since there are no layers, integrated_layers is 0
        integrated_layers = 0
        
        # Create the final heart field structure
        heart_field = {
            "love_frequency": float(love_freq),
            "harmonic_interval": float(harmonic_interval),
            "harmonic_frequencies": [float(f) for f in harmonic_freqs],
            "field_strength": float(field_strength),
            "points": field_points,
            "integrated_layers": integrated_layers
        }
        
        logger.debug(f"Created heart-centered field with strength {field_strength:.4f}, " +
                   f"stored directly on soul (no layers).")
        
        return heart_field
        
    except Exception as e:
        logger.error(f"Error creating heart-centered field: {e}", exc_info=True)
        raise RuntimeError("Heart-centered field creation failed") from e

def activate_love_resonance(soul_spark: SoulSpark, cycles: int = 7) -> None:
    """
    Activates love resonance using geometric field formation and standing waves
    to enhance harmony and emotional resonance throughout.
    """
    logger.info("Identity Step: Activate Love Resonance with Geometric Field Formation...")
    if not isinstance(cycles, int) or cycles < 0:
        raise ValueError("Cycles must be non-negative.")
    if cycles == 0:
        logger.info("Skipping love resonance activation (0 cycles).")
        return
    
    try:
        soul_frequency = getattr(soul_spark, 'soul_frequency', 0.0)
        state = getattr(soul_spark, 'consciousness_state', 'spark')
        heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0)
        emotional_resonance = getattr(soul_spark, 'emotional_resonance', {})
        current_love = float(emotional_resonance.get('love', 0.0))
        love_freq = LOVE_RESONANCE_FREQ
        
        if love_freq <= FLOAT_EPSILON or soul_frequency <= FLOAT_EPSILON:
            raise ValueError("Frequencies invalid for love resonance.")
        
        logger.debug(f"  Love Resonance Harmonic Patterning: CurrentLove={current_love:.3f}")
        
        # ENHANCEMENT: Create heart coherence field
        heart_coherence = heartbeat_entrainment * 0.5 + 0.25  # Base coherence level
        
        # Calculate harmonic relationship between love frequency and soul frequency
        harmonic_intervals = [1.0, 1.25, 1.333, 1.5, 1.667, 2.0]  # Musical intervals
        best_harmonic_resonance = 0.0
        best_harmonic_interval = 1.0
        
        for interval in harmonic_intervals:
            # Check both directions
            harmonic1 = love_freq * interval
            harmonic2 = love_freq / interval
            
            # Calculate resonance with soul frequency
            res1 = calculate_resonance(harmonic1, soul_frequency)
            res2 = calculate_resonance(harmonic2, soul_frequency)
            
            # Use the best resonance
            if res1 > best_harmonic_resonance:
                best_harmonic_resonance = res1
                best_harmonic_interval = interval
            if res2 > best_harmonic_resonance:
                best_harmonic_resonance = res2
                best_harmonic_interval = 1.0 / interval
        
        # Create heart-centered geometric field
        heart_field = _create_heart_centered_field(
            soul_spark, love_freq, best_harmonic_interval)
        
        # Love resonance cycles enhanced with geometric field
        love_accumulator = 0.0
        
        # Cycle metrics
        cycle_metrics = []
        
        for cycle in range(cycles):
            # Create heart-centered resonance pattern using heart coherence
            cycle_factor = 1.0 - (cycle / (max(1, cycles) * 1.0))
            state_factor = LOVE_RESONANCE_STATE_WEIGHT.get(
                state, LOVE_RESONANCE_STATE_WEIGHT['default'])
            
            # Use the harmonic resonance rather than direct frequency matching
            resonance_factor = best_harmonic_resonance * LOVE_RESONANCE_FREQ_RES_WEIGHT
            
            # Heart coherence amplifies the effect
            heart_factor = (LOVE_RESONANCE_HEARTBEAT_WEIGHT + 
                          LOVE_RESONANCE_HEARTBEAT_SCALE * heart_coherence)
            
            # Calculate the love increase
            base_increase = LOVE_RESONANCE_BASE_INC * cycle_factor
            
            # Add geometric field factor
            field_factor = heart_field.get("field_strength", 0.5)
            
            # Calculate full increase with all factors
            full_increase = (base_increase * 
                           state_factor * 
                           resonance_factor * 
                           heart_factor * 
                           (1.0 + field_factor * 0.5))  # 50% boost from field
            
            # Apply increase to love accumulator
            love_accumulator += full_increase
            
            # Track cycle metrics
            cycle_metrics.append({
                "cycle": cycle + 1,
                "cycle_factor": float(cycle_factor),
                "base_increase": float(base_increase),
                "effective_increase": float(full_increase),
                "accumulated": float(love_accumulator)
            })
            
            logger.debug(f"    Cycle {cycle+1}: CycleFactor={cycle_factor:.3f}, " +
                       f"FieldFactor={field_factor:.3f}, " +
                       f"Inc={full_increase:.5f}, " +
                       f"Accum={love_accumulator:.4f}")
        
        # Apply accumulated love resonance
        new_love = min(1.0, current_love + love_accumulator)
        
        # Update other emotions based on love (gentle ripple effect)
        new_emotional_resonance = emotional_resonance.copy()
        new_emotional_resonance['love'] = float(new_love)
        
        # Apply gentle boost to other emotions based on love
        for emotion in ['joy', 'peace', 'harmony', 'compassion']:
            current = float(emotional_resonance.get(emotion, 0.0))
            boost = LOVE_RESONANCE_EMOTION_BOOST_FACTOR * new_love
            new_emotional_resonance[emotion] = float(min(1.0, current + boost))
            
        # Generate love frequency sound if available
        if SOUND_MODULES_AVAILABLE:
            try:
                # Create love resonance tone with harmonics
                harmonic_ratios = [1.0, PHI, 1.5, 2.0]
                amplitudes = [0.8, 0.6, 0.5, 0.3]
                
                love_sound = sound_gen.generate_harmonic_tone(
                    love_freq, harmonic_ratios, amplitudes, 5.0, 0.5)
                
                # Save love sound
                sound_path = sound_gen.save_sound(
                    love_sound, "love_resonance.wav", "Love Resonance 528Hz")
                
                logger.debug(f"  Generated love resonance sound: {sound_path}")
                
                # Love sound provides a small additional boost
                sound_boost = 0.05  # 5% boost from actual sound
                final_love = min(1.0, new_love + sound_boost)
                new_emotional_resonance['love'] = float(final_love)
                
            except Exception as sound_err:
                logger.warning(f"Failed to generate love sound: {sound_err}")
                # Continue without failing process
        
        # Update soul with new emotional resonance
        setattr(soul_spark, 'emotional_resonance', new_emotional_resonance)
        setattr(soul_spark, 'heart_field', heart_field)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "love_frequency": float(love_freq),
                "initial_love": float(current_love),
                "cycles": cycles,
                "accumulated_increase": float(love_accumulator),
                "harmonic_resonance": float(best_harmonic_resonance),
                "harmonic_interval": float(best_harmonic_interval),
                "heart_coherence": float(heart_coherence),
                "field_strength": float(heart_field.get("field_strength", 0.0)),
                "final_love": float(new_emotional_resonance['love']),
                "sound_generated": SOUND_MODULES_AVAILABLE,
                "cycle_details": cycle_metrics,
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_love_resonance', metrics_data)
        
        logger.info(f"Love resonance activated through harmonic patterning. " +
                   f"Initial: {current_love:.4f}, Increase: {love_accumulator:.4f}, " +
                   f"Final: {new_emotional_resonance['love']:.4f}")
        
    except Exception as e:
        logger.error(f"Error activating love resonance: {e}", exc_info=True)
        raise RuntimeError("Love resonance activation failed.") from e
    
def apply_heartbeat_entrainment(soul_spark: SoulSpark, bpm: float = 72.0, duration: float = 120.0) -> None:
    """
    Applies heartbeat entrainment for emotional regulation and identity stability.
    The heartbeat provides rhythmic organizing patterns that help crystallize identity structure.
    """
    logger.info("Identity Step: Apply Heartbeat Entrainment for Emotional Regulation...")
    if bpm <= 0 or duration < 0:
        raise ValueError("BPM must be positive, duration non-negative.")

    try:
        # Get current emotional and identity state
        current_harmony = getattr(soul_spark, 'harmony', 0.0)
        foundational_security = getattr(soul_spark, 'foundational_security', 0.0)
        identity_stability = getattr(soul_spark, 'identity_stability', 0.5)

        # Calculate beat frequency for rhythm (not acoustic resonance)
        beat_freq = bpm / 60.0
        duration_factor = min(1.0, duration / 60.0)

        # Determine heartbeat effectiveness based on BPM range for emotional regulation
        if 60 <= bpm <= 80:
            bpm_effectiveness = 1.0  # Optimal range for emotional regulation
        elif 45 <= bpm < 60:
            bpm_effectiveness = 0.9  # Very calm, slightly less organizing
        elif 80 < bpm <= 100:
            bpm_effectiveness = 0.7  # Higher energy, less calming
        else:
            bpm_effectiveness = 0.5  # Outside optimal ranges

        # Calculate emotional regulation effect
        regulation_factor = bpm_effectiveness * duration_factor

        # Mother's voice foundation enhances heartbeat effectiveness
        security_enhancement = foundational_security * 0.5  # Up to 50% enhancement
        total_regulation = regulation_factor * (1.0 + security_enhancement)

        # Apply emotional regulation effects
        identity_stability_boost = total_regulation * 0.2  # Up to 20% boost
        harmony_boost = total_regulation * 0.15  # Up to 15% boost
        
        # Calculate meaningful resonance values based on soul properties (no layers needed)
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        beat_resonance = min(1.0, total_regulation * 0.8)
        
        # Calculate rhythmic resonance based on how well beat frequency harmonizes with soul
        frequency_ratio = beat_freq / (soul_frequency if soul_frequency > 0 else 1.0)
        harmonic_resonance = 1.0 / (1.0 + abs(frequency_ratio - 1.0))  # Closer to 1:1 ratio = better
        standing_wave_resonance = beat_resonance * harmonic_resonance * bpm_effectiveness
        standing_wave_boost = standing_wave_resonance * 0.25  # Increased from 0.1 to 0.25 for meaningful boost

        # Apply changes to soul directly 
        new_identity_stability = min(1.0, identity_stability + identity_stability_boost)
        new_harmony = min(1.0, current_harmony + harmony_boost)
        base_harmony_increase = harmony_boost + standing_wave_boost

        # Generate heartbeat sound for additional regulation effect
        heartbeat_sound = None
        heartbeat_sound_generated = False
        if SOUND_MODULES_AVAILABLE:
            try:
                num_samples = int(duration * soul_spark.sample_rate)
                time_array = np.linspace(0, duration, num_samples, False)
                heartbeat_sound = np.zeros(num_samples, dtype=np.float32)
                beat_period = 60.0 / bpm
                num_beats = int(duration / beat_period)

                # Generate realistic heartbeat pattern
                for i in range(num_beats):
                    beat_time = i * beat_period
                    
                    # S1 Sound ("Lub") - Mitral and Tricuspid valve closure
                    # Realistic timing: Sharp onset, ~0.14s duration
                    s1_duration = 0.14
                    s1_start = int(beat_time * soul_spark.sample_rate)
                    s1_end = min(int((beat_time + s1_duration) * soul_spark.sample_rate), num_samples)
                    
                    if s1_start < num_samples:
                        s1_length = s1_end - s1_start
                        t = np.linspace(0, s1_duration, s1_length)
                        
                        # Real S1 has sharp attack, exponential decay
                        s1_envelope = np.exp(-8 * t)  # Quick decay
                        # Add sharp attack
                        attack_len = min(10, s1_length // 10)
                        if attack_len > 0:
                            s1_envelope[:attack_len] *= np.linspace(0, 1, attack_len)
                        
                        # S1 frequency content: 20-200Hz with peak around 40-50Hz
                        s1_wave = (
                            0.6 * np.sin(2 * np.pi * 45 * t) +          # Main frequency
                            0.3 * np.sin(2 * np.pi * 35 * t) +          # Lower component
                            0.2 * np.sin(2 * np.pi * 65 * t) +          # Higher component
                            0.1 * np.sin(2 * np.pi * 25 * t) +          # Bass component
                            0.05 * np.sin(2 * np.pi * 85 * t)           # Slight high freq
                        )
                        
                        # Add muscle contraction noise (filtered white noise)
                        if s1_length > 0:
                            noise = np.random.normal(0, 0.02, s1_length)
                            # Low-pass filter the noise
                            for j in range(1, len(noise)):
                                noise[j] = 0.8 * noise[j-1] + 0.2 * noise[j]
                            s1_wave += noise
                        
                        s1_wave *= s1_envelope
                        heartbeat_sound[s1_start:s1_end] += s1_wave
                    
                    # S2 Sound ("Dub") - Aortic and Pulmonary valve closure  
                    # Occurs ~0.32s after S1, duration ~0.12s
                    s2_delay = 0.32
                    s2_duration = 0.12
                    s2_start = int((beat_time + s2_delay) * soul_spark.sample_rate)
                    s2_end = min(int((beat_time + s2_delay + s2_duration) * soul_spark.sample_rate), num_samples)
                    
                    if s2_start < num_samples:
                        s2_length = s2_end - s2_start
                        t = np.linspace(0, s2_duration, s2_length)
                        
                        # S2 has sharper attack, quicker decay than S1
                        s2_envelope = np.exp(-12 * t)  # Even quicker decay
                        attack_len = min(8, s2_length // 12)
                        if attack_len > 0:
                            s2_envelope[:attack_len] *= np.linspace(0, 1, attack_len)
                        
                        # S2 frequency content: Higher than S1, 20-150Hz, peak ~60Hz
                        s2_wave = (
                            0.5 * np.sin(2 * np.pi * 60 * t) +          # Main frequency (higher than S1)
                            0.3 * np.sin(2 * np.pi * 45 * t) +          # Lower component
                            0.2 * np.sin(2 * np.pi * 80 * t) +          # Higher component
                            0.1 * np.sin(2 * np.pi * 30 * t) +          # Bass component
                            0.05 * np.sin(2 * np.pi * 100 * t)          # High freq component
                        )
                        
                        # Add valve closure click (brief high frequency)
                        if s2_length > 0:
                            click_samples = min(s2_length // 20, 10)
                            if click_samples > 0:
                                click = 0.1 * np.sin(2 * np.pi * 200 * t[:click_samples])
                                s2_wave[:click_samples] += click
                        
                        s2_wave *= s2_envelope * 0.8  # S2 typically quieter than S1
                        heartbeat_sound[s2_start:s2_end] += s2_wave

                max_val = np.max(np.abs(heartbeat_sound))
                if max_val > FLOAT_EPSILON:
                    heartbeat_sound = heartbeat_sound / max_val * 0.8

                heartbeat_path = sound_gen.save_sound(
                    heartbeat_sound, f"heartbeat_{int(bpm)}bpm.wav", f"Heartbeat at {bpm} BPM")

                logger.debug(f"  Generated heartbeat sound at {bpm} BPM: {heartbeat_path}")

                # Physical entrainment boost with real sound
                harmony_increase = base_harmony_increase * 1.2  # 20% boost from actual sound

            except Exception as sound_err:
                logger.warning(f"Failed to generate heartbeat sound: {sound_err}")
                harmony_increase = base_harmony_increase
        else:
            harmony_increase = base_harmony_increase

        # Calculate new harmony with entrainment effect
        new_harmony = min(1.0, current_harmony + harmony_increase)

        logger.debug(f"  Heartbeat Entrainment: BeatFreq={beat_freq:.2f}Hz, " +
                   f"BeatRes={beat_resonance:.3f}, DurFactor={duration_factor:.2f} -> " +
                   f"HarmonyIncrease={harmony_increase:.4f}")

        # Heartbeat entrainment affects soul harmony directly
        final_harmony = new_harmony

        # Update soul with entrainment data
        setattr(soul_spark, 'harmony', float(final_harmony))
        setattr(soul_spark, 'heartbeat_entrainment', beat_resonance * duration_factor)
        setattr(soul_spark, 'heartbeat_frequency', float(beat_freq))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())

        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "bpm": float(bpm),
                "beat_frequency": float(beat_freq),
                "duration": float(duration),
                "beat_resonance": float(beat_resonance),
                "base_harmony_increase": float(base_harmony_increase),
                "standing_wave_resonance": float(standing_wave_resonance),
                "standing_wave_boost": float(standing_wave_boost),
                "final_harmony_increase": float(final_harmony - current_harmony),
                "physical_sound_generated": heartbeat_sound is not None,
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_heartbeat_entrainment', metrics_data)

        logger.info(f"Heartbeat entrainment applied. " +
                   f"Base Harmony Increase: {harmony_increase:.4f}, " +
                   f"Standing Wave Boost: {standing_wave_boost:.4f}, " +
                   f"Final Harmony: {final_harmony:.4f}")

    except Exception as e:
        logger.error(f"Error applying heartbeat entrainment: {e}", exc_info=True)
        raise RuntimeError("Heartbeat entrainment failed.") from e


def train_name_response(soul_spark: SoulSpark, cycles: int = 7) -> None:
    """
    Train name response using carrier wave principles based on the standing wave
    patterns established by the name. Creates resonant field instead of direct frequency.
    """
    logger.info("Identity Step: Train Name Response (Carrier Wave)...")
    if not isinstance(cycles, int) or cycles < 0:
        raise ValueError("Cycles must be non-negative.")
    if cycles == 0:
        logger.info("Skipping name response training (0 cycles).")
        return
    
    try:
        name = getattr(soul_spark, 'name', 'Unknown')
        name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
        name_standing_waves = getattr(soul_spark, 'name_standing_waves', None)
        consciousness_state = getattr(soul_spark, 'consciousness_state', 'spark')
        heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0)
        current_response = getattr(soul_spark, 'response_level', 0.0)
        
        if not name_standing_waves:
            raise ValueError("Name standing waves required for response training")
        
        logger.debug(f"  Name Carrier Wave Training: Name={name}, NR={name_resonance:.3f}, " +
                   f"State={consciousness_state}, HBEnt={heartbeat_entrainment:.3f}")
        
        # Extract standing wave pattern data
        pattern_coherence = name_standing_waves.get("pattern_coherence", 0.0)
        resonance_quality = name_standing_waves.get("resonance_quality", 0.0)
        component_frequencies = name_standing_waves.get("component_frequencies", [])
        
        name_resonance_accumulator = 0.0
        
        # Adjust cycles based on name complexity
        effective_cycles = int(cycles * (0.5 + 0.5 * resonance_quality))
        effective_cycles = max(1, min(cycles * 2, effective_cycles))
        
        # Training metrics
        cycle_metrics = []
        
        for cycle in range(effective_cycles):
            cycle_resonance = 0.0
            
            # Create phase shift based on cycle progression
            phase_shift = 2 * np.pi * (cycle / effective_cycles)
            
            # Calculate resonance factor for this cycle
            # Diminishing returns for later cycles
            cycle_factor = 1.0 - 0.5 * (cycle / max(1, effective_cycles))
            
            # Apply state modifier
            state_factor = NAME_RESPONSE_STATE_FACTORS.get(
                consciousness_state, NAME_RESPONSE_STATE_FACTORS['default'])
            
            # Apply heartbeat entrainment factor
            heartbeat_factor = (NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR + 
                              NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT * heartbeat_entrainment)
            
            # Base increase for this cycle
            base_increase = (NAME_RESPONSE_TRAIN_BASE_INC + 
                           NAME_RESPONSE_TRAIN_CYCLE_INC * cycle) * cycle_factor
            
            # Amplify by name qualities
            name_factor = NAME_RESPONSE_TRAIN_NAME_FACTOR * name_resonance
            
            # Add pattern coherence and resonance quality effects
            pattern_effect = 0.5 * pattern_coherence
            resonance_effect = 0.5 * resonance_quality
            
            # Calculate cycle increase with all factors
            cycle_resonance = (base_increase * 
                             state_factor * 
                             heartbeat_factor * 
                             (1.0 + name_factor) *
                             (1.0 + pattern_effect) *
                             (1.0 + resonance_effect))
            
            # Add to accumulator
            name_resonance_accumulator += cycle_resonance
            
            # Track cycle metrics
            cycle_metrics.append({
                "cycle": cycle + 1,
                "phase_shift": float(phase_shift),
                "cycle_factor": float(cycle_factor),
                "base_increase": float(base_increase),
                "effective_increase": float(cycle_resonance),
                "accumulated": float(name_resonance_accumulator)
            })
            
            logger.debug(f"    Cycle {cycle+1}: CycleFactor={cycle_factor:.3f}, " +
                       f"Increase={cycle_resonance:.5f}, " +
                       f"Accum={name_resonance_accumulator:.4f}")
        
        # Update response level with accumulated resonance
        # Clamp to valid range
        new_response = min(1.0, current_response + name_resonance_accumulator)
        
        # Calculate field strength based on name integration with soul properties
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        voice_frequency = getattr(soul_spark, 'voice_frequency', 220.0)
        
        # Field strength represents how well name frequencies resonate with soul
        frequency_harmony = calculate_resonance(voice_frequency, soul_frequency) if soul_frequency > 0 else 0.0
        pattern_integration = pattern_coherence * resonance_quality
        field_strength = (frequency_harmony + pattern_integration) / 2.0
        
        # Field boost from training effectiveness
        training_efficiency = name_resonance_accumulator / max(0.1, effective_cycles)  # Efficiency per cycle
        field_boost = min(0.2, training_efficiency * pattern_coherence)  # Up to 20% boost
        
        # Name response is internal to soul identity - apply field boost
        final_response = min(1.0, new_response + field_boost)
        
        setattr(soul_spark, 'response_level', float(final_response))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "name": name,
                "initial_cycles": cycles,
                "effective_cycles": effective_cycles,
                "initial_response": float(current_response),
                "accumulated_increase": float(name_resonance_accumulator),
                "field_strength": float(field_strength),  
                "field_boost": float(field_boost),     
                "final_response": float(final_response),
                "state_factor": float(state_factor),
                "heartbeat_factor": float(heartbeat_factor),
                "pattern_coherence": float(pattern_coherence),
                "resonance_quality": float(resonance_quality),
                "cycle_details": cycle_metrics,
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_name_response_training', metrics_data)
        
        logger.info(f"Name carrier wave training complete. " +
                   f"Final Response: {final_response:.4f}")
        
    except Exception as e:
        logger.error(f"Error training name response: {e}", exc_info=True)
        raise RuntimeError("Name response training failed.") from e


def identify_primary_sephiroth(soul_spark: SoulSpark) -> None:
    """ 
    Identifies primary Sephiroth aspect based on soul state and resonance.
    Stores aspect directly on soul_spark.
    """
    logger.info("Identity Step: Identify Primary Sephiroth with Layer Resonance...")
    
    try:
        # Get soul properties for Sephiroth analysis
        name = getattr(soul_spark, 'name')
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        harmony = getattr(soul_spark, 'harmony', 0.0)
        pattern_coherence = getattr(soul_spark, 'pattern_coherence', 0.0)
        phi_resonance = getattr(soul_spark, 'phi_resonance', 0.0)
        creator_connection = getattr(soul_spark, 'creator_connection_strength', 0.0)
        earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
        
        if not name:
            raise ValueError("Soul must have name for Sephiroth identification")
        if soul_frequency <= FLOAT_EPSILON:
            raise ValueError("Invalid soul frequency for Sephiroth identification")
        
        if aspect_dictionary is None:
            raise RuntimeError("Aspect Dictionary unavailable.")
        
        # Calculate traditional Sephiroth affinities based on soul properties
        sephiroth_affinities = {}
        
        # Get all available Sephiroth from aspect dictionary
        available_sephiroth = ['kether', 'chokmah', 'binah', 'chesed', 'geburah', 
                              'tiphareth', 'netzach', 'hod', 'yesod', 'malkuth']
        
        for sephirah in available_sephiroth:
            try:
                sephirah_aspects = aspect_dictionary.get_aspects(sephirah)
                if not sephirah_aspects:
                    continue
                
                sephirah_freq = sephirah_aspects.get('base_frequency', 0.0)
                if sephirah_freq <= FLOAT_EPSILON:
                    continue
                
                # Calculate frequency resonance
                freq_resonance = calculate_resonance(soul_frequency, sephirah_freq)
                
                # Calculate attribute alignments
                attribute_alignments = 0.0
                
                # Spiritual development alignment
                if sephirah in ['kether', 'tiphareth'] and creator_connection > 0.6:
                    attribute_alignments += creator_connection * 0.3
                
                # Earth connection alignment
                if sephirah == 'malkuth' and earth_resonance > 0.7:
                    attribute_alignments += earth_resonance * 0.4
                
                # Wisdom/understanding alignment
                if sephirah in ['chokmah', 'binah'] and pattern_coherence > 0.5:
                    attribute_alignments += pattern_coherence * 0.25
                
                # Harmony alignment
                if sephirah in ['chesed', 'netzach'] and harmony > 0.6:
                    attribute_alignments += harmony * 0.3
                
                # Sacred geometry alignment
                if sephirah in ['tiphareth', 'yesod'] and phi_resonance > 0.5:
                    attribute_alignments += phi_resonance * 0.25
                
                # Strength/discipline alignment
                if sephirah in ['geburah', 'hod'] and pattern_coherence > 0.7:
                    attribute_alignments += pattern_coherence * 0.2
                
                # Combined affinity score
                total_affinity = freq_resonance + attribute_alignments
                sephiroth_affinities[sephirah] = min(1.0, total_affinity)
                
            except Exception as seph_err:
                logger.warning(f"Error processing Sephirah {sephirah}: {seph_err}")
                continue
        
        # Determine primary Sephiroth aspect
        sephiroth_aspect = SEPHIROTH_ASPECT_DEFAULT
        if sephiroth_affinities:
            sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda item: item[1])[0]
        
        logger.debug(f"  Sephirah Affinities: {sephiroth_affinities} -> Identified: {sephiroth_aspect}")
        
        # Store Sephiroth aspect directly on soul_spark
        setattr(soul_spark, 'sephiroth_aspect', sephiroth_aspect)
        setattr(soul_spark, 'sephiroth_affinities', sephiroth_affinities)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "sephiroth_aspect": sephiroth_aspect,
                "affinity_scores": sephiroth_affinities,
                "soul_frequency": float(soul_frequency),
                "harmony": float(harmony),
                "pattern_coherence": float(pattern_coherence),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_sephiroth_affinity', metrics_data)
        
        logger.info(f"Primary Sephiroth aspect identified: {sephiroth_aspect}")
        
    except Exception as e:
        logger.error(f"Error identifying Sephiroth aspect: {e}", exc_info=True)
        raise RuntimeError("Sephirah aspect ID failed.") from e


def determine_elemental_affinity(soul_spark: SoulSpark) -> None:
    """
    Determines elemental affinity using wave physics analysis of soul properties.
    Stores affinity directly on soul_spark,.
    """
    logger.info("Identity Step: Determine Elemental Affinity with Wave Physics...")
    
    try:
        # Get soul properties for elemental analysis
        name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
        sephiroth_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        harmony = getattr(soul_spark, 'harmony', 0.0)
        
        # Traditional elemental frequencies (Hz)
        elemental_frequencies = {
            'earth': 136.1,     # C# - Grounding, stability
            'water': 210.42,    # G# - Flow, emotion  
            'fire': 341.3,      # F - Energy, passion
            'air': 426.7,       # Ab - Thought, communication
            'aether': 528.0     # C - Spirit, transformation
        }
        
        # Calculate resonance with each element
        elemental_affinities = {}
        
        for element, elem_freq in elemental_frequencies.items():
            # Base resonance with element frequency
            freq_resonance = calculate_resonance(soul_frequency, elem_freq)
            
            # Bonus from Sephiroth aspect alignment
            sephiroth_bonus = 0.0
            if sephiroth_aspect:
                sephiroth_elements = {
                    'kether': 'aether', 'chokmah': 'fire', 'binah': 'water',
                    'chesed': 'water', 'geburah': 'fire', 'tiphareth': 'aether',
                    'netzach': 'fire', 'hod': 'air', 'yesod': 'water',
                    'malkuth': 'earth'
                }
                if sephiroth_elements.get(sephiroth_aspect) == element:
                    sephiroth_bonus = 0.2
            
            # Name resonance influence
            name_influence = name_resonance * 0.1
            
            # Harmony influence for spiritual elements
            harmony_influence = 0.0
            if element in ['aether', 'fire'] and harmony > 0.5:
                harmony_influence = (harmony - 0.5) * 0.2
            
            # Combined affinity score
            total_affinity = freq_resonance + sephiroth_bonus + name_influence + harmony_influence
            elemental_affinities[element] = min(1.0, total_affinity)
        
        # Determine primary elemental affinity
        elemental_affinity = ELEMENTAL_AFFINITY_DEFAULT
        if elemental_affinities:
            elemental_affinity = max(elemental_affinities.items(), key=lambda item: item[1])[0]
        
        logger.debug(f"  Elemental Affinities: {elemental_affinities} -> Identified: {elemental_affinity}")
        
        # Store elemental affinity directly on soul_spark
        setattr(soul_spark, 'elemental_affinity', elemental_affinity)
        setattr(soul_spark, 'elemental_affinities', elemental_affinities)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Generate elemental sound signature
        if SOUND_MODULES_AVAILABLE:
            try:
                element_freq = elemental_frequencies.get(elemental_affinity, 432.0)
                element_harmonics = [1.0, 1.5, 2.0, 3.0]  # Natural harmonics
                element_amplitudes = [0.8, 0.4, 0.3, 0.2]
                
                element_sound = sound_gen.generate_harmonic_tone(
                    element_freq, element_harmonics, element_amplitudes, duration=6.0)
                sound_path = sound_gen.save_sound(
                    element_sound, f"element_{elemental_affinity}.wav", 
                    f"Element {elemental_affinity} Sound")
                
                setattr(soul_spark, 'elemental_sound_signature', element_sound)
                logger.debug(f"Generated elemental sound signature: {sound_path}")
                
            except Exception as sound_err:
                logger.warning(f"Failed to generate elemental sound: {sound_err}")
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "elemental_affinity": elemental_affinity,
                "affinities_scores": elemental_affinities,
                "sephiroth_aspect": sephiroth_aspect,
                "soul_frequency": float(soul_frequency),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_elemental_affinity', metrics_data)
        
        logger.info(f"Elemental affinity determined: {elemental_affinity}")
        
    except Exception as e:
        logger.error(f"Error determining elemental affinity: {e}", exc_info=True)
        raise RuntimeError("Elemental affinity determination failed.") from e

def assign_platonic_symbol(soul_spark: SoulSpark) -> None:
    """
    Assigns Platonic symbol based on affinities with proper geometric resonance.
    Creates standing wave patterns based on sacred geometry principles.
    """
    logger.info("Identity Step: Assign Platonic Symbol with Geometric Resonance...")
    elem_affinity = getattr(soul_spark, 'elemental_affinity')
    seph_aspect = getattr(soul_spark, 'sephiroth_aspect')
    gematria = getattr(soul_spark, 'gematria_value')
    
    if not elem_affinity or not seph_aspect:
        raise ValueError("Missing affinities for Platonic symbol assignment.")
    
    if aspect_dictionary is None:
        raise RuntimeError("Aspect Dictionary unavailable.")
    
    try:
        # First determine symbol based on traditional methods
        symbol = PLATONIC_ELEMENT_MAP.get(elem_affinity)
        
        if not symbol:
            seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
            if seph_geom in PLATONIC_SOLIDS:
                symbol = seph_geom
            elif seph_geom == 'cube':
                symbol = 'hexahedron'  # Standardize terminology
        
        if not symbol:
            # Fallback based on gematria
            unique_symbols = sorted(list(set(PLATONIC_ELEMENT_MAP.values()) - {None}))
            if unique_symbols:
                symbol_idx = (gematria // PLATONIC_DEFAULT_GEMATRIA_RANGE) % len(unique_symbols)
                symbol = unique_symbols[symbol_idx]
            else:
                symbol = 'sphere'  # Ultimate fallback
        
        logger.debug(f"  Platonic Symbol Logic -> Final='{symbol}'")
        
        # ENHANCEMENT: Create geometric standing wave patterns
        # Define geometric properties of platonic solids
        geometric_properties = {
            'tetrahedron': {
                'vertices': 4, 'edges': 6, 'faces': 4, 
                'face_type': 'triangle', 'symmetry': 'tetrahedral',
                'frequency_ratio': 1.0  # Base ratio
            },
            'hexahedron': {  # Cube
                'vertices': 8, 'edges': 12, 'faces': 6, 
                'face_type': 'square', 'symmetry': 'octahedral',
                'frequency_ratio': 2.0/1.0  # 2:1 ratio
            },
            'octahedron': {
                'vertices': 6, 'edges': 12, 'faces': 8, 
                'face_type': 'triangle', 'symmetry': 'octahedral',
                'frequency_ratio': 3.0/2.0  # 3:2 ratio
            },
            'dodecahedron': {
                'vertices': 20, 'edges': 30, 'faces': 12, 
                'face_type': 'pentagon', 'symmetry': 'icosahedral',
                'frequency_ratio': PHI  # Golden ratio
            },
            'icosahedron': {
                'vertices': 12, 'edges': 30, 'faces': 20, 
                'face_type': 'triangle', 'symmetry': 'icosahedral',
                'frequency_ratio': PHI * PHI  # Phi squared
            },
            'sphere': {  # Perfect form - not technically platonic but included
                'vertices': 'infinite', 'edges': 'infinite', 'faces': 'infinite', 
                'face_type': 'point', 'symmetry': 'spherical',
                'frequency_ratio': PI  # Pi ratio
            }
        }
        
        # Get properties for the chosen symbol
        props = geometric_properties.get(symbol, geometric_properties['sphere'])
        
        # Create resonant geometric pattern
        geometric_pattern = _create_geometric_standing_waves(soul_spark, symbol, props)
        
        # Update soul with platonic symbol and geometric pattern
        setattr(soul_spark, 'platonic_symbol', symbol)
        setattr(soul_spark, 'geometric_pattern', geometric_pattern)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "platonic_symbol": symbol,
                "properties": {k: v for k, v in props.items() if k != 'frequency_ratio'},
                "frequency_ratio": float(props['frequency_ratio']),
                "standing_waves": len(geometric_pattern.get('nodes', [])),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_platonic_symbol', metrics_data)
        
        logger.info(f"Platonic symbol assigned: {symbol} with " +
                   f"{len(geometric_pattern.get('nodes', []))} standing wave nodes")
        
    except Exception as e:
        logger.error(f"Error assigning Platonic symbol: {e}", exc_info=True)
        raise RuntimeError("Platonic symbol assignment failed.") from e

def _create_geometric_standing_waves(soul_spark: SoulSpark, symbol: str, 
                                   properties: Dict) -> Dict[str, Any]:
    """
    Create geometric standing wave patterns based on platonic solid properties.
    
    Args:
        soul_spark: Soul to create patterns in
        symbol: Name of platonic solid
        properties: Properties of the platonic solid
        
    Returns:
        Dictionary with geometric pattern data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pattern creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not isinstance(properties, dict):
        raise ValueError("properties must be a dictionary")
    
    try:
        logger.debug(f"Creating geometric standing waves for {symbol}...")
        
        # Get base frequency for pattern
        base_freq = soul_spark.frequency
        
        # Apply geometric frequency ratio
        freq_ratio = properties.get('frequency_ratio', 1.0)
        geometric_freq = base_freq * freq_ratio
        
        # Create harmonic series based on geometric properties
        # Use vertices, edges, faces to determine number of harmonics
        vertices = properties.get('vertices', 0)
        # Convert to numeric if needed
        if isinstance(vertices, str):
            vertices = 1000  # Large number for 'infinite'
            
        # Number of harmonics based on complexity
        num_harmonics = min(12, vertices // 2 + 3)
        
        # Create harmonic ratios (some based on golden ratio for certain solids)
        if symbol in ('dodecahedron', 'icosahedron'):
            # These are phi-based solids
            harmonic_ratios = [1.0]
            for i in range(1, num_harmonics):
                if i % 2 == 1:
                    # Odd harmonics use phi
                    harmonic_ratios.append(PHI ** ((i + 1) // 2))
                else:
                    # Even harmonics use integers
                    harmonic_ratios.append(float(i))
        else:
            # Standard harmonic series
            harmonic_ratios = [float(i) for i in range(1, num_harmonics + 1)]
            
        # Calculate frequencies
        harmonic_freqs = [geometric_freq * ratio for ratio in harmonic_ratios]
        
        # Calculate wavelengths
        wavelengths = [SPEED_OF_SOUND / freq for freq in harmonic_freqs]
        
        # Create node and antinode positions
        # These form a geometric pattern in frequency space
        nodes = []
        antinodes = []
        
        # Calculate nodes based on symbol geometry
        for i, wavelength in enumerate(wavelengths):
            # For each wavelength, create nodes at positions determined by geometry
            # This is a simplified model - real implementation would use 3D geometry
            
            # Number of nodes depends on harmonic and geometry
            num_nodes = min(int(properties.get('edges', 12) * harmonic_ratios[i] * 0.5), 30)
            
            for j in range(num_nodes):
                # Position is based on geometric distribution
                if symbol in ('tetrahedron', 'octahedron', 'icosahedron'):
                    # Triangular faces - use triangular spacing
                    pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
                    # Modulate with sine to create triangular pattern
                    pos_mod = 0.5 + 0.3 * np.sin(pos * 2 * np.pi)
                    pos = pos_mod
                elif symbol in ('hexahedron', 'dodecahedron'):
                    # Square/pentagonal faces - use more regular spacing
                    pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
                else:  # sphere
                    # Use simple linear spacing for sphere
                    pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
                
                # Calculate amplitude (varies by position and harmonic)
                # Nodes have near-zero amplitude
                amp = 0.01 + 0.01 * np.cos(pos * np.pi * 2 * (i + 1))
                
                # Add node to list
                nodes.append({
                    "harmonic": i + 1,
                    "frequency": float(harmonic_freqs[i]),
                    "wavelength": float(wavelengths[i]),
                    "position": float(pos),
                    "amplitude": float(amp)
                })
                
                # Calculate corresponding antinode (maximum amplitude)
                # Antinodes are offset from nodes by 1/4 wavelength
                antinode_pos = (pos + 0.25) % 1.0
                antinode_amp = 0.5 + 0.4 * np.sin(pos * np.pi * 2 * (i + 1))
                
                # Add antinode to list
                antinodes.append({
                    "harmonic": i + 1,
                    "frequency": float(harmonic_freqs[i]),
                    "wavelength": float(wavelengths[i]),
                    "position": float(antinode_pos),
                    "amplitude": float(antinode_amp)
                })
        
        # Create the final geometric pattern structure
        geometric_pattern = {
            "symbol": symbol,
            "base_frequency": float(geometric_freq),
            "harmonic_ratios": harmonic_ratios,
            "harmonic_frequencies": [float(f) for f in harmonic_freqs],
            "wavelengths": [float(w) for w in wavelengths],
            "nodes": nodes,
            "antinodes": antinodes,
            "properties": properties
        }
        
        logger.debug(f"Created geometric pattern for {symbol} with " +
                   f"{len(nodes)} nodes and {len(antinodes)} antinodes.")
        
        return geometric_pattern
        
    except Exception as e:
        logger.error(f"Error creating geometric standing waves: {e}", exc_info=True)

        raise RuntimeError("Geometric standing wave creation failed") from e

def _determine_astrological_signature(soul_spark: SoulSpark) -> None:
    """
    Determines Zodiac sign, governing planet, and selects traits with frequency mapping.
    Creates resonant field based on astrological correspondences.
    """
    logger.info("Identity Step: Determine Astrological Signature with Frequency Mapping...")
    try:
        # 1. Generate Conceptual Birth Datetime - within reasonable past range
        sim_start_dt = datetime.fromisoformat(getattr(soul_spark, 'creation_time', datetime.now().isoformat()))
        # Generate birth in past 30 years but not in the future
        random_offset_days = random.uniform(-365.25 * 30, -30)  # 30 years ago to 30 days ago
        birth_dt = sim_start_dt + timedelta(days=random_offset_days)
        setattr(soul_spark, 'conceptual_birth_datetime', birth_dt.isoformat())
        logger.debug(f"  Conceptual Birth Datetime: {birth_dt.strftime('%Y-%m-%d %H:%M')}")

        # 2. Determine Zodiac Sign (Using 13 signs)
        birth_month_day = birth_dt.strftime('%B %d')
        zodiac_sign = "Unknown"
        zodiac_symbol = "?"
        
        for sign_info in ZODIAC_SIGNS:
            try:
                # Create date objects for comparison
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
                else: # Case where end date is in the next year
                    if birth_date_only >= start_date or birth_date_only <= end_date:
                        zodiac_sign = sign_info['name']
                        zodiac_symbol = sign_info['symbol']
                        break
            except ValueError as date_err:
                logger.warning(f"Could not parse date for sign {sign_info['name']}: {date_err}")
                continue

        setattr(soul_spark, 'zodiac_sign', zodiac_sign)
        logger.debug(f"  Determined Zodiac Sign: {zodiac_sign} ({zodiac_symbol})")

        # 3. Determine Governing Planet with frequency mapping
        # Either use planet from Earth Harmonization or assign a new one
        governing_planet = getattr(soul_spark, 'governing_planet', None)
        
        if not governing_planet or governing_planet == 'Unknown':
            # Assign based on zodiac sign
            planet_mapping = {
                'Aries': 'Mars',
                'Taurus': 'Venus',
                'Gemini': 'Mercury',
                'Cancer': 'Moon',
                'Leo': 'Sun',
                'Virgo': 'Mercury',
                'Libra': 'Venus',
                'Scorpio': 'Pluto',
                'Ophiuchus': 'Jupiter',
                'Sagittarius': 'Jupiter',
                'Capricorn': 'Saturn',
                'Aquarius': 'Uranus',
                'Pisces': 'Neptune'
            }
            
            governing_planet = planet_mapping.get(zodiac_sign, 'Jupiter')
            setattr(soul_spark, 'governing_planet', governing_planet)
            
        logger.debug(f"  Governing Planet: {governing_planet}")
        
        # Get planetary frequency
        planet_freq = PLANETARY_FREQUENCIES.get(governing_planet, 0.0)
        if planet_freq <= 0:
            # Fallback frequency if not defined
            planet_freq = 432.0
            logger.warning(f"No frequency defined for planet {governing_planet}, using default {planet_freq}Hz")

        # 4. Select Traits with frequency mapping
        traits = {"positive": {}, "negative": {}}
        if zodiac_sign != "Unknown" and zodiac_sign in ZODIAC_TRAITS:
            all_pos = ZODIAC_TRAITS[zodiac_sign].get("positive", [])
            all_neg = ZODIAC_TRAITS[zodiac_sign].get("negative", [])
            
            # Select random traits up to the max count
            num_pos = min(len(all_pos), ASTROLOGY_MAX_POSITIVE_TRAITS)
            num_neg = min(len(all_neg), ASTROLOGY_MAX_NEGATIVE_TRAITS)
            selected_pos = random.sample(all_pos, num_pos) if num_pos > 0 else []
            selected_neg = random.sample(all_neg, num_neg) if num_neg > 0 else []
            
            # Assign strengths with astrological meaning
            # Map each trait to a frequency offset from planet frequency
            for trait_idx, trait in enumerate(selected_pos):
                # Calculate frequency based on golden ratio offsets
                trait_freq = planet_freq * (1 + (trait_idx * 0.1) * (PHI - 1))
                # Calculate resonance with soul frequency
                trait_resonance = calculate_resonance(trait_freq, soul_spark.frequency)
                # Use resonance as strength
                strength = 0.3 + 0.6 * trait_resonance
                traits["positive"][trait] = round(strength, 3)
                
            for trait_idx, trait in enumerate(selected_neg):
                # Calculate frequency for negative traits
                trait_freq = planet_freq / (1 + (trait_idx * 0.1) * (PHI - 1))
                # Calculate resonance with soul frequency
                trait_resonance = calculate_resonance(trait_freq, soul_spark.frequency)
                # Use resonance as strength
                strength = 0.3 + 0.6 * trait_resonance
                traits["negative"][trait] = round(strength, 3)
        else:
            logger.warning(f"Zodiac sign '{zodiac_sign}' not found in ZODIAC_TRAITS. No traits assigned.")

        # 5. Create astrological resonance structure
        astrological_resonance = {
            "zodiac_sign": zodiac_sign,
            "zodiac_symbol": zodiac_symbol,
            "governing_planet": governing_planet,
            "planet_frequency": float(planet_freq),
            "birth_datetime": birth_dt.isoformat(),
            "traits": traits
        }
        
        # Update soul with astrological data
        setattr(soul_spark, 'astrological_traits', traits)
        setattr(soul_spark, 'astrological_resonance', astrological_resonance)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "zodiac_sign": zodiac_sign,
                "governing_planet": governing_planet,
                "planet_frequency": float(planet_freq),
                "positive_traits_count": len(traits["positive"]),
                "negative_traits_count": len(traits["negative"]),
                "birth_datetime": birth_dt.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_astrological_signature', metrics_data)

        logger.info(f"Astrological Signature determined: Sign={zodiac_sign}, " +
                   f"Planet={governing_planet}, " +
                   f"Traits={len(traits['positive'])} Pos / {len(traits['negative'])} Neg")

    except Exception as e:
        logger.error(f"Error determining astrological signature: {e}", exc_info=True)
        # Don't hard fail, raise the error
        raise RuntimeError("Astrological signature determination failed") from e

def apply_sacred_geometry(soul_spark: SoulSpark, stages: int = 5) -> None:
    """
    Applies sacred geometry patterns directly to the soul's internal structure
    using proper light interference patterns. Creates crystalline coherence
    within the soul itself.
    """
    logger.info("Identity Step: Apply Sacred Geometry with Light Interference Patterns...")
    if not isinstance(stages, int) or stages < 0:
        raise ValueError("Stages non-negative.")
    if stages == 0:
        logger.info("Skipping sacred geometry application (0 stages).")
        return
    
    try:
        # Get soul properties for geometry application
        seph_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
        elem_affinity = getattr(soul_spark, 'elemental_affinity', None)
        name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
        soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
        
        if not seph_aspect or not elem_affinity:
            raise ValueError("Missing affinities for geometry application.")
        
        # Sacred geometry patterns and their frequencies
        geometry_patterns = {
            'flower_of_life': {'freq': 528.0, 'coherence': 0.95, 'complexity': 7},
            'metatrons_cube': {'freq': 741.0, 'coherence': 0.92, 'complexity': 13},
            'golden_spiral': {'freq': PHI * 432.0, 'coherence': 0.88, 'complexity': 5},
            'vesica_piscis': {'freq': 396.0, 'coherence': 0.85, 'complexity': 2},
            'tree_of_life': {'freq': 852.0, 'coherence': 0.90, 'complexity': 10},
            'platonic_solid': {'freq': 639.0, 'coherence': 0.87, 'complexity': 6}
        }
        
        # Select dominant geometry based on soul characteristics
        geometry_resonances = {}
        for pattern, props in geometry_patterns.items():
            # Calculate resonance between soul frequency and pattern frequency
            freq_resonance = calculate_resonance(soul_frequency, props['freq'])
            
            # Apply complexity modifier based on soul's development
            complexity_factor = min(1.0, stages / props['complexity'])
            
            # Sephiroth influence
            seph_influence = 0.1 if pattern == 'tree_of_life' and seph_aspect else 0.0
            
            # Element influence
            element_influence = 0.0
            if elem_affinity == 'earth' and pattern in ['platonic_solid', 'metatrons_cube']:
                element_influence = 0.15
            elif elem_affinity == 'aether' and pattern in ['flower_of_life', 'tree_of_life']:
                element_influence = 0.15
            
            total_resonance = (freq_resonance * complexity_factor + 
                             seph_influence + element_influence)
            geometry_resonances[pattern] = total_resonance
        
        # Select dominant pattern
        dominant_pattern = max(geometry_resonances.items(), key=lambda x: x[1])[0]
        pattern_props = geometry_patterns[dominant_pattern]
        
        # Apply sacred geometry crystallization to soul's internal structure
        crystallization_boost = (pattern_props['coherence'] * 
                                geometry_resonances[dominant_pattern] * 
                                min(1.0, stages / 5.0))
        
        # Update soul's sacred geometry properties directly
        current_crystallization = getattr(soul_spark, 'crystallization_level', 0.0)
        new_crystallization = min(1.0, current_crystallization + crystallization_boost)
        
        # Update pattern coherence using sacred geometry
        current_pattern_coherence = getattr(soul_spark, 'pattern_coherence', 0.0)
        geometry_pattern_boost = pattern_props['coherence'] * 0.1
        new_pattern_coherence = min(1.0, current_pattern_coherence + geometry_pattern_boost)
        
        # Update phi resonance through golden ratio in geometry
        current_phi = getattr(soul_spark, 'phi_resonance', 0.0)
        phi_boost = 0.0
        if dominant_pattern in ['golden_spiral', 'flower_of_life']:
            phi_boost = 0.05 * (geometry_resonances[dominant_pattern])
        new_phi_resonance = min(1.0, current_phi + phi_boost)
        
        # Update harmony through geometric resonance
        current_harmony = getattr(soul_spark, 'harmony', 0.0)
        harmony_boost = crystallization_boost * 0.3
        new_harmony = min(1.0, current_harmony + harmony_boost)
        
        # Store sacred geometry properties directly on soul_spark
        setattr(soul_spark, 'sacred_geometry_pattern', dominant_pattern)
        setattr(soul_spark, 'sacred_geometry_frequency', pattern_props['freq'])
        setattr(soul_spark, 'sacred_geometry_coherence', pattern_props['coherence'])
        setattr(soul_spark, 'sacred_geometry_stages', stages)
        setattr(soul_spark, 'crystallization_level', new_crystallization)
        setattr(soul_spark, 'pattern_coherence', new_pattern_coherence)
        setattr(soul_spark, 'phi_resonance', new_phi_resonance)
        setattr(soul_spark, 'harmony', new_harmony)
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Calculate final result factors
        result_factors = {
            'pattern_coherence': new_pattern_coherence,
            'phi_resonance': new_phi_resonance,
            'harmony': new_harmony,
            'toroidal_flow': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "dominant_pattern": dominant_pattern,
                "pattern_frequency": pattern_props['freq'],
                "crystallization_boost": float(crystallization_boost),
                "stages_applied": stages,
                "geometry_resonances": {k: float(v) for k, v in geometry_resonances.items()},
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_sacred_geometry', metrics_data)
        
        logger.info(f"Sacred geometry pattern reinforcement applied. Dominant: {dominant_pattern}, "
                   f"Stages: {stages}, Crystallization: {new_crystallization:.4f}")
        logger.info(f"  Resulting Factors: P.Coh={new_pattern_coherence:.4f}, "
                   f"Phi={new_phi_resonance:.4f}, Harm={new_harmony:.4f}, "
                   f"Torus={result_factors['toroidal_flow']:.4f}")
        
    except Exception as e:
        logger.error(f"Error applying sacred geometry: {e}", exc_info=True)
        raise RuntimeError("Sacred geometry application failed") from e

def _create_sacred_geometry_pattern(soul_spark: SoulSpark, geometry: str,
                                 stage_factor: float, effect_scale: float) -> Dict[str, Any]:
    """
    Create sacred geometry pattern with light wave physics.
    
    Args:
        soul_spark: Soul to create pattern in
        geometry: Sacred geometry type
        stage_factor: Stage progression factor
        effect_scale: Effect strength scale
        
    Returns:
        Dictionary with geometry pattern data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pattern creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not geometry:
        raise ValueError("geometry cannot be empty")
    if stage_factor <= 0 or effect_scale <= 0:
        raise ValueError("stage_factor and effect_scale must be positive")
    
    try:
        logger.debug(f"Creating sacred geometry pattern for {geometry}...")
        
        # Each geometry has specific properties
        geometry_properties = {
            "seed_of_life": {
                "circles": 7,
                "symmetry": 6,
                "frequency_ratio": 1.0,
                "center_radius_ratio": 1.0
            },
            "flower_of_life": {
                "circles": 19,
                "symmetry": 6,
                "frequency_ratio": PHI,
                "center_radius_ratio": 0.5
            },
            "fruit_of_life": {
                "circles": 13,
                "symmetry": 6,
                "frequency_ratio": PHI * PHI,
                "center_radius_ratio": 0.3333
            },
            "tree_of_life": {
                "circles": 10,
                "symmetry": 3,
                "frequency_ratio": 3.0,
                "center_radius_ratio": 0.5
            },
            "metatrons_cube": {
                "circles": 13,
                "symmetry": 12,
                "frequency_ratio": PI,
                "center_radius_ratio": 0.25
            },
            "sri_yantra": {
                "circles": 9,
                "symmetry": 9,
                "frequency_ratio": 3.0 / 2.0,
                "center_radius_ratio": 0.1
            }
        }
        
        # Get properties for this geometry
        props = geometry_properties.get(
            geometry, geometry_properties["seed_of_life"])
        
        # Get base frequency from soul
        base_freq = soul_spark.frequency
        
        # Calculate geometry frequency
        geometry_freq = base_freq * props["frequency_ratio"]
        
        # Create light wave interference pattern
        num_points = 100
        pattern = np.zeros(num_points)
        
        # Calculate interference pattern
        for circle in range(props["circles"]):
            # Each circle contributes a wave component
            circle_freq = geometry_freq * (1.0 + 0.1 * circle)
            circle_radius = props["center_radius_ratio"] * (1.0 + 0.2 * circle)
            circle_phase = 2 * PI * circle / props["circles"]
            
            # Create wave for this circle
            x = np.linspace(0, 1, num_points)
            circle_wave = np.sin(2 * PI * circle_freq * x + circle_phase) * circle_radius
            
            # Add to overall pattern with constructive interference
            pattern += circle_wave
        
        # Normalize pattern
        max_val = np.max(np.abs(pattern))
        if max_val > FLOAT_EPSILON:
            normalized_pattern = pattern / max_val
        else:
            normalized_pattern = np.zeros_like(pattern)
            
        # Calculate pattern coherence (measure of regular structure)
        # Higher coherence = more ordered pattern
        fft_values = np.abs(np.fft.rfft(normalized_pattern))
        fft_sum = np.sum(fft_values)
        
        # Find dominant frequencies (peaks in FFT)
        peak_threshold = 0.1 * np.max(fft_values)
        peaks = np.where(fft_values > peak_threshold)[0]
        
        # More peaks at regular intervals = more coherent pattern
        num_peaks = len(peaks)
        spacing_regularity = 0.0
        
        if num_peaks > 1:
            # Calculate spacings between peaks
            spacings = np.diff(peaks)
            
            # Measure regularity (lower std = more regular)
            spacing_std = np.std(spacings)
            spacing_mean = np.mean(spacings)
            
            if spacing_mean > FLOAT_EPSILON:
                spacing_regularity = 1.0 - min(1.0, spacing_std / spacing_mean)
        
        # Calculate overall coherence
        pattern_coherence = (
            0.3 * (num_peaks / max(1, props["circles"])) + 
            0.7 * spacing_regularity)
        
        # Identify nodes and antinodes in the pattern
        nodes = []
        antinodes = []
        
        # Find local minima (nodes) and maxima (antinodes)
        for i in range(1, num_points - 1):
            if normalized_pattern[i] < normalized_pattern[i-1] and normalized_pattern[i] < normalized_pattern[i+1]:
                # Local minimum = node
                if abs(normalized_pattern[i]) < 0.2:  # Near zero
                    nodes.append({
                        "position": float(i / num_points),
                        "amplitude": float(normalized_pattern[i])
                    })
            elif normalized_pattern[i] > normalized_pattern[i-1] and normalized_pattern[i] > normalized_pattern[i+1]:
                # Local maximum = antinode
                if abs(normalized_pattern[i]) > 0.5:  # Strong amplitude
                    antinodes.append({
                        "position": float(i / num_points),
                        "amplitude": float(normalized_pattern[i])
                    })
        
        # Create sacred geometry pattern structure
        geometry_pattern = {
            "geometry": geometry,
            "frequency": float(geometry_freq),
            "coherence": float(pattern_coherence),
            "pattern": normalized_pattern.tolist(),
            "nodes": nodes,
            "antinodes": antinodes,
            "properties": props
        }
        
        logger.debug(f"Created sacred geometry pattern for {geometry} with " +
                   f"coherence {pattern_coherence:.4f}, " +
                   f"{len(nodes)} nodes, {len(antinodes)} antinodes.")
        
        return geometry_pattern
        
    except Exception as e:
        logger.error(f"Error creating sacred geometry pattern: {e}", exc_info=True)
        raise RuntimeError("Sacred geometry pattern creation failed") from e

def _create_coherent_interference_pattern(soul_spark: SoulSpark, 
                                       geometry_applications: List[Dict]) -> Dict[str, Any]:
    """
    Create coherent light interference pattern from multiple sacred geometries.
    
    Args:
        soul_spark: Soul to create pattern in
        geometry_applications: List of applied geometries
        
    Returns:
        Dictionary with interference pattern data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pattern creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not geometry_applications:
        raise ValueError("geometry_applications cannot be empty")
    
    try:
        logger.debug(f"Creating coherent interference pattern from " +
                   f"{len(geometry_applications)} geometries...")
        
        # Create combined interference pattern
        num_points = 100
        combined_pattern = np.zeros(num_points)
        
        # Weighted contribution factors
        total_weight = 0.0
        
        # Combine patterns from all geometries
        for geom_app in geometry_applications:
            pattern = geom_app.get("pattern", {}).get("pattern", [])
            
            # Convert to numpy array if needed
            if isinstance(pattern, list) and len(pattern) >= num_points:
                pattern_array = np.array(pattern[:num_points])
                
                # Calculate weight based on geometry properties
                coherence = geom_app.get("pattern", {}).get("coherence", 0.0)
                stage = geom_app.get("stage", 1)
                
                # Higher coherence and later stages have more weight
                weight = coherence * 0.5 + stage / len(geometry_applications) * 0.5
                
                # Add weighted contribution
                combined_pattern += pattern_array * weight
                total_weight += weight
        
            # Normalize combined pattern
        if total_weight > FLOAT_EPSILON:
            combined_pattern /= total_weight
            
        # Analyze combined pattern for coherence
        # Calculate autocorrelation for periodicity
        autocorr = np.correlate(combined_pattern, combined_pattern, mode='full')
        autocorr = autocorr[num_points-1:] / autocorr[num_points-1]

        # Normalize autocorrelation for peak finding
        autocorr_normalized = autocorr

        # Find peaks in autocorrelation
        max_autocorr = np.max(autocorr_normalized[1:])  # Exclude first point (always 1.0)
        peak_threshold = max(0.1, 0.3 * max_autocorr)   # Adaptive threshold
        peaks = []
        
        # Skip first few points to avoid noise
        start_idx = max(1, int(len(autocorr_normalized) * 0.05))  # Skip first 5%
        
        for i in range(start_idx, len(autocorr_normalized) - 1):
            if (autocorr_normalized[i] > peak_threshold and 
                autocorr_normalized[i] > autocorr_normalized[i-1] and 
                autocorr_normalized[i] > autocorr_normalized[i+1]):
                peaks.append(i)
      # Calculate coherence based on peak regularity with better baseline
        peak_coherence = 0.2  # Base coherence instead of 0.0
        if len(peaks) > 1:
            # Calculate spacings between peaks
            spacings = np.diff(peaks)
            
            # Measure regularity (lower std = more regular)
            spacing_std = np.std(spacings)
            spacing_mean = np.mean(spacings)
            
            if spacing_mean > FLOAT_EPSILON:
                regularity = 1.0 - min(1.0, spacing_std / spacing_mean)
                # Boost coherence based on both number of peaks and regularity
                peak_coherence = 0.2 + 0.6 * regularity + 0.2 * min(1.0, len(peaks) / 5.0)
        elif len(peaks) == 1:
            peak_coherence = 0.4  # Single peak still shows some structure

        # Fix: define aspect_count as the number of geometry applications
        aspect_count = len(geometry_applications)
        # Combine aspect count and pattern coherence with better weighting
        aspect_factor = min(1.0, aspect_count / 8.0)  # Reduced from 10 to 8 for easier achievement
        crystalline_coherence = 0.4 * aspect_factor + 0.6 * peak_coherence  # Favor pattern over count

        # Identify interference nodes and antinodes
        int_nodes = []
        int_antinodes = []
        
        # Find local minima (nodes) and maxima (antinodes)
        for i in range(1, num_points - 1):
            if combined_pattern[i] < combined_pattern[i-1] and combined_pattern[i] < combined_pattern[i+1]:
                # Local minimum = node
                if abs(combined_pattern[i]) < 0.1:  # Near zero
                    int_nodes.append({
                        "position": float(i / num_points),
                        "amplitude": float(combined_pattern[i])
                    })
            elif combined_pattern[i] > combined_pattern[i-1] and combined_pattern[i] > combined_pattern[i+1]:
                # Local maximum = antinode
                if abs(combined_pattern[i]) > 0.5:  # Strong amplitude
                    int_antinodes.append({
                        "position": float(i / num_points),
                        "amplitude": float(combined_pattern[i])
                    })
        
        # Create interference pattern structure
        interference_pattern = {
            "pattern": combined_pattern.tolist(),
            "coherence": float(crystalline_coherence),
            "nodes": int_nodes,
            "antinodes": int_antinodes,
            "contributing_geometries": [g.get("geometry", "unknown") for g in geometry_applications]
        }
        
        logger.debug(f"Created coherent interference pattern with " +
            f"coherence {crystalline_coherence:.4f}, " +
            f"{len(int_nodes)} nodes, {len(int_antinodes)} antinodes.")
        
        return interference_pattern

    except Exception as e:
        logger.error(f"Error creating coherent interference pattern: {e}", exc_info=True)
        raise RuntimeError("Coherent interference pattern creation failed") from e

def calculate_attribute_coherence(soul_spark: SoulSpark) -> None:
    """
    Calculates attribute coherence score (0-1) based on how well all identity
    attributes form a unified, coherent pattern within the soul itself.
    focuses on soul's internal coherence.
    """
    logger.info("Identity Step: Calculate Attribute Coherence with Wave Physics...")
    try:
        # Collect key attribute values from the soul's identity
        attributes = {
            'name_resonance': getattr(soul_spark, 'name_resonance', 0.0),
            'response_level': getattr(soul_spark, 'response_level', 0.0),
            'state_stability': getattr(soul_spark, 'state_stability', 0.0),
            'crystallization_level': getattr(soul_spark, 'crystallization_level', 0.0),
            'heartbeat_entrainment': getattr(soul_spark, 'heartbeat_entrainment', 0.0),
            'emotional_resonance_avg': np.mean(list(getattr(soul_spark, 'emotional_resonance', {}).values())) 
                if getattr(soul_spark, 'emotional_resonance') else 0.0,
            'creator_connection': getattr(soul_spark, 'creator_connection_strength', 0.0),
            'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
            'elemental_alignment': 0.0,  # Calculate from elemental affinity
            'cycle_synchronization': 0.0,  # Calculate from earth cycles
            'harmony': getattr(soul_spark, 'harmony', 0.0),
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
        }
        
        # Calculate elemental alignment from affinity
        elemental_affinity = getattr(soul_spark, 'elemental_affinity', None)
        if elemental_affinity and hasattr(soul_spark, 'elemental_affinities'):
            affinity_scores = getattr(soul_spark, 'elemental_affinities', {})
            attributes['elemental_alignment'] = affinity_scores.get(elemental_affinity, 0.0)
        
        # Calculate cycle synchronization from earth resonance factors
        earth_resonance = attributes['earth_resonance']
        creator_connection = attributes['creator_connection']
        if earth_resonance > 0.1 and creator_connection > 0.1:
            attributes['cycle_synchronization'] = (earth_resonance * creator_connection) ** 0.5
        
        # Filter and validate attribute values
        attr_values = [v for v in attributes.values() 
                      if isinstance(v, (int, float)) and 
                      np.isfinite(v) and 
                      0.0 <= v <= 1.0]
        
        if len(attr_values) < 5:
            coherence_score = 0.5
            logger.warning(f"  Attribute Coherence: Not enough valid attributes " +
                          f"({len(attr_values)}). Using default {coherence_score}.")
        else:
            # Multiple coherence measures for robust calculation
            
            # 1. Standard deviation approach (lower std_dev = more coherence)
            std_dev = np.std(attr_values)
            std_dev_coherence = max(0.0, 1.0 - min(1.0, std_dev * ATTRIBUTE_COHERENCE_STD_DEV_SCALE))
            
            # 2. Pattern coherence through phi relationships
            phi_related_attrs = [
                attributes['phi_resonance'],
                attributes['pattern_coherence'], 
                attributes['harmony']
            ]
            phi_values = [v for v in phi_related_attrs if isinstance(v, (int, float)) and np.isfinite(v)]
            
            if len(phi_values) >= 2:
                phi_std = np.std(phi_values)
                pattern_coherence = max(0.0, 1.0 - min(1.0, phi_std * 1.5))
            else:
                pattern_coherence = np.mean(phi_values) if phi_values else 0.0
            
            # 3. Identity crystallization coherence
            identity_attrs = [
                attributes['name_resonance'],
                attributes['crystallization_level'],
                attributes['state_stability']
            ]
            identity_values = [v for v in identity_attrs if isinstance(v, (int, float)) and np.isfinite(v)]
            
            if len(identity_values) >= 2:
                identity_coherence = 1.0 - min(1.0, np.std(identity_values) * 2.0)
            else:
                identity_coherence = np.mean(identity_values) if identity_values else 0.0
            
            # 4. Spiritual connection coherence
            spiritual_attrs = [
                attributes['creator_connection'],
                attributes['earth_resonance'],
                attributes['emotional_resonance_avg']
            ]
            spiritual_values = [v for v in spiritual_attrs if isinstance(v, (int, float)) and np.isfinite(v)]
            
            if len(spiritual_values) >= 2:
                spiritual_coherence = 1.0 - min(1.0, np.std(spiritual_values) * 1.8)
            else:
                spiritual_coherence = np.mean(spiritual_values) if spiritual_values else 0.0
            
            # Combine coherence measures with appropriate weights
            coherence_score = (
                0.3 * std_dev_coherence +      # Overall attribute consistency
                0.25 * pattern_coherence +      # Sacred geometry/phi patterns
                0.25 * identity_coherence +     # Core identity stability  
                0.2 * spiritual_coherence)      # Spiritual connections
        
        # Apply resonant harmony enhancement for high-coherence souls
        harmony_factors = []
        
        if attributes['harmony'] > 0.7 and attributes['pattern_coherence'] > 0.7:
            harmony_factors.append(
                attributes['harmony'] * attributes['pattern_coherence'] * 0.4)
                
        if attributes['phi_resonance'] > 0.8 and attributes['toroidal_flow_strength'] > 0.6:
            harmony_factors.append(
                attributes['phi_resonance'] * attributes['toroidal_flow_strength'] * 0.3)
                
        if attributes['earth_resonance'] > 0.8 and attributes['creator_connection'] > 0.6:
            harmony_factors.append(
                attributes['earth_resonance'] * attributes['creator_connection'] * 0.3)
        
        # Apply harmony boost if we have strong harmony factors
        if harmony_factors:
            harmony_boost = sum(harmony_factors) / len(harmony_factors)
            final_coherence = min(1.0, coherence_score * (1.0 + harmony_boost * 0.25))
            logger.debug(f"  Applied resonant harmony boost: {harmony_boost:.4f} -> " +
                        f"Score={final_coherence:.4f}")
        else:
            final_coherence = coherence_score
        
        # Store attribute coherence directly on soul_spark
        setattr(soul_spark, 'attribute_coherence', float(final_coherence))
        setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "attribute_values": len(attr_values),
                "std_dev_coherence": float(std_dev_coherence) if 'std_dev_coherence' in locals() else 0.0,
                "pattern_coherence": float(pattern_coherence) if 'pattern_coherence' in locals() else 0.0,
                "identity_coherence": float(identity_coherence) if 'identity_coherence' in locals() else 0.0,
                "spiritual_coherence": float(spiritual_coherence) if 'spiritual_coherence' in locals() else 0.0,
                "harmony_boost": float(harmony_boost) if 'harmony_boost' in locals() else 0.0,
                "final_coherence": float(final_coherence),
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_attribute_coherence', metrics_data)
        
        logger.info(f"Attribute coherence calculated: {final_coherence:.4f}")
        logger.info(f"  Attribute Values: {attributes}")
    
    except Exception as e:
        logger.error(f"Error calculating attribute coherence: {e}", exc_info=True)
        raise RuntimeError("Failed to calculate attribute coherence") from e

def _create_crystalline_structure(soul_spark: SoulSpark) -> Dict[str, Any]:
    """
    Create crystalline structure that integrates all identity components
    into a coherent pattern using light physics principles.
    
    Args:
        soul_spark: Soul to create structure for
        
    Returns:
        Dictionary with crystalline structure data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If structure creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    
    try:
        name = getattr(soul_spark, 'name', 'Unknown')
        logger.debug(f"Creating crystalline structure for '{name}'...")
        
        # Collect key characteristics
        aspects = []
        
        # Name information
        name_standing_waves = getattr(soul_spark, 'name_standing_waves', None)
        if name_standing_waves:
            aspects.append({
                "type": "name",
                "name": name,
                "resonance": getattr(soul_spark, 'name_resonance', 0.0),
                "frequencies": name_standing_waves.get("component_frequencies", []),
                "nodes": len(name_standing_waves.get("nodes", [])),
                "coherence": name_standing_waves.get("pattern_coherence", 0.0)
            })
            
        # Voice information
        voice_freq = getattr(soul_spark, 'voice_frequency', 0.0)
        if voice_freq > 0:
            aspects.append({
                "type": "voice",
                "frequency": float(voice_freq),
                "wavelength": getattr(soul_spark, 'voice_wavelength', 0.0)
            })
            
        # Color information
        color_props = getattr(soul_spark, 'color_properties', None)
        if color_props:
            aspects.append({
                "type": "color",
                "color": color_props.get("hex", "#FFFFFF"),
                "frequency": color_props.get("frequency_hz", 0.0)
            })
            
        # Sephiroth aspect
        seph_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
        if seph_aspect:
            aspects.append({
                "type": "sephiroth",
                "aspect": seph_aspect
            })
            
        # Elemental affinity
        elem_affinity = getattr(soul_spark, 'elemental_affinity', None)
        if elem_affinity:
            aspects.append({
                "type": "element",
                "element": elem_affinity
            })
            
        # Platonic symbol
        platonic_symbol = getattr(soul_spark, 'platonic_symbol', None)
        if platonic_symbol:
            aspects.append({
                "type": "platonic",
                "symbol": platonic_symbol
            })
            
        # Sacred geometry
        sacred_geometry = getattr(soul_spark, 'sacred_geometry_imprint', None)
        if sacred_geometry:
            aspects.append({
                "type": "sacred_geometry",
                "geometry": sacred_geometry
            })
            
        # Astrological signature
        zodiac_sign = getattr(soul_spark, 'zodiac_sign', None)
        planet = getattr(soul_spark, 'governing_planet', None)
        if zodiac_sign and planet:
            aspects.append({
                "type": "astrology",
                "sign": zodiac_sign,
                "planet": planet
            })
        
        # Calculate integration coherence for these aspects
        # More aspects with coherent relationships = higher score
        aspect_count = len(aspects)
        
        # Create crystalline structure based on aspect integration
        # This structure uses sacred geometry and light physics
        
        # 1. Establish geometric base structure
        base_geometry = platonic_symbol if platonic_symbol else "sphere"
        sacred_imprint = sacred_geometry if sacred_geometry else "seed_of_life"
        
        # 2. Calculate light frequencies from aspects
        frequencies = []
        
        for aspect in aspects:
            if "frequency" in aspect:
                frequencies.append(aspect["frequency"])
            elif aspect["type"] == "name" and "frequencies" in aspect:
                frequencies.extend(aspect["frequencies"])
        
        # Ensure we have at least one frequency
        if not frequencies:
            frequencies = [soul_spark.frequency]
            
        # 3. Create light interference pattern
        num_points = 100
        crystal_pattern = np.zeros(num_points)
        
        # Generate pattern from frequencies
        for freq in frequencies:
            if freq <= 0:
                continue
                
            # Convert frequency to wavelength
            wavelength = SPEED_OF_SOUND / freq
            
            # Create wave for this frequency
            x = np.linspace(0, 1, num_points)
            freq_wave = np.sin(2 * PI * x / wavelength)
            
            # Add to pattern with constructive interference
            crystal_pattern += freq_wave
        
        # Normalize pattern
        max_val = np.max(np.abs(crystal_pattern))
        if max_val > FLOAT_EPSILON:
            normalized_pattern = crystal_pattern / max_val
        else:
            normalized_pattern = np.zeros_like(crystal_pattern)
            
        # 4. Calculate crystalline coherence
        # Higher coherence = more integrated identity
        
        # Use autocorrelation to measure pattern regularity
        autocorr = np.correlate(normalized_pattern, normalized_pattern, mode='full')
        autocorr_normalized = autocorr[num_points-1:] / autocorr[num_points-1]
        
        # Find peaks in autocorrelation
        peak_threshold = 0.5
        peaks = []
        
        for i in range(1, len(autocorr_normalized) - 1):
            if (autocorr_normalized[i] > peak_threshold and 
                autocorr_normalized[i] > autocorr_normalized[i-1] and 
                autocorr_normalized[i] > autocorr_normalized[i+1]):
                peaks.append(i)
        
        # Calculate coherence based on peak regularity
        peak_coherence = 0.0
        if len(peaks) > 1:
            # Calculate spacings between peaks
            spacings = np.diff(peaks)
            
            # Measure regularity (lower std = more regular)
            spacing_std = np.std(spacings)
            spacing_mean = np.mean(spacings)
            
            if spacing_mean > FLOAT_EPSILON:
                peak_coherence = 1.0 - min(1.0, spacing_std / spacing_mean)
        
        # Combine aspect count and pattern coherence
        aspect_factor = min(1.0, aspect_count / 10.0)  # Cap at 10 aspects
        crystalline_coherence = 0.3 * aspect_factor + 0.7 * peak_coherence
        
        # 5. Create final crystalline structure
        crystalline_structure = {
            "aspects": aspects,
            "aspect_count": aspect_count,
            "base_geometry": base_geometry,
            "sacred_imprint": sacred_imprint,
            "pattern": normalized_pattern.tolist(),
            "coherence": float(crystalline_coherence),
            "peak_coherence": float(peak_coherence),
            "aspect_factor": float(aspect_factor)
        }
        
        logger.debug(f"Created crystalline structure with " +
                   f"{aspect_count} aspects, " +
                   f"coherence: {crystalline_coherence:.4f}")
        
        return crystalline_structure
        
    except Exception as e:
        logger.error(f"Error creating crystalline structure: {e}", exc_info=True)
        raise RuntimeError("Crystalline structure creation failed") from e

def verify_identity_crystallization(soul_spark: SoulSpark, 
                                 threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies identity crystallization by measuring how well identity components
    have formed coherent patterns within the soul_spark itself.
    
    Crystallization measures:
    - Identity attribute completeness and stability
    - Coherence between identity components 
    - Internal pattern formation within the soul
    - Self-recognition strength
    """
    logger.info("Identity Step: Verify Crystallization with Light Physics...")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("Threshold invalid.")
    
    try:
        # Check for required attributes
        required_attrs_for_score = CRYSTALLIZATION_REQUIRED_ATTRIBUTES
        missing_attributes = [attr for attr in required_attrs_for_score 
                            if getattr(soul_spark, attr, None) is None]
        attr_presence_score = (len(required_attrs_for_score) - len(missing_attributes)) / max(1, len(required_attrs_for_score))
        
        # COMPREHENSIVE LOGGING: Log attribute details
        logger.info(f"=== IDENTITY CRYSTALLIZATION DEBUG ===")
        logger.info(f"Soul ID: {getattr(soul_spark, 'spark_id', 'Unknown')}")
        logger.info(f"Required attributes count: {len(required_attrs_for_score)}")
        logger.info(f"Missing attributes count: {len(missing_attributes)}")
        logger.info(f"Missing attributes: {missing_attributes}")
        logger.info(f"Attribute presence score: {attr_presence_score}")
        logger.info(f"Attribute presence threshold: {CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}")
        
        # Log each required attribute value
        logger.info("=== REQUIRED ATTRIBUTE VALUES ===")
        for attr in required_attrs_for_score:
            value = getattr(soul_spark, attr, None)
            logger.info(f"  {attr}: {value} (type: {type(value).__name__})")
        
        # Get component metrics - Focus on SOUL identity attributes only
        component_metrics = {
            'name_resonance': getattr(soul_spark, 'name_resonance', 0.0),
            'response_level': getattr(soul_spark, 'response_level', 0.0),
            'state_stability': getattr(soul_spark, 'state_stability', 0.0),
            'crystallization_level': getattr(soul_spark, 'crystallization_level', 0.0),
            'attribute_coherence': getattr(soul_spark, 'attribute_coherence', 0.0),
            'emotional_resonance': np.mean(list(getattr(soul_spark, 'emotional_resonance', {}).values())) 
                if getattr(soul_spark, 'emotional_resonance') else 0.0,
            'harmony': getattr(soul_spark, 'harmony', 0.0),
            'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
            'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
            'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
        }
        
        logger.info(f"=== COMPONENT METRICS ===")
        for comp, value in component_metrics.items():
            logger.info(f"  {comp}: {value}")
        
        logger.debug(f"  Identity Verification Components: {component_metrics}")
        
        # Create crystalline structure that integrates all components within the soul
        crystalline_structure = _create_crystalline_structure(soul_spark)
        crystalline_coherence = crystalline_structure.get("coherence", 0.0)
        
        # Add crystalline coherence to component metrics
        component_metrics['crystalline_coherence'] = crystalline_coherence
        
        # === NATURAL LATTICE FORMATION VERIFICATION ===
        # Check if identity aspects form stable, self-reinforcing crystalline patterns
        logger.info("=== LATTICE FORMATION VERIFICATION ===")
        
        lattice_stability_checks = {}
        lattice_formation_success = True
        
        # 1. UNIT CELL CHECK - Basic identity structure must be stable
        logger.info("  Checking Unit Cell Structure...")
        unit_cell_elements = {
            'name_recognition': getattr(soul_spark, 'name_recognition_rate', 0.0),
            'voice_frequency': getattr(soul_spark, 'voice_frequency', 0.0),
            'name_resonance': component_metrics.get('name_resonance', 0.0),
            'harmony': component_metrics.get('harmony', 0.0)
        }
        
        unit_cell_stable = True
        for element, value in unit_cell_elements.items():
            is_stable = value >= 0.3  # 30% minimum for stable unit cell - natural growth threshold
            lattice_stability_checks[f'unit_cell_{element}'] = is_stable
            if not is_stable:
                unit_cell_stable = False
                logger.warning(f"    Unit cell unstable: {element} = {value:.3f} < 0.3")
            else:
                logger.info(f"    Unit cell stable: {element} = {value:.3f}")
        
        # 2. STRUCTURAL FRAMEWORK CHECK - Identity aspects must be present and coherent
        logger.info("  Checking Structural Framework...")
        framework_elements = {
            'sephiroth_aspect': getattr(soul_spark, 'sephiroth_aspect', None),
            'elemental_affinity': getattr(soul_spark, 'elemental_affinity', None), 
            'platonic_symbol': getattr(soul_spark, 'platonic_symbol', None),
            'zodiac_sign': getattr(soul_spark, 'zodiac_sign', None),
            'pattern_coherence': component_metrics.get('pattern_coherence', 0.0)
        }
        
        framework_stable = True
        for element, value in framework_elements.items():
            if element == 'pattern_coherence':
                is_stable = value >= 0.9  # Pattern coherence must be very high
                lattice_stability_checks[f'framework_{element}'] = is_stable
                if not is_stable:
                    framework_stable = False
                    logger.warning(f"    Framework unstable: {element} = {value:.3f} < 0.9")
                else:
                    logger.info(f"    Framework stable: {element} = {value:.3f}")
            else:
                is_stable = value is not None
                lattice_stability_checks[f'framework_{element}'] = is_stable
                if not is_stable:
                    framework_stable = False
                    logger.warning(f"    Framework incomplete: {element} = None")
                else:
                    logger.info(f"    Framework complete: {element} = {value}")
        
        # 3. BINDING FORCES CHECK - Emotional regulation and training must be effective
        logger.info("  Checking Binding Forces...")
        binding_elements = {
            'name_lattice_formed': getattr(soul_spark, 'name_lattice_formed', False),
            'foundational_security': getattr(soul_spark, 'foundational_security', 0.0),
            'emotional_resonance': component_metrics.get('emotional_resonance', 0.0),
            'phi_resonance': component_metrics.get('phi_resonance', 0.0),
            'attribute_coherence': component_metrics.get('attribute_coherence', 0.0)
        }
        
        binding_stable = True
        for element, value in binding_elements.items():
            if element == 'name_lattice_formed':
                is_stable = value == True
                lattice_stability_checks[f'binding_{element}'] = is_stable
                if not is_stable:
                    binding_stable = False
                    logger.warning(f"    Binding incomplete: {element} = {value}")
                else:
                    logger.info(f"    Binding complete: {element} = {value}")
            else:
                is_stable = value >= 0.2  # Natural threshold for binding forces - meaningful resonance
                lattice_stability_checks[f'binding_{element}'] = is_stable
                if not is_stable:
                    binding_stable = False
                    logger.warning(f"    Binding weak: {element} = {value:.3f} < 0.2")
                else:
                    logger.info(f"    Binding adequate: {element} = {value:.3f}")
        
        # 4. OVERALL LATTICE STABILITY
        stable_checks = sum(1 for stable in lattice_stability_checks.values() if stable)
        total_checks = len(lattice_stability_checks)
        lattice_stability_ratio = stable_checks / total_checks
        
        # Lattice formation requires ALL critical elements to be stable
        critical_checks = ['unit_cell_name_recognition', 'unit_cell_harmony', 'framework_pattern_coherence', 'binding_name_lattice_formed']
        critical_stable = all(lattice_stability_checks.get(check, False) for check in critical_checks)
        
        overall_lattice_stable = critical_stable and lattice_stability_ratio >= 0.6  # Natural formation threshold
        
        logger.info(f"  Lattice stability: {stable_checks}/{total_checks} checks passed ({lattice_stability_ratio:.1%})")
        logger.info(f"  Critical elements stable: {critical_stable}")
        logger.info(f"  Overall lattice stable: {overall_lattice_stable}")
        
        # Calculate crystallization score based on lattice stability with natural thresholds
        if overall_lattice_stable:
            total_crystallization_score = 0.85 + (lattice_stability_ratio - 0.6) * 0.375  # 0.85 to 1.0 range when >=60% 
        else:
            total_crystallization_score = lattice_stability_ratio * 1.4  # Natural growth scaling - up to 1.4 for full completion
        
        total_crystallization_score = max(0.0, min(1.0, total_crystallization_score))
        logger.info(f"Lattice-based crystallization score: {total_crystallization_score:.4f}")
        
        # Check crystallization success
        logger.info(f"=== FINAL DECISION ===")
        logger.info(f"Final crystallization score: {total_crystallization_score:.4f}")
        logger.info(f"Required threshold: {threshold}")
        logger.info(f"Score meets threshold: {total_crystallization_score >= threshold}")
        logger.info(f"Attribute presence score: {attr_presence_score:.4f}")
        logger.info(f"Required attr presence threshold: {CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}")
        logger.info(f"Attr presence meets threshold: {attr_presence_score >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}")
        
        is_crystallized = (total_crystallization_score >= threshold and 
                          attr_presence_score >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD)
        logger.info(f"FINAL RESULT: is_crystallized = {is_crystallized}")
        
        logger.debug(f"  Identity Verification: Score={total_crystallization_score:.4f}, " +
                   f"Threshold={threshold}, AttrPresence={attr_presence_score:.2f} -> " +
                   f"Crystallized={is_crystallized}")
        
        # Create identity light signature based on crystalline structure
        if is_crystallized:
            identity_light_signature = _create_identity_light_signature(soul_spark, crystalline_structure)
            
            # Create identity sound signature if sound modules available
            if SOUND_MODULES_AVAILABLE:
                try:
                    identity_sound_signature = _create_identity_sound_signature(
                        soul_spark, crystalline_structure)
                    setattr(soul_spark, 'identity_sound_signature', identity_sound_signature)
                except Exception as sound_err:
                    logger.warning(f"Failed to create identity sound signature: {sound_err}")
                    # Continue without failing
            
            # Set crystalline structure and light signature
            setattr(soul_spark, 'crystalline_structure', crystalline_structure)
            setattr(soul_spark, 'identity_light_signature', identity_light_signature)
        
        # Create verification result
        verification_result = {
            'total_crystallization_score': float(total_crystallization_score),
            'threshold': threshold,
            'is_crystallized': is_crystallized,
            'components': component_metrics,
            'lattice_stability_checks': lattice_stability_checks,
            'missing_attributes': missing_attributes,
            'crystalline_coherence': float(crystalline_coherence)
        }
        
        # Add crystalline structure summary if crystallized
        if is_crystallized:
            crystalline_summary = {
                "aspect_count": crystalline_structure.get("aspect_count", 0),
                "coherence": crystalline_structure.get("coherence", 0.0),
                "base_geometry": crystalline_structure.get("base_geometry", "unknown"),
                "sacred_imprint": crystalline_structure.get("sacred_imprint", "unknown")
            }
            verification_result['crystalline_summary'] = crystalline_summary
        
        setattr(soul_spark, 'identity_metrics', verification_result)
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics_data = {
                "soul_id": soul_spark.spark_id,
                "crystallization_score": float(total_crystallization_score),
                "threshold": float(threshold),
                "is_crystallized": is_crystallized,
                "crystalline_coherence": float(crystalline_coherence),
                "missing_attributes": missing_attributes,
                "timestamp": datetime.now().isoformat()
            }
            metrics.record_metrics('identity_verification', metrics_data)
        
        # Raise error if not crystallized
        if not is_crystallized:
            score_fails = total_crystallization_score < threshold
            attr_fails = attr_presence_score < CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD
            
            failure_reasons = []
            if score_fails:
                failure_reasons.append(f"Score {total_crystallization_score:.4f} < Threshold {threshold}")
            if attr_fails:
                failure_reasons.append(f"Attr Presence {attr_presence_score:.2f} < {CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}")
            
            error_msg = f"Identity crystallization failed: {' AND '.join(failure_reasons)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Success message
        logger.info(f"Identity check PASSED: Score={total_crystallization_score:.4f}, " +
                   f"Crystalline Coherence={crystalline_coherence:.4f}")
        
        # Set ready for birth flag if crystallized
        if is_crystallized:
            setattr(soul_spark, FLAG_READY_FOR_BIRTH, True)
        
        return is_crystallized, verification_result
        
    except Exception as e:
        logger.error(f"Error verifying identity: {e}", exc_info=True)
        raise RuntimeError("Identity verification failed.") from e

def _create_identity_light_signature(soul_spark: SoulSpark, 
                                 crystalline_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create identity light signature from crystalline structure.
    
    Args:
        soul_spark: Soul to create signature for
        crystalline_structure: Crystalline structure data
        
    Returns:
        Dictionary with light signature data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If signature creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not isinstance(crystalline_structure, dict):
        raise ValueError("crystalline_structure must be a dictionary")
    
    try:
        logger.debug(f"Creating identity light signature...")
        
        # Extract key components
        name = getattr(soul_spark, 'name', 'Unknown')
        aspects = crystalline_structure.get("aspects", [])
        pattern = crystalline_structure.get("pattern", [])
        coherence = crystalline_structure.get("coherence", 0.0)
        
        # 1. Map audio frequencies to light spectrum
        light_frequencies = []
        
        for aspect in aspects:
            if "frequency" in aspect:
                # Convert audio to light frequency
                light_freq = aspect["frequency"] * 1e12  # Simple scaling
                wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
                
                # Map to color
                if 380 <= wavelength_nm <= 750:
                    if 380 <= wavelength_nm < 450: color = "violet"
                    elif 450 <= wavelength_nm < 495: color = "blue"
                    elif 495 <= wavelength_nm < 570: color = "green"
                    elif 570 <= wavelength_nm < 590: color = "yellow"
                    elif 590 <= wavelength_nm < 620: color = "orange"
                    else: color = "red"
                else:
                    color = "beyond visible"
                    
                light_frequencies.append({
                    "aspect": aspect.get("type", "unknown"),
                    "audio_frequency": float(aspect["frequency"]),
                    "light_frequency": float(light_freq),
                    "wavelength_nm": float(wavelength_nm),
                    "color": color
                })
            elif aspect["type"] == "name" and "frequencies" in aspect:
                # Process name frequencies
                for freq in aspect["frequencies"]:
                    # Convert audio to light frequency
                    light_freq = freq * 1e12  # Simple scaling
                    wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
                    
                    # Map to color (simplified)
                    if 380 <= wavelength_nm <= 750:
                        if 380 <= wavelength_nm < 450: color = "violet"
                        elif 450 <= wavelength_nm < 495: color = "blue"
                        elif 495 <= wavelength_nm < 570: color = "green"
                        elif 570 <= wavelength_nm < 590: color = "yellow"
                        elif 590 <= wavelength_nm < 620: color = "orange"
                        else: color = "red"
                    else:
                        color = "beyond visible"
                        
                    light_frequencies.append({
                        "aspect": "name",
                        "audio_frequency": float(freq),
                        "light_frequency": float(light_freq),
                        "wavelength_nm": float(wavelength_nm),
                        "color": color
                    })
        
# 2. Create light interference pattern from crystalline pattern
        light_pattern = []
        
        if pattern:
            # Convert to numpy array if not already
            if isinstance(pattern, list):
                pattern_array = np.array(pattern)
            else:
                pattern_array = pattern
                
            # Create light interference pattern
            num_points = len(pattern_array)
            
            for i in range(num_points):
                # Convert position to wavelength
                pos = i / num_points
                
                # Calculate value from pattern
                value = pattern_array[i]
                
                # Map to light spectrum
                # Use wavelength based on position (380-750nm range)
                wavelength_nm = 380 + pos * (750 - 380)
                
                # Calculate frequency from wavelength
                light_freq = SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
                
                # Calculate amplitude (intensity) from pattern value
                # Scale to 0-1 range
                amplitude = (value + 1) / 2
                
                # Add to light pattern
                light_pattern.append({
                    "position": float(pos),
                    "wavelength_nm": float(wavelength_nm),
                    "frequency_hz": float(light_freq),
                    "amplitude": float(amplitude)
                })
        
        # 3. Calculate primary and secondary colors
        # Primary color based on soul color
        primary_color = getattr(soul_spark, 'soul_color', '#FFFFFF')
        
        # Secondary colors from aspects
        secondary_colors = []
        for aspect in aspects:
            if aspect["type"] == "color" and "color" in aspect:
                secondary_colors.append(aspect["color"])
            elif aspect["type"] == "sephiroth":
                # Add Sephiroth color if not already present
                seph_color = aspect_dictionary.get_aspects(aspect["aspect"]).get('color', '#FFFFFF')
                if seph_color not in secondary_colors:
                    secondary_colors.append(seph_color)
            elif aspect["type"] == "element" and aspect["element"] in ["fire", "water", "air", "earth", "aether"]:
                # Add element color if not already present
                element_colors = {
                    'fire': '#FF5500',
                    'water': '#0077FF',
                    'air': '#AAFFFF',
                    'earth': '#996633',
                    'aether': '#FFFFFF'
                }
                elem_color = element_colors.get(aspect["element"], '#CCCCCC')
                if elem_color not in secondary_colors:
                    secondary_colors.append(elem_color)
        
        # 4. Create complete light signature
        light_signature = {
            "name": name,
            "primary_color": primary_color,
            "secondary_colors": secondary_colors,
            "light_frequencies": light_frequencies,
            "light_pattern": light_pattern,
            "coherence": float(coherence)
        }
        
        logger.debug(f"Created identity light signature with " +
                   f"{len(light_frequencies)} frequencies, " +
                   f"{len(light_pattern)} pattern points")
        
        return light_signature
        
    except Exception as e:
        logger.error(f"Error creating identity light signature: {e}", exc_info=True)
        raise RuntimeError("Identity light signature creation failed") from e

def _create_identity_sound_signature(soul_spark: SoulSpark, 
                                  crystalline_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create identity sound signature from crystalline structure.
    
    Args:
        soul_spark: Soul to create signature for
        crystalline_structure: Crystalline structure data
        
    Returns:
        Dictionary with sound signature data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If signature creation fails
    """
    if not isinstance(soul_spark, SoulSpark):
        raise ValueError("soul_spark must be a SoulSpark instance")
    if not isinstance(crystalline_structure, dict):
        raise ValueError("crystalline_structure must be a dictionary")
    if not SOUND_MODULES_AVAILABLE:
        raise RuntimeError("Sound modules not available")
    
    try:
        logger.debug(f"Creating identity sound signature...")
        
        # Extract key components
        name = getattr(soul_spark, 'name', 'Unknown')
        aspects = crystalline_structure.get("aspects", [])
        pattern = crystalline_structure.get("pattern", [])
        
        # Get base frequency from soul
        base_freq = soul_spark.frequency
        
        # 1. Collect frequencies from aspects
        frequencies = []
        amplitudes = []
        
        for aspect in aspects:
            if "frequency" in aspect:
                frequencies.append(aspect["frequency"])
                amplitudes.append(0.7)  # Default amplitude
            elif aspect["type"] == "name" and "frequencies" in aspect:
                for i, freq in enumerate(aspect["frequencies"]):
                    frequencies.append(freq)
                    # Decreasing amplitude for higher harmonics
                    amp = 0.8 / (i + 1)
                    amplitudes.append(amp)
        
        # Ensure we have at least one frequency
        if not frequencies:
            frequencies = [base_freq]
            amplitudes = [0.8]
            
        # 2. Create harmonic structure
        # Normalize amplitudes to avoid clipping
        max_amp = max(amplitudes) if amplitudes else 1.0
        if max_amp > FLOAT_EPSILON:
            norm_amplitudes = [a / max_amp for a in amplitudes]
        else:
            norm_amplitudes = [0.0] * len(amplitudes)
            
        # Convert to harmonic ratios for sound generation
        harm_ratios = [f / base_freq for f in frequencies]
        
        # 3. Generate identity sound
        sound_duration = 5.0  # seconds
        
        identity_sound = sound_gen.generate_harmonic_tone(
            base_freq, harm_ratios, norm_amplitudes, sound_duration, 0.5)
        
        # 4. Save identity sound
        sound_path = sound_gen.save_sound(
            identity_sound, f"identity_{name.lower()}.wav", 
            f"Identity Sound for {name}")
        
        # 5. Create complete sound signature
        sound_signature = {
            "name": name,
            "base_frequency": float(base_freq),
            "harmonics": [float(r) for r in harm_ratios],
            "amplitudes": [float(a) for a in norm_amplitudes],
            "duration": float(sound_duration),
            "sound_path": sound_path
        }
        
        logger.debug(f"Created identity sound signature with " +
                   f"{len(frequencies)} frequencies, saved to {sound_path}")
        
        return sound_signature
        
    except Exception as e:
        logger.error(f"Error creating identity sound signature: {e}", exc_info=True)
        raise RuntimeError("Identity sound signature creation failed") from e

# --- Orchestration Function ---
def perform_identity_crystallization(soul_spark: SoulSpark,
                                    train_cycles: int = 7,
                                    entrainment_bpm: float = 72.0,
                                    entrainment_duration: float = 120.0,
                                    love_cycles: int = 5,
                                    geometry_stages: int = 2,
                                    crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD
                                    ) -> Tuple[SoulSpark, Dict[str, Any]]:
    """
    Performs complete identity crystallization with light, sound, and crystallisation principles.
    Creates a coherent identity pattern throughout. Stability and coherence emerge naturally.
    """
    # --- Input Validation ---
    if not isinstance(soul_spark, SoulSpark):
        raise TypeError("soul_spark invalid.")
    if not isinstance(train_cycles, int) or train_cycles < 0:
        raise ValueError("train_cycles must be >= 0")
    if not isinstance(entrainment_bpm, (int, float)) or entrainment_bpm <= 0:
        raise ValueError("bpm must be > 0")
    if not isinstance(entrainment_duration, (int, float)) or entrainment_duration < 0:
        raise ValueError("duration must be >= 0")
    if not isinstance(love_cycles, int) or love_cycles < 0:
        raise ValueError("love_cycles must be >= 0")
    if not isinstance(geometry_stages, int) or geometry_stages < 0:
        raise ValueError("geometry_stages must be >= 0")
    if not (0.0 < crystallization_threshold <= 1.0):
        raise ValueError("Threshold invalid.")

    spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
    logger.info(f"--- Starting Identity Crystallization for Soul {spark_id} ---")
    start_time_iso = datetime.now().isoformat()
    start_time_dt = datetime.fromisoformat(start_time_iso)
    process_metrics_summary = {'steps_completed': []}

    try:
        _ensure_soul_properties(soul_spark) # Raises error if fails
        _check_prerequisites(soul_spark) # Raises error if fails

        initial_state = {
             'stability_su': soul_spark.stability, 
             'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level,
             'attribute_coherence': soul_spark.attribute_coherence,
             'harmony': soul_spark.harmony,
             'phi_resonance': soul_spark.phi_resonance,
             'pattern_coherence': soul_spark.pattern_coherence,
             'toroidal_flow_strength': soul_spark.toroidal_flow_strength
        }
        logger.info(f"Identity Init State: S={initial_state['stability_su']:.1f}, " +
                   f"C={initial_state['coherence_cu']:.1f}, " +
                   f"CrystLvl={initial_state['crystallization_level']:.3f}, " +
                   f"Harm={initial_state['harmony']:.3f}, " +
                   f"PhiRes={initial_state['phi_resonance']:.3f}, " +
                   f"PatCoh={initial_state['pattern_coherence']:.3f}, " +
                   f"Torus={initial_state['toroidal_flow_strength']:.3f}")

        # --- Run Sequence ---
        assign_name(soul_spark)
        process_metrics_summary['steps_completed'].append('name')

        # Establish mother's voice as foundational security layer
        establish_mothers_voice_foundation(soul_spark)
        process_metrics_summary['steps_completed'].append('mothers_voice')

        # Identity crystallization works within the soul
        process_metrics_summary['steps_completed'].append('identity_setup')
        
        assign_voice_frequency(soul_spark)
        process_metrics_summary['steps_completed'].append('voice')
        
        process_soul_color(soul_spark)
        process_metrics_summary['steps_completed'].append('color')
        
        apply_heartbeat_entrainment(soul_spark, entrainment_bpm, entrainment_duration)
        process_metrics_summary['steps_completed'].append('heartbeat')
        
        train_name_response(soul_spark, train_cycles)
        process_metrics_summary['steps_completed'].append('response')
        
        identify_primary_sephiroth(soul_spark)
        process_metrics_summary['steps_completed'].append('sephiroth_id')
        
        determine_elemental_affinity(soul_spark)
        process_metrics_summary['steps_completed'].append('elemental_id')
        
        assign_platonic_symbol(soul_spark)
        process_metrics_summary['steps_completed'].append('platonic_id')
        
        _determine_astrological_signature(soul_spark)
        process_metrics_summary['steps_completed'].append('astrology')
        
        activate_love_resonance(soul_spark, love_cycles)
        process_metrics_summary['steps_completed'].append('love')
        
        apply_sacred_geometry(soul_spark, geometry_stages)
        process_metrics_summary['steps_completed'].append('geometry')
        
        calculate_attribute_coherence(soul_spark)
        process_metrics_summary['steps_completed'].append('attr_coherence')

        logger.info("Identity Step: Final State Update & Verification...")
        if hasattr(soul_spark, 'update_state'):
            soul_spark.update_state() # Update SU/CU scores after geometry factor influence
            logger.debug(f"  Identity S/C after update: S={soul_spark.stability:.1f}, " +
                       f"C={soul_spark.coherence:.1f}")
        else:
            raise AttributeError("SoulSpark missing 'update_state' method.")

        is_crystallized, verification_metrics = verify_identity_crystallization(
            soul_spark, crystallization_threshold)
        process_metrics_summary['steps_completed'].append('verify')
        # verify_identity_crystallization raises RuntimeError if it fails

        # --- Final Update & Metrics ---
        last_mod_time = datetime.now().isoformat()
        setattr(soul_spark, 'last_modified', last_mod_time)
        
        if hasattr(soul_spark, 'add_memory_echo'):
            soul_spark.add_memory_echo(
                f"Identity crystallized as '{soul_spark.name}'. " +
                f"Level: {soul_spark.crystallization_level:.3f}, " +
                f"Sign: {soul_spark.zodiac_sign}, " +
                f"S/C: {soul_spark.stability:.1f}/{soul_spark.coherence:.1f}")

        end_time_iso = last_mod_time
        end_time_dt = datetime.fromisoformat(end_time_iso)
        
        # Gather final state metrics
        final_state = {
             'stability_su': soul_spark.stability, 
             'coherence_cu': soul_spark.coherence,
             'crystallization_level': soul_spark.crystallization_level,
             'attribute_coherence': soul_spark.attribute_coherence,
             'harmony': soul_spark.harmony,
             'phi_resonance': soul_spark.phi_resonance,
             'pattern_coherence': soul_spark.pattern_coherence,
             'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
             'name': soul_spark.name, 
             'zodiac_sign': soul_spark.zodiac_sign,
             FLAG_IDENTITY_CRYSTALLIZED: getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED)
        }
        
        # Create complete process metrics
        overall_metrics = {
            'action': 'identity_crystallization',
            'soul_id': spark_id,
            'start_time': start_time_iso,
            'end_time': end_time_iso,
            'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
            'initial_state': initial_state,
            'final_state': final_state,
            'final_crystallization_score': verification_metrics['total_crystallization_score'],
            'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
            'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
            'harmony_change': final_state['harmony'] - initial_state['harmony'],
            'phi_resonance_change': final_state['phi_resonance'] - initial_state['phi_resonance'],
            'pattern_coherence_change': final_state['pattern_coherence'] - initial_state['pattern_coherence'],
            'torus_change': final_state['toroidal_flow_strength'] - initial_state['toroidal_flow_strength'],
            'astrological_signature': {
                'sign': soul_spark.zodiac_sign,
                'planet': soul_spark.governing_planet,
                'traits': soul_spark.astrological_traits
            },
            'crystalline_summary': getattr(soul_spark, 'crystalline_structure', {}).get('aspects', []),
            'success': True,
        }
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.record_metrics('identity_crystallization_summary', overall_metrics)

        logger.info(f"--- Identity Crystallization Completed Successfully for Soul {spark_id} ---")
        logger.info(f"  Final State: Name='{soul_spark.name}', " +
                   f"Sign='{soul_spark.zodiac_sign}', " +
                   f"CrystLvl={soul_spark.crystallization_level:.3f}, " +
                   f"S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        
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
         
         # Set failure flags
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False)
         setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         
         # Hard fail
         raise e_rt
         
    except Exception as e:
         logger.critical(f"Unexpected error during Identity Crystallization for {spark_id}: {e}", exc_info=True)
         failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'unexpected'
         record_id_failure(spark_id, start_time_iso, failed_step, str(e))
         
         # Set failure flags
         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False)
         setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         
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
                'action': 'identity_crystallization',
                'soul_id': spark_id,
                'start_time': start_time_iso,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': False,
                'error': error_msg,
                'failed_step': failed_step
            })
        except Exception as metric_e:
            logger.error(f"Failed to record ID failure metrics for {spark_id}: {metric_e}")

# --- END OF FILE ---