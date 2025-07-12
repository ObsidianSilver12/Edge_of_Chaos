
      















# """
# Identity Crystallization Module (Refactored V4.4.0 - Light-Sound Principles)

# Crystallizes identity after Earth attunement. Implements Name (user input), Voice, 
# Color, Affinities (Seph, Elem, Platonic), and Astrological Signature using complete
# light and sound principles. Implements proper light physics for crystalline formation 
# and uses aura layers for resonance rather than direct frequency modification.

# Heartbeat/Love cycles implement proper acoustic physics for harmony enhancement.
# Sacred Geometry uses proper light interference patterns and standing waves.
# Modifies SoulSpark via aura layers. Uses strict validation. Hard fails only.
# """

# import logging
# import numpy as np
# import os
# import sys
# from datetime import datetime, timedelta
# import time
# import random
# import uuid
# import math
# import re
# from typing import Dict, List, Any, Tuple, Optional
# from math import pi as PI, sqrt, exp, sin, cos, tanh

# # Direct import of constants without try blocks
# from constants.constants import *

# # --- Logging ---
# logger = logging.getLogger(__name__)
# if not logger.handlers:
#     logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# # --- Dependency Imports ---
# from stage_1.soul_spark.soul_spark import SoulSpark
# from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
# from stage_1.soul_formation.creator_entanglement import calculate_resonance

# # Sound module imports - These modules actively generate actual sound
# try:
#     from sound.sound_generator import SoundGenerator
#     from sound.sounds_of_universe import UniverseSounds
#     SOUND_MODULES_AVAILABLE = True
#     sound_gen = SoundGenerator(output_dir="output/sounds/identity")
#     universe_sounds = UniverseSounds(output_dir="output/sounds/identity")
# except ImportError:
#     logger.critical("Sound modules not available. Identity crystallization REQUIRES sound generation.")
#     SOUND_MODULES_AVAILABLE = False
#     raise ImportError("Critical sound modules missing.") from None

# # --- Metrics Tracking ---
# try:
#     import metrics_tracking as metrics
#     METRICS_AVAILABLE = True
# except ImportError:
#     logger.critical("Metrics tracking module not found. Identity crystallization REQUIRES metrics.")
#     METRICS_AVAILABLE = False
#     raise ImportError("Critical metrics module missing.") from None

# def log_soul_identity_summary(soul_spark: SoulSpark) -> None:
#     """
#     Log key soul identity properties to terminal before name assignment.
#     Shows the soul's fundamental characteristics.
#     """
#     print("\n" + "="*80)
#     print("SOUL IDENTITY SUMMARY - PRE-NAMING")
#     print("="*80)
    
#     # Basic soul info
#     print(f"Soul ID: {getattr(soul_spark, 'spark_id', 'Unknown')}")
#     print(f"Base Frequency: {getattr(soul_spark, 'frequency', 0.0):.2f} Hz")
#     print(f"Soul Frequency: {getattr(soul_spark, 'soul_frequency', 0.0):.2f} Hz")
    
#     # Soul color
#     soul_color = getattr(soul_spark, 'soul_color', None)
#     if soul_color:
#         print(f"Soul Color: {soul_color}")
#     else:
#         print("Soul Color: Not yet determined")
    
#     # Current state
#     print(f"Stability: {getattr(soul_spark, 'stability', 0.0):.1f} SU")
#     print(f"Coherence: {getattr(soul_spark, 'coherence', 0.0):.1f} CU")
#     print(f"Energy: {getattr(soul_spark, 'energy', 0.0):.1f} SEU")
    
#     # Sephiroth aspects
#     aspects = getattr(soul_spark, 'aspects', {})
#     if aspects:
#         print(f"\nSephiroth Aspects ({len(aspects)} total):")
#         # Show top 5 strongest aspects
#         sorted_aspects = sorted(aspects.items(), 
#                               key=lambda x: x[1].get('strength', 0.0), 
#                               reverse=True)[:5]
#         for aspect_name, aspect_data in sorted_aspects:
#             strength = aspect_data.get('strength', 0.0)
#             source = aspect_data.get('source', 'Unknown')
#             print(f"  • {aspect_name}: {strength:.3f} (from {source})")
#         if len(aspects) > 5:
#             print(f"  ... and {len(aspects) - 5} more aspects")
#     else:
#         print("\nSephiroth Aspects: None acquired yet")
    
#     # Astrological info (if available)
#     zodiac_sign = getattr(soul_spark, 'zodiac_sign', None)
#     governing_planet = getattr(soul_spark, 'governing_planet', None)
#     birth_datetime = getattr(soul_spark, 'conceptual_birth_datetime', None)
    
#     if zodiac_sign or governing_planet or birth_datetime:
#         print(f"\nAstrological Signature:")
#         if birth_datetime:
#             try:
#                 birth_dt = datetime.fromisoformat(birth_datetime)
#                 print(f"  Birth DateTime: {birth_dt.strftime('%Y-%m-%d %H:%M:%S')}")
#             except:
#                 print(f"  Birth DateTime: {birth_datetime}")
#         if zodiac_sign:
#             print(f"  Zodiac Sign: {zodiac_sign}")
#         if governing_planet:
#             print(f"  Governing Planet: {governing_planet}")
    
#     # Elemental balance
#     elements = getattr(soul_spark, 'elements', {})
#     if elements:
#         print(f"\nElemental Balance:")
#         for element, value in sorted(elements.items(), key=lambda x: x[1], reverse=True):
#             percentage = value * 100
#             print(f"  {element.capitalize()}: {percentage:.1f}%")
    
#     # Earth connection
#     earth_resonance = getattr(soul_spark, 'earth_resonance', 0.0)
#     gaia_connection = getattr(soul_spark, 'gaia_connection', 0.0)
#     if earth_resonance > 0 or gaia_connection > 0:
#         print(f"\nEarth Connection:")
#         print(f"  Earth Resonance: {earth_resonance:.3f}")
#         print(f"  Gaia Connection: {gaia_connection:.3f}")
    
#     # Layer count
#     layers = getattr(soul_spark, 'layers', [])
#     print(f"\nAura Layers: {len(layers)} formed")
    
#     print("="*80)
#     print("READY FOR NAME ASSIGNMENT")
#     print("="*80 + "\n")

# # --- Helper Functions ---

# def _check_prerequisites(soul_spark: SoulSpark) -> bool:
#     """ Checks prerequisites using SU/CU thresholds and 0-1 factor threshold. Raises ValueError on failure. """
#     logger.debug(f"Checking identity prerequisites for soul {soul_spark.spark_id}...")
#     if not isinstance(soul_spark, SoulSpark): raise TypeError("Invalid SoulSpark object.")

#     # 1. Stage Flag Check (Use FLAG_EARTH_ATTUNED)
#     if not getattr(soul_spark, FLAG_EARTH_ATTUNED, False):
#         msg = f"Prerequisite failed: Soul not marked {FLAG_EARTH_ATTUNED}."
#         logger.error(msg); raise ValueError(msg)

#     # 2. State Thresholds
#     stability_su = getattr(soul_spark, 'stability', -1.0)
#     coherence_cu = getattr(soul_spark, 'coherence', -1.0)
#     earth_resonance = getattr(soul_spark, 'earth_resonance', -1.0)
#     if stability_su < 0 or coherence_cu < 0 or earth_resonance < 0:
#         msg = "Prerequisite failed: Soul missing stability, coherence, or earth_resonance."
#         logger.error(msg); raise AttributeError(msg)

#     if stability_su < IDENTITY_STABILITY_THRESHOLD_SU:
#         msg = f"Prerequisite failed: Stability ({stability_su:.1f} SU) < {IDENTITY_STABILITY_THRESHOLD_SU} SU."
#         logger.error(msg); raise ValueError(msg)
#     if coherence_cu < IDENTITY_COHERENCE_THRESHOLD_CU:
#         msg = f"Prerequisite failed: Coherence ({coherence_cu:.1f} CU) < {IDENTITY_COHERENCE_THRESHOLD_CU} CU."
#         logger.error(msg); raise ValueError(msg)
#     if earth_resonance < IDENTITY_EARTH_RESONANCE_THRESHOLD:
#         msg = f"Prerequisite failed: Earth Resonance ({earth_resonance:.3f}) < {IDENTITY_EARTH_RESONANCE_THRESHOLD}."
#         logger.error(msg); raise ValueError(msg)

#     # 3. Essential Attributes for Calculation
#     if getattr(soul_spark, 'soul_color', None) is None:
#         msg = "Prerequisite failed: SoulSpark missing 'soul_color'."
#         logger.error(msg); raise AttributeError(msg)
#     if getattr(soul_spark, 'soul_frequency', 0.0) <= FLOAT_EPSILON:
#         msg = f"Prerequisite failed: SoulSpark missing valid 'soul_frequency' ({getattr(soul_spark, 'soul_frequency', 0.0)})."
#         logger.error(msg); raise ValueError(msg)
#     if getattr(soul_spark, 'frequency', 0.0) <= FLOAT_EPSILON:
#         msg = f"Prerequisite failed: SoulSpark missing valid base 'frequency'."
#         logger.error(msg); raise ValueError(msg)

#     # 4. Layer Structure Check
#     if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
#         msg = "Prerequisite failed: Soul must have aura layers established."
#         logger.error(msg); raise AttributeError(msg)
    
#     # 5. Check for existing crystallization
#     if getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False):
#         logger.warning(f"Soul {soul_spark.spark_id} already marked {FLAG_IDENTITY_CRYSTALLIZED}. Re-running.")

#     logger.debug("Identity prerequisites met.")
#     return True

# def _ensure_soul_properties(soul_spark: SoulSpark):
#     """ Ensure soul has necessary properties. Raises error if missing/invalid. """
#     logger.debug(f"Ensuring properties for identity process (Soul {soul_spark.spark_id})...")
#     required = ['stability', 'coherence', 'earth_resonance', 'soul_color', 'soul_frequency', 'frequency',
#                 'yin_yang_balance', 'aspects', 'consciousness_state', 'phi_resonance',
#                 'pattern_coherence', 'harmony', 'toroidal_flow_strength', 'layers']
#     if not all(hasattr(soul_spark, attr) for attr in required):
#         missing = [attr for attr in required if not hasattr(soul_spark, attr)]
#         raise AttributeError(f"SoulSpark missing essential attributes for Identity: {missing}")

#     if soul_spark.soul_color is None: raise ValueError("SoulColor cannot be None.")
#     if soul_spark.soul_frequency <= FLOAT_EPSILON: raise ValueError("SoulFrequency must be positive.")
#     if soul_spark.frequency <= FLOAT_EPSILON: raise ValueError("Base Frequency must be positive.")

#     # Initialize attributes set during this stage if missing
#     defaults = {
#         "name": None, "gematria_value": 0, "name_resonance": 0.0, "voice_frequency": 0.0,
#         "response_level": 0.0, "heartbeat_entrainment": 0.0, "color_frequency": 0.0,
#         "sephiroth_aspect": None, "elemental_affinity": None, "platonic_symbol": None,
#         "emotional_resonance": {'love': 0.0, 'joy': 0.0, 'peace': 0.0, 'harmony': 0.0, 'compassion': 0.0},
#         "crystallization_level": 0.0, "attribute_coherence": 0.0,
#         "identity_metrics": None, "sacred_geometry_imprint": None,
#         "conceptual_birth_datetime": None, "zodiac_sign": None, "astrological_traits": None,
#         "identity_light_signature": None, "identity_sound_signature": None, 
#         "crystalline_structure": None, "name_standing_waves": None
#     }
#     for attr, default in defaults.items():
#         if not hasattr(soul_spark, attr) or getattr(soul_spark, attr) is None:
#              if attr == 'emotional_resonance': setattr(soul_spark, attr, default.copy())
#              elif attr == 'astrological_traits': setattr(soul_spark, attr, {}) # Init as empty dict
#              else: setattr(soul_spark, attr, default)

#     if hasattr(soul_spark, '_validate_attributes'): soul_spark._validate_attributes()
#     logger.debug("Soul properties ensured for Identity Crystallization.")

# # --- Name Calculation Helpers with Light Physics ---
# def _calculate_gematria(name: str) -> int:
#     """Calculate gematria value using full character encoding for accuracy."""
#     if not isinstance(name, str): raise TypeError("Name must be a string.")
#     return sum(ord(char) - ord('a') + 1 for char in name.lower() if 'a' <= char <= 'z')

# def _calculate_name_resonance(name: str, gematria: int) -> float:
#     """Calculate name resonance (0-1) based on advanced harmonic principles."""
#     if not name: return 0.0
#     try:
#         vowels=sum(1 for c in name.lower() if c in 'aeiouy')
#         consonants=sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz')
#         total_letters=vowels+consonants
#         if total_letters == 0: return 0.0
        
#         # Vowel ratio compared to golden ratio
#         vowel_ratio=vowels/total_letters
#         phi_inv = 1.0 / GOLDEN_RATIO
#         vowel_factor = max(0.0, 1.0 - abs(vowel_ratio - phi_inv) * 2.5)
        
#         # Unique letters and distribution
#         unique_letters = len(set(c for c in name.lower() if 'a' <= c <= 'z'))
#         letter_factor = unique_letters / max(1, len(name))
        
#         # Gematria-based resonance factor
#         gematria_factor = 0.5
#         if gematria > 0:
#             for num in NAME_GEMATRIA_RESONANT_NUMBERS:
#                  if gematria % num == 0: gematria_factor = 0.9; break
        
#         # Frequency analysis - look for patterns
#         frequency_factor = 0.0
#         if len(name) >= 3:
#             # Check for repeating patterns
#             for pattern_len in range(1, min(5, len(name) // 2 + 1)):
#                 patterns = set()
#                 for i in range(0, len(name) - pattern_len + 1):
#                     pattern = name[i:i+pattern_len].lower()
#                     patterns.add(pattern)
#                 # More unique patterns = higher complexity = better resonance
#                 complexity = len(patterns) / (len(name) - pattern_len + 1)
#                 frequency_factor = max(frequency_factor, complexity * 0.1)
        
#         # Combine all factors with weights
#         total_weight = (NAME_RESONANCE_WEIGHT_VOWEL + NAME_RESONANCE_WEIGHT_LETTER + 
#                        NAME_RESONANCE_WEIGHT_GEMATRIA + 0.1)  # 0.1 for frequency factor
        
#         resonance_contrib = (NAME_RESONANCE_WEIGHT_VOWEL * vowel_factor + 
#                            NAME_RESONANCE_WEIGHT_LETTER * letter_factor + 
#                            NAME_RESONANCE_WEIGHT_GEMATRIA * gematria_factor +
#                            0.1 * frequency_factor)
        
#         resonance = NAME_RESONANCE_BASE + (1.0 - NAME_RESONANCE_BASE) * (resonance_contrib / total_weight)
#         return float(max(0.0, min(1.0, resonance)))
#     except Exception as e:
#         logger.error(f"Error calculating name resonance for '{name}': {e}")
#         raise RuntimeError("Name resonance calculation failed.") from e

# def _calculate_name_standing_waves(name: str, base_frequency: float) -> Dict[str, Any]:
#     """
#     Calculate standing wave patterns generated by the name.
#     Creates a light-sound signature that forms basis for identity.
    
#     Args:
#         name: Soul's name
#         base_frequency: Base frequency to anchor the standing waves
        
#     Returns:
#         Dictionary with standing wave properties
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If calculation fails
#     """
#     if not name: raise ValueError("Name cannot be empty")
#     if base_frequency <= 0: raise ValueError("Base frequency must be positive")
    
#     try:
#         # Create frequency array from name
#         frequencies = []
#         amplitudes = []
#         phases = []
        
#         # Map each character to a frequency offset
#         # Using a Phi-based mapping for harmonic relationships
#         for i, char in enumerate(name.lower()):
#             if 'a' <= char <= 'z':
#                 # Calculate offset based on character position in alphabet
#                 char_value = ord(char) - ord('a') + 1
                
#                 # Create frequency that relates to base via Phi-based ratios
#                 # This creates harmonically related frequencies
#                 power = (char_value % 12) / 12.0  # Normalize to 0-1 range
#                 ratio = 1.0 + (PHI - 1.0) * power
                
#                 # Calculate frequency, ensure it's within human hearing range
#                 freq = base_frequency * ratio
#                 freq = min(20000, max(20, freq))  # Limit to audible range
                
#                 # Calculate amplitude based on character position
#                 # Vowels have higher amplitude
#                 if char in 'aeiouy':
#                     amp = 0.7 - 0.3 * (i / len(name))
#                 else:
#                     amp = 0.5 - 0.3 * (i / len(name))
                
#                 # Phase based on character position for wave interference
#                 phase = (char_value / 26.0) * 2 * PI
                
#                 frequencies.append(freq)
#                 amplitudes.append(amp)
#                 phases.append(phase)
        
#         # Calculate interference pattern at specific points
#         num_points = 100
#         interference_pattern = np.zeros(num_points, dtype=np.float32)
#         x_values = np.linspace(0, 1, num_points)
        
#         for i, x in enumerate(x_values):
#             # Sum waves at this point
#             for freq, amp, phase in zip(frequencies, amplitudes, phases):
#                 # Create standing wave pattern
#                 wavelength = SPEED_OF_SOUND / freq
#                 k = 2 * PI / wavelength
#                 interference_pattern[i] += amp * np.sin(k * x + phase)
                
#         # Find nodes and antinodes
#         nodes = []
#         antinodes = []
        
#         # Finding approximate nodes (close to zero amplitude)
#         for i in range(1, len(interference_pattern) - 1):
#             if abs(interference_pattern[i]) < 0.1:
#                 if abs(interference_pattern[i-1]) > 0.1 or abs(interference_pattern[i+1]) > 0.1:
#                     nodes.append({"position": float(x_values[i]), 
#                                  "amplitude": float(interference_pattern[i])})
            
#             # Finding approximate antinodes (local maxima/minima)
#             if (interference_pattern[i] > interference_pattern[i-1] and 
#                 interference_pattern[i] > interference_pattern[i+1]):
#                 antinodes.append({"position": float(x_values[i]), 
#                                  "amplitude": float(interference_pattern[i])})
#             elif (interference_pattern[i] < interference_pattern[i-1] and 
#                   interference_pattern[i] < interference_pattern[i+1]):
#                 antinodes.append({"position": float(x_values[i]), 
#                                  "amplitude": float(interference_pattern[i])})
                
#         # Calculate resonance quality based on number of stable nodes/antinodes
#         resonance_quality = min(1.0, (len(nodes) + len(antinodes)) / 20.0)
        
#         # Create light signature - mapping frequencies to light spectrum
#         light_frequencies = []
#         for freq in frequencies:
#             # Scale audio to light frequency
#             light_freq = freq * 1e12  # Simple scaling for example
#             wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
            
#             # Map to visible spectrum color
#             if 380 <= wavelength_nm <= 750:
#                 if 380 <= wavelength_nm < 450: color = "violet"
#                 elif 450 <= wavelength_nm < 495: color = "blue"
#                 elif 495 <= wavelength_nm < 570: color = "green"
#                 elif 570 <= wavelength_nm < 590: color = "yellow"
#                 elif 590 <= wavelength_nm < 620: color = "orange"
#                 else: color = "red"
#             else:
#                 color = "beyond visible"
                
#             light_frequencies.append({
#                 "audio_frequency": float(freq),
#                 "light_frequency": float(light_freq),
#                 "wavelength_nm": float(wavelength_nm),
#                 "color": color
#             })
        
#         # Calculate overall coherence of the pattern
#         # More regular patterns have higher coherence
#         amplitude_variance = np.var(interference_pattern)
#         pattern_coherence = max(0.0, min(1.0, 1.0 - amplitude_variance))
        
#         # Create the final standing wave structure
#         standing_waves = {
#             "base_frequency": float(base_frequency),
#             "component_frequencies": frequencies,
#             "component_amplitudes": amplitudes,
#             "component_phases": phases,
#             "nodes": nodes,
#             "antinodes": antinodes,
#             "pattern_coherence": float(pattern_coherence),
#             "resonance_quality": float(resonance_quality),
#             "light_frequencies": light_frequencies,
#             "interference_pattern": interference_pattern.tolist()
#         }
        
#         logger.debug(f"Calculated name standing waves: {len(frequencies)} components, " +
#                     f"Nodes: {len(nodes)}, Antinodes: {len(antinodes)}, " +
#                     f"Coherence: {pattern_coherence:.4f}")
        
#         return standing_waves
        
#     except Exception as e:
#         logger.error(f"Error calculating name standing waves: {e}", exc_info=True)
#         raise RuntimeError("Name standing wave calculation failed") from e

# def _generate_name_frequency_signature(name: str, base_frequency: float, duration: float = 10.0) -> np.ndarray:
#     """
#     Generate audio signature from name's standing wave pattern.
#     Uses actual sound synthesis based on name frequencies.
    
#     Args:
#         name: Soul's name
#         base_frequency: Base frequency for the signature
#         duration: Length of audio in seconds
        
#     Returns:
#         NumPy array containing the audio samples
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If generation fails
#     """
#     if not name: raise ValueError("Name cannot be empty")
#     if base_frequency <= 0: raise ValueError("Base frequency must be positive")
#     if duration <= 0: raise ValueError("Duration must be positive")
    
#     if not SOUND_MODULES_AVAILABLE:
#         raise RuntimeError("Sound generation requires sound modules")
    
#     try:
#         # Calculate name standing wave properties
#         wave_properties = _calculate_name_standing_waves(name, base_frequency)
        
#         # Use sound generator to create harmonics
#         frequencies = wave_properties["component_frequencies"]
#         amplitudes = wave_properties["component_amplitudes"]
        
#         # Ensure lengths match
#         if len(frequencies) != len(amplitudes):
#             raise ValueError("Frequency and amplitude arrays must have same length")
        
#         # Normalize amplitudes to avoid clipping
#         max_amp = max(amplitudes) if amplitudes else 1.0
#         if max_amp > FLOAT_EPSILON:
#             normalized_amps = [a / max_amp * 0.8 for a in amplitudes]  # Scale to 0.8 max
#         else:
#             normalized_amps = [0.0] * len(amplitudes)
        
#         # Create harmonic structure from frequencies
#         harmonics = [f / base_frequency for f in frequencies]
        
#         logger.debug(f"Generating name audio signature: Name={name}, " +
#                     f"Base={base_frequency:.2f}Hz, Harmonics={len(harmonics)}")
        
#         # Generate sound using sound_generator
#         name_sound = sound_gen.generate_harmonic_tone(base_frequency, harmonics, 
#                                                    normalized_amps, duration, 0.5)
        
#         # Save the sound for future reference
#         sound_filename = f"name_signature_{name.lower()}.wav"
#         sound_path = sound_gen.save_sound(name_sound, sound_filename, 
#                                        f"Name Signature for '{name}'")
        
#         logger.info(f"Generated name frequency signature. Saved to: {sound_path}")
#         return name_sound
        
#     except Exception as e:
#         logger.error(f"Error generating name frequency signature: {e}", exc_info=True)
#         raise RuntimeError("Name frequency signature generation failed") from e

# # --- Color Processing with Light Physics ---
# def _generate_color_from_frequency(frequency: float) -> str:
#     """Generate a color hex code from a frequency using light physics."""
#     if frequency <= FLOAT_EPSILON:
#         raise ValueError("Frequency must be positive for color generation")
    
#     try:
#         # Map the frequency to visible light spectrum wavelengths (380-750nm)
#         # Using a logarithmic mapping to better distribute across spectrum
#         normalized_freq = np.log(frequency) / np.log(20000)  # Normalize using log scale
#         normalized_freq = max(0.0, min(1.0, normalized_freq))  # Clamp to 0-1
        
#         # Map to wavelength (nm) - higher frequencies = shorter wavelengths
#         wavelength = 750 - normalized_freq * (750 - 380)
        
#         # Convert wavelength to RGB using physics-based approach
#         r, g, b = 0, 0, 0
        
#         # Visible spectrum wavelength to RGB conversion (physics-based)
#         if 380 <= wavelength < 440:
#             # Violet -> Blue
#             r = -(wavelength - 440) / (440 - 380)
#             g = 0.0
#             b = 1.0
#         elif 440 <= wavelength < 490:
#             # Blue -> Cyan
#             r = 0.0
#             g = (wavelength - 440) / (490 - 440)
#             b = 1.0
#         elif 490 <= wavelength < 510:
#             # Cyan -> Green
#             r = 0.0
#             g = 1.0
#             b = -(wavelength - 510) / (510 - 490)
#         elif 510 <= wavelength < 580:
#             # Green -> Yellow
#             r = (wavelength - 510) / (580 - 510)
#             g = 1.0
#             b = 0.0
#         elif 580 <= wavelength < 645:
#             # Yellow -> Red
#             r = 1.0
#             g = -(wavelength - 645) / (645 - 580)
#             b = 0.0
#         elif 645 <= wavelength <= 750:
#             # Red
#             r = 1.0
#             g = 0.0
#             b = 0.0
        
#         # Adjust intensity with wavelength (drops at edges of visible spectrum)
#         if wavelength < 420:
#             intensity = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
#         elif wavelength > 700:
#             intensity = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
#         else:
#             intensity = 1.0
            
#         # Scale RGB values and convert to hex
#         r = int(255 * r * intensity)
#         g = int(255 * g * intensity)
#         b = int(255 * b * intensity)
        
#         hex_color = f"#{r:02X}{g:02X}{b:02X}"
#         return hex_color
        
#     except Exception as e:
#         logger.error(f"Error generating color from frequency {frequency}: {e}")
#         raise RuntimeError(f"Color generation failed: {e}") from e

# def _calculate_color_frequency(color_hex: str) -> float:
#     """Calculate a frequency in Hz associated with a color hex code or color name."""
#     if not isinstance(color_hex, str):
#         raise ValueError(f"Color must be a string, got {type(color_hex)}")
    
#     # Color name to hex mapping
#     color_name_map = {
#         'red': '#FF0000',
#         'green': '#00FF00', 
#         'blue': '#0000FF',
#         'yellow': '#FFFF00',
#         'orange': '#FFA500',
#         'purple': '#800080',
#         'violet': '#8A2BE2',
#         'indigo': '#4B0082',
#         'cyan': '#00FFFF',
#         'magenta': '#FF00FF',
#         'pink': '#FFC0CB',
#         'brown': '#A52A2A',
#         'black': '#000000',
#         'white': '#FFFFFF',
#         'gray': '#808080',
#         'grey': '#808080'
#     }
    
#     # Convert color name to hex if needed
#     original_color = color_hex
#     if color_hex.lower() in color_name_map:
#         color_hex = color_name_map[color_hex.lower()]
#         logger.debug(f"Converted color name '{original_color}' to hex '{color_hex}'")
    
#     # Validate hex format
#     if not re.match(r'^#[0-9A-F]{6}$', color_hex.upper()):
#         raise ValueError(f"Invalid color format: '{original_color}' -> '{color_hex}'. Expected hex format like #FF0000 or color name.")
    
#     try:
#         # Convert hex to RGB
#         r = int(color_hex[1:3], 16) / 255.0
#         g = int(color_hex[3:5], 16) / 255.0
#         b = int(color_hex[5:7], 16) / 255.0
        
#         # Calculate dominant color
#         max_val = max(r, g, b)
        
#         # Estimate wavelength based on dominant color
#         if max_val <= FLOAT_EPSILON:
#             wavelength = 550  # Default to middle (green)
#         elif max_val == r:
#             # Red dominant (620-750nm)
#             if g > b:
#                 # More yellow (570-620nm)
#                 wavelength = 620 - (g / max_val) * 50
#             else:
#                 wavelength = 650 + (b / max_val) * 100
#         elif max_val == g:
#             # Green dominant (495-570nm)
#             if r > b:
#                 # More yellow (570-590nm)
#                 wavelength = 570 - (r / max_val) * 75
#             else:
#                 # More cyan (490-495nm)
#                 wavelength = 495 + (b / max_val) * 5
#         else:  # max_val == b
#             # Blue dominant (450-495nm)
#             if g > r:
#                 # More cyan (495-520nm)
#                 wavelength = 495 - (g / max_val) * 45
#             else:
#                 # More violet (380-450nm)
#                 wavelength = 450 - (r / max_val) * 70
                
#         # Convert wavelength to frequency (f = c/λ)
#         frequency = SPEED_OF_LIGHT / (wavelength * 1e-9)
        
#         # Scale to audio range (typical 20Hz-20kHz)
#         # Using logarithmic scaling to distribute frequencies more naturally
#         visible_min = SPEED_OF_LIGHT / (750e-9)  # ~400 THz
#         visible_max = SPEED_OF_LIGHT / (380e-9)  # ~790 THz
        
#         log_min = np.log(visible_min)
#         log_max = np.log(visible_max)
#         log_freq = np.log(frequency)
        
#         # Normalize to 0-1 in log space
#         normalized = (log_freq - log_min) / (log_max - log_min)
        
#         # Map to audio range
#         audio_min = 20.0  # Hz
#         audio_max = 20000.0  # Hz
        
#         # Use log mapping for better distribution
#         audio_freq = audio_min * np.power(audio_max / audio_min, normalized)
        
#         return float(audio_freq)
        
#     except Exception as e:
#         logger.error(f"Error calculating frequency from color {original_color}: {e}")
#         raise RuntimeError(f"Color frequency calculation failed: {e}") from e


# def _find_closest_spectrum_color(hex_color: str, color_spectrum) -> str:
#     """Find the closest color in the spectrum to the given hex color."""
#     if not isinstance(hex_color, str):
#         raise ValueError(f"Hex color must be a string, got {type(hex_color)}")
    
#     try:
#         # Convert target hex to RGB
#         if not re.match(r'^#[0-9A-F]{6}$', hex_color.upper()):
#             raise ValueError(f"Invalid hex color format: {hex_color}")
        
#         r = int(hex_color[1:3], 16)
#         g = int(hex_color[3:5], 16)
#         b = int(hex_color[5:7], 16)
#         target_rgb = (r, g, b)
        
#         # Find closest color by Euclidean distance
#         min_distance = float('inf')
#         closest_color = None
        
#         for spectrum_color in color_spectrum.keys():
#             try:
#                 # Convert spectrum color to RGB
#                 sr = int(spectrum_color[1:3], 16)
#                 sg = int(spectrum_color[3:5], 16)
#                 sb = int(spectrum_color[5:7], 16)
#                 spectrum_rgb = (sr, sg, sb)
                
#                 # Calculate Euclidean distance
#                 distance = sum((c1 - c2) ** 2 for c1, c2 in zip(target_rgb, spectrum_rgb)) ** 0.5
                
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_color = spectrum_color
#             except ValueError:
#                 # Skip invalid color formats in spectrum
#                 continue
        
#         if not closest_color:
#             # Default to first color in spectrum if none found
#             closest_color = next(iter(color_spectrum.keys()), hex_color)
        
#         return closest_color
        
#     except Exception as e:
#         logger.error(f"Error finding closest spectrum color: {e}")
#         # Return original color instead of failing
#         return hex_color

# # --- Core Identity Crystallization Functions ---

# def assign_name(soul_spark: SoulSpark) -> None:
#     """ 
#     Assigns name via user input, calculates gematria/resonance(0-1), and 
#     establishes standing wave patterns and frequency signature.
#     """
#     log_soul_identity_summary(soul_spark)
#     logger.info("Identity Step: Assign Name with Light-Sound Signature...")
#     name_to_use = None
#     print("-" * 30+"\nIDENTITY CRYSTALLIZATION: SOUL NAMING"+"\nSoul ID: "+soul_spark.spark_id)
#     print(f"  Color: {getattr(soul_spark, 'soul_color', 'N/A')}, Freq: {getattr(soul_spark, 'soul_frequency', 0.0):.1f}Hz")
#     print(f"  S/C: {soul_spark.stability:.1f}SU / {soul_spark.coherence:.1f}CU")
#     print("-" * 30)
#     while not name_to_use:
#         try:
#             user_name = input(f"*** Please enter the name for this soul: ").strip()
#             if not user_name: print("Name cannot be empty.")
#             else: name_to_use = user_name
#         except EOFError: raise RuntimeError("Failed to get soul name from user input (EOF).")
#         except Exception as e: raise RuntimeError(f"Failed to get soul name: {e}")
#     print("-" * 30)

#     try:
#         # Calculate traditional gematria value
#         gematria = _calculate_gematria(name_to_use)
        
#         # Calculate name resonance 
#         name_resonance = _calculate_name_resonance(name_to_use, gematria)
        
# # Calculate standing wave patterns for the name
#         standing_waves = _calculate_name_standing_waves(name_to_use, soul_spark.frequency)
        
#         # Generate actual sound signature - only if sound modules available
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 sound_signature = _generate_name_frequency_signature(
#                     name_to_use, soul_spark.frequency)
#                 logger.debug(f"  Generated name sound signature: {len(sound_signature)} samples")
#             except Exception as sound_err:
#                 logger.error(f"Failed to generate name sound: {sound_err}")
#                 # Don't fail whole process, just log the error
        
#         # Create light signature from standing waves
#         light_signature = {
#             "primary_color": _generate_color_from_frequency(soul_spark.frequency),
#             "harmonic_colors": [f["color"] for f in standing_waves["light_frequencies"]],
#             "resonance_quality": standing_waves["resonance_quality"],
#             "coherence": standing_waves["pattern_coherence"]
#         }
        
#         logger.debug(f"  Name assigned: '{name_to_use}', Gematria: {gematria}, " +
#                    f"Resonance Factor: {name_resonance:.4f}, " +
#                    f"Standing Waves: {len(standing_waves['nodes'])} nodes")
        
#         # Update soul with name information
#         setattr(soul_spark, 'name', name_to_use)
#         setattr(soul_spark, 'gematria_value', gematria)
#         setattr(soul_spark, 'name_resonance', name_resonance)
#         setattr(soul_spark, 'name_standing_waves', standing_waves)
#         setattr(soul_spark, 'identity_light_signature', light_signature)
#         timestamp = datetime.now().isoformat()
#         setattr(soul_spark, 'last_modified', timestamp)
        
#         # Add memory echo for future reference
#         if hasattr(soul_spark, 'add_memory_echo'): 
#             soul_spark.add_memory_echo(f"Name assigned: {name_to_use} (G:{gematria}, " +
#                                      f"NR:{name_resonance:.3f}, StandingWaves:{len(standing_waves['nodes'])})")
        
#         # Create resonant integration with aura layers
#         _integrate_name_with_aura_layers(soul_spark, standing_waves)
        
#         logger.info(f"Name assignment complete: {name_to_use} (ResFactor: {name_resonance:.4f}, " +
#                    f"LightSig: {len(light_signature['harmonic_colors'])} colors)")
        
#     except Exception as e:
#         logger.error(f"Error processing assigned name: {e}", exc_info=True)
#         raise RuntimeError("Name processing failed.") from e
    
# # --- Create Aura Layer ---
# def _create_identity_aura_layer(soul_spark: SoulSpark) -> Dict[str, Any]:
#     """
#     Creates a specialized identity aura layer that encodes the soul's 
#     light physics frequency and identity signature.
    
#     Args:
#         soul_spark: The soul to create identity layer for
        
#     Returns:
#         Dictionary with created layer data
#     """
#     logger.info("Identity Step: Creating Identity Aura Layer...")
    
#     # Get soul properties
#     soul_frequency = getattr(soul_spark, 'soul_frequency', soul_spark.frequency)
#     soul_color = getattr(soul_spark, 'soul_color', '#FFFFFF')
#     name = getattr(soul_spark, 'name', 'Unknown')
    
#     # Calculate light physics frequency (MHz range)
#     light_physics_freq = soul_frequency * 1e6
    
#     # Convert soul frequency to wavelength
#     wavelength_m = SPEED_OF_LIGHT / light_physics_freq if light_physics_freq > 0 else 0
    
#     # Convert soul color to spectral components
#     r = int(soul_color[1:3], 16) / 255.0
#     g = int(soul_color[3:5], 16) / 255.0
#     b = int(soul_color[5:7], 16) / 255.0
    
#     # Calculate dominant wavelength from RGB
#     # Simplified spectral calculation
#     if max(r, g, b) <= FLOAT_EPSILON:
#         dominant_wavelength_nm = 550  # Default green
#     elif r >= g and r >= b:
#         # Red dominant (620-750nm)
#         dominant_wavelength_nm = 620 + (r * 130)
#     elif g >= r and g >= b:
#         # Green dominant (495-570nm)
#         dominant_wavelength_nm = 495 + (g * 75)
#     else:
#         # Blue dominant (450-495nm)
#         dominant_wavelength_nm = 450 + (b * 45)
    
#     # Create identity layer
#     identity_layer = {
#         "type": "identity",
#         "name": f"Identity Layer - {name}",
#         "light_physics_frequency": float(light_physics_freq),
#         "wavelength_m": float(wavelength_m),
#         "soul_color": soul_color,
#         "dominant_wavelength_nm": float(dominant_wavelength_nm),
#         "spectral_components": {
#             "red": float(r),
#             "green": float(g),
#             "blue": float(b)
#         },
#         "resonant_frequencies": [
#             float(soul_frequency),
#             float(soul_frequency * PHI),
#             float(soul_frequency * 2.0)
#         ],
#         "creation_timestamp": datetime.now().isoformat()
#     }
    
#     # Add identity layer to soul's aura layers
#     if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
#         setattr(soul_spark, 'layers', [])
    
#     # Append to existing layers
#     soul_spark.layers.append(identity_layer)
    
#     logger.info(f"Created Identity Aura Layer with Light Physics Frequency: {light_physics_freq:.2f}MHz, "
#                f"Wavelength: {wavelength_m:.2e}m, Dominant Wavelength: {dominant_wavelength_nm:.2f}nm")
    
#     return identity_layer

# def _integrate_name_with_aura_layers(soul_spark: SoulSpark, 
#                                    standing_waves: Dict[str, Any]) -> None:
#     """
#     Integrate name-based standing waves with aura layers to create resonance.
#     This establishes identity patterns without modifying core frequency.
    
#     Args:
#         soul_spark: The soul to integrate with
#         standing_waves: Standing wave patterns from name
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be SoulSpark instance")
#     if not isinstance(standing_waves, dict):
#         raise ValueError("standing_waves must be a dictionary")
#     if not hasattr(soul_spark, 'layers') or not soul_spark.layers:
#         raise ValueError("Soul must have aura layers for integration")
    
#     name = getattr(soul_spark, 'name', None)
#     if not name:
#         raise ValueError("Soul must have a name assigned")
    
#     try:
#         logger.debug(f"Integrating name '{name}' standing waves with aura layers...")
        
#         # Get key frequency components from standing waves
#         frequencies = standing_waves.get("component_frequencies", [])
#         nodes = standing_waves.get("nodes", [])
#         antinodes = standing_waves.get("antinodes", [])
        
#         if not frequencies:
#             raise ValueError("Standing waves missing frequency components")
        
#         # Find resonant layers for each frequency component
#         resonant_connections = []
        
#         for freq_idx, freq in enumerate(frequencies):
#             # Look for most resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create name resonance field in layer
#                 if 'name_resonance' not in layer:
#                     layer['name_resonance'] = {}
                
#                 # Position in layer is based on node/antinode positions
#                 # Find closest node/antinode
#                 closest_pos = 0.5  # Default to middle
#                 closest_dist = 1.0
                
#                 # Check nodes
#                 for node in nodes:
#                     position = node.get("position", 0)
#                     dist = abs(position - (freq_idx / max(1, len(frequencies))))
#                     if dist < closest_dist:
#                         closest_dist = dist
#                         closest_pos = position
                
#                 # Check antinodes
#                 for antinode in antinodes:
#                     position = antinode.get("position", 0)
#                     dist = abs(position - (freq_idx / max(1, len(frequencies))))
#                     if dist < closest_dist:
#                         closest_dist = dist
#                         closest_pos = position
                
#                 # Add resonance component to layer
#                 layer['name_resonance'][f"component_{freq_idx}"] = {
#                     "frequency": float(freq),
#                     "resonance": float(best_resonance),
#                     "position": float(closest_pos),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(freq))
                
#                 # Track connection for metrics
#                 resonant_connections.append({
#                     "layer_idx": best_layer_idx,
#                     "frequency": float(freq),
#                     "resonance": float(best_resonance),
#                     "position": float(closest_pos)
#                 })
        
#         # Log integration statistics
#         logger.debug(f"Name-layer integration complete: {len(resonant_connections)} connections " +
#                     f"across {len(set(c['layer_idx'] for c in resonant_connections))} layers")
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "name": name,
#                 "connections_count": len(resonant_connections),
#                 "layers_affected": len(set(c['layer_idx'] for c in resonant_connections)),
#                 "frequencies_integrated": len(frequencies),
#                 "avg_resonance": sum(c['resonance'] for c in resonant_connections) / 
#                                max(1, len(resonant_connections)),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_name_layer_integration', metrics_data)
        
#     except Exception as e:
#         logger.error(f"Error integrating name with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Name-layer integration failed") from e

# def assign_voice_frequency(soul_spark: SoulSpark) -> None:
#     """
#     Assigns voice frequency (Hz) based on name/attributes/aura layers.
#     Implements proper acoustic physics and creates standing wave patterns.
#     """
#     logger.info("Identity Step: Assign Voice Frequency with Acoustic Physics...")
#     name = getattr(soul_spark, 'name')
#     gematria = getattr(soul_spark, 'gematria_value')
#     name_resonance = getattr(soul_spark, 'name_resonance')
#     yin_yang = getattr(soul_spark, 'yin_yang_balance')
#     standing_waves = getattr(soul_spark, 'name_standing_waves')
    
#     if not name:
#         raise ValueError("Cannot assign voice frequency without a name.")
#     if not standing_waves:
#         raise ValueError("Name standing waves required for voice frequency calculation.")
    
#     try:
#         # Base voice frequency calculation
#         length_factor = len(name) / 10.0
#         vowels = sum(1 for c in name.lower() if c in 'aeiouy')
#         total_letters = len(name)
#         vowel_ratio = vowels / max(1, total_letters)
#         gematria_factor = (gematria % 100) / 100.0
#         resonance_factor = name_resonance
        
#         # Calculate base frequency through traditional factors
#         voice_frequency = (VOICE_FREQ_BASE + 
#                          VOICE_FREQ_ADJ_LENGTH_FACTOR * (length_factor - 0.5) + 
#                          VOICE_FREQ_ADJ_VOWEL_FACTOR * (vowel_ratio - 0.5) + 
#                          VOICE_FREQ_ADJ_GEMATRIA_FACTOR * (gematria_factor - 0.5) + 
#                          VOICE_FREQ_ADJ_RESONANCE_FACTOR * (resonance_factor - 0.5) + 
#                          VOICE_FREQ_ADJ_YINYANG_FACTOR * (yin_yang - 0.5))
        
#         logger.debug(f"  Voice Freq Base Calc -> Raw={voice_frequency:.2f}Hz")
        
#         # Enhance with standing wave resonance
#         # Find the most coherent antinode frequency
#         coherent_freq = None
#         max_coherence = 0.0
        
#         if standing_waves and "antinodes" in standing_waves:
#             antinodes = standing_waves["antinodes"]
#             component_freqs = standing_waves.get("component_frequencies", [])
            
#             # Find most coherent antinode position
#             for antinode in antinodes:
#                 position = antinode.get("position", 0)
#                 amplitude = abs(antinode.get("amplitude", 0))
                
#                 # Higher amplitude = more coherent
#                 if amplitude > max_coherence:
#                     max_coherence = amplitude
                    
#                     # Find closest frequency component
#                     if component_freqs:
#                         # Normalize position to index
#                         idx = int(position * len(component_freqs))
#                         idx = max(0, min(len(component_freqs) - 1, idx))
#                         coherent_freq = component_freqs[idx]
        
#         # Incorporate coherent frequency if found
#         if coherent_freq and max_coherence > 0.3:
#             # Blend with base calculation - weighted by coherence
#             voice_frequency = (voice_frequency * (1.0 - max_coherence * 0.7) + 
#                              coherent_freq * max_coherence * 0.7)
            
#             logger.debug(f"  Enhanced with standing wave: {coherent_freq:.2f}Hz " +
#                        f"(Coherence: {max_coherence:.3f}) -> New={voice_frequency:.2f}Hz")
        
#         # Snap to nearest Solfeggio frequency if close
#         solfeggio_values = list(SOLFEGGIO_FREQUENCIES.values())
#         if solfeggio_values:
#             closest_solfeggio = min(solfeggio_values, key=lambda x: abs(x - voice_frequency))
            
#             if abs(closest_solfeggio - voice_frequency) < VOICE_FREQ_SOLFEGGIO_SNAP_HZ:
#                 voice_frequency = closest_solfeggio
#                 logger.debug(f"  Voice Freq Snapped to Solfeggio: {voice_frequency:.2f}Hz")
        
#         # Ensure frequency is within valid range
#         voice_frequency = min(VOICE_FREQ_MAX_HZ, max(VOICE_FREQ_MIN_HZ, voice_frequency))
        
#         # Generate voice sound pattern if sound modules available
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 # Create voice tone with harmonics
#                 harmonics = [1.0, 2.0, 3.0, 1.5]  # Include perfect fifth (1.5)
#                 amplitudes = [0.7, 0.3, 0.15, 0.25]  # Typical vocal harmonic structure
                
#                 voice_sound = sound_gen.generate_harmonic_tone(
#                     voice_frequency, harmonics, amplitudes, 3.0, 0.3)
                
#                 # Save voice sound
#                 voice_sound_path = sound_gen.save_sound(
#                     voice_sound, f"voice_{soul_spark.spark_id}.wav", f"Voice for {name}")
                
#                 logger.debug(f"  Generated voice sound: {voice_sound_path}")
                
#                 # Create a sound signature for the voice
#                 setattr(soul_spark, 'identity_sound_signature', {
#                     "type": "voice",
#                     "fundamental_frequency": float(voice_frequency),
#                     "harmonics": harmonics,
#                     "sound_path": voice_sound_path
#                 })
                
#             except Exception as sound_err:
#                 logger.error(f"Error generating voice sound: {sound_err}")
#                 # Continue without failing the process
        
#         # Calculate the acoustic wavelength for this frequency
#         wavelength_meters = SPEED_OF_SOUND / voice_frequency
        
#         # Update soul with voice frequency data
#         setattr(soul_spark, 'voice_frequency', float(voice_frequency))
#         setattr(soul_spark, 'voice_wavelength', float(wavelength_meters))
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Integrate voice frequency with aura layers
#         _integrate_voice_with_aura_layers(soul_spark, voice_frequency, harmonics=[1.0, 2.0, 3.0, 1.5])
        
#         logger.info(f"Voice frequency assigned: {voice_frequency:.2f}Hz, " +
#                    f"Wavelength: {wavelength_meters:.2f}m")
        
#     except Exception as e:
#         logger.error(f"Error assigning voice frequency: {e}", exc_info=True)
#         raise RuntimeError("Voice frequency assignment failed.") from e

# def _integrate_voice_with_aura_layers(soul_spark: SoulSpark, voice_freq: float, 
#                                     harmonics: List[float] = None) -> None:
#     """
#     Integrate voice frequency with aura layers to establish resonance.
    
#     Args:
#         soul_spark: Soul to integrate with
#         voice_freq: Voice fundamental frequency
#         harmonics: List of harmonic ratios to integrate
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if voice_freq <= 0:
#         raise ValueError("voice_freq must be positive")
#     if harmonics is None:
#         harmonics = [1.0, 2.0, 3.0]  # Default harmonics
    
#     try:
#         logger.debug(f"Integrating voice frequency {voice_freq:.2f}Hz with aura layers...")
        
#         # Generate the harmonic frequencies
#         harmonic_freqs = [voice_freq * h for h in harmonics]
        
#         # Track integration statistics
#         connected_layers = set()
#         total_resonance = 0.0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create voice resonance field in layer
#                 if 'voice_resonance' not in layer:
#                     layer['voice_resonance'] = {}
                
#                 # Add harmonic to layer
#                 layer['voice_resonance'][f"harmonic_{harm_idx}"] = {
#                     "frequency": float(freq),
#                     "harmonic_ratio": float(harmonics[harm_idx]),
#                     "resonance": float(best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(freq))
                
#                 # Track statistics
#                 connected_layers.add(best_layer_idx)
#                 total_resonance += best_resonance
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, len(connected_layers))
#         logger.debug(f"Voice frequency integrated with {len(connected_layers)} layers, " +
#                     f"Avg resonance: {avg_resonance:.4f}")
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "voice_frequency": float(voice_freq),
#                 "harmonics_count": len(harmonics),
#                 "layers_connected": len(connected_layers),
#                 "average_resonance": float(avg_resonance),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_voice_integration', metrics_data)
        
#     except Exception as e:
#         logger.error(f"Error integrating voice with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Voice-layer integration failed") from e

# def process_soul_color(soul_spark: SoulSpark) -> None:
#     """
#     Process the soul's color using light physics and electromagnetic theory.
#     Creates proper light-frequency relationships and integrates with aura layers.
    
#     This function creates the soul's spectral signature, mapping its frequency
#     to visible light and electromagnetic ranges, establishing quantum coherence
#     between frequency bands.
#     """
#     logger.info("Identity Step: Processing Soul Color with Light Physics...")
    
#     # Ensure soul color is set
#     if not hasattr(soul_spark, 'soul_color') or soul_spark.soul_color is None:
#         # Generate color from soul frequency if not set
#         try:
#             soul_color = _generate_color_from_frequency(soul_spark.frequency)
#             setattr(soul_spark, 'soul_color', soul_color)
#             logger.debug(f"  Generated soul_color {soul_color} based on frequency {soul_spark.frequency:.1f}Hz")
#         except Exception as color_err:
#             logger.error(f"Failed to generate soul color: {color_err}")
#             raise RuntimeError("Soul color generation failed") from color_err
    
#     soul_color = soul_spark.soul_color
    
#     # CRITICAL FIX: Validate hex format before using soul_color[3:5] to prevent line 1301 error
#     if not soul_color or not isinstance(soul_color, str) or len(soul_color) < 7:
#         logger.warning(f"Soul color '{soul_color}' is invalid, regenerating")
#         try:
#             soul_color = _generate_color_from_frequency(soul_spark.frequency)
#             soul_spark.soul_color = soul_color
#             logger.debug(f"  Regenerated color: {soul_color}")
#         except Exception as color_err:
#             logger.error(f"Failed to regenerate soul color: {color_err}")
#             raise RuntimeError("Soul color regeneration failed") from color_err
    
#     # Validate color format
#     if not re.match(r'^#[0-9A-F]{6}$', soul_color.upper()):
#         logger.warning(f"Soul color '{soul_color}' not a valid hex color, regenerating")
#         try:
#             # Regenerate using frequency
#             soul_color = _generate_color_from_frequency(soul_spark.frequency)
#             soul_spark.soul_color = soul_color
#             logger.debug(f"  Regenerated color: {soul_color}")
#         except Exception as color_err:
#             logger.error(f"Failed to regenerate soul color: {color_err}")
#             raise RuntimeError("Soul color regeneration failed") from color_err
    
#     try:
#         # Map to predefined spectrum if needed
#         if COLOR_SPECTRUM:
#             # Find closest color in spectrum
#             closest_color = _find_closest_spectrum_color(soul_color, COLOR_SPECTRUM)
            
#             if closest_color:
#                 logger.debug(f"  Mapped to closest spectrum color: {closest_color}")
#                 soul_spark.soul_color = closest_color
#                 soul_color = closest_color
        
#         # Calculate color frequency in sound spectrum
#         color_frequency = _calculate_color_frequency(soul_color)
#         setattr(soul_spark, 'color_frequency', float(color_frequency))
        
#         # Split into RGB components for more detailed analysis - SAFE now after validation
#         r = int(soul_color[1:3], 16) / 255.0
#         g = int(soul_color[3:5], 16) / 255.0
#         b = int(soul_color[5:7], 16) / 255.0
        
#         # Calculate color properties with light physics
#         # Hue (dominant wavelength)
#         max_component = max(r, g, b)
#         if max_component <= FLOAT_EPSILON:
#             hue = 0  # Default to red if black
#         elif max_component == r:
#             hue = (g - b) / (max_component - min(r, g, b) + FLOAT_EPSILON) % 6
#         elif max_component == g:
#             hue = (b - r) / (max_component - min(r, g, b) + FLOAT_EPSILON) + 2
#         else:  # max_component == b
#             hue = (r - g) / (max_component - min(r, g, b) + FLOAT_EPSILON) + 4
        
#         hue = (hue * 60) % 360  # Convert to degrees
        
#         # Saturation (purity of color)
#         if max_component <= FLOAT_EPSILON:
#             saturation = 0
#         else:
#             saturation = (max_component - min(r, g, b)) / (max_component + FLOAT_EPSILON)
        
#         # Value/Brightness
#         value = max_component
        
#         # Create color properties structure
#         color_properties = {
#             "hex": soul_color,
#             "rgb": [float(r), float(g), float(b)],
#             "hue": float(hue),
#             "saturation": float(saturation),
#             "brightness": float(value),
#             "frequency_hz": float(color_frequency),
#             "wavelength_nm": float(SPEED_OF_LIGHT / (color_frequency * 1e12)) if color_frequency > 0 else 0
#         }
        
#         # Calculate photon energy according to E=hf
#         # Planck's constant h = 6.626e-34 J⋅s
#         # Convert to eV for more intuitive scale
#         planck_constant = 6.626e-34  # J⋅s
#         electron_volt = 1.602e-19     # J/eV
        
#         photon_energy_joules = planck_constant * (color_frequency * 1e12)  # Convert to THz
#         photon_energy_ev = photon_energy_joules / electron_volt
        
#         color_properties["photon_energy_ev"] = float(photon_energy_ev)
#         color_properties["quantum_field_strength"] = float(saturation * value * 0.8 + 0.2)
        
#         # Generate sound based on color frequency if sound modules available
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 color_sound = sound_gen.generate_tone(
#                     color_frequency, 2.0, amplitude=0.6, fade_in_out=0.3)
                
#                 color_sound_path = sound_gen.save_sound(
#                     color_sound, f"color_{soul_color[1:]}.wav", f"Color Tone for {soul_color}")
                
#                 color_properties["sound_path"] = color_sound_path
#                 logger.debug(f"  Generated color sound: {color_sound_path}")
                
#             except Exception as sound_err:
#                 logger.warning(f"Failed to generate color sound: {sound_err}")
#                 # Continue without failing process
        
#         # Integrate color with aura layers
#         _integrate_color_with_aura_layers(soul_spark, color_properties)
        
#         setattr(soul_spark, 'color_properties', color_properties)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         logger.info(f"Soul color processed: {soul_color}, " +
#                    f"Frequency: {color_frequency:.2f}Hz, " +
#                    f"Wavelength: {color_properties['wavelength_nm']:.2f}nm, " +
#                    f"Photon Energy: {photon_energy_ev:.4f}eV")
        
#     except Exception as e:
#         logger.error(f"Error processing soul color: {e}", exc_info=True)
#         raise RuntimeError("Soul color processing failed") from e

# def _integrate_color_with_aura_layers(soul_spark: SoulSpark, color_properties: Dict) -> None:
#     """
#     Integrate color frequency with aura layers to establish resonance.
    
#     Args:
#         soul_spark: Soul to integrate with
#         color_properties: Color properties dictionary
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not isinstance(color_properties, dict):
#         raise ValueError("color_properties must be a dictionary")
    
#     try:
#         color_freq = color_properties.get("frequency_hz", 0)
#         if color_freq <= 0:
#             raise ValueError("Color frequency must be positive")
        
#         logger.debug(f"Integrating color frequency {color_freq:.2f}Hz with aura layers...")
        
#         # Track integration statistics
#         connected_layers = set()
#         total_resonance = 0.0
        
#         # Create harmonics for color frequency
#         # Using Phi-based harmonics for more natural resonance
#         harmonic_ratios = [1.0, PHI, 2.0, PHI * 2, 3.0]
#         harmonic_freqs = [color_freq * ratio for ratio in harmonic_ratios]
        
#         # Find resonant layers for each harmonic
#         for harm_idx, freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create color resonance field in layer
#                 if 'color_resonance' not in layer:
#                     layer['color_resonance'] = {}
                
#                 # Add harmonic to layer
#                 layer['color_resonance'][f"harmonic_{harm_idx}"] = {
#                     "frequency": float(freq),
#                     "harmonic_ratio": float(harmonic_ratios[harm_idx]),
#                     "resonance": float(best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # Store layer color influence
#                 if 'color_influence' not in layer:
#                     layer['color_influence'] = color_properties.get('hex', '#FFFFFF')
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(freq))
                
#                 # Track statistics
#                 connected_layers.add(best_layer_idx)
#                 total_resonance += best_resonance
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, len(connected_layers))
#         logger.debug(f"Color frequency integrated with {len(connected_layers)} layers, " +
#                     f"Avg resonance: {avg_resonance:.4f}")
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "color": color_properties.get('hex', 'unknown'),
#                 "color_frequency": float(color_freq),
#                 "harmonics_count": len(harmonic_ratios),
#                 "layers_connected": len(connected_layers),
#                 "average_resonance": float(avg_resonance),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_color_integration', metrics_data)
        
#     except Exception as e:
#         logger.error(f"Error integrating color with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Color-layer integration failed") from e

# def apply_heartbeat_entrainment(soul_spark: SoulSpark, bpm: float = 72.0, duration: float = 120.0) -> None:
#     """
#     Applies heartbeat entrainment using proper acoustic physics, creates
#     resonant standing waves that enhance harmony factor throughout aura layers.
#     """
#     logger.info("Identity Step: Apply Heartbeat Entrainment with Acoustic Physics...")
#     if bpm <= 0 or duration < 0:
#         raise ValueError("BPM must be positive, duration non-negative.")
    
#     try:
#         # Get current harmony and resonance state
#         current_harmony = getattr(soul_spark, 'harmony', 0.0)
#         voice_frequency = getattr(soul_spark, 'voice_frequency', 0.0)
        
#         # Calculate beat frequency in Hz (not BPM)
#         beat_freq = bpm / 60.0
        
#         # Calculate resonance between beat and voice frequency
#         if beat_freq <= FLOAT_EPSILON or voice_frequency <= FLOAT_EPSILON:
#             beat_resonance = 0.0
#         else:
#             beat_resonance = calculate_resonance(beat_freq, voice_frequency)
        
#         # Calculate entrainment factor based on duration
#         duration_factor = min(1.0, duration / HEARTBEAT_ENTRAINMENT_DURATION_CAP)
        
#         # Calculate harmony increase based on resonance and duration
#         base_harmony_increase = beat_resonance * duration_factor * HEARTBEAT_ENTRAINMENT_INC_FACTOR
        
#         # Generate actual heartbeat sound if available
#         heartbeat_sound = None
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 # Generate heartbeat waveform - more complex than simple sine
#                 # Create a more realistic heartbeat waveform (using a shaped envelope)
#                 num_samples = int(duration * soul_spark.sample_rate)
#                 time_array = np.linspace(0, duration, num_samples, False)
                
#                 # Empty sound array
#                 heartbeat_sound = np.zeros(num_samples, dtype=np.float32)
                
#                 # Generate individual beats
#                 beat_period = 60.0 / bpm
#                 num_beats = int(duration / beat_period)
                
#                 for i in range(num_beats):
#                     beat_time = i * beat_period
                    
#                     # First part of beat (lub) - higher frequency
#                     lub_duration = 0.15  # seconds
#                     lub_start = int((beat_time) * soul_spark.sample_rate)
#                     lub_end = min(int((beat_time + lub_duration) * soul_spark.sample_rate), num_samples)
                    
#                     if lub_start < num_samples:
#                         lub_envelope = np.zeros(lub_end - lub_start)
#                         t = np.linspace(0, lub_duration, lub_end - lub_start)
#                         # Shaped envelope: fast attack, slower decay
#                         attack = 0.1 * lub_duration
#                         attack_samples = int(attack * len(t))
#                         if attack_samples > 0:
#                             lub_envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
#                         decay_samples = len(t) - attack_samples
#                         if decay_samples > 0:
#                             lub_envelope[attack_samples:] = np.exp(-3 * np.linspace(0, 1, decay_samples))
                        
#                         # Main heartbeat frequency component (higher pitch for "lub")
#                         lub_freq = 60.0 
#                         lub_wave = 0.8 * np.sin(2 * np.pi * lub_freq * t)
                        
#                         # Add harmonics for richness
#                         lub_wave += 0.4 * np.sin(2 * np.pi * lub_freq * 2 * t)
#                         lub_wave += 0.2 * np.sin(2 * np.pi * lub_freq * 3 * t)
                        
#                         # Apply envelope
#                         lub_wave *= lub_envelope
                        
#                         # Add to main sound
#                         heartbeat_sound[lub_start:lub_end] += lub_wave
                    
#                     # Second part of beat (dub) - lower frequency, after a delay
#                     dub_delay = 0.2  # seconds after lub starts
#                     dub_duration = 0.25  # seconds
#                     dub_start = int((beat_time + dub_delay) * soul_spark.sample_rate)
#                     dub_end = min(int((beat_time + dub_delay + dub_duration) * soul_spark.sample_rate), num_samples)
                    
#                     if dub_start < num_samples:
#                         dub_envelope = np.zeros(dub_end - dub_start)
#                         t = np.linspace(0, dub_duration, dub_end - dub_start)
#                         # Shaped envelope: moderate attack, slower decay
#                         attack = 0.15 * dub_duration
#                         attack_samples = int(attack * len(t))
#                         if attack_samples > 0:
#                             dub_envelope[:attack_samples] = np.linspace(0, 0.7, attack_samples)
#                         decay_samples = len(t) - attack_samples
#                         if decay_samples > 0:
#                             dub_envelope[attack_samples:] = 0.7 * np.exp(-2 * np.linspace(0, 1, decay_samples))
                        
#                         # Main heartbeat frequency component (lower pitch for "dub")
#                         dub_freq = 40.0
#                         dub_wave = 0.6 * np.sin(2 * np.pi * dub_freq * t)
                        
#                         # Add harmonics for richness
#                         dub_wave += 0.3 * np.sin(2 * np.pi * dub_freq * 2 * t)
#                         dub_wave += 0.15 * np.sin(2 * np.pi * dub_freq * 3 * t)
                        
#                         # Apply envelope
#                         dub_wave *= dub_envelope
                        
#                         # Add to main sound
#                         heartbeat_sound[dub_start:dub_end] += dub_wave
                
#                 # Normalize to avoid clipping
#                 max_val = np.max(np.abs(heartbeat_sound))
#                 if max_val > FLOAT_EPSILON:
#                     heartbeat_sound = heartbeat_sound / max_val * 0.8
                
#                 # Save the heartbeat sound
#                 heartbeat_path = sound_gen.save_sound(
#                     heartbeat_sound, f"heartbeat_{int(bpm)}bpm.wav", f"Heartbeat at {bpm} BPM")
                
#                 logger.debug(f"  Generated heartbeat sound at {bpm} BPM: {heartbeat_path}")
                
#                 # Physical entrainment boost with real sound
#                 # The actual sound generation increases the entrainment effect
#                 harmony_increase = base_harmony_increase * 1.2  # 20% boost from actual sound
                
#             except Exception as sound_err:
#                 logger.warning(f"Failed to generate heartbeat sound: {sound_err}")
#                 # Use default calculation without sound boost
#                 harmony_increase = base_harmony_increase
#         else:
#             # No sound modules available
#             harmony_increase = base_harmony_increase
        
#         # Calculate new harmony with entrainment effect
#         new_harmony = min(1.0, current_harmony + harmony_increase)
        
#         logger.debug(f"  Heartbeat Entrainment: BeatFreq={beat_freq:.2f}Hz, " +
#                    f"BeatRes={beat_resonance:.3f}, DurFactor={duration_factor:.2f} -> " +
#                    f"HarmonyIncrease={harmony_increase:.4f}")
        
#         # Create standing wave patterns in aura
#         standing_wave_resonance = _create_heartbeat_standing_waves(
#             soul_spark, beat_freq, beat_resonance)
        
#         # The standing wave resonance enhances the harmony effect
#         standing_wave_boost = standing_wave_resonance * 0.1  # Up to 10% additional boost
#         final_harmony = min(1.0, new_harmony + standing_wave_boost)
        
#         # Update soul with entrainment data
#         setattr(soul_spark, 'harmony', float(final_harmony))
#         setattr(soul_spark, 'heartbeat_entrainment', beat_resonance * duration_factor)
#         setattr(soul_spark, 'heartbeat_frequency', float(beat_freq))
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "bpm": float(bpm),
#                 "beat_frequency": float(beat_freq),
#                 "duration": float(duration),
#                 "beat_resonance": float(beat_resonance),
#                 "base_harmony_increase": float(base_harmony_increase),
#                 "standing_wave_resonance": float(standing_wave_resonance),
#                 "standing_wave_boost": float(standing_wave_boost),
#                 "final_harmony_increase": float(final_harmony - current_harmony),
#                 "physical_sound_generated": heartbeat_sound is not None,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_heartbeat_entrainment', metrics_data)
        
#         logger.info(f"Heartbeat entrainment applied. " +
#                    f"Base Harmony Increase: {harmony_increase:.4f}, " +
#                    f"Standing Wave Boost: {standing_wave_boost:.4f}, " +
#                    f"Final Harmony: {final_harmony:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error applying heartbeat entrainment: {e}", exc_info=True)
#         raise RuntimeError("Heartbeat entrainment failed.") from e

# def _create_heartbeat_standing_waves(soul_spark: SoulSpark, beat_freq: float, 
#                                    resonance: float) -> float:
#     """
#     Create standing wave patterns in aura layers based on heartbeat frequency.
    
#     Args:
#         soul_spark: Soul to create standing waves in
#         beat_freq: Heartbeat frequency in Hz
#         resonance: Initial resonance level
        
#     Returns:
#         Standing wave resonance factor (0-1)
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If standing wave creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if beat_freq <= 0:
#         raise ValueError("beat_freq must be positive")
#     if not 0 <= resonance <= 1:
#         raise ValueError("resonance must be between 0 and 1")
    
#     try:
#         logger.debug(f"Creating heartbeat standing waves at {beat_freq:.2f}Hz...")
        
#         # Calculate wavelength in meters
#         wavelength = SPEED_OF_SOUND / beat_freq
        
#         # Create harmonic series from beat frequency
#         harmonic_ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
#         harmonic_freqs = [beat_freq * ratio for ratio in harmonic_ratios]
        
#         # Track layer integration
#         connected_layers = 0
#         total_standing_wave_res = 0.0
        
#         for layer_idx, layer in enumerate(soul_spark.layers):
#             if not isinstance(layer, dict):
#                 continue
                
#             # Create heartbeat resonance field in layer
#             if 'heartbeat_resonance' not in layer:
#                 layer['heartbeat_resonance'] = {}
            
#             # Determine position in layer (different for each layer)
#             # Create natural variability for standing wave positions
#             layer_position = (layer_idx / max(1, len(soul_spark.layers) - 1))
            
#             # Calculate best resonant harmonic for this layer
#             best_harmonic_idx = 0
#             best_harmonic_res = 0.0
            
#             # Get layer frequencies
#             layer_freqs = []
#             if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                 layer_freqs.extend(layer['resonant_frequencies'])
            
#             # Find best resonance across harmonics
#             for h_idx, h_freq in enumerate(harmonic_freqs):
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(h_freq, layer_freq)
#                     if res > best_harmonic_res:
#                         best_harmonic_res = res
#                         best_harmonic_idx = h_idx
            
#             # Skip layers with poor resonance
#             if best_harmonic_res < 0.1:
#                 continue
                
#             # Use the most resonant harmonic for this layer
#             harmonic_freq = harmonic_freqs[best_harmonic_idx]
#             harmonic_ratio = harmonic_ratios[best_harmonic_idx]
#             harmonic_wavelength = wavelength / harmonic_ratio
            
#             # Calculate optimal node positions for standing waves
#             # For standing waves, nodes occur at 1/4, 3/4, etc. of the wavelength
#             primary_node_pos = 0.25 * harmonic_wavelength
#             secondary_node_pos = 0.75 * harmonic_wavelength
            
#             # Calculate antinode positions (maximum amplitude)
#             # Antinodes at 0, 1/2, 1, etc. of wavelength
#             primary_antinode_pos = 0.0
#             secondary_antinode_pos = 0.5 * harmonic_wavelength
            
#             # Create standing wave in layer
#             layer['heartbeat_resonance']['standing_wave'] = {
#                 "frequency": float(harmonic_freq),
#                 "harmonic_ratio": float(harmonic_ratio),
#                 "wavelength_meters": float(harmonic_wavelength),
#                 "resonance": float(best_harmonic_res),
#                 "nodes": [
#                     {"position": float(primary_node_pos), "amplitude": 0.0},
#                     {"position": float(secondary_node_pos), "amplitude": 0.0}
#                 ],
#                 "antinodes": [
#                     {"position": float(primary_antinode_pos), "amplitude": float(best_harmonic_res)},
#                     {"position": float(secondary_antinode_pos), "amplitude": float(best_harmonic_res)}
#                 ],
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             # Add to layer's resonant frequencies if not present
#             if 'resonant_frequencies' not in layer:
#                 layer['resonant_frequencies'] = []
#             if harmonic_freq not in layer['resonant_frequencies']:
#                 layer['resonant_frequencies'].append(float(harmonic_freq))
            
#             # Update tracking stats
#             connected_layers += 1
#             total_standing_wave_res += best_harmonic_res
            
#         # Calculate overall standing wave resonance quality
#         if connected_layers > 0:
#             avg_standing_wave_res = total_standing_wave_res / connected_layers
#             # Scale by number of connected layers (more layers = stronger effect)
#             # But with diminishing returns
#             standing_wave_factor = avg_standing_wave_res * min(1.0, connected_layers / 5.0)
#         else:
#             standing_wave_factor = 0.0
            
#         logger.debug(f"Created heartbeat standing waves in {connected_layers} layers, " +
#                    f"Resonance factor: {standing_wave_factor:.4f}")
        
#         return standing_wave_factor
        
#     except Exception as e:
#         logger.error(f"Error creating heartbeat standing waves: {e}", exc_info=True)
#         raise RuntimeError("Heartbeat standing wave creation failed") from e

# def train_name_response(soul_spark: SoulSpark, cycles: int = 7) -> None:
#     """
#     Train name response using carrier wave principles based on the standing wave
#     patterns established by the name. Creates resonant field instead of direct frequency.
#     """
#     logger.info("Identity Step: Train Name Response (Carrier Wave)...")
#     if not isinstance(cycles, int) or cycles < 0:
#         raise ValueError("Cycles must be non-negative.")
#     if cycles == 0:
#         logger.info("Skipping name response training (0 cycles).")
#         return
    
#     try:
#         name = getattr(soul_spark, 'name', 'Unknown')
#         name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
#         name_standing_waves = getattr(soul_spark, 'name_standing_waves', None)
#         consciousness_state = getattr(soul_spark, 'consciousness_state', 'spark')
#         heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0)
#         current_response = getattr(soul_spark, 'response_level', 0.0)
        
#         if not name_standing_waves:
#             raise ValueError("Name standing waves required for response training")
        
#         logger.debug(f"  Name Carrier Wave Training: Name={name}, NR={name_resonance:.3f}, " +
#                    f"State={consciousness_state}, HBEnt={heartbeat_entrainment:.3f}")
        
#         # Extract standing wave pattern data
#         pattern_coherence = name_standing_waves.get("pattern_coherence", 0.0)
#         resonance_quality = name_standing_waves.get("resonance_quality", 0.0)
#         component_frequencies = name_standing_waves.get("component_frequencies", [])
        
#         # Create resonant field based on name's standing waves
#         # Instead of directly modifying frequency, we create reference points
#         # that generate resonance throughout the aura
#         name_resonance_accumulator = 0.0
        
#         # Adjust cycles based on name complexity
#         effective_cycles = int(cycles * (0.5 + 0.5 * resonance_quality))
#         effective_cycles = max(1, min(cycles * 2, effective_cycles))
        
#         # Training metrics
#         cycle_metrics = []
        
#         for cycle in range(effective_cycles):
#             cycle_resonance = 0.0
            
#             # Create phase shift based on cycle progression
#             phase_shift = 2 * np.pi * (cycle / effective_cycles)
            
#             # Calculate resonance factor for this cycle
#             # Diminishing returns for later cycles
#             cycle_factor = 1.0 - 0.5 * (cycle / max(1, effective_cycles))
            
#             # Apply state modifier
#             state_factor = NAME_RESPONSE_STATE_FACTORS.get(
#                 consciousness_state, NAME_RESPONSE_STATE_FACTORS['default'])
            
#             # Apply heartbeat entrainment factor
#             heartbeat_factor = (NAME_RESPONSE_TRAIN_HEARTBEAT_FACTOR + 
#                               NAME_RESPONSE_TRAIN_HEARTBEAT_WEIGHT * heartbeat_entrainment)
            
#             # Base increase for this cycle
#             base_increase = (NAME_RESPONSE_TRAIN_BASE_INC + 
#                            NAME_RESPONSE_TRAIN_CYCLE_INC * cycle) * cycle_factor
            
#             # Amplify by name qualities
#             name_factor = NAME_RESPONSE_TRAIN_NAME_FACTOR * name_resonance
            
#             # Add pattern coherence and resonance quality effects
#             pattern_effect = 0.5 * pattern_coherence
#             resonance_effect = 0.5 * resonance_quality
            
#             # Calculate cycle increase with all factors
#             cycle_resonance = (base_increase * 
#                              state_factor * 
#                              heartbeat_factor * 
#                              (1.0 + name_factor) *
#                              (1.0 + pattern_effect) *
#                              (1.0 + resonance_effect))
            
#             # Add to accumulator
#             name_resonance_accumulator += cycle_resonance
            
#             # Track cycle metrics
#             cycle_metrics.append({
#                 "cycle": cycle + 1,
#                 "phase_shift": float(phase_shift),
#                 "cycle_factor": float(cycle_factor),
#                 "base_increase": float(base_increase),
#                 "effective_increase": float(cycle_resonance),
#                 "accumulated": float(name_resonance_accumulator)
#             })
            
#             logger.debug(f"    Cycle {cycle+1}: CycleFactor={cycle_factor:.3f}, " +
#                        f"Increase={cycle_resonance:.5f}, " +
#                        f"Accum={name_resonance_accumulator:.4f}")
        
#         # Update response level with accumulated resonance
#         # Clamp to valid range
#         new_response = min(1.0, current_response + name_resonance_accumulator)
        
#         # Create resonant field throughout aura layers
#         resonant_field_strength = _create_name_response_field(
#             soul_spark, component_frequencies, new_response)
        
#         # The resonant field provides a small additional boost to response
#         final_response = min(1.0, new_response + resonant_field_strength * 0.05)
        
#         setattr(soul_spark, 'response_level', float(final_response))
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "name": name,
#                 "initial_cycles": cycles,
#                 "effective_cycles": effective_cycles,
#                 "initial_response": float(current_response),
#                 "accumulated_increase": float(name_resonance_accumulator),
#                 "field_strength": float(resonant_field_strength),
#                 "field_boost": float(final_response - new_response),
#                 "final_response": float(final_response),
#                 "state_factor": float(state_factor),
#                 "heartbeat_factor": float(heartbeat_factor),
#                 "pattern_coherence": float(pattern_coherence),
#                 "resonance_quality": float(resonance_quality),
#                 "cycle_details": cycle_metrics,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_name_response_training', metrics_data)
        
#         logger.info(f"Name carrier wave training complete. " +
#                    f"Base Response Level: {new_response:.4f}, " +
#                    f"Field Boost: {final_response - new_response:.4f}, " +
#                    f"Final Response: {final_response:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error training name response: {e}", exc_info=True)
#         raise RuntimeError("Name response training failed.") from e

# def _create_name_response_field(soul_spark: SoulSpark, frequencies: List[float],
#                               response_level: float) -> float:
#     """
#     Create a resonant field throughout aura layers based on name frequencies.
    
#     Args:
#         soul_spark: Soul to create field in
#         frequencies: List of frequencies to establish resonance with
#         response_level: Current response level (0-1)
        
#     Returns:
#         Field strength factor (0-1)
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If field creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not frequencies:
#         raise ValueError("frequencies list cannot be empty")
#     if not 0 <= response_level <= 1:
#         raise ValueError("response_level must be between 0 and 1")
    
#     try:
#         logger.debug(f"Creating name response field with {len(frequencies)} frequencies...")
        
#         # Track layer integration
#         connected_layers = 0
#         total_field_strength = 0.0
        
#         # Start with primary frequency
#         primary_freq = frequencies[0] if frequencies else 0
        
#         for layer_idx, layer in enumerate(soul_spark.layers):
#             if not isinstance(layer, dict):
#                 continue
                
#             # Create name field in layer
#             if 'name_field' not in layer:
#                 layer['name_field'] = {}
            
#             # Find most resonant frequency for this layer
#             best_freq = primary_freq
#             best_resonance = 0.0
            
#             # Get layer frequencies
#             layer_freqs = []
#             if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                 layer_freqs.extend(layer['resonant_frequencies'])
            
#             # Find best resonance
#             for name_freq in frequencies:
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(name_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_freq = name_freq
            
#             # Skip layers with poor resonance
#             if best_resonance < 0.1:
#                 continue
                
#             # Calculate field strength based on resonance and response level
#             field_strength = best_resonance * response_level
            
#             # Create field in layer
#             layer['name_field']['resonance'] = {
#                 "frequency": float(best_freq),
#                 "resonance": float(best_resonance),
#                 "field_strength": float(field_strength),
#                 "response_level": float(response_level),
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             # Add to layer's resonant frequencies if not present
#             if 'resonant_frequencies' not in layer:
#                 layer['resonant_frequencies'] = []
#             if best_freq not in layer['resonant_frequencies']:
#                 layer['resonant_frequencies'].append(float(best_freq))
            
#             # Update tracking stats
#             connected_layers += 1
#             total_field_strength += field_strength
        
#         # Calculate overall field strength
#         if connected_layers > 0:
#             avg_field_strength = total_field_strength / connected_layers
#             # Scale by number of connected layers (more layers = stronger field)
#             # But with diminishing returns
#             field_factor = avg_field_strength * min(1.0, connected_layers / 5.0)
#         else:
#             field_factor = 0.0
            
#         logger.debug(f"Created name response field in {connected_layers} layers, " +
#                    f"Field strength: {field_factor:.4f}")
        
#         return field_factor
        
#     except Exception as e:
#         logger.error(f"Error creating name response field: {e}", exc_info=True)
#         raise RuntimeError("Name response field creation failed") from e

# def identify_primary_sephiroth(soul_spark: SoulSpark) -> None:
#     """ 
#     Identifies primary Sephiroth aspect based on soul state (Hz, factors),
#     using layer resonance rather than direct frequency matching.
#     """
#     logger.info("Identity Step: Identify Primary Sephiroth with Layer Resonance...")
#     soul_frequency = getattr(soul_spark, 'soul_frequency')
#     soul_color = getattr(soul_spark, 'soul_color')
#     consciousness_state = getattr(soul_spark, 'consciousness_state')
#     gematria = getattr(soul_spark, 'gematria_value')
#     yin_yang = getattr(soul_spark, 'yin_yang_balance')
    
#     if soul_frequency <= FLOAT_EPSILON or soul_color is None:
#         raise ValueError("Missing required soul_frequency or soul_color.")
    
#     if aspect_dictionary is None:
#         raise RuntimeError("Aspect Dictionary unavailable.")
    
#     try:
#         # Starting with traditional affinity calculations
#         sephiroth_affinities = {}
        
#         # Gematria-based affinity
#         for gem_range, sephirah in SEPHIROTH_AFFINITY_GEMATRIA_RANGES.items():
#             if gematria in gem_range:
#                 sephiroth_affinities[sephirah] = (
#                     sephiroth_affinities.get(sephirah, 0.0) + 
#                     SEPHIROTH_AFFINITY_GEMATRIA_WEIGHT)
#                 break
        
#         # Color-based affinity
#         color_match = SEPHIROTH_AFFINITY_COLOR_MAP.get(soul_color.lower())
#         if color_match:
#             sephiroth_affinities[color_match] = (
#                 sephiroth_affinities.get(color_match, 0.0) + 
#                 SEPHIROTH_AFFINITY_COLOR_WEIGHT)
        
#         # State-based affinity
#         state_match = SEPHIROTH_AFFINITY_STATE_MAP.get(consciousness_state)
#         if state_match:
#             sephiroth_affinities[state_match] = (
#                 sephiroth_affinities.get(state_match, 0.0) + 
#                 SEPHIROTH_AFFINITY_STATE_WEIGHT)
        
#         # Direct frequency-based affinity (traditional)
#         for sephirah_name in aspect_dictionary.sephiroth_names:
#             sephirah_freq = aspect_dictionary.get_aspects(sephirah_name).get('base_frequency', 0.0)
#             if sephirah_freq > FLOAT_EPSILON:
#                 resonance = calculate_resonance(soul_frequency, sephirah_freq)
#                 if resonance > SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD:
#                     sephiroth_affinities[sephirah_name] = (
#                         sephiroth_affinities.get(sephirah_name, 0.0) + 
#                         resonance * 0.5)
        
#         # Yin-Yang balance affinity
#         if yin_yang < SEPHIROTH_AFFINITY_YINYANG_LOW_THRESHOLD:
#             for sephirah in SEPHIROTH_AFFINITY_YIN_SEPHIROTH:
#                 sephiroth_affinities[sephirah] = (
#                     sephiroth_affinities.get(sephirah, 0.0) + 
#                     SEPHIROTH_AFFINITY_YINYANG_WEIGHT * (1.0 - yin_yang))
#         elif yin_yang > SEPHIROTH_AFFINITY_YINYANG_HIGH_THRESHOLD:
#             for sephirah in SEPHIROTH_AFFINITY_YANG_SEPHIROTH:
#                 sephiroth_affinities[sephirah] = (
#                     sephiroth_affinities.get(sephirah, 0.0) + 
#                     SEPHIROTH_AFFINITY_YINYANG_WEIGHT * yin_yang)
#         else:
#             balance_factor = 1.0 - abs(yin_yang - 0.5) * 2
#             for sephirah in SEPHIROTH_AFFINITY_BALANCED_SEPHIROTH:
#                 sephiroth_affinities[sephirah] = (
#                     sephiroth_affinities.get(sephirah, 0.0) + 
#                     SEPHIROTH_AFFINITY_BALANCE_WEIGHT * balance_factor)
        
#         # ENHANCEMENT: Add layer-based resonance calculation
#         # Check how layers resonate with each Sephiroth's frequencies
#         layer_resonance_weights = {}
        
#         # For each Sephirah, check resonance with aura layers
#         for sephirah_name in aspect_dictionary.sephiroth_names:
#             # Get Sephirah frequency and harmonics
#             sephirah_freq = aspect_dictionary.get_aspects(sephirah_name).get('base_frequency', 0.0)
#             if sephirah_freq <= FLOAT_EPSILON:
#                 continue
                
#             # Create harmonic series
#             sephirah_freqs = [sephirah_freq, sephirah_freq * PHI, sephirah_freq * 2.0]
            
#             total_layer_resonance = 0.0
#             resonant_layers = 0
            
#             # Check each layer
#             for layer in soul_spark.layers:
#                 if not isinstance(layer, dict):
#                     continue
                    
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 if not layer_freqs:
#                     continue
                    
#                 # Calculate best resonance between Sephirah and layer
#                 best_resonance = 0.0
#                 for s_freq in sephirah_freqs:
#                     for l_freq in layer_freqs:
#                         resonance = calculate_resonance(s_freq, l_freq)
#                         best_resonance = max(best_resonance, resonance)
                
#                 # Count as resonant if above threshold
#                 if best_resonance > SEPHIROTH_AFFINITY_FREQ_RESONANCE_THRESHOLD:
#                     total_layer_resonance += best_resonance
#                     resonant_layers += 1
            
#             # Calculate layer resonance weight
#             if resonant_layers > 0:
#                 avg_resonance = total_layer_resonance / resonant_layers
#                 layer_weight = avg_resonance * min(1.0, resonant_layers / 3.0)
#                 layer_resonance_weights[sephirah_name] = layer_weight * 0.5  # 50% weight to layer resonance
#             else:
#                 layer_resonance_weights[sephirah_name] = 0.0
       
#         # Add layer resonance weights to affinities
#         for sephirah, weight in layer_resonance_weights.items():
#             sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + weight
        
#         # Determine primary Sephiroth aspect
#         sephiroth_aspect = SEPHIROTH_ASPECT_DEFAULT
#         if sephiroth_affinities:
#             sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda item: item[1])[0]
        
#         logger.debug(f"  Sephirah Affinities: {sephiroth_affinities} -> Identified: {sephiroth_aspect}")
        
#         # Update soul with Sephiroth aspect
#         setattr(soul_spark, 'sephiroth_aspect', sephiroth_aspect)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Create Sephiroth resonance in aura layers
#         _integrate_sephiroth_with_aura_layers(soul_spark, sephiroth_aspect)
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "sephiroth_aspect": sephiroth_aspect,
#                 "traditional_affinities": {k: v for k, v in sephiroth_affinities.items() 
#                                          if k not in layer_resonance_weights},
#                 "layer_resonance_weights": layer_resonance_weights,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_sephiroth_affinity', metrics_data)
        
#         logger.info(f"Primary Sephiroth aspect identified: {sephiroth_aspect}")
        
#     except Exception as e:
#         logger.error(f"Error identifying Sephiroth aspect: {e}", exc_info=True)
#         raise RuntimeError("Sephirah aspect ID failed.") from e

# def _integrate_sephiroth_with_aura_layers(soul_spark: SoulSpark, sephiroth_name: str) -> None:
#     """
#     Integrate Sephiroth energetic pattern with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         sephiroth_name: Name of the Sephiroth aspect
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not sephiroth_name:
#         raise ValueError("sephiroth_name cannot be empty")
    
#     try:
#         # Get Sephiroth information
#         sephirah_aspects = aspect_dictionary.get_aspects(sephiroth_name)
#         if not sephirah_aspects:
#             raise ValueError(f"No aspects found for Sephiroth {sephiroth_name}")
        
#         sephirah_freq = sephirah_aspects.get('base_frequency', 0.0)
#         if sephirah_freq <= FLOAT_EPSILON:
#             raise ValueError(f"Invalid base frequency for Sephiroth {sephiroth_name}")
        
#         logger.debug(f"Integrating Sephiroth {sephiroth_name} ({sephirah_freq:.2f}Hz) with aura layers...")
        
#         # Create harmonic series based on Sephiroth frequency
#         harmonic_ratios = [1.0, PHI, 2.0, PHI*2, 3.0]
#         harmonic_freqs = [sephirah_freq * ratio for ratio in harmonic_ratios]
        
#         # Get Sephiroth properties for integration
#         sephirah_color = sephirah_aspects.get('color', '#FFFFFF')
#         sephirah_element = sephirah_aspects.get('element', 'generic').lower()
#         sephirah_geometry = sephirah_aspects.get('geometric_correspondence', 'generic').lower()
        
#         # Track layer integration
#         connected_layers = 0
#         total_resonance = 0.0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, harm_freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(harm_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create Sephiroth resonance field in layer
#                 if 'sephiroth_resonance' not in layer:
#                     layer['sephiroth_resonance'] = {}
                
#                 # Add harmonic to layer
#                 layer['sephiroth_resonance'][f"harmonic_{harm_idx}"] = {
#                     "sephiroth": sephiroth_name,
#                     "frequency": float(harm_freq),
#                     "harmonic_ratio": float(harmonic_ratios[harm_idx]),
#                     "resonance": float(best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # Add Sephiroth properties to layer
#                 if 'sephiroth_properties' not in layer:
#                     layer['sephiroth_properties'] = {}
                
#                 layer['sephiroth_properties'] = {
#                     "name": sephiroth_name,
#                     "color": sephirah_color,
#                     "element": sephirah_element,
#                     "geometry": sephirah_geometry
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if harm_freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(harm_freq))
                
#                 # Track statistics
#                 connected_layers += 1
#                 total_resonance += best_resonance
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, connected_layers)
#         logger.debug(f"Sephiroth {sephiroth_name} integrated with {connected_layers} layers, " +
#                    f"Avg resonance: {avg_resonance:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error integrating Sephiroth with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Sephiroth-layer integration failed") from e

# def determine_elemental_affinity(soul_spark: SoulSpark) -> None:
#     """
#     Determines elemental affinity based on soul state and aura layer resonance.
#     Uses wave physics to establish elemental signature throughout aura.
#     """
#     logger.info("Identity Step: Determine Elemental Affinity with Wave Physics...")
#     name = getattr(soul_spark, 'name')
#     seph_aspect = getattr(soul_spark, 'sephiroth_aspect')
#     color = getattr(soul_spark, 'soul_color')
#     state = getattr(soul_spark, 'consciousness_state')
#     freq = getattr(soul_spark, 'soul_frequency')
    
#     if not all([name, seph_aspect, color, state]) or freq <= FLOAT_EPSILON:
#         raise ValueError("Missing attributes for elemental affinity determination.")
    
#     if aspect_dictionary is None:
#         raise RuntimeError("Aspect Dictionary unavailable.")
    
#     try:
#         # Traditional affinity calculation
#         elemental_affinities = {}
        
#         # Name composition analysis
#         vowels = sum(1 for c in name.lower() if c in 'aeiouy')
#         consonants = sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxz')
#         total = vowels + consonants
        
#         if total > 0:
#             v_ratio = vowels / total
#             c_ratio = consonants / total
#             elem = 'fire'  # default
            
#             if v_ratio > ELEMENTAL_AFFINITY_VOWEL_THRESHOLD:
#                 elem = 'air'
#             elif c_ratio > ELEMENTAL_AFFINITY_CONSONANT_THRESHOLD:
#                 elem = 'earth'
#             elif 0.4 <= v_ratio <= 0.6:
#                 elem = 'water'
                
#             elemental_affinities[elem] = elemental_affinities.get(
#                 elem, 0.0) + ELEMENTAL_AFFINITY_VOWEL_MAP.get(elem, 0.1)
        
#         # Sephiroth-based elemental affinity
#         seph_element = aspect_dictionary.get_aspects(seph_aspect).get('element', '').lower()
#         if '/' in seph_element:
#             elements = seph_element.split('/')
#             weight = ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT / len(elements)
#             for e in elements:
#                 elemental_affinities[e] = elemental_affinities.get(e, 0.0) + weight
#         elif seph_element:
#             elemental_affinities[seph_element] = elemental_affinities.get(
#                 seph_element, 0.0) + ELEMENTAL_AFFINITY_SEPHIROTH_WEIGHT
        
#         # Color-based elemental affinity
#         color_element = ELEMENTAL_AFFINITY_COLOR_MAP.get(color.lower())
#         if color_element:
#             if '/' in color_element:
#                 elements = color_element.split('/')
#                 weight = ELEMENTAL_AFFINITY_COLOR_WEIGHT / len(elements)
#                 for e in elements:
#                     elemental_affinities[e] = elemental_affinities.get(e, 0.0) + weight
#             else:
#                 elemental_affinities[color_element] = elemental_affinities.get(
#                     color_element, 0.0) + ELEMENTAL_AFFINITY_COLOR_WEIGHT
        
#         # State-based elemental affinity
#         state_element = ELEMENTAL_AFFINITY_STATE_MAP.get(state)
#         if state_element:
#             elemental_affinities[state_element] = elemental_affinities.get(
#                 state_element, 0.0) + ELEMENTAL_AFFINITY_STATE_WEIGHT
        
#         # Frequency-based elemental affinity
#         assigned_element = ELEMENTAL_AFFINITY_DEFAULT
#         for upper_bound, element in ELEMENTAL_AFFINITY_FREQ_RANGES:
#             if freq < upper_bound:
#                 assigned_element = element
#                 break
                
#         elemental_affinities[assigned_element] = elemental_affinities.get(
#             assigned_element, 0.0) + ELEMENTAL_AFFINITY_FREQ_WEIGHT
        
#         # ENHANCEMENT: Add layer-based resonance calculation
#         # Calculate how aura layers resonate with elemental frequencies
#         layer_resonance_weights = {}
        
#         # Define elemental base frequencies (example)
#         elemental_frequencies = {
#             'fire': 396.0,  # Base frequencies for each element
#             'water': 417.0,
#             'air': 528.0,
#             'earth': 639.0,
#             'aether': 741.0
#         }
        
#         # For each element, check resonance with aura layers
#         for element, elem_freq in elemental_frequencies.items():
#             # Create harmonic series
#             elem_freqs = [elem_freq, elem_freq * PHI, elem_freq * 2.0]
            
#             total_layer_resonance = 0.0
#             resonant_layers = 0
            
#             # Check each layer
#             for layer in soul_spark.layers:
#                 if not isinstance(layer, dict):
#                     continue
                    
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 if not layer_freqs:
#                     continue
                    
#                 # Calculate best resonance between element and layer
#                 best_resonance = 0.0
#                 for e_freq in elem_freqs:
#                     for l_freq in layer_freqs:
#                         resonance = calculate_resonance(e_freq, l_freq)
#                         best_resonance = max(best_resonance, resonance)
                
#                 # Count as resonant if above threshold
#                 if best_resonance > 0.2:  # Lower threshold for elements
#                     total_layer_resonance += best_resonance
#                     resonant_layers += 1
            
#             # Calculate layer resonance weight
#             if resonant_layers > 0:
#                 avg_resonance = total_layer_resonance / resonant_layers
#                 layer_weight = avg_resonance * min(1.0, resonant_layers / 3.0)
#                 layer_resonance_weights[element] = layer_weight * 0.3  # 30% weight to layer resonance
#             else:
#                 layer_resonance_weights[element] = 0.0
        
#         # Add layer resonance weights to affinities
#         for element, weight in layer_resonance_weights.items():
#             elemental_affinities[element] = elemental_affinities.get(element, 0.0) + weight
        
#         # Determine primary elemental affinity
#         elemental_affinity = ELEMENTAL_AFFINITY_DEFAULT
#         if elemental_affinities:
#             elemental_affinity = max(elemental_affinities.items(), key=lambda item: item[1])[0]
        
#         logger.debug(f"  Elemental Affinities: {elemental_affinities} -> Identified: {elemental_affinity}")
        
#         # Update soul with elemental affinity
#         setattr(soul_spark, 'elemental_affinity', elemental_affinity)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Create elemental resonance in aura layers
#         _integrate_element_with_aura_layers(
#             soul_spark, elemental_affinity, elemental_frequencies.get(elemental_affinity, 0.0))
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "elemental_affinity": elemental_affinity,
#                 "traditional_affinities": {k: v for k, v in elemental_affinities.items() 
#                                          if k not in layer_resonance_weights},
#                 "layer_resonance_weights": layer_resonance_weights,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_elemental_affinity', metrics_data)
        
#         logger.info(f"Elemental affinity determined: {elemental_affinity}")
        
#     except Exception as e:
#         logger.error(f"Error determining elemental affinity: {e}", exc_info=True)
#         raise RuntimeError("Elemental affinity determination failed.") from e

# def _integrate_element_with_aura_layers(soul_spark: SoulSpark, element_name: str,
#                                       element_freq: float) -> None:
#     """
#     Integrate elemental pattern with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         element_name: Name of elemental affinity
#         element_freq: Base frequency for the element
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not element_name:
#         raise ValueError("element_name cannot be empty")
#     if element_freq <= 0:
#         logger.warning(f"Invalid element frequency ({element_freq}), using default 432Hz")
#         element_freq = 432.0
    
#     try:
#         logger.debug(f"Integrating element {element_name} ({element_freq:.2f}Hz) with aura layers...")
        
#         # Create harmonic series based on elemental frequency
#         harmonic_ratios = [1.0, PHI, 2.0, 3.0]
#         harmonic_freqs = [element_freq * ratio for ratio in harmonic_ratios]
        
#         # Define element properties
#         # NOTE: In a full implementation, these would come from a comprehensive
#         # elemental database like aspect_dictionary for Sephiroth
#         element_properties = {
#             'fire': {'color': '#FF5500', 'sound': 'crackling', 'geometry': 'tetrahedron'},
#             'water': {'color': '#0077FF', 'sound': 'flowing', 'geometry': 'icosahedron'},
#             'air': {'color': '#AAFFFF', 'sound': 'whistling', 'geometry': 'octahedron'},
#             'earth': {'color': '#996633', 'sound': 'rumbling', 'geometry': 'cube'},
#             'aether': {'color': '#FFFFFF', 'sound': 'humming', 'geometry': 'dodecahedron'},
#         }
        
#         # Get properties for this element
#         props = element_properties.get(element_name, 
#                                      {'color': '#CCCCCC', 'sound': 'generic', 'geometry': 'sphere'})
        
#         # Track layer integration
#         connected_layers = 0
#         total_resonance = 0.0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, harm_freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(harm_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create element resonance field in layer
#                 if 'element_resonance' not in layer:
#                     layer['element_resonance'] = {}
                
#                 # Add harmonic to layer
#                 layer['element_resonance'][f"harmonic_{harm_idx}"] = {
#                     "element": element_name,
#                     "frequency": float(harm_freq),
#                     "harmonic_ratio": float(harmonic_ratios[harm_idx]),
#                     "resonance": float(best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # Add element properties to layer
#                 if 'element_properties' not in layer:
#                     layer['element_properties'] = {}
                
#                 layer['element_properties'] = {
#                     "name": element_name,
#                     "color": props['color'],
#                     "sound": props['sound'],
#                     "geometry": props['geometry']
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if harm_freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(harm_freq))
                
#                 # Track statistics
#                 connected_layers += 1
#                 total_resonance += best_resonance
        
#         # Generate element sound if available
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 # Create element tone with harmonics
#                 element_sound = sound_gen.generate_harmonic_tone(
#                     element_freq, harmonic_ratios, 
#                     [0.8, 0.5, 0.3, 0.2],  # Amplitudes
#                     5.0,  # Duration
#                     0.5)  # Fade
                
#                 # Save element sound
#                 sound_path = sound_gen.save_sound(
#                     element_sound, f"element_{element_name}.wav", 
#                     f"Element {element_name} Sound")
                
#                 logger.debug(f"  Generated element sound: {sound_path}")
                
#             except Exception as sound_err:
#                 logger.warning(f"Failed to generate element sound: {sound_err}")
#                 # Continue without failing process
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, connected_layers)
#         logger.debug(f"Element {element_name} integrated with {connected_layers} layers, " +
#                    f"Avg resonance: {avg_resonance:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error integrating element with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Element-layer integration failed") from e

# def assign_platonic_symbol(soul_spark: SoulSpark) -> None:
#     """
#     Assigns Platonic symbol based on affinities with proper geometric resonance.
#     Creates standing wave patterns based on sacred geometry principles.
#     """
#     logger.info("Identity Step: Assign Platonic Symbol with Geometric Resonance...")
#     elem_affinity = getattr(soul_spark, 'elemental_affinity')
#     seph_aspect = getattr(soul_spark, 'sephiroth_aspect')
#     gematria = getattr(soul_spark, 'gematria_value')
    
#     if not elem_affinity or not seph_aspect:
#         raise ValueError("Missing affinities for Platonic symbol assignment.")
    
#     if aspect_dictionary is None:
#         raise RuntimeError("Aspect Dictionary unavailable.")
    
#     try:
#         # First determine symbol based on traditional methods
#         symbol = PLATONIC_ELEMENT_MAP.get(elem_affinity)
        
#         if not symbol:
#             seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
#             if seph_geom in PLATONIC_SOLIDS:
#                 symbol = seph_geom
#             elif seph_geom == 'cube':
#                 symbol = 'hexahedron'  # Standardize terminology
        
#         if not symbol:
#             # Fallback based on gematria
#             unique_symbols = sorted(list(set(PLATONIC_ELEMENT_MAP.values()) - {None}))
#             if unique_symbols:
#                 symbol_idx = (gematria // PLATONIC_DEFAULT_GEMATRIA_RANGE) % len(unique_symbols)
#                 symbol = unique_symbols[symbol_idx]
#             else:
#                 symbol = 'sphere'  # Ultimate fallback
        
#         logger.debug(f"  Platonic Symbol Logic -> Final='{symbol}'")
        
#         # ENHANCEMENT: Create geometric standing wave patterns
#         # Define geometric properties of platonic solids
#         geometric_properties = {
#             'tetrahedron': {
#                 'vertices': 4, 'edges': 6, 'faces': 4, 
#                 'face_type': 'triangle', 'symmetry': 'tetrahedral',
#                 'frequency_ratio': 1.0  # Base ratio
#             },
#             'hexahedron': {  # Cube
#                 'vertices': 8, 'edges': 12, 'faces': 6, 
#                 'face_type': 'square', 'symmetry': 'octahedral',
#                 'frequency_ratio': 2.0/1.0  # 2:1 ratio
#             },
#             'octahedron': {
#                 'vertices': 6, 'edges': 12, 'faces': 8, 
#                 'face_type': 'triangle', 'symmetry': 'octahedral',
#                 'frequency_ratio': 3.0/2.0  # 3:2 ratio
#             },
#             'dodecahedron': {
#                 'vertices': 20, 'edges': 30, 'faces': 12, 
#                 'face_type': 'pentagon', 'symmetry': 'icosahedral',
#                 'frequency_ratio': PHI  # Golden ratio
#             },
#             'icosahedron': {
#                 'vertices': 12, 'edges': 30, 'faces': 20, 
#                 'face_type': 'triangle', 'symmetry': 'icosahedral',
#                 'frequency_ratio': PHI * PHI  # Phi squared
#             },
#             'sphere': {  # Perfect form - not technically platonic but included
#                 'vertices': 'infinite', 'edges': 'infinite', 'faces': 'infinite', 
#                 'face_type': 'point', 'symmetry': 'spherical',
#                 'frequency_ratio': PI  # Pi ratio
#             }
#         }
        
#         # Get properties for the chosen symbol
#         props = geometric_properties.get(symbol, geometric_properties['sphere'])
        
#         # Create resonant geometric pattern
#         geometric_pattern = _create_geometric_standing_waves(soul_spark, symbol, props)
        
#         # Update soul with platonic symbol and geometric pattern
#         setattr(soul_spark, 'platonic_symbol', symbol)
#         setattr(soul_spark, 'geometric_pattern', geometric_pattern)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "platonic_symbol": symbol,
#                 "properties": {k: v for k, v in props.items() if k != 'frequency_ratio'},
#                 "frequency_ratio": float(props['frequency_ratio']),
#                 "standing_waves": len(geometric_pattern.get('nodes', [])),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_platonic_symbol', metrics_data)
        
#         logger.info(f"Platonic symbol assigned: {symbol} with " +
#                    f"{len(geometric_pattern.get('nodes', []))} standing wave nodes")
        
#     except Exception as e:
#         logger.error(f"Error assigning Platonic symbol: {e}", exc_info=True)
#         raise RuntimeError("Platonic symbol assignment failed.") from e

# def _create_geometric_standing_waves(soul_spark: SoulSpark, symbol: str, 
#                                    properties: Dict) -> Dict[str, Any]:
#     """
#     Create geometric standing wave patterns based on platonic solid properties.
    
#     Args:
#         soul_spark: Soul to create patterns in
#         symbol: Name of platonic solid
#         properties: Properties of the platonic solid
        
#     Returns:
#         Dictionary with geometric pattern data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If pattern creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not symbol:
#         raise ValueError("symbol cannot be empty")
#     if not isinstance(properties, dict):
#         raise ValueError("properties must be a dictionary")
    
#     try:
#         logger.debug(f"Creating geometric standing waves for {symbol}...")
        
#         # Get base frequency for pattern
#         base_freq = soul_spark.frequency
        
#         # Apply geometric frequency ratio
#         freq_ratio = properties.get('frequency_ratio', 1.0)
#         geometric_freq = base_freq * freq_ratio
        
#         # Create harmonic series based on geometric properties
#         # Use vertices, edges, faces to determine number of harmonics
#         vertices = properties.get('vertices', 0)
#         # Convert to numeric if needed
#         if isinstance(vertices, str):
#             vertices = 1000  # Large number for 'infinite'
            
#         # Number of harmonics based on complexity
#         num_harmonics = min(12, vertices // 2 + 3)
        
#         # Create harmonic ratios (some based on golden ratio for certain solids)
#         if symbol in ('dodecahedron', 'icosahedron'):
#             # These are phi-based solids
#             harmonic_ratios = [1.0]
#             for i in range(1, num_harmonics):
#                 if i % 2 == 1:
#                     # Odd harmonics use phi
#                     harmonic_ratios.append(PHI ** ((i + 1) // 2))
#                 else:
#                     # Even harmonics use integers
#                     harmonic_ratios.append(float(i))
#         else:
#             # Standard harmonic series
#             harmonic_ratios = [float(i) for i in range(1, num_harmonics + 1)]
            
#         # Calculate frequencies
#         harmonic_freqs = [geometric_freq * ratio for ratio in harmonic_ratios]
        
#         # Calculate wavelengths
#         wavelengths = [SPEED_OF_SOUND / freq for freq in harmonic_freqs]
        
#         # Create node and antinode positions
#         # These form a geometric pattern in frequency space
#         nodes = []
#         antinodes = []
        
#         # Calculate nodes based on symbol geometry
#         for i, wavelength in enumerate(wavelengths):
#             # For each wavelength, create nodes at positions determined by geometry
#             # This is a simplified model - real implementation would use 3D geometry
            
#             # Number of nodes depends on harmonic and geometry
#             num_nodes = min(int(properties.get('edges', 12) * harmonic_ratios[i] * 0.5), 30)
            
#             for j in range(num_nodes):
#                 # Position is based on geometric distribution
#                 if symbol in ('tetrahedron', 'octahedron', 'icosahedron'):
#                     # Triangular faces - use triangular spacing
#                     pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
#                     # Modulate with sine to create triangular pattern
#                     pos_mod = 0.5 + 0.3 * np.sin(pos * 2 * np.pi)
#                     pos = pos_mod
#                 elif symbol in ('hexahedron', 'dodecahedron'):
#                     # Square/pentagonal faces - use more regular spacing
#                     pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
#                 else:  # sphere
#                     # Use simple linear spacing for sphere
#                     pos = j / (num_nodes - 1) if num_nodes > 1 else 0.5
                
#                 # Calculate amplitude (varies by position and harmonic)
#                 # Nodes have near-zero amplitude
#                 amp = 0.01 + 0.01 * np.cos(pos * np.pi * 2 * (i + 1))
                
#                 # Add node to list
#                 nodes.append({
#                     "harmonic": i + 1,
#                     "frequency": float(harmonic_freqs[i]),
#                     "wavelength": float(wavelengths[i]),
#                     "position": float(pos),
#                     "amplitude": float(amp)
#                 })
                
#                 # Calculate corresponding antinode (maximum amplitude)
#                 # Antinodes are offset from nodes by 1/4 wavelength
#                 antinode_pos = (pos + 0.25) % 1.0
#                 antinode_amp = 0.5 + 0.4 * np.sin(pos * np.pi * 2 * (i + 1))
                
#                 # Add antinode to list
#                 antinodes.append({
#                     "harmonic": i + 1,
#                     "frequency": float(harmonic_freqs[i]),
#                     "wavelength": float(wavelengths[i]),
#                     "position": float(antinode_pos),
#                     "amplitude": float(antinode_amp)
#                 })
        
#         # Integrate geometric pattern with aura layers
#         _integrate_geometry_with_aura_layers(
#             soul_spark, symbol, geometric_freq, harmonic_freqs, nodes, antinodes)
        
#         # Create the final geometric pattern structure
#         geometric_pattern = {
#             "symbol": symbol,
#             "base_frequency": float(geometric_freq),
#             "harmonic_ratios": harmonic_ratios,
#             "harmonic_frequencies": [float(f) for f in harmonic_freqs],
#             "wavelengths": [float(w) for w in wavelengths],
#             "nodes": nodes,
#             "antinodes": antinodes,
#             "properties": properties
#         }
        
#         logger.debug(f"Created geometric pattern for {symbol} with " +
#                    f"{len(nodes)} nodes and {len(antinodes)} antinodes.")
        
#         return geometric_pattern
        
#     except Exception as e:
#         logger.error(f"Error creating geometric standing waves: {e}", exc_info=True)

#         raise RuntimeError("Geometric standing wave creation failed") from e

# def _integrate_geometry_with_aura_layers(soul_spark: SoulSpark, symbol: str,
#                                        base_freq: float, harmonic_freqs: List[float],
#                                        nodes: List[Dict], antinodes: List[Dict]) -> None:
#     """
#     Integrate geometric pattern with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         symbol: Platonic symbol name
#         base_freq: Base frequency for the geometry
#         harmonic_freqs: List of harmonic frequencies
#         nodes: List of node positions
#         antinodes: List of antinode positions
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not symbol:
#         raise ValueError("symbol cannot be empty")
#     if base_freq <= 0:
#         raise ValueError("base_freq must be positive")
#     if not harmonic_freqs:
#         raise ValueError("harmonic_freqs cannot be empty")
    
#     try:
#         logger.debug(f"Integrating geometric pattern for {symbol} with aura layers...")
        
#         # Track layer integration
#         connected_layers = 0
#         total_resonance = 0.0
        
#         # Find resonant layers for each harmonic frequency
#         for freq_idx, freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create geometry resonance field in layer
#                 if 'geometry_resonance' not in layer:
#                     layer['geometry_resonance'] = {}
                
#                 # Find nodes and antinodes for this frequency
#                 freq_nodes = [n for n in nodes if abs(n["frequency"] - freq) < 0.1]
#                 freq_antinodes = [n for n in antinodes if abs(n["frequency"] - freq) < 0.1]
                
#                 # Add geometry to layer
#                 layer['geometry_resonance'][f"harmonic_{freq_idx}"] = {
#                     "symbol": symbol,
#                     "frequency": float(freq),
#                     "resonance": float(best_resonance),
#                     "nodes": [{"position": n["position"], "amplitude": n["amplitude"]} 
#                             for n in freq_nodes],
#                     "antinodes": [{"position": n["position"], "amplitude": n["amplitude"]} 
#                                for n in freq_antinodes],
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(freq))
                
#                 # Track statistics
#                 connected_layers += 1
#                 total_resonance += best_resonance
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, connected_layers)
#         logger.debug(f"Geometry {symbol} integrated with {connected_layers} layers, " +
#                    f"Avg resonance: {avg_resonance:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error integrating geometry with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Geometry-layer integration failed") from e

# def _determine_astrological_signature(soul_spark: SoulSpark) -> None:
#     """
#     Determines Zodiac sign, governing planet, and selects traits with frequency mapping.
#     Creates resonant field based on astrological correspondences.
#     """
#     logger.info("Identity Step: Determine Astrological Signature with Frequency Mapping...")
#     try:
#         # 1. Generate Conceptual Birth Datetime
#         sim_start_dt = datetime.fromisoformat(getattr(soul_spark, 'creation_time', datetime.now().isoformat()))
#         random_offset_days = random.uniform(0, 365.25)
#         birth_dt = sim_start_dt + timedelta(days=random_offset_days)
#         setattr(soul_spark, 'conceptual_birth_datetime', birth_dt.isoformat())
#         logger.debug(f"  Conceptual Birth Datetime: {birth_dt.strftime('%Y-%m-%d %H:%M')}")

#         # 2. Determine Zodiac Sign (Using 13 signs)
#         birth_month_day = birth_dt.strftime('%B %d')
#         zodiac_sign = "Unknown"
#         zodiac_symbol = "?"
        
#         for sign_info in ZODIAC_SIGNS:
#             try:
#                 # Create date objects for comparison
#                 start_str = f"{sign_info['start_date']} {birth_dt.year}"
#                 end_str = f"{sign_info['end_date']} {birth_dt.year}"
#                 dt_format = "%B %d %Y"
#                 start_date = datetime.strptime(start_str, dt_format).date()
#                 end_date = datetime.strptime(end_str, dt_format).date()
#                 birth_date_only = birth_dt.date()

#                 if start_date <= end_date: # Normal case (e.g., April 19 - May 13)
#                     if start_date <= birth_date_only <= end_date:
#                         zodiac_sign = sign_info['name']
#                         zodiac_symbol = sign_info['symbol']
#                         break
#                 else: # Case where end date is in the next year
#                     if birth_date_only >= start_date or birth_date_only <= end_date:
#                         zodiac_sign = sign_info['name']
#                         zodiac_symbol = sign_info['symbol']
#                         break
#             except ValueError as date_err:
#                 logger.warning(f"Could not parse date for sign {sign_info['name']}: {date_err}")
#                 continue

#         setattr(soul_spark, 'zodiac_sign', zodiac_sign)
#         logger.debug(f"  Determined Zodiac Sign: {zodiac_sign} ({zodiac_symbol})")

#         # 3. Determine Governing Planet with frequency mapping
#         # Either use planet from Earth Harmonization or assign a new one
#         governing_planet = getattr(soul_spark, 'governing_planet', None)
        
#         if not governing_planet or governing_planet == 'Unknown':
#             # Assign based on zodiac sign
#             planet_mapping = {
#                 'Aries': 'Mars',
#                 'Taurus': 'Venus',
#                 'Gemini': 'Mercury',
#                 'Cancer': 'Moon',
#                 'Leo': 'Sun',
#                 'Virgo': 'Mercury',
#                 'Libra': 'Venus',
#                 'Scorpio': 'Pluto',
#                 'Ophiuchus': 'Jupiter',
#                 'Sagittarius': 'Jupiter',
#                 'Capricorn': 'Saturn',
#                 'Aquarius': 'Uranus',
#                 'Pisces': 'Neptune'
#             }
            
#             governing_planet = planet_mapping.get(zodiac_sign, 'Jupiter')
#             setattr(soul_spark, 'governing_planet', governing_planet)
            
#         logger.debug(f"  Governing Planet: {governing_planet}")
        
#         # Get planetary frequency
#         planet_freq = PLANETARY_FREQUENCIES.get(governing_planet, 0.0)
#         if planet_freq <= 0:
#             # Fallback frequency if not defined
#             planet_freq = 432.0
#             logger.warning(f"No frequency defined for planet {governing_planet}, using default {planet_freq}Hz")

#         # 4. Select Traits with frequency mapping
#         traits = {"positive": {}, "negative": {}}
#         if zodiac_sign != "Unknown" and zodiac_sign in ZODIAC_TRAITS:
#             all_pos = ZODIAC_TRAITS[zodiac_sign].get("positive", [])
#             all_neg = ZODIAC_TRAITS[zodiac_sign].get("negative", [])
            
#             # Select random traits up to the max count
#             num_pos = min(len(all_pos), ASTROLOGY_MAX_POSITIVE_TRAITS)
#             num_neg = min(len(all_neg), ASTROLOGY_MAX_NEGATIVE_TRAITS)
#             selected_pos = random.sample(all_pos, num_pos) if num_pos > 0 else []
#             selected_neg = random.sample(all_neg, num_neg) if num_neg > 0 else []
            
#             # Assign strengths with astrological meaning
#             # Map each trait to a frequency offset from planet frequency
#             for trait_idx, trait in enumerate(selected_pos):
#                 # Calculate frequency based on golden ratio offsets
#                 trait_freq = planet_freq * (1 + (trait_idx * 0.1) * (PHI - 1))
#                 # Calculate resonance with soul frequency
#                 trait_resonance = calculate_resonance(trait_freq, soul_spark.frequency)
#                 # Use resonance as strength
#                 strength = 0.3 + 0.6 * trait_resonance
#                 traits["positive"][trait] = round(strength, 3)
                
#             for trait_idx, trait in enumerate(selected_neg):
#                 # Calculate frequency for negative traits
#                 trait_freq = planet_freq / (1 + (trait_idx * 0.1) * (PHI - 1))
#                 # Calculate resonance with soul frequency
#                 trait_resonance = calculate_resonance(trait_freq, soul_spark.frequency)
#                 # Use resonance as strength
#                 strength = 0.3 + 0.6 * trait_resonance
#                 traits["negative"][trait] = round(strength, 3)
#         else:
#             logger.warning(f"Zodiac sign '{zodiac_sign}' not found in ZODIAC_TRAITS. No traits assigned.")

#         # 5. Create astrological resonance structure
#         astrological_resonance = {
#             "zodiac_sign": zodiac_sign,
#             "zodiac_symbol": zodiac_symbol,
#             "governing_planet": governing_planet,
#             "planet_frequency": float(planet_freq),
#             "birth_datetime": birth_dt.isoformat(),
#             "traits": traits
#         }
        
#         # 6. Integrate with aura layers
#         _integrate_astrology_with_aura_layers(soul_spark, astrological_resonance)

#         # Update soul with astrological data
#         setattr(soul_spark, 'astrological_traits', traits)
#         setattr(soul_spark, 'astrological_resonance', astrological_resonance)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "zodiac_sign": zodiac_sign,
#                 "governing_planet": governing_planet,
#                 "planet_frequency": float(planet_freq),
#                 "positive_traits_count": len(traits["positive"]),
#                 "negative_traits_count": len(traits["negative"]),
#                 "birth_datetime": birth_dt.isoformat(),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_astrological_signature', metrics_data)

#         logger.info(f"Astrological Signature determined: Sign={zodiac_sign}, " +
#                    f"Planet={governing_planet}, " +
#                    f"Traits={len(traits['positive'])} Pos / {len(traits['negative'])} Neg")

#     except Exception as e:
#         logger.error(f"Error determining astrological signature: {e}", exc_info=True)
#         # Don't hard fail, raise the error
#         raise RuntimeError("Astrological signature determination failed") from e

# def _integrate_astrology_with_aura_layers(soul_spark: SoulSpark, 
#                                         astro_data: Dict[str, Any]) -> None:
#     """
#     Integrate astrological frequencies with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         astro_data: Astrological data dictionary
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not isinstance(astro_data, dict):
#         raise ValueError("astro_data must be a dictionary")
    
#     try:
#         planet = astro_data.get("governing_planet", "Unknown")
#         planet_freq = astro_data.get("planet_frequency", 0.0)
        
#         if planet_freq <= 0:
#             raise ValueError("Planet frequency must be positive")
        
#         logger.debug(f"Integrating astrological pattern for {planet} " +
#                    f"({planet_freq:.2f}Hz) with aura layers...")
        
#         # Create harmonic series based on planetary frequency
#         harmonic_ratios = [1.0, PHI, 2.0, PHI*2]
#         harmonic_freqs = [planet_freq * ratio for ratio in harmonic_ratios]
        
#         # Track layer integration
#         connected_layers = 0
#         total_resonance = 0.0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, harm_freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(harm_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create astrology resonance field in layer
#                 if 'astrology_resonance' not in layer:
#                     layer['astrology_resonance'] = {}
                
#                 # Add harmonic to layer
#                 layer['astrology_resonance'][f"harmonic_{harm_idx}"] = {
#                     "planet": planet,
#                     "frequency": float(harm_freq),
#                     "harmonic_ratio": float(harmonic_ratios[harm_idx]),
#                     "resonance": float(best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # Add astrological properties to layer
#                 if 'astrology_properties' not in layer:
#                     layer['astrology_properties'] = {}
                
#                 layer['astrology_properties'] = {
#                     "zodiac_sign": astro_data.get("zodiac_sign", "Unknown"),
#                     "planet": planet,
#                     "zodiac_symbol": astro_data.get("zodiac_symbol", "?")
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if harm_freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(harm_freq))
                
#                 # Track statistics
#                 connected_layers += 1
#                 total_resonance += best_resonance
        
#         # Log integration results
#         avg_resonance = total_resonance / max(1, connected_layers)
#         logger.debug(f"Astrological pattern integrated with {connected_layers} layers, " +
#                    f"Avg resonance: {avg_resonance:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error integrating astrology with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Astrology-layer integration failed") from e

# def activate_love_resonance(soul_spark: SoulSpark, cycles: int = 7) -> None:
#     """
#     Activates love resonance using geometric field formation and standing waves
#     to enhance harmony and emotional resonance throughout aura layers.
#     """
#     logger.info("Identity Step: Activate Love Resonance with Geometric Field Formation...")
#     if not isinstance(cycles, int) or cycles < 0:
#         raise ValueError("Cycles must be non-negative.")
#     if cycles == 0:
#         logger.info("Skipping love resonance activation (0 cycles).")
#         return
    
#     try:
#         soul_frequency = getattr(soul_spark, 'soul_frequency', 0.0)
#         state = getattr(soul_spark, 'consciousness_state', 'spark')
#         heartbeat_entrainment = getattr(soul_spark, 'heartbeat_entrainment', 0.0)
#         emotional_resonance = getattr(soul_spark, 'emotional_resonance', {})
#         current_love = float(emotional_resonance.get('love', 0.0))
#         love_freq = LOVE_RESONANCE_FREQ
        
#         if love_freq <= FLOAT_EPSILON or soul_frequency <= FLOAT_EPSILON:
#             raise ValueError("Frequencies invalid for love resonance.")
        
#         logger.debug(f"  Love Resonance Harmonic Patterning: CurrentLove={current_love:.3f}")
        
#         # ENHANCEMENT: Create heart coherence field
#         heart_coherence = heartbeat_entrainment * 0.5 + 0.25  # Base coherence level
        
#         # Calculate harmonic relationship between love frequency and soul frequency
#         harmonic_intervals = [1.0, 1.25, 1.333, 1.5, 1.667, 2.0]  # Musical intervals
#         best_harmonic_resonance = 0.0
#         best_harmonic_interval = 1.0
        
#         for interval in harmonic_intervals:
#             # Check both directions
#             harmonic1 = love_freq * interval
#             harmonic2 = love_freq / interval
            
#             # Calculate resonance with soul frequency
#             res1 = calculate_resonance(harmonic1, soul_frequency)
#             res2 = calculate_resonance(harmonic2, soul_frequency)
            
#             # Use the best resonance
#             if res1 > best_harmonic_resonance:
#                 best_harmonic_resonance = res1
#                 best_harmonic_interval = interval
#             if res2 > best_harmonic_resonance:
#                 best_harmonic_resonance = res2
#                 best_harmonic_interval = 1.0 / interval
        
#         # Create heart-centered geometric field
#         heart_field = _create_heart_centered_field(
#             soul_spark, love_freq, best_harmonic_interval)
        
#         # Love resonance cycles enhanced with geometric field
#         love_accumulator = 0.0
        
#         # Cycle metrics
#         cycle_metrics = []
        
#         for cycle in range(cycles):
#             # Create heart-centered resonance pattern using heart coherence
#             cycle_factor = 1.0 - (cycle / (max(1, cycles) * 1.0))
#             state_factor = LOVE_RESONANCE_STATE_WEIGHT.get(
#                 state, LOVE_RESONANCE_STATE_WEIGHT['default'])
            
#             # Use the harmonic resonance rather than direct frequency matching
#             resonance_factor = best_harmonic_resonance * LOVE_RESONANCE_FREQ_RES_WEIGHT
            
#             # Heart coherence amplifies the effect
#             heart_factor = (LOVE_RESONANCE_HEARTBEAT_WEIGHT + 
#                           LOVE_RESONANCE_HEARTBEAT_SCALE * heart_coherence)
            
#             # Calculate the love increase
#             base_increase = LOVE_RESONANCE_BASE_INC * cycle_factor
            
#             # Add geometric field factor
#             field_factor = heart_field.get("field_strength", 0.5)
            
#             # Calculate full increase with all factors
#             full_increase = (base_increase * 
#                            state_factor * 
#                            resonance_factor * 
#                            heart_factor * 
#                            (1.0 + field_factor * 0.5))  # 50% boost from field
            
#             # Apply increase to love accumulator
#             love_accumulator += full_increase
            
#             # Track cycle metrics
#             cycle_metrics.append({
#                 "cycle": cycle + 1,
#                 "cycle_factor": float(cycle_factor),
#                 "base_increase": float(base_increase),
#                 "effective_increase": float(full_increase),
#                 "accumulated": float(love_accumulator)
#             })
            
#             logger.debug(f"    Cycle {cycle+1}: CycleFactor={cycle_factor:.3f}, " +
#                        f"FieldFactor={field_factor:.3f}, " +
#                        f"Inc={full_increase:.5f}, " +
#                        f"Accum={love_accumulator:.4f}")
        
#         # Apply accumulated love resonance
#         new_love = min(1.0, current_love + love_accumulator)
        
#         # Update other emotions based on love (gentle ripple effect)
#         new_emotional_resonance = emotional_resonance.copy()
#         new_emotional_resonance['love'] = float(new_love)
        
#         # Apply gentle boost to other emotions based on love
#         for emotion in ['joy', 'peace', 'harmony', 'compassion']:
#             current = float(emotional_resonance.get(emotion, 0.0))
#             boost = LOVE_RESONANCE_EMOTION_BOOST_FACTOR * new_love
#             new_emotional_resonance[emotion] = float(min(1.0, current + boost))
            
#         # Generate love frequency sound if available
#         if SOUND_MODULES_AVAILABLE:
#             try:
#                 # Create love resonance tone with harmonics
#                 harmonic_ratios = [1.0, PHI, 1.5, 2.0]
#                 amplitudes = [0.8, 0.6, 0.5, 0.3]
                
#                 love_sound = sound_gen.generate_harmonic_tone(
#                     love_freq, harmonic_ratios, amplitudes, 5.0, 0.5)
                
#                 # Save love sound
#                 sound_path = sound_gen.save_sound(
#                     love_sound, "love_resonance.wav", "Love Resonance 528Hz")
                
#                 logger.debug(f"  Generated love resonance sound: {sound_path}")
                
#                 # Love sound provides a small additional boost
#                 sound_boost = 0.05  # 5% boost from actual sound
#                 final_love = min(1.0, new_love + sound_boost)
#                 new_emotional_resonance['love'] = float(final_love)
                
#             except Exception as sound_err:
#                 logger.warning(f"Failed to generate love sound: {sound_err}")
#                 # Continue without failing process
        
#         # Update soul with new emotional resonance
#         setattr(soul_spark, 'emotional_resonance', new_emotional_resonance)
#         setattr(soul_spark, 'heart_field', heart_field)
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "love_frequency": float(love_freq),
#                 "initial_love": float(current_love),
#                 "cycles": cycles,
#                 "accumulated_increase": float(love_accumulator),
#                 "harmonic_resonance": float(best_harmonic_resonance),
#                 "harmonic_interval": float(best_harmonic_interval),
#                 "heart_coherence": float(heart_coherence),
#                 "field_strength": float(heart_field.get("field_strength", 0.0)),
#                 "final_love": float(new_emotional_resonance['love']),
#                 "sound_generated": SOUND_MODULES_AVAILABLE,
#                 "cycle_details": cycle_metrics,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_love_resonance', metrics_data)
        
#         logger.info(f"Love resonance activated through harmonic patterning. " +
#                    f"Initial: {current_love:.4f}, Increase: {love_accumulator:.4f}, " +
#                    f"Final: {new_emotional_resonance['love']:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error activating love resonance: {e}", exc_info=True)
#         raise RuntimeError("Love resonance activation failed.") from e

# def _create_heart_centered_field(soul_spark: SoulSpark, love_freq: float,
#                               harmonic_interval: float) -> Dict[str, Any]:
#     """
#     Create heart-centered geometric field for love resonance.
    
#     Args:
#         soul_spark: Soul to create field in
#         love_freq: Love frequency (528Hz default)
#         harmonic_interval: Optimal harmonic interval
        
#     Returns:
#         Dictionary with heart field data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If field creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if love_freq <= 0:
#         raise ValueError("love_freq must be positive")
#     if harmonic_interval <= 0:
#         raise ValueError("harmonic_interval must be positive")
    
#     try:
#         logger.debug(f"Creating heart-centered field at {love_freq:.2f}Hz " +
#                    f"with harmonic interval {harmonic_interval:.3f}...")
        
#         # Create harmonic series based on love frequency
#         harmonic_ratios = [1.0, harmonic_interval, PHI, 2.0]
#         harmonic_freqs = [love_freq * ratio for ratio in harmonic_ratios]
        
#         # Heart field properties - heart shape in phase space
#         num_points = 36
#         radius = 0.8
#         field_points = []
        
#         # Create heart shape points in phase space
#         for i in range(num_points):
#             angle = 2 * PI * i / num_points
            
#             # Heart shape parametric equation
#             # x = 16 * sin(t)^3
#             # y = 13 * cos(t) - 5 * cos(2t) - 2 * cos(3t) - cos(4t)
#             # Normalized to -1 to 1 range
            
#             # t parameter (0 to 2π)
#             t = angle
            
#             # Heart shape coordinates
#             x = 16 * np.sin(t)**3
#             y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            
#             # Normalize to 0-1 range
#             x_norm = (x / 16 + 1) / 2
#             y_norm = (y / 16 + 1) / 2
            
#             # Scale by radius
#             x_scaled = 0.5 + (x_norm - 0.5) * radius
#             y_scaled = 0.5 + (y_norm - 0.5) * radius
            
#             # Calculate value at this point
#             # Phase space value based on position
#             phase_value = (1.0 - 0.5 * np.sqrt((x_scaled - 0.5)**2 + (y_scaled - 0.5)**2)) ** 2
            
#             # Add point to field
#             field_points.append({
#                 "x": float(x_scaled),
#                 "y": float(y_scaled),
#                 "value": float(phase_value)
#             })
        
#         # Calculate field strength based on points
#         field_strength = sum(p["value"] for p in field_points) / len(field_points)
        
#         # Integrate heart field with aura layers
#         integrated_layers = _integrate_heart_field_with_aura_layers(
#             soul_spark, love_freq, harmonic_freqs, field_points, field_strength)
        
#         # Create the final heart field structure
#         heart_field = {
#             "love_frequency": float(love_freq),
#             "harmonic_interval": float(harmonic_interval),
#             "harmonic_frequencies": [float(f) for f in harmonic_freqs],
#             "field_strength": float(field_strength),
#             "points": field_points,
#             "integrated_layers": integrated_layers
#         }
        
#         logger.debug(f"Created heart-centered field with strength {field_strength:.4f}, " +
#                    f"integrated with {integrated_layers} layers.")
        
#         return heart_field
        
#     except Exception as e:
#         logger.error(f"Error creating heart-centered field: {e}", exc_info=True)
#         raise RuntimeError("Heart-centered field creation failed") from e

# def _integrate_heart_field_with_aura_layers(soul_spark: SoulSpark, love_freq: float,
#                                          harmonic_freqs: List[float],
#                                          field_points: List[Dict],
#                                          field_strength: float) -> int:
#     """
#     Integrate heart field with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         love_freq: Primary love frequency
#         harmonic_freqs: List of harmonic frequencies
#         field_points: List of field points in phase space
#         field_strength: Overall field strength
        
#     Returns:
#         Number of integrated layers
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if love_freq <= 0:
#         raise ValueError("love_freq must be positive")
#     if not harmonic_freqs:
#         raise ValueError("harmonic_freqs cannot be empty")
#     if not field_points:
#         raise ValueError("field_points cannot be empty")
    
#     try:
#         logger.debug(f"Integrating heart field with aura layers...")
        
#         # Track layer integration
#         connected_layers = 0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, harm_freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(harm_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create heart field in layer
#                 if 'heart_field' not in layer:
#                     layer['heart_field'] = {}
                
#                 # Add harmonic to layer
#                 layer['heart_field'][f"harmonic_{harm_idx}"] = {
#                     "frequency": float(harm_freq),
#                     "resonance": float(best_resonance),
#                     "field_strength": float(field_strength * best_resonance),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # Add field properties to layer
#                 if 'field_properties' not in layer:
#                     layer['field_properties'] = {}
                
#                 # Add love resonance to properties
#                 layer['field_properties']['love_resonance'] = {
#                     "frequency": float(love_freq),
#                     "strength": float(field_strength),
#                     "shape": "heart"
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if harm_freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(harm_freq))
                
#                 # Count connected layer
#                 connected_layers += 1
        
#         logger.debug(f"Heart field integrated with {connected_layers} layers.")
#         return connected_layers
        
#     except Exception as e:
#         logger.error(f"Error integrating heart field with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Heart field-layer integration failed") from e

# def apply_sacred_geometry(soul_spark: SoulSpark, stages: int = 5) -> None:
#     """
#     Applies sacred geometry with proper light interference patterns that reinforce
#     coherence through constructive interference in the aura's standing waves.
    
#     This function ensures that the geometry is encoded directly in the soul's structure,
#     regardless of stability/coherence levels.
#     """
#     logger.info("Identity Step: Apply Sacred Geometry with Light Interference Patterns...")
#     if not isinstance(stages, int) or stages < 0:
#         raise ValueError("Stages non-negative.")
#     if stages == 0:
#         logger.info("Skipping sacred geometry application (0 stages).")
#         return
    
#     try:
#         seph_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
#         elem_affinity = getattr(soul_spark, 'elemental_affinity', None)
#         name_resonance = getattr(soul_spark, 'name_resonance', 0.0)
        
#         if not seph_aspect or not elem_affinity:
#             raise ValueError("Missing affinities for geometry application.")
        
#         # Available sacred geometries
#         geometries = SACRED_GEOMETRY_STAGES
#         actual_stages = min(stages, len(geometries))
#         current_cryst_level = getattr(soul_spark, 'crystallization_level', 0.0)
#         dominant_geometry = None
#         max_increase = -1.0
        
#         # Track factor boosts for patterns
#         total_pcoh_boost = 0.0
#         total_phi_boost = 0.0
#         total_harmony_boost = 0.0
#         total_torus_boost = 0.0
        
#         # Track each geometry application
#         geometry_applications = []
        
#         logger.debug("  Applying Sacred Geometry Pattern Reinforcement:")
        
#         for i in range(actual_stages):
#             geometry = geometries[i]
            
#             # Calculate stage factor (progressive complexity)
#             stage_factor = (SACRED_GEOMETRY_STAGE_FACTOR_BASE + 
#                           SACRED_GEOMETRY_STAGE_FACTOR_SCALE * (i / max(1, actual_stages - 1)))
            
#             # Calculate base increase
#             base_increase = SACRED_GEOMETRY_BASE_INC_BASE + SACRED_GEOMETRY_BASE_INC_SCALE * i
            
#             # Calculate sephiroth and elemental factors
#             seph_geom = aspect_dictionary.get_aspects(seph_aspect).get('geometric_correspondence', '').lower()
#             seph_weight = SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT.get(
#                 seph_geom, SACRED_GEOMETRY_SYMBOL_MATCH_WEIGHT['default'])
#             sephiroth_factor = seph_weight
            
#             elem_weight = SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT.get(
#                 elem_affinity, SACRED_GEOMETRY_ELEMENT_MATCH_WEIGHT['default'])
#             elemental_factor = elem_weight
            
#             # Calculate Fibonacci-based name factor
#             fib_idx = min(i, len(FIBONACCI_SEQUENCE) - 1)
#             fib_val = FIBONACCI_SEQUENCE[fib_idx]
#             fib_norm_idx = min(SACRED_GEOMETRY_FIB_MAX_IDX, len(FIBONACCI_SEQUENCE) - 1)
#             fib_norm = FIBONACCI_SEQUENCE[fib_norm_idx]
            
#             name_factor = (SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_BASE + 
#                          SACRED_GEOMETRY_NAME_RESONANCE_FACTOR_SCALE * name_resonance * 
#                          (fib_val / max(1, fib_norm)))
            
#             # Calculate total increase for this stage
#             increase = base_increase * stage_factor * sephiroth_factor * elemental_factor * name_factor
#             current_cryst_level = min(1.0, current_cryst_level + increase)
            
#             # Track dominant geometry
#             if increase > max_increase:
#                 dominant_geometry = geometry
#                 max_increase = increase
            
#             logger.debug(f"    Stage {i+1} ({geometry}): CrystInc={increase:.5f}")
            
#             # Get effects for this geometry
#             geom_effects = GEOMETRY_EFFECTS.get(geometry, DEFAULT_GEOMETRY_EFFECT)
#             stage_effect_scale = increase * 0.5
            
#             # Apply geometric effects to patterns rather than directly changing frequency
#             stage_pcoh_boost = geom_effects.get('pattern_coherence_boost', 0.0) * stage_effect_scale
#             stage_phi_boost = geom_effects.get('phi_resonance_boost', 0.0) * stage_effect_scale
#             stage_harmony_boost = geom_effects.get('harmony_boost', 0.0) * stage_effect_scale
#             stage_torus_boost = geom_effects.get('toroidal_flow_strength_boost', 0.0) * stage_effect_scale
            
#             total_pcoh_boost += stage_pcoh_boost
#             total_phi_boost += stage_phi_boost
#             total_harmony_boost += stage_harmony_boost
#             total_torus_boost += stage_torus_boost
            
#             # Create sacred geometry light interference pattern
#             geom_pattern = _create_sacred_geometry_pattern(
#                 soul_spark, geometry, stage_factor, stage_effect_scale)
            
#             # Track geometry application
#             geometry_applications.append({
#                 "geometry": geometry,
#                 "stage": i + 1,
#                 "increase": float(increase),
#                 "pattern_coherence_boost": float(stage_pcoh_boost),
#                 "phi_resonance_boost": float(stage_phi_boost),
#                 "harmony_boost": float(stage_harmony_boost),
#                 "torus_boost": float(stage_torus_boost),
#                 "pattern": geom_pattern
#             })
            
#             logger.debug(f"      Factor Boosts (Cum): dPcoh={total_pcoh_boost:.5f}, " +
#                        f"dPhi={total_phi_boost:.5f}, " +
#                        f"dHarm={total_harmony_boost:.5f}, " +
#                        f"dTorus={total_torus_boost:.5f}")
        
#         # ENHANCEMENT: Create coherent light interference pattern
#         # that reinforces the sacred geometry throughout the aura
#         interference_pattern = _create_coherent_interference_pattern(
#             soul_spark, geometry_applications)
        
#         # Apply additional boost from coherent interference
#         interference_boost_factor = interference_pattern.get("coherence", 0.0) * 0.2
        
#         # Apply final pattern reinforcement boosts with interference enhancement
#         # MODIFICATION: Even if soul already has perfect SU/CU, we still apply sacred geometry
#         soul_spark.pattern_coherence = min(1.0, soul_spark.pattern_coherence + 
#                                          total_pcoh_boost * (1.0 + interference_boost_factor))
        
#         soul_spark.phi_resonance = min(1.0, soul_spark.phi_resonance + 
#                                      total_phi_boost * (1.0 + interference_boost_factor))
        
#         soul_spark.harmony = min(1.0, soul_spark.harmony + 
#                                total_harmony_boost * (1.0 + interference_boost_factor))
        
#         soul_spark.toroidal_flow_strength = min(1.0, soul_spark.toroidal_flow_strength + 
#                                              total_torus_boost * (1.0 + interference_boost_factor))
        
#         # Update soul with sacred geometry data - ALWAYS STORE THIS
#         setattr(soul_spark, 'crystallization_level', float(current_cryst_level))
#         setattr(soul_spark, 'sacred_geometry_imprint', dominant_geometry)
#         setattr(soul_spark, 'sacred_geometry_applications', geometry_applications)
#         setattr(soul_spark, 'interference_pattern', interference_pattern)
        
#         # Add sacred geometry to soul's core properties
#         # Ensure this is always saved regardless of SU/CU
#         if not hasattr(soul_spark, 'core_properties'):
#             setattr(soul_spark, 'core_properties', {})
        
#         soul_spark.core_properties['sacred_geometry'] = {
#             'dominant_geometry': dominant_geometry,
#             'pattern_coherence': float(soul_spark.pattern_coherence),
#             'phi_resonance': float(soul_spark.phi_resonance),
#             'interference_pattern_coherence': float(interference_pattern.get("coherence", 0.0))
#         }
        
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "stages_applied": actual_stages,
#                 "dominant_geometry": dominant_geometry,
#                 "crystallization_level": float(current_cryst_level),
#                 "pattern_coherence_boost": float(total_pcoh_boost),
#                 "phi_resonance_boost": float(total_phi_boost),
#                 "harmony_boost": float(total_harmony_boost),
#                 "torus_boost": float(total_torus_boost),
#                 "interference_boost_factor": float(interference_boost_factor),
#                 "final_pattern_coherence": float(soul_spark.pattern_coherence),
#                 "final_phi_resonance": float(soul_spark.phi_resonance),
#                 "final_harmony": float(soul_spark.harmony),
#                 "final_torus_strength": float(soul_spark.toroidal_flow_strength),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_sacred_geometry', metrics_data)
        
#         logger.info(f"Sacred geometry pattern reinforcement applied. " +
#                    f"Dominant: {dominant_geometry}, " +
#                    f"Stages: {actual_stages}, " +
#                    f"Crystallization: {current_cryst_level:.4f}")
#         logger.info(f"  Resulting Factors: P.Coh={soul_spark.pattern_coherence:.4f}, " +
#                    f"Phi={soul_spark.phi_resonance:.4f}, " +
#                    f"Harm={soul_spark.harmony:.4f}, " +
#                    f"Torus={soul_spark.toroidal_flow_strength:.4f}")
        
#     except Exception as e:
#         logger.error(f"Error applying sacred geometry: {e}", exc_info=True)
#         raise RuntimeError("Sacred geometry application failed.") from e

# def _create_sacred_geometry_pattern(soul_spark: SoulSpark, geometry: str,
#                                  stage_factor: float, effect_scale: float) -> Dict[str, Any]:
#     """
#     Create sacred geometry pattern with light wave physics.
    
#     Args:
#         soul_spark: Soul to create pattern in
#         geometry: Sacred geometry type
#         stage_factor: Stage progression factor
#         effect_scale: Effect strength scale
        
#     Returns:
#         Dictionary with geometry pattern data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If pattern creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not geometry:
#         raise ValueError("geometry cannot be empty")
#     if stage_factor <= 0 or effect_scale <= 0:
#         raise ValueError("stage_factor and effect_scale must be positive")
    
#     try:
#         logger.debug(f"Creating sacred geometry pattern for {geometry}...")
        
#         # Each geometry has specific properties
#         geometry_properties = {
#             "seed_of_life": {
#                 "circles": 7,
#                 "symmetry": 6,
#                 "frequency_ratio": 1.0,
#                 "center_radius_ratio": 1.0
#             },
#             "flower_of_life": {
#                 "circles": 19,
#                 "symmetry": 6,
#                 "frequency_ratio": PHI,
#                 "center_radius_ratio": 0.5
#             },
#             "fruit_of_life": {
#                 "circles": 13,
#                 "symmetry": 6,
#                 "frequency_ratio": PHI * PHI,
#                 "center_radius_ratio": 0.3333
#             },
#             "tree_of_life": {
#                 "circles": 10,
#                 "symmetry": 3,
#                 "frequency_ratio": 3.0,
#                 "center_radius_ratio": 0.5
#             },
#             "metatrons_cube": {
#                 "circles": 13,
#                 "symmetry": 12,
#                 "frequency_ratio": PI,
#                 "center_radius_ratio": 0.25
#             },
#             "sri_yantra": {
#                 "circles": 9,
#                 "symmetry": 9,
#                 "frequency_ratio": 3.0 / 2.0,
#                 "center_radius_ratio": 0.1
#             }
#         }
        
#         # Get properties for this geometry
#         props = geometry_properties.get(
#             geometry, geometry_properties["seed_of_life"])
        
#         # Get base frequency from soul
#         base_freq = soul_spark.frequency
        
#         # Calculate geometry frequency
#         geometry_freq = base_freq * props["frequency_ratio"]
        
#         # Create light wave interference pattern
#         num_points = 100
#         pattern = np.zeros(num_points)
        
#         # Calculate interference pattern
#         for circle in range(props["circles"]):
#             # Each circle contributes a wave component
#             circle_freq = geometry_freq * (1.0 + 0.1 * circle)
#             circle_radius = props["center_radius_ratio"] * (1.0 + 0.2 * circle)
#             circle_phase = 2 * PI * circle / props["circles"]
            
#             # Create wave for this circle
#             x = np.linspace(0, 1, num_points)
#             circle_wave = np.sin(2 * PI * circle_freq * x + circle_phase) * circle_radius
            
#             # Add to overall pattern with constructive interference
#             pattern += circle_wave
        
#         # Normalize pattern
#         max_val = np.max(np.abs(pattern))
#         if max_val > FLOAT_EPSILON:
#             normalized_pattern = pattern / max_val
#         else:
#             normalized_pattern = np.zeros_like(pattern)
            
#         # Calculate pattern coherence (measure of regular structure)
#         # Higher coherence = more ordered pattern
#         fft_values = np.abs(np.fft.rfft(normalized_pattern))
#         fft_sum = np.sum(fft_values)
        
#         # Find dominant frequencies (peaks in FFT)
#         peak_threshold = 0.1 * np.max(fft_values)
#         peaks = np.where(fft_values > peak_threshold)[0]
        
#         # More peaks at regular intervals = more coherent pattern
#         num_peaks = len(peaks)
#         spacing_regularity = 0.0
        
#         if num_peaks > 1:
#             # Calculate spacings between peaks
#             spacings = np.diff(peaks)
            
#             # Measure regularity (lower std = more regular)
#             spacing_std = np.std(spacings)
#             spacing_mean = np.mean(spacings)
            
#             if spacing_mean > FLOAT_EPSILON:
#                 spacing_regularity = 1.0 - min(1.0, spacing_std / spacing_mean)
        
#         # Calculate overall coherence
#         pattern_coherence = (
#             0.3 * (num_peaks / max(1, props["circles"])) + 
#             0.7 * spacing_regularity)
        
#         # Identify nodes and antinodes in the pattern
#         nodes = []
#         antinodes = []
        
#         # Find local minima (nodes) and maxima (antinodes)
#         for i in range(1, num_points - 1):
#             if normalized_pattern[i] < normalized_pattern[i-1] and normalized_pattern[i] < normalized_pattern[i+1]:
#                 # Local minimum = node
#                 if abs(normalized_pattern[i]) < 0.2:  # Near zero
#                     nodes.append({
#                         "position": float(i / num_points),
#                         "amplitude": float(normalized_pattern[i])
#                     })
#             elif normalized_pattern[i] > normalized_pattern[i-1] and normalized_pattern[i] > normalized_pattern[i+1]:
#                 # Local maximum = antinode
#                 if abs(normalized_pattern[i]) > 0.5:  # Strong amplitude
#                     antinodes.append({
#                         "position": float(i / num_points),
#                         "amplitude": float(normalized_pattern[i])
#                     })
        
#         # Integrate sacred geometry with aura layers
#         _integrate_sacred_geometry_with_aura_layers(
#             soul_spark, geometry, geometry_freq, pattern_coherence, 
#             normalized_pattern.tolist(), nodes, antinodes)
        
#         # Create sacred geometry pattern structure
#         geometry_pattern = {
#             "geometry": geometry,
#             "frequency": float(geometry_freq),
#             "coherence": float(pattern_coherence),
#             "pattern": normalized_pattern.tolist(),
#             "nodes": nodes,
#             "antinodes": antinodes,
#             "properties": props
#         }
        
#         logger.debug(f"Created sacred geometry pattern for {geometry} with " +
#                    f"coherence {pattern_coherence:.4f}, " +
#                    f"{len(nodes)} nodes, {len(antinodes)} antinodes.")
        
#         return geometry_pattern
        
#     except Exception as e:
#         logger.error(f"Error creating sacred geometry pattern: {e}", exc_info=True)
#         raise RuntimeError("Sacred geometry pattern creation failed") from e

# def _integrate_sacred_geometry_with_aura_layers(soul_spark: SoulSpark, geometry: str,
#                                              geometry_freq: float, coherence: float,
#                                              pattern: List[float], nodes: List[Dict],
#                                              antinodes: List[Dict]) -> None:
#     """
#     Integrate sacred geometry pattern with aura layers.
    
#     Args:
#         soul_spark: Soul to integrate with
#         geometry: Sacred geometry name
#         geometry_freq: Frequency of the geometry
#         coherence: Coherence of the pattern
#         pattern: List of pattern values
#         nodes: List of node positions
#         antinodes: List of antinode positions
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If integration fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not geometry:
#         raise ValueError("geometry cannot be empty")
#     if geometry_freq <= 0:
#         raise ValueError("geometry_freq must be positive")
#     if not pattern:
#         raise ValueError("pattern cannot be empty")
    
#     try:
#         logger.debug(f"Integrating sacred geometry {geometry} with aura layers...")
        
#         # Create harmonic series based on geometry frequency
#         harmonic_ratios = [1.0, PHI, 2.0, 3.0]
#         harmonic_freqs = [geometry_freq * ratio for ratio in harmonic_ratios]
        
#         # Track layer integration
#         connected_layers = 0
        
#         # Find resonant layers for each harmonic
#         for harm_idx, harm_freq in enumerate(harmonic_freqs):
#             # Find best resonant layer
#             best_layer_idx = -1
#             best_resonance = 0.1  # Minimum threshold
            
#             for layer_idx, layer in enumerate(soul_spark.layers):
#                 if not isinstance(layer, dict):
#                     continue
                
#                 # Get layer frequencies
#                 layer_freqs = []
#                 if 'resonant_frequencies' in layer and isinstance(layer['resonant_frequencies'], list):
#                     layer_freqs.extend(layer['resonant_frequencies'])
                
#                 # Skip empty layers
#                 if not layer_freqs:
#                     continue
                
#                 # Find best resonance with this layer
#                 for layer_freq in layer_freqs:
#                     res = calculate_resonance(harm_freq, layer_freq)
#                     if res > best_resonance:
#                         best_resonance = res
#                         best_layer_idx = layer_idx
            
#             # Create resonant connection if found
#             if best_layer_idx >= 0:
#                 # Get layer for modification
#                 layer = soul_spark.layers[best_layer_idx]
                
#                 # Create sacred geometry field in layer
#                 if 'sacred_geometry' not in layer:
#                     layer['sacred_geometry'] = {}
                
#                 # Add harmonic to layer with subset of pattern data
#                 # Store a simplified version to avoid excessive data
#                 layer['sacred_geometry'][f"{geometry}_{harm_idx}"] = {
#                     "name": geometry,
#                     "frequency": float(harm_freq),
#                     "harmonic_ratio": float(harmonic_ratios[harm_idx]),
#                     "coherence": float(coherence),
#                     "resonance": float(best_resonance),
#                     "nodes_count": len(nodes),
#                     "antinodes_count": len(antinodes),
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # If not already present, add to layer's resonant frequencies
#                 if 'resonant_frequencies' not in layer:
#                     layer['resonant_frequencies'] = []
#                 if harm_freq not in layer['resonant_frequencies']:
#                     layer['resonant_frequencies'].append(float(harm_freq))
                
#                 # Count connected layer
#                 connected_layers += 1
        
#         logger.debug(f"Sacred geometry {geometry} integrated with {connected_layers} layers.")
        
#     except Exception as e:
#         logger.error(f"Error integrating sacred geometry with aura layers: {e}", exc_info=True)
#         raise RuntimeError("Sacred geometry-layer integration failed") from e

# def _create_coherent_interference_pattern(soul_spark: SoulSpark, 
#                                        geometry_applications: List[Dict]) -> Dict[str, Any]:
#     """
#     Create coherent light interference pattern from multiple sacred geometries.
    
#     Args:
#         soul_spark: Soul to create pattern in
#         geometry_applications: List of applied geometries
        
#     Returns:
#         Dictionary with interference pattern data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If pattern creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not geometry_applications:
#         raise ValueError("geometry_applications cannot be empty")
    
#     try:
#         logger.debug(f"Creating coherent interference pattern from " +
#                    f"{len(geometry_applications)} geometries...")
        
#         # Create combined interference pattern
#         num_points = 100
#         combined_pattern = np.zeros(num_points)
        
#         # Weighted contribution factors
#         total_weight = 0.0
        
#         # Combine patterns from all geometries
#         for geom_app in geometry_applications:
#             pattern = geom_app.get("pattern", {}).get("pattern", [])
            
#             # Convert to numpy array if needed
#             if isinstance(pattern, list) and len(pattern) >= num_points:
#                 pattern_array = np.array(pattern[:num_points])
                
#                 # Calculate weight based on geometry properties
#                 coherence = geom_app.get("pattern", {}).get("coherence", 0.0)
#                 stage = geom_app.get("stage", 1)
                
#                 # Higher coherence and later stages have more weight
#                 weight = coherence * 0.5 + stage / len(geometry_applications) * 0.5
                
#                 # Add weighted contribution
#                 combined_pattern += pattern_array * weight
#                 total_weight += weight
        
#         # Normalize combined pattern
#         if total_weight > FLOAT_EPSILON:
#             combined_pattern /= total_weight
            
#         # Analyze combined pattern for coherence
#         # Calculate autocorrelation for periodicity
#         autocorr = np.correlate(combined_pattern, combined_pattern, mode='full')
#         autocorr = autocorr[num_points-1:] / autocorr[num_points-1]
        
#         # Find peaks in autocorrelation
#         peak_threshold = 0.5
#         peaks = []
        
#         for i in range(1, len(autocorr) - 1):
#             if autocorr[i] > peak_threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
#                 peaks.append(i)
        
#         # Calculate coherence based on peak regularity
#         if len(peaks) > 1:
#             # Calculate spacings between peaks
#             spacings = np.diff(peaks)
            
#             # Measure regularity (lower std = more regular)
#             spacing_std = np.std(spacings)
#             spacing_mean = np.mean(spacings)
            
#             if spacing_mean > FLOAT_EPSILON:
#                 peak_regularity = 1.0 - min(1.0, spacing_std / spacing_mean)
#             else:
#                 peak_regularity = 0.0
                
#             # More peaks and more regular spacing = higher coherence
#             interference_coherence = (
#                 0.4 * (len(peaks) / 10.0) +  # Up to 10 peaks for max score
#                 0.6 * peak_regularity)
#         else:
#             interference_coherence = 0.1  # Base coherence with few peaks
        
#         # Identify interference nodes and antinodes
#         int_nodes = []
#         int_antinodes = []
        
#         # Find local minima (nodes) and maxima (antinodes)
#         for i in range(1, num_points - 1):
#             if combined_pattern[i] < combined_pattern[i-1] and combined_pattern[i] < combined_pattern[i+1]:
#                 # Local minimum = node
#                 if abs(combined_pattern[i]) < 0.1:  # Near zero
#                     int_nodes.append({
#                         "position": float(i / num_points),
#                         "amplitude": float(combined_pattern[i])
#                     })
#             elif combined_pattern[i] > combined_pattern[i-1] and combined_pattern[i] > combined_pattern[i+1]:
#                 # Local maximum = antinode
#                 if abs(combined_pattern[i]) > 0.5:  # Strong amplitude
#                     int_antinodes.append({
#                         "position": float(i / num_points),
#                         "amplitude": float(combined_pattern[i])
#                     })
        
#         # Create interference pattern structure
#         interference_pattern = {
#             "pattern": combined_pattern.tolist(),
#             "coherence": float(interference_coherence),
#             "nodes": int_nodes,
#             "antinodes": int_antinodes,
#             "contributing_geometries": [g.get("geometry", "unknown") for g in geometry_applications]
#         }
        
#         logger.debug(f"Created coherent interference pattern with " +
#                    f"coherence {interference_coherence:.4f}, " +
#                    f"{len(int_nodes)} nodes, {len(int_antinodes)} antinodes.")
        
#         return interference_pattern
        
#     except Exception as e:
#         logger.error(f"Error creating coherent interference pattern: {e}", exc_info=True)
#         raise RuntimeError("Coherent interference pattern creation failed") from e

# def calculate_attribute_coherence(soul_spark: SoulSpark) -> None:
#     """
#     Calculates attribute coherence score (0-1) based on aura layer resonance
#     and the coherent integration of all identity attributes.
#     """
#     logger.info("Identity Step: Calculate Attribute Coherence with Wave Physics...")
#     try:
#         # Collect key attribute values throughout the identity
#         attributes = {
#                     'name_resonance': getattr(soul_spark, 'name_resonance', 0.0),
#                     'response_level': getattr(soul_spark, 'response_level', 0.0),
#                     'state_stability': getattr(soul_spark, 'state_stability', 0.0),
#                     'crystallization_level': getattr(soul_spark, 'crystallization_level', 0.0),
#                     'heartbeat_entrainment': getattr(soul_spark, 'heartbeat_entrainment', 0.0),
#                     'emotional_resonance_avg': np.mean(list(getattr(soul_spark, 'emotional_resonance', {}).values())) 
#                         if getattr(soul_spark, 'emotional_resonance') else 0.0,
#                     'creator_connection': getattr(soul_spark, 'creator_connection_strength', 0.0),
#                     'earth_resonance': getattr(soul_spark, 'earth_resonance', 0.0),
#                     'elemental_alignment': getattr(soul_spark, 'elemental_alignment', 0.0),
#                     'cycle_synchronization': getattr(soul_spark, 'cycle_synchronization', 0.0),
#                     'harmony': getattr(soul_spark, 'harmony', 0.0),
#                     'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
#                     'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
#                     'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
#                 }
                
#         # Filter and validate attribute values
#         attr_values = [v for v in attributes.values() 
#                     if isinstance(v, (int, float)) and 
#                     np.isfinite(v) and 
#                     0.0 <= v <= 1.0]
        
#         if len(attr_values) < 5:
#             coherence_score = 0.5
#             logger.warning(f"  Attribute Coherence: Not enough valid attributes " +
#                         f"({len(attr_values)}). Using default {coherence_score}.")
#         else:
#             # ENHANCEMENT: Use multiple coherence measures
            
#             # 1. Standard deviation approach (lower std_dev = more coherence)
#             std_dev = np.std(attr_values)
#             std_dev_coherence = max(0.0, 1.0 - min(1.0, std_dev * ATTRIBUTE_COHERENCE_STD_DEV_SCALE))
            
#             # 2. Pattern-based approach (look for golden ratio relationships)
#             # Sort values for pattern detection
#             sorted_values = sorted(attr_values)
#             pattern_coherence = 0.0
            
#             if len(sorted_values) >= 3:
#                 # Look for phi-based patterns in adjacent triplets
#                 phi_pattern_scores = []
                
#                 for i in range(len(sorted_values) - 2):
#                     # Get three consecutive values
#                     a, b, c = sorted_values[i:i+3]
                    
#                     # Skip if values too small for meaningful ratio
#                     if a < 0.1:
#                         continue
                    
#                     # Calculate ratios
#                     ratio1 = b / a if a > FLOAT_EPSILON else 0.0
#                     ratio2 = c / b if b > FLOAT_EPSILON else 0.0
                    
#                     # Check proximity to phi
#                     phi_match1 = 1.0 - min(1.0, abs(ratio1 - PHI) / PHI)
#                     phi_match2 = 1.0 - min(1.0, abs(ratio2 - PHI) / PHI)
                    
#                     # Average phi match for this triplet
#                     triplet_phi_score = (phi_match1 + phi_match2) / 2.0
#                     phi_pattern_scores.append(triplet_phi_score)
                
#                 # Average phi pattern score
#                 if phi_pattern_scores:
#                     pattern_coherence = sum(phi_pattern_scores) / len(phi_pattern_scores)
#                 else:
#                     pattern_coherence = 0.0
            
#             # 3. Layer-based coherence approach
#             # Calculate how integrated the identity is across aura layers
#             layer_coherence = _calculate_layer_integration_coherence(soul_spark)
            
#             # Combine the different coherence measures
#             coherence_score = (
#                 0.4 * std_dev_coherence +  # Standard deviation
#                 0.3 * pattern_coherence +  # Phi patterns
#                 0.3 * layer_coherence)     # Layer integration
                
#             logger.debug(f"  Attr Coh Calc: " +
#                     f"StdDev={std_dev:.4f} -> {std_dev_coherence:.4f}, " +
#                     f"Pattern={pattern_coherence:.4f}, " +
#                     f"Layer={layer_coherence:.4f} -> " +
#                     f"Final={coherence_score:.4f}")
        
#         # Create resonant harmony enhancement
#         # Consider resonant harmony between different attributes
#         harmony_factors = []
        
#         if attributes['harmony'] > 0.1 and attributes['pattern_coherence'] > 0.1:
#             harmony_factors.append(
#                 attributes['harmony'] * attributes['pattern_coherence'] * 0.5)
                
#         if attributes['phi_resonance'] > 0.1 and attributes['toroidal_flow_strength'] > 0.1:
#             harmony_factors.append(
#                 attributes['phi_resonance'] * attributes['toroidal_flow_strength'] * 0.5)
                
#         if attributes['earth_resonance'] > 0.1 and attributes.get('elemental_alignment', 0.0) > 0.1:
#             harmony_factors.append(
#                 attributes['earth_resonance'] * attributes.get('elemental_alignment', 0.0) * 0.5)
        
#         # Apply harmony boost if we have harmony factors
#         if harmony_factors:
#             harmony_boost = sum(harmony_factors) / len(harmony_factors)
#             final_coherence = min(1.0, coherence_score * (1.0 + harmony_boost * 0.3))
#             logger.debug(f"  Applied resonant harmony boost: {harmony_boost:.4f} -> " +
#                     f"Score={final_coherence:.4f}")
#         else:
#             final_coherence = coherence_score
        
#         # Update soul with attribute coherence
#         setattr(soul_spark, 'attribute_coherence', float(final_coherence))
#         setattr(soul_spark, 'last_modified', datetime.now().isoformat())
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "attribute_values": len(attr_values),
#                 "std_dev_coherence": float(std_dev_coherence) if 'std_dev_coherence' in locals() else 0.0,
#                 "pattern_coherence": float(pattern_coherence) if 'pattern_coherence' in locals() else 0.0,
#                 "layer_coherence": float(layer_coherence) if 'layer_coherence' in locals() else 0.0,
#                 "harmony_boost": float(harmony_boost) if 'harmony_boost' in locals() else 0.0,
#                 "final_coherence": float(final_coherence),
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_attribute_coherence', metrics_data)
        
#         logger.info(f"Attribute coherence calculated: {final_coherence:.4f}")
#         logger.info(f"  Attribute Values: {attributes}")
    
#     except Exception as e:
#         logger.error(f"Error calculating attribute coherence: {e}", exc_info=True)
#         raise RuntimeError("Failed to calculate attribute coherence") from e

# def _calculate_layer_integration_coherence(soul_spark: SoulSpark) -> float:
#     """
#     Calculate coherence based on how well integrated the identity is across aura layers.
    
#     Args:
#         soul_spark: Soul to analyze
        
#     Returns:
#         Layer integration coherence factor (0-1)
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If calculation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
    
#     try:
#         logger.debug(f"Calculating layer integration coherence...")
        
#         # Collect identity components that should be integrated in layers
#         identity_components = [
#             'name_resonance',
#             'voice_resonance',
#             'color_resonance',
#             'sephiroth_resonance',
#             'element_resonance',
#             'geometry_resonance',
#             'heart_field',
#             'astrology_resonance',
#             'sacred_geometry'
#         ]
        
#         # Track integration
#         layer_scores = []
        
#         # For each layer, calculate how many identity components are integrated
#         for layer_idx, layer in enumerate(soul_spark.layers):
#             if not isinstance(layer, dict):
#                 continue
                
#             # Count integrated components
#             components_found = []
            
#             for component in identity_components:
#                 if component in layer and layer[component]:
#                     components_found.append(component)
            
#             # Calculate score for this layer
#             if identity_components:
#                 layer_integration = len(components_found) / len(identity_components)
#             else:
#                 layer_integration = 0.0
            
#             # Adjust by layer's position (outer layers less important than inner)
#             layer_pos = layer_idx / max(1, len(soul_spark.layers) - 1)
#             layer_weight = 1.0 - 0.5 * layer_pos  # Inner layers have more weight
            
#             # Add weighted score
#             layer_scores.append(layer_integration * layer_weight)
        
#         # Calculate overall coherence from layer integration
#         if layer_scores:
#             # Use sum with diminishing returns rather than average
#             # This rewards having more layers with some integration
#             # rather than just a few fully integrated layers
#             total_score = sum(layer_scores)
#             max_possible = len(soul_spark.layers)  # Perfect score
            
#             # Apply diminishing returns
#             coherence = 1.0 - exp(-total_score / max_possible * 3.0)
#             coherence = max(0.0, min(1.0, coherence))
#         else:
#             coherence = 0.0
            
#         logger.debug(f"Layer integration coherence: {coherence:.4f} " +
#                    f"from {len(layer_scores)} layers")
        
#         return coherence
        
#     except Exception as e:
#         logger.error(f"Error calculating layer integration coherence: {e}", exc_info=True)
#         raise RuntimeError("Layer integration coherence calculation failed") from e

# def _create_crystalline_structure(soul_spark: SoulSpark) -> Dict[str, Any]:
#     """
#     Create crystalline structure that integrates all identity components
#     into a coherent pattern using light physics principles.
    
#     Args:
#         soul_spark: Soul to create structure for
        
#     Returns:
#         Dictionary with crystalline structure data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If structure creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
    
#     try:
#         name = getattr(soul_spark, 'name', 'Unknown')
#         logger.debug(f"Creating crystalline structure for '{name}'...")
        
#         # Collect key characteristics
#         aspects = []
        
#         # Name information
#         name_standing_waves = getattr(soul_spark, 'name_standing_waves', None)
#         if name_standing_waves:
#             aspects.append({
#                 "type": "name",
#                 "name": name,
#                 "resonance": getattr(soul_spark, 'name_resonance', 0.0),
#                 "frequencies": name_standing_waves.get("component_frequencies", []),
#                 "nodes": len(name_standing_waves.get("nodes", [])),
#                 "coherence": name_standing_waves.get("pattern_coherence", 0.0)
#             })
            
#         # Voice information
#         voice_freq = getattr(soul_spark, 'voice_frequency', 0.0)
#         if voice_freq > 0:
#             aspects.append({
#                 "type": "voice",
#                 "frequency": float(voice_freq),
#                 "wavelength": getattr(soul_spark, 'voice_wavelength', 0.0)
#             })
            
#         # Color information
#         color_props = getattr(soul_spark, 'color_properties', None)
#         if color_props:
#             aspects.append({
#                 "type": "color",
#                 "color": color_props.get("hex", "#FFFFFF"),
#                 "frequency": color_props.get("frequency_hz", 0.0)
#             })
            
#         # Sephiroth aspect
#         seph_aspect = getattr(soul_spark, 'sephiroth_aspect', None)
#         if seph_aspect:
#             aspects.append({
#                 "type": "sephiroth",
#                 "aspect": seph_aspect
#             })
            
#         # Elemental affinity
#         elem_affinity = getattr(soul_spark, 'elemental_affinity', None)
#         if elem_affinity:
#             aspects.append({
#                 "type": "element",
#                 "element": elem_affinity
#             })
            
#         # Platonic symbol
#         platonic_symbol = getattr(soul_spark, 'platonic_symbol', None)
#         if platonic_symbol:
#             aspects.append({
#                 "type": "platonic",
#                 "symbol": platonic_symbol
#             })
            
#         # Sacred geometry
#         sacred_geometry = getattr(soul_spark, 'sacred_geometry_imprint', None)
#         if sacred_geometry:
#             aspects.append({
#                 "type": "sacred_geometry",
#                 "geometry": sacred_geometry
#             })
            
#         # Astrological signature
#         zodiac_sign = getattr(soul_spark, 'zodiac_sign', None)
#         planet = getattr(soul_spark, 'governing_planet', None)
#         if zodiac_sign and planet:
#             aspects.append({
#                 "type": "astrology",
#                 "sign": zodiac_sign,
#                 "planet": planet
#             })
        
#         # Calculate integration coherence for these aspects
#         # More aspects with coherent relationships = higher score
#         aspect_count = len(aspects)
        
#         # Create crystalline structure based on aspect integration
#         # This structure uses sacred geometry and light physics
        
#         # 1. Establish geometric base structure
#         base_geometry = platonic_symbol if platonic_symbol else "sphere"
#         sacred_imprint = sacred_geometry if sacred_geometry else "seed_of_life"
        
#         # 2. Calculate light frequencies from aspects
#         frequencies = []
        
#         for aspect in aspects:
#             if "frequency" in aspect:
#                 frequencies.append(aspect["frequency"])
#             elif aspect["type"] == "name" and "frequencies" in aspect:
#                 frequencies.extend(aspect["frequencies"])
        
#         # Ensure we have at least one frequency
#         if not frequencies:
#             frequencies = [soul_spark.frequency]
            
#         # 3. Create light interference pattern
#         num_points = 100
#         crystal_pattern = np.zeros(num_points)
        
#         # Generate pattern from frequencies
#         for freq in frequencies:
#             if freq <= 0:
#                 continue
                
#             # Convert frequency to wavelength
#             wavelength = SPEED_OF_SOUND / freq
            
#             # Create wave for this frequency
#             x = np.linspace(0, 1, num_points)
#             freq_wave = np.sin(2 * PI * x / wavelength)
            
#             # Add to pattern with constructive interference
#             crystal_pattern += freq_wave
        
#         # Normalize pattern
#         max_val = np.max(np.abs(crystal_pattern))
#         if max_val > FLOAT_EPSILON:
#             normalized_pattern = crystal_pattern / max_val
#         else:
#             normalized_pattern = np.zeros_like(crystal_pattern)
            
#         # 4. Calculate crystalline coherence
#         # Higher coherence = more integrated identity
        
#         # Use autocorrelation to measure pattern regularity
#         autocorr = np.correlate(normalized_pattern, normalized_pattern, mode='full')
#         autocorr_normalized = autocorr[num_points-1:] / autocorr[num_points-1]
        
#         # Find peaks in autocorrelation
#         peak_threshold = 0.5
#         peaks = []
        
#         for i in range(1, len(autocorr_normalized) - 1):
#             if (autocorr_normalized[i] > peak_threshold and 
#                 autocorr_normalized[i] > autocorr_normalized[i-1] and 
#                 autocorr_normalized[i] > autocorr_normalized[i+1]):
#                 peaks.append(i)
        
#         # Calculate coherence based on peak regularity
#         peak_coherence = 0.0
#         if len(peaks) > 1:
#             # Calculate spacings between peaks
#             spacings = np.diff(peaks)
            
#             # Measure regularity (lower std = more regular)
#             spacing_std = np.std(spacings)
#             spacing_mean = np.mean(spacings)
            
#             if spacing_mean > FLOAT_EPSILON:
#                 peak_coherence = 1.0 - min(1.0, spacing_std / spacing_mean)
        
#         # Combine aspect count and pattern coherence
#         aspect_factor = min(1.0, aspect_count / 10.0)  # Cap at 10 aspects
#         crystalline_coherence = 0.3 * aspect_factor + 0.7 * peak_coherence
        
#         # 5. Create final crystalline structure
#         crystalline_structure = {
#             "aspects": aspects,
#             "aspect_count": aspect_count,
#             "base_geometry": base_geometry,
#             "sacred_imprint": sacred_imprint,
#             "pattern": normalized_pattern.tolist(),
#             "coherence": float(crystalline_coherence),
#             "peak_coherence": float(peak_coherence),
#             "aspect_factor": float(aspect_factor)
#         }
        
#         logger.debug(f"Created crystalline structure with " +
#                    f"{aspect_count} aspects, " +
#                    f"coherence: {crystalline_coherence:.4f}")
        
#         return crystalline_structure
        
#     except Exception as e:
#         logger.error(f"Error creating crystalline structure: {e}", exc_info=True)
#         raise RuntimeError("Crystalline structure creation failed") from e

# def verify_identity_crystallization(soul_spark: SoulSpark, 
#                                  threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD) -> Tuple[bool, Dict[str, Any]]:
#     """
#     Verifies identity crystallization with enhanced light physics component weights.
#     Creates finalized crystalline structure for identity and light signature.
#     """
#     logger.info("Identity Step: Verify Crystallization with Light Physics...")
#     if not (0.0 < threshold <= 1.0):
#         raise ValueError("Threshold invalid.")
    
#     try:
#         # Check for required attributes
#         required_attrs_for_score = CRYSTALLIZATION_REQUIRED_ATTRIBUTES
#         missing_attributes = [attr for attr in required_attrs_for_score 
#                             if getattr(soul_spark, attr, None) is None]
#         attr_presence_score = (len(required_attrs_for_score) - len(missing_attributes)) / max(1, len(required_attrs_for_score))
        
#         # Get component metrics
#         component_metrics = {
#             'name_resonance': getattr(soul_spark, 'name_resonance', 0.0),
#             'response_level': getattr(soul_spark, 'response_level', 0.0),
#             'state_stability': getattr(soul_spark, 'state_stability', 0.0),
#             'crystallization_level': getattr(soul_spark, 'crystallization_level', 0.0),
#             'attribute_coherence': getattr(soul_spark, 'attribute_coherence', 0.0),
#             'attribute_presence': attr_presence_score,
#             'emotional_resonance': np.mean(list(getattr(soul_spark, 'emotional_resonance', {}).values())) 
#                 if getattr(soul_spark, 'emotional_resonance') else 0.0,
#             'harmony': getattr(soul_spark, 'harmony', 0.0),
#             'pattern_coherence': getattr(soul_spark, 'pattern_coherence', 0.0),
#             'phi_resonance': getattr(soul_spark, 'phi_resonance', 0.0),
#             'toroidal_flow_strength': getattr(soul_spark, 'toroidal_flow_strength', 0.0)
#         }
        
#         logger.debug(f"  Identity Verification Components: {component_metrics}")
        
#         # Create crystalline structure that integrates all components
#         crystalline_structure = _create_crystalline_structure(soul_spark)
#         crystalline_coherence = crystalline_structure.get("coherence", 0.0)
        
#         # Add crystalline coherence to component metrics
#         component_metrics['crystalline_coherence'] = crystalline_coherence
        
#         # Calculate crystallization score with enhanced weights
#         # These weights emphasize pattern integrity and light coherence
#         component_weights = {
#             'name_resonance': 0.15,
#             'response_level': 0.10,
#             'state_stability': 0.05,
#             'crystallization_level': 0.15,
#             'attribute_coherence': 0.05,
#             'emotional_resonance': 0.05,
#             'harmony': 0.10,
#             'pattern_coherence': 0.15,
#             'phi_resonance': 0.10,
#             'toroidal_flow_strength': 0.10,
#             'crystalline_coherence': 0.20  # New component with strong weight
#         }
        
#         # Calculate weighted score
#         total_crystallization_score = 0.0
#         total_weight = 0.0
        
#         for component, weight in component_weights.items():
#             value = component_metrics.get(component, 0.0)
#             if isinstance(value, (int, float)) and np.isfinite(value):
#                 total_crystallization_score += value * weight
#                 total_weight += weight
#             else:
#                 logger.warning(f"    Invalid value for component '{component}': {value}")
        
#         # Normalize if needed
#         if total_weight > FLOAT_EPSILON:
#             total_crystallization_score /= total_weight
        
#         # Apply final adjustments
#         # The identity must be properly integrated across aura layers
#         layer_integration = _calculate_layer_integration_coherence(soul_spark)
        
#         # If layer integration is poor, reduce the score
#         if layer_integration < 0.3:
#             adjustment = layer_integration / 0.3  # 0-1 scale
#             total_crystallization_score *= adjustment
#             logger.warning(f"    Poor layer integration ({layer_integration:.3f}) " +
#                          f"reduced score by factor {adjustment:.3f}")
        
#         # Clamp to valid range
#         total_crystallization_score = max(0.0, min(1.0, total_crystallization_score))
        
#         # Check crystallization success
#         is_crystallized = (total_crystallization_score >= threshold and 
#                           attr_presence_score >= CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD)
        
#         logger.debug(f"  Identity Verification: Score={total_crystallization_score:.4f}, " +
#                    f"Threshold={threshold}, AttrPresence={attr_presence_score:.2f}, " +
#                    f"LayerIntegration={layer_integration:.3f} -> " +
#                    f"Crystallized={is_crystallized}")
        
#         # Create identity light signature based on crystalline structure
#         if is_crystallized:
#             identity_light_signature = _create_identity_light_signature(soul_spark, crystalline_structure)
            
#             # Create identity sound signature if sound modules available
#             if SOUND_MODULES_AVAILABLE:
#                 try:
#                     identity_sound_signature = _create_identity_sound_signature(
#                         soul_spark, crystalline_structure)
#                     setattr(soul_spark, 'identity_sound_signature', identity_sound_signature)
#                 except Exception as sound_err:
#                     logger.warning(f"Failed to create identity sound signature: {sound_err}")
#                     # Continue without failing
            
#             # Set crystalline structure and light signature
#             setattr(soul_spark, 'crystalline_structure', crystalline_structure)
#             setattr(soul_spark, 'identity_light_signature', identity_light_signature)
            
#             # Copy structure summary to identity metrics to maintain compatibility
#             crystalline_summary = {
#                 "aspect_count": crystalline_structure.get("aspect_count", 0),
#                 "coherence": crystalline_structure.get("coherence", 0.0),
#                 "base_geometry": crystalline_structure.get("base_geometry", "unknown"),
#                 "sacred_imprint": crystalline_structure.get("sacred_imprint", "unknown")
#             }
            
#         # Update soul with crystallization results
#         setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, is_crystallized)
#         setattr(soul_spark, 'crystallization_level', float(total_crystallization_score))
#         timestamp = datetime.now().isoformat()
#         setattr(soul_spark, 'last_modified', timestamp)
        
#         # Create verification result
#         verification_result = {
#             'total_crystallization_score': float(total_crystallization_score),
#             'threshold': threshold,
#             'is_crystallized': is_crystallized,
#             'components': component_metrics,
#             'weights': component_weights,
#             'missing_attributes': missing_attributes,
#             'layer_integration': float(layer_integration),
#             'crystalline_coherence': float(crystalline_coherence)
#         }
        
#              # Add crystalline structure summary if crystallized
#         if is_crystallized:
#             crystalline_summary = {
#                 "aspect_count": crystalline_structure.get("aspect_count", 0),
#                 "coherence": crystalline_structure.get("coherence", 0.0),
#                 "base_geometry": crystalline_structure.get("base_geometry", "unknown"),
#                 "sacred_imprint": crystalline_structure.get("sacred_imprint", "unknown")
#             }
#             verification_result['crystalline_summary'] = crystalline_summary
       
#         setattr(soul_spark, 'identity_metrics', verification_result)
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics_data = {
#                 "soul_id": soul_spark.spark_id,
#                 "crystallization_score": float(total_crystallization_score),
#                 "threshold": float(threshold),
#                 "is_crystallized": is_crystallized,
#                 "layer_integration": float(layer_integration),
#                 "crystalline_coherence": float(crystalline_coherence),
#                 "missing_attributes": missing_attributes,
#                 "timestamp": datetime.now().isoformat()
#             }
#             metrics.record_metrics('identity_verification', metrics_data)
        
#         # Raise error if not crystallized
#         if not is_crystallized:
#             error_msg = (f"Identity crystallization failed: " +
#                        f"Score {total_crystallization_score:.4f} < " +
#                        f"Threshold {threshold} or " +
#                        f"Attr Presence {attr_presence_score:.2f} < " +
#                        f"{CRYSTALLIZATION_ATTR_PRESENCE_THRESHOLD}.")
#             logger.error(error_msg)
#             raise RuntimeError(error_msg)
        
#         # Success message
#         logger.info(f"Identity check PASSED: Score={total_crystallization_score:.4f}, " +
#                    f"Crystalline Coherence={crystalline_coherence:.4f}")
        
#         # Set ready for birth flag if crystallized
#         if is_crystallized:
#             setattr(soul_spark, FLAG_READY_FOR_BIRTH, True)
        
#         return is_crystallized, verification_result
        
#     except Exception as e:
#         logger.error(f"Error verifying identity: {e}", exc_info=True)
#         raise RuntimeError("Identity verification failed.") from e

# def _create_identity_light_signature(soul_spark: SoulSpark, 
#                                  crystalline_structure: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Create identity light signature from crystalline structure.
    
#     Args:
#         soul_spark: Soul to create signature for
#         crystalline_structure: Crystalline structure data
        
#     Returns:
#         Dictionary with light signature data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If signature creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not isinstance(crystalline_structure, dict):
#         raise ValueError("crystalline_structure must be a dictionary")
    
#     try:
#         logger.debug(f"Creating identity light signature...")
        
#         # Extract key components
#         name = getattr(soul_spark, 'name', 'Unknown')
#         aspects = crystalline_structure.get("aspects", [])
#         pattern = crystalline_structure.get("pattern", [])
#         coherence = crystalline_structure.get("coherence", 0.0)
        
#         # 1. Map audio frequencies to light spectrum
#         light_frequencies = []
        
#         for aspect in aspects:
#             if "frequency" in aspect:
#                 # Convert audio to light frequency
#                 light_freq = aspect["frequency"] * 1e12  # Simple scaling
#                 wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
                
#                 # Map to color
#                 if 380 <= wavelength_nm <= 750:
#                     if 380 <= wavelength_nm < 450: color = "violet"
#                     elif 450 <= wavelength_nm < 495: color = "blue"
#                     elif 495 <= wavelength_nm < 570: color = "green"
#                     elif 570 <= wavelength_nm < 590: color = "yellow"
#                     elif 590 <= wavelength_nm < 620: color = "orange"
#                     else: color = "red"
#                 else:
#                     color = "beyond visible"
                    
#                 light_frequencies.append({
#                     "aspect": aspect.get("type", "unknown"),
#                     "audio_frequency": float(aspect["frequency"]),
#                     "light_frequency": float(light_freq),
#                     "wavelength_nm": float(wavelength_nm),
#                     "color": color
#                 })
#             elif aspect["type"] == "name" and "frequencies" in aspect:
#                 # Process name frequencies
#                 for freq in aspect["frequencies"]:
#                     # Convert audio to light frequency
#                     light_freq = freq * 1e12  # Simple scaling
#                     wavelength_nm = (SPEED_OF_LIGHT / light_freq) * 1e9
                    
#                     # Map to color (simplified)
#                     if 380 <= wavelength_nm <= 750:
#                         if 380 <= wavelength_nm < 450: color = "violet"
#                         elif 450 <= wavelength_nm < 495: color = "blue"
#                         elif 495 <= wavelength_nm < 570: color = "green"
#                         elif 570 <= wavelength_nm < 590: color = "yellow"
#                         elif 590 <= wavelength_nm < 620: color = "orange"
#                         else: color = "red"
#                     else:
#                         color = "beyond visible"
                        
#                     light_frequencies.append({
#                         "aspect": "name",
#                         "audio_frequency": float(freq),
#                         "light_frequency": float(light_freq),
#                         "wavelength_nm": float(wavelength_nm),
#                         "color": color
#                     })
        
# # 2. Create light interference pattern from crystalline pattern
#         light_pattern = []
        
#         if pattern:
#             # Convert to numpy array if not already
#             if isinstance(pattern, list):
#                 pattern_array = np.array(pattern)
#             else:
#                 pattern_array = pattern
                
#             # Create light interference pattern
#             num_points = len(pattern_array)
            
#             for i in range(num_points):
#                 # Convert position to wavelength
#                 pos = i / num_points
                
#                 # Calculate value from pattern
#                 value = pattern_array[i]
                
#                 # Map to light spectrum
#                 # Use wavelength based on position (380-750nm range)
#                 wavelength_nm = 380 + pos * (750 - 380)
                
#                 # Calculate frequency from wavelength
#                 light_freq = SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
                
#                 # Calculate amplitude (intensity) from pattern value
#                 # Scale to 0-1 range
#                 amplitude = (value + 1) / 2
                
#                 # Add to light pattern
#                 light_pattern.append({
#                     "position": float(pos),
#                     "wavelength_nm": float(wavelength_nm),
#                     "frequency_hz": float(light_freq),
#                     "amplitude": float(amplitude)
#                 })
        
#         # 3. Calculate primary and secondary colors
#         # Primary color based on soul color
#         primary_color = getattr(soul_spark, 'soul_color', '#FFFFFF')
        
#         # Secondary colors from aspects
#         secondary_colors = []
#         for aspect in aspects:
#             if aspect["type"] == "color" and "color" in aspect:
#                 secondary_colors.append(aspect["color"])
#             elif aspect["type"] == "sephiroth":
#                 # Add Sephiroth color if not already present
#                 seph_color = aspect_dictionary.get_aspects(aspect["aspect"]).get('color', '#FFFFFF')
#                 if seph_color not in secondary_colors:
#                     secondary_colors.append(seph_color)
#             elif aspect["type"] == "element" and aspect["element"] in ["fire", "water", "air", "earth", "aether"]:
#                 # Add element color if not already present
#                 element_colors = {
#                     'fire': '#FF5500',
#                     'water': '#0077FF',
#                     'air': '#AAFFFF',
#                     'earth': '#996633',
#                     'aether': '#FFFFFF'
#                 }
#                 elem_color = element_colors.get(aspect["element"], '#CCCCCC')
#                 if elem_color not in secondary_colors:
#                     secondary_colors.append(elem_color)
        
#         # 4. Create complete light signature
#         light_signature = {
#             "name": name,
#             "primary_color": primary_color,
#             "secondary_colors": secondary_colors,
#             "light_frequencies": light_frequencies,
#             "light_pattern": light_pattern,
#             "coherence": float(coherence)
#         }
        
#         logger.debug(f"Created identity light signature with " +
#                    f"{len(light_frequencies)} frequencies, " +
#                    f"{len(light_pattern)} pattern points")
        
#         return light_signature
        
#     except Exception as e:
#         logger.error(f"Error creating identity light signature: {e}", exc_info=True)
#         raise RuntimeError("Identity light signature creation failed") from e

# def _create_identity_sound_signature(soul_spark: SoulSpark, 
#                                   crystalline_structure: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Create identity sound signature from crystalline structure.
    
#     Args:
#         soul_spark: Soul to create signature for
#         crystalline_structure: Crystalline structure data
        
#     Returns:
#         Dictionary with sound signature data
        
#     Raises:
#         ValueError: If inputs are invalid
#         RuntimeError: If signature creation fails
#     """
#     if not isinstance(soul_spark, SoulSpark):
#         raise ValueError("soul_spark must be a SoulSpark instance")
#     if not isinstance(crystalline_structure, dict):
#         raise ValueError("crystalline_structure must be a dictionary")
#     if not SOUND_MODULES_AVAILABLE:
#         raise RuntimeError("Sound modules not available")
    
#     try:
#         logger.debug(f"Creating identity sound signature...")
        
#         # Extract key components
#         name = getattr(soul_spark, 'name', 'Unknown')
#         aspects = crystalline_structure.get("aspects", [])
#         pattern = crystalline_structure.get("pattern", [])
        
#         # Get base frequency from soul
#         base_freq = soul_spark.frequency
        
#         # 1. Collect frequencies from aspects
#         frequencies = []
#         amplitudes = []
        
#         for aspect in aspects:
#             if "frequency" in aspect:
#                 frequencies.append(aspect["frequency"])
#                 amplitudes.append(0.7)  # Default amplitude
#             elif aspect["type"] == "name" and "frequencies" in aspect:
#                 for i, freq in enumerate(aspect["frequencies"]):
#                     frequencies.append(freq)
#                     # Decreasing amplitude for higher harmonics
#                     amp = 0.8 / (i + 1)
#                     amplitudes.append(amp)
        
#         # Ensure we have at least one frequency
#         if not frequencies:
#             frequencies = [base_freq]
#             amplitudes = [0.8]
            
#         # 2. Create harmonic structure
#         # Normalize amplitudes to avoid clipping
#         max_amp = max(amplitudes) if amplitudes else 1.0
#         if max_amp > FLOAT_EPSILON:
#             norm_amplitudes = [a / max_amp for a in amplitudes]
#         else:
#             norm_amplitudes = [0.0] * len(amplitudes)
            
#         # Convert to harmonic ratios for sound generation
#         harm_ratios = [f / base_freq for f in frequencies]
        
#         # 3. Generate identity sound
#         sound_duration = 5.0  # seconds
        
#         identity_sound = sound_gen.generate_harmonic_tone(
#             base_freq, harm_ratios, norm_amplitudes, sound_duration, 0.5)
        
#         # 4. Save identity sound
#         sound_path = sound_gen.save_sound(
#             identity_sound, f"identity_{name.lower()}.wav", 
#             f"Identity Sound for {name}")
        
#         # 5. Create complete sound signature
#         sound_signature = {
#             "name": name,
#             "base_frequency": float(base_freq),
#             "harmonics": [float(r) for r in harm_ratios],
#             "amplitudes": [float(a) for a in norm_amplitudes],
#             "duration": float(sound_duration),
#             "sound_path": sound_path
#         }
        
#         logger.debug(f"Created identity sound signature with " +
#                    f"{len(frequencies)} frequencies, saved to {sound_path}")
        
#         return sound_signature
        
#     except Exception as e:
#         logger.error(f"Error creating identity sound signature: {e}", exc_info=True)
#         raise RuntimeError("Identity sound signature creation failed") from e

# # --- Orchestration Function ---
# def perform_identity_crystallization(soul_spark: SoulSpark,
#                                     train_cycles: int = 7,
#                                     entrainment_bpm: float = 72.0,
#                                     entrainment_duration: float = 120.0,
#                                     love_cycles: int = 5,
#                                     geometry_stages: int = 2,
#                                     crystallization_threshold: float = IDENTITY_CRYSTALLIZATION_THRESHOLD
#                                     ) -> Tuple[SoulSpark, Dict[str, Any]]:
#     """
#     Performs complete identity crystallization with light, sound, and aura principles.
#     Creates a coherent identity pattern throughout the aura layers rather than
#     directly modifying core frequency. Stability and coherence emerge naturally.
#     """
#     # --- Input Validation ---
#     if not isinstance(soul_spark, SoulSpark):
#         raise TypeError("soul_spark invalid.")
#     if not isinstance(train_cycles, int) or train_cycles < 0:
#         raise ValueError("train_cycles must be >= 0")
#     if not isinstance(entrainment_bpm, (int, float)) or entrainment_bpm <= 0:
#         raise ValueError("bpm must be > 0")
#     if not isinstance(entrainment_duration, (int, float)) or entrainment_duration < 0:
#         raise ValueError("duration must be >= 0")
#     if not isinstance(love_cycles, int) or love_cycles < 0:
#         raise ValueError("love_cycles must be >= 0")
#     if not isinstance(geometry_stages, int) or geometry_stages < 0:
#         raise ValueError("geometry_stages must be >= 0")
#     if not (0.0 < crystallization_threshold <= 1.0):
#         raise ValueError("Threshold invalid.")

#     spark_id = getattr(soul_spark, 'spark_id', 'unknown_spark')
#     logger.info(f"--- Starting Identity Crystallization for Soul {spark_id} ---")
#     start_time_iso = datetime.now().isoformat()
#     start_time_dt = datetime.fromisoformat(start_time_iso)
#     process_metrics_summary = {'steps_completed': []}

#     try:
#         _ensure_soul_properties(soul_spark) # Raises error if fails
#         _check_prerequisites(soul_spark) # Raises error if fails

#         initial_state = {
#              'stability_su': soul_spark.stability, 
#              'coherence_cu': soul_spark.coherence,
#              'crystallization_level': soul_spark.crystallization_level,
#              'attribute_coherence': soul_spark.attribute_coherence,
#              'harmony': soul_spark.harmony,
#              'phi_resonance': soul_spark.phi_resonance,
#              'pattern_coherence': soul_spark.pattern_coherence,
#              'toroidal_flow_strength': soul_spark.toroidal_flow_strength
#         }
#         logger.info(f"Identity Init State: S={initial_state['stability_su']:.1f}, " +
#                    f"C={initial_state['coherence_cu']:.1f}, " +
#                    f"CrystLvl={initial_state['crystallization_level']:.3f}, " +
#                    f"Harm={initial_state['harmony']:.3f}, " +
#                    f"PhiRes={initial_state['phi_resonance']:.3f}, " +
#                    f"PatCoh={initial_state['pattern_coherence']:.3f}, " +
#                    f"Torus={initial_state['toroidal_flow_strength']:.3f}")

#         # --- Run Sequence ---
#         assign_name(soul_spark)
#         process_metrics_summary['steps_completed'].append('name')

#         _create_identity_aura_layer(soul_spark)
#         process_metrics_summary['steps_completed'].append('identity_layer')
        
#         assign_voice_frequency(soul_spark)
#         process_metrics_summary['steps_completed'].append('voice')
        
#         process_soul_color(soul_spark)
#         process_metrics_summary['steps_completed'].append('color')
        
#         apply_heartbeat_entrainment(soul_spark, entrainment_bpm, entrainment_duration)
#         process_metrics_summary['steps_completed'].append('heartbeat')
        
#         train_name_response(soul_spark, train_cycles)
#         process_metrics_summary['steps_completed'].append('response')
        
#         identify_primary_sephiroth(soul_spark)
#         process_metrics_summary['steps_completed'].append('sephiroth_id')
        
#         determine_elemental_affinity(soul_spark)
#         process_metrics_summary['steps_completed'].append('elemental_id')
        
#         assign_platonic_symbol(soul_spark)
#         process_metrics_summary['steps_completed'].append('platonic_id')
        
#         _determine_astrological_signature(soul_spark)
#         process_metrics_summary['steps_completed'].append('astrology')
        
#         activate_love_resonance(soul_spark, love_cycles)
#         process_metrics_summary['steps_completed'].append('love')
        
#         apply_sacred_geometry(soul_spark, geometry_stages)
#         process_metrics_summary['steps_completed'].append('geometry')
        
#         calculate_attribute_coherence(soul_spark)
#         process_metrics_summary['steps_completed'].append('attr_coherence')

#         logger.info("Identity Step: Final State Update & Verification...")
#         if hasattr(soul_spark, 'update_state'):
#             soul_spark.update_state() # Update SU/CU scores after geometry factor influence
#             logger.debug(f"  Identity S/C after update: S={soul_spark.stability:.1f}, " +
#                        f"C={soul_spark.coherence:.1f}")
#         else:
#             raise AttributeError("SoulSpark missing 'update_state' method.")

#         is_crystallized, verification_metrics = verify_identity_crystallization(
#             soul_spark, crystallization_threshold)
#         process_metrics_summary['steps_completed'].append('verify')
#         # verify_identity_crystallization raises RuntimeError if it fails

#         # --- Final Update & Metrics ---
#         last_mod_time = datetime.now().isoformat()
#         setattr(soul_spark, 'last_modified', last_mod_time)
        
#         if hasattr(soul_spark, 'add_memory_echo'):
#             soul_spark.add_memory_echo(
#                 f"Identity crystallized as '{soul_spark.name}'. " +
#                 f"Level: {soul_spark.crystallization_level:.3f}, " +
#                 f"Sign: {soul_spark.zodiac_sign}, " +
#                 f"S/C: {soul_spark.stability:.1f}/{soul_spark.coherence:.1f}")

#         end_time_iso = last_mod_time
#         end_time_dt = datetime.fromisoformat(end_time_iso)
        
#         # Gather final state metrics
#         final_state = {
#              'stability_su': soul_spark.stability, 
#              'coherence_cu': soul_spark.coherence,
#              'crystallization_level': soul_spark.crystallization_level,
#              'attribute_coherence': soul_spark.attribute_coherence,
#              'harmony': soul_spark.harmony,
#              'phi_resonance': soul_spark.phi_resonance,
#              'pattern_coherence': soul_spark.pattern_coherence,
#              'toroidal_flow_strength': soul_spark.toroidal_flow_strength,
#              'name': soul_spark.name, 
#              'zodiac_sign': soul_spark.zodiac_sign,
#              FLAG_IDENTITY_CRYSTALLIZED: getattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED)
#         }
        
#         # Create complete process metrics
#         overall_metrics = {
#             'action': 'identity_crystallization',
#             'soul_id': spark_id,
#             'start_time': start_time_iso,
#             'end_time': end_time_iso,
#             'duration_seconds': (end_time_dt - start_time_dt).total_seconds(),
#             'initial_state': initial_state,
#             'final_state': final_state,
#             'final_crystallization_score': verification_metrics['total_crystallization_score'],
#             'stability_change_su': final_state['stability_su'] - initial_state['stability_su'],
#             'coherence_change_cu': final_state['coherence_cu'] - initial_state['coherence_cu'],
#             'harmony_change': final_state['harmony'] - initial_state['harmony'],
#             'phi_resonance_change': final_state['phi_resonance'] - initial_state['phi_resonance'],
#             'pattern_coherence_change': final_state['pattern_coherence'] - initial_state['pattern_coherence'],
#             'torus_change': final_state['toroidal_flow_strength'] - initial_state['toroidal_flow_strength'],
#             'astrological_signature': {
#                 'sign': soul_spark.zodiac_sign,
#                 'planet': soul_spark.governing_planet,
#                 'traits': soul_spark.astrological_traits
#             },
#             'crystalline_summary': getattr(soul_spark, 'crystalline_structure', {}).get('aspects', []),
#             'success': True,
#         }
        
#         # Record metrics
#         if METRICS_AVAILABLE:
#             metrics.record_metrics('identity_crystallization_summary', overall_metrics)

#         logger.info(f"--- Identity Crystallization Completed Successfully for Soul {spark_id} ---")
#         logger.info(f"  Final State: Name='{soul_spark.name}', " +
#                    f"Sign='{soul_spark.zodiac_sign}', " +
#                    f"CrystLvl={soul_spark.crystallization_level:.3f}, " +
#                    f"S={soul_spark.stability:.1f}, C={soul_spark.coherence:.1f}")
        
#         return soul_spark, overall_metrics

#     except (ValueError, TypeError, AttributeError) as e_val:
#          logger.error(f"Identity Crystallization failed for {spark_id}: {e_val}", exc_info=True)
#          failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'prerequisites/validation'
#          record_id_failure(spark_id, start_time_iso, failed_step, str(e_val))
#          # Hard fail
#          raise e_val
         
#     except RuntimeError as e_rt:
#          logger.critical(f"Identity Crystallization failed critically for {spark_id}: {e_rt}", exc_info=True)
#          failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'verification/runtime'
#          record_id_failure(spark_id, start_time_iso, failed_step, str(e_rt))
         
#          # Set failure flags
#          setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False)
#          setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         
#          # Hard fail
#          raise e_rt
         
#     except Exception as e:
#          logger.critical(f"Unexpected error during Identity Crystallization for {spark_id}: {e}", exc_info=True)
#          failed_step = process_metrics_summary['steps_completed'][-1] if process_metrics_summary['steps_completed'] else 'unexpected'
#          record_id_failure(spark_id, start_time_iso, failed_step, str(e))
         
#          # Set failure flags
#          setattr(soul_spark, FLAG_IDENTITY_CRYSTALLIZED, False)
#          setattr(soul_spark, FLAG_READY_FOR_BIRTH, False)
         
#          # Hard fail
#          raise RuntimeError(f"Unexpected Identity Crystallization failure: {e}") from e

# # --- Failure Metric Helper ---
# def record_id_failure(spark_id: str, start_time_iso: str, failed_step: str, error_msg: str):
#     """ Helper to record failure metrics consistently. """
#     if METRICS_AVAILABLE:
#         try:
#             end_time = datetime.now().isoformat()
#             duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time_iso)).total_seconds()
#             metrics.record_metrics('identity_crystallization_summary', {
#                 'action': 'identity_crystallization',
#                 'soul_id': spark_id,
#                 'start_time': start_time_iso,
#                 'end_time': end_time,
#                 'duration_seconds': duration,
#                 'success': False,
#                 'error': error_msg,
#                 'failed_step': failed_step
#             })
#         except Exception as metric_e:
#             logger.error(f"Failed to record ID failure metrics for {spark_id}: {metric_e}")

# # --- END OF FILE ---