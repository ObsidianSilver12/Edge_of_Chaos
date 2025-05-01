"""
Sounds of Sephiroth Module

This module implements specialized sound generation for the Sephiroth dimensions.
It creates specific tones, frequencies, and harmonic structures for each Sephiroth
based on their metaphysical properties and position in the Tree of Life.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import numpy as np
import types  # For method injection
from datetime import datetime

# --- Constants ---
OUTPUT_DIR_BASE = "output/sounds"  # Define this first since it's used in class initialization
try:
    from constants.constants import *
except ImportError:
    logging.warning("Could not import constants from src.constants. Using fallback values in SoundsOfSephiroth.")
    PHI = (1 + np.sqrt(5)) / 2
    FUNDAMENTAL_FREQUENCY_432 = 432.0
    SAMPLE_RATE = 44100
    MAX_AMPLITUDE = 0.8
    PI = np.pi

from constants.constants import *

# --- Import Dependencies ---
try:
    # Import the base SoundGenerator
    from sound.sound_generator import SoundGenerator
    SOUND_GENERATOR_AVAILABLE = True
except ImportError:
    logging.error("Failed to import base SoundGenerator from sound.sound_generator. Placeholder will be used.")
    # Define a minimal placeholder if the base class is absolutely essential for structure
    class SoundGenerator:
        def __init__(self, sample_rate=SAMPLE_RATE, output_dir="output/sounds"):
            self.sample_rate = sample_rate
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            logging.warning("Using placeholder SoundGenerator class.")
        
        def generate_harmonic_tone(self, *args, **kwargs):
            return np.array([])
        
        def generate_dimensional_transition(self, *args, **kwargs):
            return np.array([])
        
        def generate_sacred_chord(self, *args, **kwargs):
            return np.array([])
        
        def save_sound(self, *args, **kwargs):
            return None
        
        def create_sound(self, *args, **kwargs):
            return None  # Placeholder for compatibility
    
    SOUND_GENERATOR_AVAILABLE = False

# Import the Sephiroth aspect dictionary loader
try:
    # Adjust path based on your structure
    from stage_1.fields.sephiroth_aspect_dictionary import aspect_dictionary
    ASPECT_DICT_AVAILABLE = True
except ImportError:
    logging.critical("Failed to import SephirothAspectDictionary. Sephiroth sound generation cannot function.")
    aspect_dictionary = None
    ASPECT_DICT_AVAILABLE = False

# Configure logging
log_file_path = os.path.join("logs", "sephiroth_sounds.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file_path
)
logger = logging.getLogger('sephiroth_sounds')

class SephirothFrequencies:
    """
    Class containing the frequency properties for each Sephiroth.
    Frequencies are loaded from Sephiroth Aspect classes via aspect_dictionary.
    """
    
    def __init__(self):
        """Initialize Sephiroth frequencies by loading them from aspect definitions."""
        if not ASPECT_DICT_AVAILABLE:
            raise RuntimeError("SephirothAspectDictionary failed to load. Cannot initialize SephirothFrequencies.")
        
        self.sephiroth_names = aspect_dictionary.sephiroth_names
        self.frequencies = {}
        missing_aspects = []
        
        logger.info("Loading Sephiroth base frequencies from Aspect classes...")
        for sephirah_name in self.sephiroth_names:
            aspect_instance = aspect_dictionary[sephirah_name]
            if aspect_instance and hasattr(aspect_instance, 'base_frequency'):
                freq = aspect_instance.base_frequency
                if isinstance(freq, (int, float)) and freq > 0:
                    self.frequencies[sephirah_name] = freq
                    logger.debug(f"Loaded frequency for {sephirah_name}: {freq} Hz")
                else:
                    missing_aspects.append(sephirah_name)
                    logger.error(f"Invalid or non-positive base_frequency for {sephirah_name}.")
            else:
                missing_aspects.append(sephirah_name)
                logger.error(f"Could not load base_frequency for {sephirah_name}.")
        
        if missing_aspects:
            err_msg = f"Failed to load frequencies for: {', '.join(missing_aspects)}. Ensure aspect classes exist and have a positive 'base_frequency'."
            logger.error(err_msg)
            # Don't fail here to allow partial functionality, but warn severely
            logger.critical("Some Sephiroth frequencies could not be loaded. System may fail when accessing those Sephiroth.")
        
        # Default values for harmonic properties (Consider moving these to Aspect classes later)
        self.harmonic_counts = {
            "kether": 9, "chokmah": 8, "binah": 8,
            "chesed": 7, "geburah": 7, "tiphareth": 8,
            "netzach": 6, "hod": 6, "yesod": 5,
            "malkuth": 4, "daath": 8
        }
        self.phi_harmonic_counts = {
            "kether": 4, "chokmah": 3, "binah": 3,
            "chesed": 3, "geburah": 3, "tiphareth": 4,
            "netzach": 2, "hod": 2, "yesod": 2,
            "malkuth": 2, "daath": 3
        }
        self.harmonic_falloff = {
            "kether": 0.05, "chokmah": 0.08, "binah": 0.08,
            "chesed": 0.1, "geburah": 0.1, "tiphareth": 0.07,
            "netzach": 0.12, "hod": 0.12, "yesod": 0.15,
            "malkuth": 0.2, "daath": 0.08
        }
        
        # Relationship harmonics (ratios) - Keep these defined here for now
        self.relationship_harmonics = {
            ("chokmah", "chesed"): 1.5,
            ("chesed", "netzach"): 1.2,
            ("binah", "geburah"): 1.5,
            ("geburah", "hod"): 1.2,
            ("kether", "tiphareth"): PHI,
            ("tiphareth", "yesod"): 1.25,
            ("yesod", "malkuth"): 1.1,
            ("binah", "chokmah"): 1.05,
            ("geburah", "chesed"): 1.05,
            ("hod", "netzach"): 1.05,
            ("kether", "daath"): PHI / 1.2,
            ("chokmah", "daath"): 1.1,
            ("binah", "daath"): 1.1,
            ("daath", "tiphareth"): PHI / 1.1,
        }
        
        logger.info("Initialized Sephiroth frequencies using Aspect class definitions.")
    
    def get_frequency(self, sephirah_name):
        sephirah_name = sephirah_name.lower()
        freq = self.frequencies.get(sephirah_name)
        if freq is None:
            # Fail hard instead of using fallback
            error_msg = f"Frequency for {sephirah_name} not found. Check aspect definitions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return freq
    
    def get_harmonic_structure(self, sephirah):
        sephirah = sephirah.lower()
        
        # Check if the sephirah exists in our frequencies dictionary
        if sephirah not in self.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah} in get_harmonic_structure. Check aspect definitions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        primary_frequency = self.frequencies[sephirah]  # Will exist due to check above
        
        # Use .get with default values, not for fallback but for configuration flexibility
        harmonic_count = self.harmonic_counts.get(sephirah, 7)
        phi_harmonic_count = self.phi_harmonic_counts.get(sephirah, 3)
        falloff_rate = self.harmonic_falloff.get(sephirah, 0.1)
        
        return {
            'primary_frequency': primary_frequency,
            'harmonic_count': harmonic_count,
            'phi_harmonic_count': phi_harmonic_count,
            'falloff_rate': falloff_rate
        }
    
    def get_relationship_frequency(self, sephirah1, sephirah2):
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        # Verify both sephiroth exist
        freq1 = self.frequencies.get(sephirah1)
        freq2 = self.frequencies.get(sephirah2)
        
        if freq1 is None:
            error_msg = f"Unknown Sephirah: {sephirah1} in relationship. Check aspect definitions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if freq2 is None:
            error_msg = f"Unknown Sephirah: {sephirah2} in relationship. Check aspect definitions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        key1 = (sephirah1, sephirah2)
        key2 = (sephirah2, sephirah1)
        
        if key1 in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[key1]
            return freq1 * multiplier
        elif key2 in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[key2]
            return freq2 * multiplier
        else:
            # Geometric mean is a valid calculation, not a fallback
            return np.sqrt(freq1 * freq2)


class SephirothSoundIntegration:
    """
    Advanced generator for Sephiroth-specific sounds and tones.
    """
    
    def __init__(self, base_frequency=FUNDAMENTAL_FREQUENCY_432, output_dir=os.path.join(OUTPUT_DIR_BASE, "sephiroth")):
        """Initialize the Sephiroth sound generator."""
        # Attempt to initialize the frequency loader
        try:
            self.frequencies = SephirothFrequencies()
        except Exception as e:
            logger.critical(f"Failed to initialize SephirothFrequencies: {e}")
            raise RuntimeError(f"Cannot create SephirothSoundGenerator: {e}")
        
        # Initialize base SoundGenerator
        if SOUND_GENERATOR_AVAILABLE:
            try:
                self.sound_generator = SoundGenerator(output_dir=output_dir)
                self.sample_rate = self.sound_generator.sample_rate
                logger.info(f"SephirothSoundGenerator initialized with sample rate {self.sample_rate}")
            except Exception as e:
                logger.critical(f"Failed to initialize base SoundGenerator: {e}", exc_info=True)
                raise RuntimeError(f"Cannot create SephirothSoundGenerator: {e}")
        else:
            logger.critical("Base SoundGenerator class not available. Cannot initialize SephirothSoundGenerator.")
            raise RuntimeError("Base SoundGenerator class not available")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _is_generator_available(self, method_name):
        """Check if the base sound generator and a required method are available."""
        if self.sound_generator is None:
            error_msg = f"Cannot call {method_name}: Base SoundGenerator is not available."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not hasattr(self.sound_generator, method_name):
            error_msg = f"Base SoundGenerator is missing the required method: {method_name}"
            logger.error(error_msg)
            raise AttributeError(error_msg)
        
        return True
    
    def generate_sephirah_tone(self, sephirah, duration=10.0, include_field=False):
        """Generates tone using the loaded frequencies."""
        self._is_generator_available('generate_harmonic_tone')
        
        sephirah = sephirah.lower()
        
        try:
            structure = self.frequencies.get_harmonic_structure(sephirah)
            frequency = structure['primary_frequency']
            
            harmonics = self._get_sephirah_harmonic_ratios(sephirah)
            amplitudes = self._get_sephirah_amplitude_profile(sephirah, len(harmonics))
            
            tone = self.sound_generator.generate_harmonic_tone(
                base_frequency=frequency,
                harmonics=harmonics,
                duration=duration,
                amplitudes=amplitudes,
                fade_in_out=min(1.0, duration / 10)
            )
            
            if tone is None or len(tone) == 0:
                error_msg = f"generate_harmonic_tone failed for {sephirah}."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if include_field:
                field_tone = self._generate_field_characteristics(sephirah, duration)
                if field_tone is not None and len(field_tone) > 0:
                    target_len = len(tone)  # Ensure same length
                    if len(field_tone) > target_len:
                        field_tone = field_tone[:target_len]
                    else:
                        field_tone = np.pad(field_tone, (0, target_len - len(field_tone)))
                    tone = 0.7 * tone + 0.3 * field_tone  # Mix
            
            max_amp = np.max(np.abs(tone))
            if max_amp > 1e-6:
                tone = tone / max_amp * MAX_AMPLITUDE
            else:
                logger.warning(f"Very low amplitude for {sephirah} tone, normalizing may amplify noise")
                tone = np.zeros_like(tone)
            
            logger.info(f"Generated {sephirah} tone at {frequency:.2f}Hz")
            return tone
            
        except Exception as e:
            logger.error(f"Error generating {sephirah} tone: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate {sephirah} tone: {str(e)}")
    
    def _get_sephirah_harmonic_ratios(self, sephirah):
        sephirah = sephirah.lower()
        
        # Verify the sephirah exists
        if sephirah not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah} in _get_sephirah_harmonic_ratios"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        base_harmonics = [1.0, PHI, 2.0, PHI * 2, 3.0]
        harmonic_map = {
            "kether": [1.0, 1.5, PHI, 2.0, 3.0, 5.0, PHI * 3],
            "chokmah": [1.0, 1.5, 2.0, 3.0, PHI, PHI * 2],
            "binah": [1.0, 1.33, 1.66, 2.0, 2.5, PHI],
            "chesed": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            "geburah": [1.0, 1.1, 1.33, 1.66, 2.0, 2.5],
            "tiphareth": [1.0, PHI, 2.0, PHI * 2, 3.0, PHI * 3],
            "netzach": [1.0, 1.25, 1.5, 1.75, 2.0, PHI],
            "hod": [1.0, 1.2, 1.33, 1.66, 2.0, 2.5],
            "yesod": [1.0, 1.125, 1.25, 1.5, 2.0, PHI],
            "malkuth": [1.0, 1.125, 1.25, 1.33, 1.5],
            "daath": [1.0, 1.2, PHI, 2.0, 2.618, 3.0]
        }
        
        ratios = harmonic_map.get(sephirah)
        if ratios is None:
            logger.warning(f"No specific harmonic ratios defined for {sephirah}, using base harmonics")
            ratios = base_harmonics
            
        # Ensure fundamental is always included and ratios are positive
        if 1.0 not in ratios:
            ratios.insert(0, 1.0)
            
        return [r for r in ratios if isinstance(r, (int, float)) and r > 0]
    
    def _get_sephirah_amplitude_profile(self, sephirah, harmonic_count):
        sephirah = sephirah.lower()
        
        # Verify the sephirah exists
        if sephirah not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah} in _get_sephirah_amplitude_profile"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        structure = self.frequencies.get_harmonic_structure(sephirah)
        falloff = structure['falloff_rate']
        num_harmonics_defined = structure['harmonic_count']
        
        # Use count from structure
        num_to_generate = min(harmonic_count, num_harmonics_defined)
        
        # Start fundamental strong
        amplitudes = [MAX_AMPLITUDE * 0.8]
        
        if num_to_generate > 1:
            for i in range(1, num_to_generate):
                denominator = 1 + i * abs(falloff)
                if denominator < 1e-6:
                    denominator = 1e-6
                amp = amplitudes[0] / denominator
                amplitudes.append(max(0, min(amp, MAX_AMPLITUDE)))
        
        # Pad with zeros if requested harmonic_count is more than generated
        if len(amplitudes) < harmonic_count:
            amplitudes.extend([0.0] * (harmonic_count - len(amplitudes)))
            
        # Return exactly the number requested
        return amplitudes[:harmonic_count]
    
    def _generate_field_characteristics(self, sephirah, duration):
        self._is_generator_available('generate_harmonic_tone')
        
        sephirah = sephirah.lower()
        
        # Verify the sephirah exists
        if sephirah not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah} in _generate_field_characteristics"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            frequency = self.frequencies.get_frequency(sephirah)
            field_freq = frequency * 0.95
            
            num_samples = int(duration * self.sample_rate)
            if num_samples <= 0:
                raise ValueError(f"Invalid sample count: {num_samples} for duration {duration}")
                
            time = np.linspace(0, duration, num_samples, endpoint=False)
            field_tone = 0.5 * np.sin(2 * PI * field_freq * time)
            
            # Apply modulations based on Sephirah type
            if sephirah in ["kether", "chokmah", "binah", "daath"]:
                high_freq = field_freq * 4
                field_tone += 0.2 * np.sin(2 * PI * high_freq * time)
                mod_freq = 0.2
                field_tone *= (0.2 * np.sin(2 * PI * mod_freq * time) + 0.8)
                
            elif sephirah in ["chesed", "geburah", "tiphareth"]:
                phi_freq = field_freq * PHI
                field_tone += 0.3 * np.sin(2 * PI * phi_freq * time)
                pulse_freq = 0.25
                field_tone *= (0.3 * np.sin(2 * PI * pulse_freq * time) ** 2 + 0.7)
                
            else:  # Lower Sephiroth
                ground_freq = field_freq * 0.5
                field_tone += 0.4 * np.sin(2 * PI * ground_freq * time)
                wave_freq = 0.1
                field_tone *= (0.15 * np.sin(2 * PI * wave_freq * time) + 0.85)
            
            max_amp = np.max(np.abs(field_tone))
            if max_amp > 1e-6:
                field_tone = field_tone / max_amp * (MAX_AMPLITUDE * 0.7)
            else:
                logger.warning(f"Very low amplitude for {sephirah} field characteristics")
                field_tone = np.zeros_like(field_tone)
                
            return field_tone
            
        except Exception as e:
            logger.error(f"Error in _generate_field_characteristics for {sephirah}: {e}")
            raise RuntimeError(f"Failed to generate field characteristics for {sephirah}: {e}")
    
    def generate_path_sound(self, sephirah1, sephirah2, duration=15.0):
        self._is_generator_available('generate_dimensional_transition')
        
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        # Verify both sephiroth exist
        if sephirah1 not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah1} in generate_path_sound"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if sephirah2 not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah2} in generate_path_sound"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            path_freq = self.frequencies.get_relationship_frequency(sephirah1, sephirah2)
            freq1 = self.frequencies.get_frequency(sephirah1)
            freq2 = self.frequencies.get_frequency(sephirah2)
            
            num_samples = int(duration * self.sample_rate)
            if num_samples <= 0:
                raise ValueError(f"Invalid sample count: {num_samples} for duration {duration}")
                
            time = np.linspace(0, duration, num_samples, endpoint=False)
            t_norm = time / duration  # Normalized time [0, 1)
            
            # Components
            path_tone = 0.7 * np.sin(2 * PI * path_freq * time)
            
            # Linear frequency morph
            freq_morph = freq1 * (1 - t_norm) + freq2 * t_norm
            # Use cumulative phase
            morph_tone = 0.5 * np.sin(2 * PI * np.cumsum(freq_morph) / self.sample_rate)
            
            # Get approximate dimensional levels
            if sephirah1 in self.frequencies.sephiroth_names:
                s1_index = self.frequencies.sephiroth_names.index(sephirah1)
            else:
                error_msg = f"Sephirah {sephirah1} not found in sephiroth_names list"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if sephirah2 in self.frequencies.sephiroth_names:
                s2_index = self.frequencies.sephiroth_names.index(sephirah2)
            else:
                error_msg = f"Sephirah {sephirah2} not found in sephiroth_names list"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Map index 0-10 to a dimension range (e.g., 3-10)
            dim1 = 3 + (s1_index / 9.0) * 7.0
            dim2 = 3 + (s2_index / 9.0) * 7.0
            
            transition = self.sound_generator.generate_dimensional_transition(
                start_dimension=dim1,
                end_dimension=dim2,
                duration=duration,
                steps=50
            )
            
            # Mix
            path_sound = np.zeros(num_samples)
            if transition is not None and len(transition) == num_samples:
                path_sound = 0.3 * path_tone + 0.3 * morph_tone + 0.4 * transition
            else:
                logger.warning(f"Dimensional transition generation failed, using simplified path sound")
                path_sound = 0.5 * path_tone + 0.5 * morph_tone
            
            # Normalize
            max_amp = np.max(np.abs(path_sound))
            if max_amp > 1e-6:
                path_sound = path_sound / max_amp * MAX_AMPLITUDE
            else:
                logger.warning(f"Very low amplitude for path {sephirah1}-{sephirah2}")
                path_sound = np.zeros_like(path_sound)
            
            logger.info(f"Generated path sound {sephirah1}-{sephirah2}")
            return path_sound
            
        except Exception as e:
            logger.error(f"Error generating path sound {sephirah1}-{sephirah2}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate path sound {sephirah1}-{sephirah2}: {e}")
    
    def generate_tree_of_life_progression(self, duration_per_sephirah=5.0, include_paths=True, direction="ascending"):
        self._is_generator_available('generate_harmonic_tone')
        
        if direction.lower() == "ascending":
            progression_order = ["malkuth", "yesod", "hod", "netzach", "tiphareth", "geburah", "chesed", "binah", "chokmah", "kether"]
        elif direction.lower() == "descending":
            progression_order = ["kether", "chokmah", "binah", "chesed", "geburah", "tiphareth", "netzach", "hod", "yesod", "malkuth"]
        else:
            error_msg = f"Invalid direction '{direction}'. Use 'ascending' or 'descending'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if duration_per_sephirah <= 0:
            error_msg = "duration_per_sephirah must be positive."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Verify all sephiroth in the progression exist
            for sephirah in progression_order:
                if sephirah not in self.frequencies.frequencies:
                    error_msg = f"Sephirah {sephirah} in progression_order not found in frequencies"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            num_sephiroth = len(progression_order)
            path_duration = duration_per_sephirah * 0.4 if include_paths else 0  # Adjusted path duration
            total_duration = (num_sephiroth * duration_per_sephirah + (num_sephiroth - 1) * path_duration)
            total_samples = int(total_duration * self.sample_rate)
            
            if total_samples <= 0:
                error_msg = "Calculated total duration is zero or negative."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            progression_sound = np.zeros(total_samples)
            current_pos = 0
            
            for i, sephirah in enumerate(progression_order):
                # Generate Tone
                tone_samples = int(duration_per_sephirah * self.sample_rate)
                if tone_samples <= 0:
                    continue
                    
                tone = self.generate_sephirah_tone(sephirah, duration_per_sephirah, include_field=True)  # Include field for richness
                
                if tone is not None and len(tone) > 0:
                    start_idx = current_pos
                    end_idx = min(start_idx + len(tone), total_samples)
                    len_to_add = end_idx - start_idx
                    
                    if len_to_add > 0:
                        progression_sound[start_idx:end_idx] += tone[:len_to_add]
                    current_pos += tone_samples  # Advance by full intended duration
                else:
                    error_msg = f"Failed to generate tone for {sephirah}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                                
                # Add Path Sound if not the last sephirah
                if include_paths and i < num_sephiroth - 1:
                    next_sephirah = progression_order[i + 1]
                    path_samples = int(path_duration * self.sample_rate)
                    
                    if path_samples > 0:
                        path_sound = self.generate_path_sound(sephirah, next_sephirah, path_duration)
                        
                        if path_sound is not None and len(path_sound) > 0:
                            start_idx = current_pos
                            end_idx = min(start_idx + len(path_sound), total_samples)
                            len_to_add = end_idx - start_idx
                            
                            if len_to_add > 0:
                                progression_sound[start_idx:end_idx] += path_sound[:len_to_add]
                            current_pos += path_samples  # Advance by full intended duration
                        else:
                            error_msg = f"Failed to generate path sound between {sephirah} and {next_sephirah}."
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
            
            # Normalize the final sound
            max_amp = np.max(np.abs(progression_sound))
            if max_amp > 1e-6:
                progression_sound = progression_sound / max_amp * MAX_AMPLITUDE
            else:
                logger.warning(f"Very low amplitude for Tree of Life progression")
                progression_sound = np.zeros_like(progression_sound)
            
            logger.info(f"Generated Tree of Life progression in {direction} direction")
            return progression_sound
            
        except Exception as e:
            logger.error(f"Error generating Tree of Life progression: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate Tree of Life progression: {e}")
    
    def generate_middle_pillar_sound(self, duration_per_sephirah=5.0, include_paths=True):
        self._is_generator_available('generate_harmonic_tone')
        
        # Middle pillar order: Kether -> Daath -> Tiphareth -> Yesod -> Malkuth
        pillar_order = ["kether", "daath", "tiphareth", "yesod", "malkuth"]
        
        try:
            # Verify all sephiroth in the pillar exist
            for sephirah in pillar_order:
                if sephirah not in self.frequencies.frequencies:
                    error_msg = f"Sephirah {sephirah} in middle pillar not found in frequencies"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Similar to generate_tree_of_life_progression but with middle pillar order
            num_sephiroth = len(pillar_order)
            path_duration = duration_per_sephirah * 0.3 if include_paths else 0
            total_duration = (num_sephiroth * duration_per_sephirah + (num_sephiroth - 1) * path_duration)
            total_samples = int(total_duration * self.sample_rate)
            
            if total_samples <= 0:
                error_msg = "Calculated total duration is zero or negative."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            pillar_sound = np.zeros(total_samples)
            current_pos = 0
            
            for i, sephirah in enumerate(pillar_order):
                # Generate Tone
                tone_samples = int(duration_per_sephirah * self.sample_rate)
                if tone_samples <= 0:
                    continue
                    
                tone = self.generate_sephirah_tone(sephirah, duration_per_sephirah, include_field=True)
                
                if tone is not None and len(tone) > 0:
                    start_idx = current_pos
                    end_idx = min(start_idx + len(tone), total_samples)
                    len_to_add = end_idx - start_idx
                    
                    if len_to_add > 0:
                        pillar_sound[start_idx:end_idx] += tone[:len_to_add]
                    current_pos += tone_samples
                else:
                    error_msg = f"Failed to generate tone for {sephirah}."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Add Path Sound if not the last sephirah
                if include_paths and i < num_sephiroth - 1:
                    next_sephirah = pillar_order[i + 1]
                    path_samples = int(path_duration * self.sample_rate)
                    
                    if path_samples > 0:
                        path_sound = self.generate_path_sound(sephirah, next_sephirah, path_duration)
                        
                        if path_sound is not None and len(path_sound) > 0:
                            start_idx = current_pos
                            end_idx = min(start_idx + len(path_sound), total_samples)
                            len_to_add = end_idx - start_idx
                            
                            if len_to_add > 0:
                                pillar_sound[start_idx:end_idx] += path_sound[:len_to_add]
                            current_pos += path_samples
                        else:
                            error_msg = f"Failed to generate path sound between {sephirah} and {next_sephirah}."
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
            
            # Normalize the final sound
            max_amp = np.max(np.abs(pillar_sound))
            if max_amp > 1e-6:
                pillar_sound = pillar_sound / max_amp * MAX_AMPLITUDE
            else:
                logger.warning(f"Very low amplitude for Middle Pillar sound")
                pillar_sound = np.zeros_like(pillar_sound)
            
            logger.info(f"Generated Middle Pillar sound")
            return pillar_sound
            
        except Exception as e:
            logger.error(f"Error generating Middle Pillar sound: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate Middle Pillar sound: {e}")
    
    def generate_sephiroth_chord(self, sephiroth_list, duration=10.0, normalize=True):
        self._is_generator_available('generate_sacred_chord')
        
        if not sephiroth_list:
            error_msg = "sephiroth_list cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Verify all sephiroth in the list exist
            frequencies = []
            for sephirah in sephiroth_list:
                sephirah = sephirah.lower()
                if sephirah not in self.frequencies.frequencies:
                    error_msg = f"Sephirah {sephirah} in sephiroth_list not found in frequencies"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                frequencies.append(self.frequencies.get_frequency(sephirah))
            
            # Generate sacred chord from frequencies
            chord = self.sound_generator.generate_sacred_chord(
                base_frequency=frequencies[0],  # Use the first frequency if it's a list
                duration=duration,
                fade_in_out=min(1.0, duration / 10)
            )

            if chord is None or len(chord) == 0:
                error_msg = "Failed to generate sacred chord."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if normalize:
                max_amp = np.max(np.abs(chord))
                if max_amp > 1e-6:
                    chord = chord / max_amp * MAX_AMPLITUDE
                else:
                    logger.warning(f"Very low amplitude for Sephiroth chord")
                    chord = np.zeros_like(chord)
            
            logger.info(f"Generated Sephiroth chord with {len(sephiroth_list)} frequencies")
            return chord
        
        except Exception as e:
            logger.error(f"Error generating Sephiroth chord: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate Sephiroth chord: {e}")
    
    def generate_gateway_sound(self, gateway_key, duration=15.0):
        self._is_generator_available('generate_harmonic_tone')
        
        gateway_key = gateway_key.lower()
        recognized_gateways = ["tetrahedron", "octahedron", "hexahedron", "icosahedron", "dodecahedron"]
        
        if gateway_key not in recognized_gateways:
            error_msg = f"Unrecognized gateway key: {gateway_key}. Valid keys: {', '.join(recognized_gateways)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Define gateway to sephiroth mappings
            GATEWAY_SEPHIROTH_MAP = {
                "tetrahedron": ["geburah", "chesed", "netzach", "hod"],
                "octahedron": ["kether", "tiphareth", "yesod", "malkuth"],
                "hexahedron": ["binah", "chesed", "netzach", "malkuth"],
                "icosahedron": ["chokmah", "tiphareth", "hod", "yesod"],
                "dodecahedron": ["kether", "chokmah", "binah", "tiphareth", "daath"]
            }
            
            # Get sephiroth associated with this gateway
            gateway_sephiroth = GATEWAY_SEPHIROTH_MAP.get(gateway_key)
            
            if not gateway_sephiroth:
                error_msg = f"No Sephiroth associated with gateway key: {gateway_key}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Verify all sephiroth in the gateway exist
            for sephirah in gateway_sephiroth:
                if sephirah not in self.frequencies.frequencies:
                    error_msg = f"Sephirah {sephirah} in gateway_sephiroth not found in frequencies"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Base frequency calculation
            gateway_frequencies = [self.frequencies.get_frequency(s) for s in gateway_sephiroth]
            
            # Geometric means are less stable when sequence length varies
            # Use root of product instead for consistent gateway sound character
            base_freq = np.prod(gateway_frequencies) ** (1.0 / len(gateway_frequencies))
            
            # Generate component sounds
            num_samples = int(duration * self.sample_rate)
            gateway_sound = np.zeros(num_samples)
            
            # Harmonic-rich base tone
            time = np.linspace(0, duration, num_samples, endpoint=False)
            
            # Add gateway-specific modulations
            if gateway_key == "tetrahedron":  # Fire
                # Generate fire-like crackling and movement
                carrier = 0.7 * np.sin(2 * PI * base_freq * time)
                modulator = np.sin(2 * PI * 0.3 * time) + 0.3 * np.sin(2 * PI * 1.7 * time)
                modulated = carrier * (0.7 + 0.3 * modulator)
                
                # Add higher harmonics for fire quality
                fire_harmonics = [1.5, 2.0, 3.0, 5.0]
                for h in fire_harmonics:
                    h_amp = 0.3 / h
                    random_phase = np.random.rand() * 2 * PI
                    gateway_sound += h_amp * np.sin(2 * PI * base_freq * h * time + random_phase)
                
                gateway_sound += 0.7 * modulated
                
            elif gateway_key == "octahedron":  # Air
                # Airy, flowing quality with subtle modulations
                carrier = 0.6 * np.sin(2 * PI * base_freq * time)
                
                # Flowing air modulations
                flow_mod = 0.4 * np.sin(2 * PI * 0.1 * time) + 0.2 * np.sin(2 * PI * 0.23 * time)
                
                # Add breathy harmonics
                air_harmonics = [1.01, 1.5, 2.0, 4.0]
                for i, h in enumerate(air_harmonics):
                    h_amp = 0.2 / (i + 1)
                    # Slight frequency drift for air movement
                    drift = 1.0 + 0.01 * np.sin(2 * PI * 0.07 * time)
                    gateway_sound += h_amp * np.sin(2 * PI * base_freq * h * drift * time)
                
                gateway_sound += carrier * (0.8 + 0.2 * flow_mod)
                
            elif gateway_key == "hexahedron":  # Earth
                # Solid, stable quality with deep resonance
                carrier = 0.8 * np.sin(2 * PI * base_freq * time)
                
                # Add deep earth rumble
                rumble = 0.3 * np.sin(2 * PI * base_freq * 0.5 * time)
                
                # Stable harmonics
                earth_harmonics = [1.25, 1.5, 2.0]
                for h in earth_harmonics:
                    h_amp = 0.4 / h
                    gateway_sound += h_amp * np.sin(2 * PI * base_freq * h * time)
                
                gateway_sound += carrier + rumble
                
            elif gateway_key == "icosahedron":  # Water
                # Flowing, fluid quality
                carrier = 0.6 * np.sin(2 * PI * base_freq * time)
                
                # Add water-like flow modulations
                flow_speed = 0.15 + 0.1 * np.sin(2 * PI * 0.05 * time)
                flow_mod = 0.4 * np.sin(2 * PI * flow_speed * time)
                
                # Add harmonics following golden ratio for water's natural patterns
                water_harmonics = [1.0, PHI, PHI**2, PHI**3]
                for i, h in enumerate(water_harmonics):
                    h_amp = 0.35 / (i + 1)
                    phase_mod = 0.2 * np.sin(2 * PI * 0.1 * time)
                    gateway_sound += h_amp * np.sin(2 * PI * base_freq * h * time + phase_mod)
                
                gateway_sound += carrier * (0.7 + 0.3 * flow_mod)
                
            elif gateway_key == "dodecahedron":  # Aether/Spirit
                # Ethereal, otherworldly quality
                carrier = 0.5 * np.sin(2 * PI * base_freq * time)
                
                # Subtle, complex modulations
                phase_shift = np.cumsum(0.01 * np.sin(2 * PI * 0.05 * time)) / self.sample_rate
                
                # Phi-based harmonics for transcendent quality
                spirit_harmonics = [1.0, PHI, PHI**2, 2*PHI, PHI**3]
                for i, h in enumerate(spirit_harmonics):
                    h_amp = 0.4 / (i + 1)
                    gateway_sound += h_amp * np.sin(2 * PI * base_freq * h * time + phase_shift)
                
                # Add subtle choir-like effect
                choir = 0.3 * np.sin(2 * PI * base_freq * 1.5 * time) * np.sin(2 * PI * 0.2 * time)
                gateway_sound += carrier + choir
            
            # Add chord-like combination of all gateway sephiroth (subtle)
            chord = self.generate_sephiroth_chord(gateway_sephiroth, duration, normalize=False)
            if chord is not None and len(chord) == len(gateway_sound):
                gateway_sound += 0.3 * chord
            
            # Apply fade-in/out
            fade_len = int(min(duration * 0.15, 2.0) * self.sample_rate)
            if fade_len > 0 and len(gateway_sound) > 2 * fade_len:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                gateway_sound[:fade_len] *= fade_in
                gateway_sound[-fade_len:] *= fade_out
            
            # Normalize
            max_amp = np.max(np.abs(gateway_sound))
            if max_amp > 1e-6:
                gateway_sound = gateway_sound / max_amp * MAX_AMPLITUDE
            else:
                logger.warning(f"Very low amplitude for gateway sound: {gateway_key}")
                gateway_sound = np.zeros_like(gateway_sound)
            
            logger.info(f"Generated gateway sound for {gateway_key} with {len(gateway_sephiroth)} sephiroth")
            return gateway_sound
            
        except Exception as e:
            logger.error(f"Error generating gateway sound for {gateway_key}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate gateway sound for {gateway_key}: {e}")
    
    def save_sephirah_tone(self, sephirah, duration=10.0, include_field=True):
        """Generates and saves a tone for a specific Sephirah."""
        sephirah = sephirah.lower()
        
        # Verify the sephirah exists
        if sephirah not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah} in save_sephirah_tone"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            tone = self.generate_sephirah_tone(sephirah, duration, include_field)
            
            if tone is None or len(tone) == 0:
                error_msg = f"Failed to generate tone for {sephirah}."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sephirah}_tone_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the tone
            self.sound_generator.save_sound(tone, filepath, self.sample_rate)
            logger.info(f"Saved {sephirah} tone to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving {sephirah} tone: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save {sephirah} tone: {e}")
    
    def save_path_sound(self, sephirah1, sephirah2, duration=15.0):
        """Generates and saves a path sound between two Sephiroth."""
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        # Verify both sephiroth exist
        if sephirah1 not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah1} in save_path_sound"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if sephirah2 not in self.frequencies.frequencies:
            error_msg = f"Unknown Sephirah: {sephirah2} in save_path_sound"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            path_sound = self.generate_path_sound(sephirah1, sephirah2, duration)
            
            if path_sound is None or len(path_sound) == 0:
                error_msg = f"Failed to generate path sound between {sephirah1} and {sephirah2}."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"path_{sephirah1}_to_{sephirah2}_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the path sound
            self.sound_generator.save_sound(path_sound, filepath, self.sample_rate)
            logger.info(f"Saved path sound {sephirah1}-{sephirah2} to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving path sound {sephirah1}-{sephirah2}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save path sound {sephirah1}-{sephirah2}: {e}")
    
    def save_gateway_sound(self, gateway_key, duration=15.0):
        """Generates and saves a gateway sound."""
        gateway_key = gateway_key.lower()
        recognized_gateways = ["tetrahedron", "octahedron", "hexahedron", "icosahedron", "dodecahedron"]
        
        if gateway_key not in recognized_gateways:
            error_msg = f"Unrecognized gateway key: {gateway_key}. Valid keys: {', '.join(recognized_gateways)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            gateway_sound = self.generate_gateway_sound(gateway_key, duration)
            
            if gateway_sound is None or len(gateway_sound) == 0:
                error_msg = f"Failed to generate gateway sound for {gateway_key}."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gateway_{gateway_key}_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the gateway sound
            self.sound_generator.save_sound(gateway_sound, filepath, self.sample_rate)
            logger.info(f"Saved gateway sound for {gateway_key} to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving gateway sound for {gateway_key}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save gateway sound for {gateway_key}: {e}")
    
    def get_field_frequency_map(self):
        """Returns a mapping of Sephiroth names to their field frequencies."""
        field_map = {}
        
        for sephirah in self.frequencies.sephiroth_names:
            try:
                freq = self.frequencies.get_frequency(sephirah)
                field_map[sephirah] = {
                    'primary_frequency': freq,
                    'field_frequency': freq * 0.95,  # Field frequency is slightly lower
                    'harmonic_structure': self.frequencies.get_harmonic_structure(sephirah)
                }
            except Exception as e:
                logger.error(f"Error getting frequency data for {sephirah}: {e}")
                # Don't include this sephirah in the map if we can't get its frequency
        
        return field_map


def get_sound_generator():
    """Factory function to get a SephirothSoundGenerator instance."""
    try:
        return SoundGenerator()
    except Exception as e:
        logger.critical(f"Failed to create SephirothSoundGenerator: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create SephirothSoundGenerator: {e}")


# Example usage
if __name__ == "__main__":
    try:
        generator = SephirothSoundIntegration()
        
        # Generate and save a Sephirah tone
        generator.save_sephirah_tone("tiphareth", duration=10.0)
        
        # Generate and save a path sound
        generator.save_path_sound("tiphareth", "netzach", duration=8.0)
        
        # Generate and save a gateway sound
        generator.save_gateway_sound("tetrahedron", duration=12.0)
        
        print("Sound generation complete. Check output directory for WAV files.")
        
    except Exception as e:
        print(f"Error in sound generation: {e}")
        logger.critical(f"Error in main execution: {e}", exc_info=True)


