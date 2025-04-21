"""
Sounds of Sephiroth Module

This module implements specialized sound generation for the Sephiroth dimensions.
It creates specific tones, frequencies, and harmonic structures for each Sephiroth
based on their metaphysical properties and position in the Tree of Life.

This module extends the base sound_generator.py with Sephiroth-specific
functionality, allowing for more detailed and accurate sonic representations.

Author: Soul Development Framework Team
"""
from sound.sound_generator import SoundGenerator, FUNDAMENTAL_FREQUENCY, PHI
import numpy as np
import logging
import os
import sys
from datetime import datetime
import types # Added for method injection later

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sound generator
try:
    from sound.sound_generator import SoundGenerator, FUNDAMENTAL_FREQUENCY, PHI
except ImportError:
    logging.warning("sound_generator.py not available. Sephiroth sounds will not function correctly.")
    # Define fallbacks if import fails
    class SoundGenerator: # Minimal placeholder
        def __init__(self, output_dir="output/sounds"):
            self.output_dir = output_dir
            self.sample_rate = 44100 # Default sample rate
            os.makedirs(output_dir, exist_ok=True)
            logging.warning("Using placeholder SoundGenerator class.")
        def generate_harmonic_tone(self, *args, **kwargs): return None
        def generate_dimensional_transition(self, *args, **kwargs): return None
        def generate_sacred_chord(self, *args, **kwargs): return None
        def save_sound(self, *args, **kwargs): return None

    FUNDAMENTAL_FREQUENCY = 432.0
    PHI = 1.618033988749895
    SoundGenerator = None # Explicitly set to None if import failed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_sounds.log'
)
logger = logging.getLogger('sephiroth_sounds')

class SephirothFrequencies:
    """
    Class containing the frequency properties for each Sephiroth.

    These frequencies are derived from traditional correspondences and
    the spiritual properties of each Sephiroth, scaled from a base frequency.
    """

    def __init__(self, base_frequency=FUNDAMENTAL_FREQUENCY):
        """
        Initialize Sephiroth frequencies based on a given base frequency.

        Args:
            base_frequency (float): Base frequency to derive Sephiroth frequencies from
        """
        self.base_frequency = base_frequency

        # Define frequency modifiers for each Sephiroth
        # These represent the relative frequency compared to the Creator/Kether
        # Higher Sephiroth have higher frequencies as they are closer to divine source
        self.frequency_modifiers = {
            # Higher triad
            "kether": 1.0,       # Crown - highest/purest frequency
            "chokmah": 0.95,     # Wisdom
            "binah": 0.9,        # Understanding

            # Middle triad
            "chesed": 0.85,      # Mercy
            "geburah": 0.8,      # Severity
            "tiphareth": 0.75,   # Beauty - central harmonizing Sephiroth

            # Lower triad
            "netzach": 0.7,      # Victory
            "hod": 0.65,         # Glory
            "yesod": 0.6,        # Foundation

            # Material realm
            "malkuth": 0.55      # Kingdom - lowest frequency, most dense
        }

        # Define harmonic structures for each Sephiroth
        self.harmonic_counts = {
            "kether": 12,      # Most complex harmonics (divine completeness)
            "chokmah": 10,
            "binah": 10,
            "chesed": 9,
            "geburah": 8,
            "tiphareth": 10,    # Rich harmonics (central balancing point)
            "netzach": 7,
            "hod": 7,
            "yesod": 6,
            "malkuth": 4       # Simplest harmonic structure (material simplicity)
        }

        # Define phi-harmonic counts (harmonics based on golden ratio)
        self.phi_harmonic_counts = {
            "kether": 7,       # Highest phi-connection (divine proportion)
            "chokmah": 6,
            "binah": 6,
            "chesed": 5,
            "geburah": 4,
            "tiphareth": 6,    # Strong phi-resonance (balanced beauty)
            "netzach": 3,
            "hod": 3,
            "yesod": 2,
            "malkuth": 1       # Minimal phi-connection (material density)
        }

        # Amplitude falloff rates for harmonics
        # Higher values mean harmonics diminish more quickly in amplitude
        self.harmonic_falloff = {
            "kether": 0.05,    # Sustained harmonics (divine persistence)
            "chokmah": 0.06,
            "binah": 0.07,
            "chesed": 0.08,
            "geburah": 0.1,
            "tiphareth": 0.07, # Balanced falloff
            "netzach": 0.12,
            "hod": 0.12,
            "yesod": 0.15,
            "malkuth": 0.2     # Rapid falloff (material density absorbs)
        }

        # Calculate the actual frequencies based on modifiers
        self.frequencies = {
            sephirah: self.base_frequency * modifier
            for sephirah, modifier in self.frequency_modifiers.items()
        }

        # Add special frequency relationships for specific pairs
        # These reflect metaphysical relationships between Sephiroth pairs
        self.relationship_harmonics = {
            # Pillar of Mercy relationships
            ("chokmah", "chesed"): 1.5,     # Wisdom to Mercy
            ("chesed", "netzach"): 1.2,     # Mercy to Victory

            # Pillar of Severity relationships
            ("binah", "geburah"): 1.5,     # Understanding to Severity
            ("geburah", "hod"): 1.2,       # Severity to Glory

            # Middle Pillar relationships
            ("kether", "tiphareth"): PHI,  # Crown to Beauty (phi)
            ("tiphareth", "yesod"): 1.25,           # Beauty to Foundation
            ("yesod", "malkuth"): 1.1,              # Foundation to Kingdom

            # Horizontal relationships
            ("binah", "chokmah"): 1.05,        # Understanding to Wisdom
            ("geburah", "chesed"): 1.05,       # Severity to Mercy
            ("hod", "netzach"): 1.05,          # Glory to Victory
        }

        logger.info(f"Initialized Sephiroth frequencies with base {base_frequency} Hz")

    def get_frequency(self, sephirah_name):
        """
        Get the primary frequency for a specific Sephiroth.

        Args:
            sephirah_name (str): The name of the Sephiroth (lowercase)

        Returns:
            float: The primary frequency in Hz
        """
        sephirah_name = sephirah_name.lower()
        if sephirah_name in self.frequencies:
            return self.frequencies[sephirah_name]
        else:
            logger.warning(f"Unknown Sephirah: {sephirah}, returning base frequency")
            return self.base_frequency

    def get_harmonic_structure(self, sephirah):
        """
        Get the complete harmonic structure for a Sephiroth.

        Args:
            sephirah (str): The name of the Sephiroth (lowercase)

        Returns:
            dict: Dictionary with harmonic structure information
        """
        sephirah = sephirah.lower()

        if sephirah not in self.frequencies:
            logger.warning(f"Unknown Sephirah: {sephirah}, returning default structure")
            return {
                'primary_frequency': self.base_frequency,
                'harmonic_count': 7,
                'phi_harmonic_count': 3,
                'falloff_rate': 0.1
            }

        return {
            'primary_frequency': self.frequencies[sephirah],
            'harmonic_count': self.harmonic_counts[sephirah],
            'phi_harmonic_count': self.phi_harmonic_counts[sephirah],
            'falloff_rate': self.harmonic_falloff[sephirah]
        }

    def get_relationship_frequency(self, sephirah1, sephirah2):
        """
        Get the relationship frequency between two Sephiroth.

        Args:
            sephirah1 (str): First Sephiroth name
            sephirah2 (str): Second Sephiroth name

        Returns:
            float: Relationship frequency in Hz
        """
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()

        # Check direct relationship
        if (sephirah1, sephirah2) in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[(sephirah1, sephirah2)]
            return self.frequencies[sephirah1] * multiplier

        # Check reverse relationship
        if (sephirah2, sephirah1) in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[(sephirah2, sephirah1)]
            # Use freq of the *first* element in the tuple key for consistency
            return self.frequencies[sephirah2] * multiplier

        # If no direct relationship, use geometric mean of frequencies
        if sephirah1 in self.frequencies and sephirah2 in self.frequencies:
            # Ensure frequencies are positive before sqrt
            freq1 = self.frequencies[sephirah1]
            freq2 = self.frequencies[sephirah2]
            if freq1 > 0 and freq2 > 0:
                return np.sqrt(freq1 * freq2)
            else:
                 logger.warning(f"Non-positive frequency for {sephirah1} or {sephirah2}, returning base")
                 return self.base_frequency

        logger.warning(f"Unknown Sephiroth relationship: {sephirah1}-{sephirah2}")
        return self.base_frequency


class SephirothSoundGenerator:
    """
    Advanced generator for Sephiroth-specific sounds and tones.

    This extends the base SoundGenerator with specialized functionality
    for creating sounds that represent the Sephiroth dimensions and their
    unique metaphysical properties.
    """

    def __init__(self, base_frequency=FUNDAMENTAL_FREQUENCY, output_dir="output/sounds/sephiroth"):
        """
        Initialize the Sephiroth sound generator.

        Args:
            base_frequency (float): Base frequency for sound generation
            output_dir (str): Directory for saving generated sounds
        """
        self.frequencies = SephirothFrequencies(base_frequency)

        # Create sound generator if available
        if SoundGenerator is not None:
             try:
                 # Re-attempt import inside __init__ in case it became available
                 from sound.sound_generator import SoundGenerator as ActualSoundGenerator
                 self.sound_generator = ActualSoundGenerator(output_dir=output_dir)
                 logger.info("Actual Sound generator initialized for Sephiroth sounds")
             except ImportError:
                 self.sound_generator = None
                 logger.error("SoundGenerator not available. Sephiroth sounds will not function.")
        else:
            self.sound_generator = None # Was already None from initial check
            logger.error("SoundGenerator not available. Sephiroth sounds will not function.")

        # Ensure output directory exists
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_sephirah_tone(self, sephirah, duration=10.0, include_field=False):
        """
        Generate a detailed audio tone representing a specific Sephiroth.

        Args:
            sephirah (str): The name of the Sephiroth
            duration (float): Duration in seconds
            include_field (bool): Whether to include field harmonics

        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot generate Sephiroth tone: SoundGenerator not available")
            return None

        sephirah = sephirah.lower()

        # Get specific harmonic structure for this Sephiroth
        structure = self.frequencies.get_harmonic_structure(sephirah)
        frequency = structure['primary_frequency']

        # Define harmonic ratios for this Sephiroth
        harmonics = self._get_sephirah_harmonic_ratios(sephirah)

        # Define amplitude profile
        amplitudes = self._get_sephirah_amplitude_profile(sephirah, len(harmonics))

        # Generate the tone
        try:
            tone = self.sound_generator.generate_harmonic_tone(
                base_frequency=frequency,
                harmonics=harmonics,
                durations=[duration] * len(harmonics), # Assuming generate_harmonic_tone takes list of durations
                amplitudes=amplitudes,
                fade_in_out=min(1.0, duration / 10)  # 10% fade in/out
            )

            if tone is None: # Check if base tone generation failed
                 logger.error(f"Base tone generation failed for {sephirah}")
                 return None

            # If including field harmonics, add field characteristics
            if include_field:
                field_tone = self._generate_field_characteristics(sephirah, duration)
                if field_tone is not None:
                    # Ensure field_tone has the same length as tone
                    if len(field_tone) != len(tone):
                         # Resize field_tone (simple truncation/padding)
                         target_len = len(tone)
                         if len(field_tone) > target_len:
                             field_tone = field_tone[:target_len]
                         else:
                             field_tone = np.pad(field_tone, (0, target_len - len(field_tone)))

                    # Mix 70% tone, 30% field
                    tone = 0.7 * tone + 0.3 * field_tone

                    # Normalize
                    max_amp = np.max(np.abs(tone))
                    if max_amp > 1e-6: # Avoid division by zero/very small numbers
                        tone = tone / max_amp * 0.8  # Keep some headroom
                    else:
                         tone = np.zeros_like(tone) # Set to zeros if amplitude is negligible


            logger.info(f"Generated {sephirah} tone at {frequency:.2f}Hz with {len(harmonics)} harmonics")
            return tone

        except Exception as e:
            logger.error(f"Error generating {sephirah} tone: {str(e)}", exc_info=True)
            return None

    def _get_sephirah_harmonic_ratios(self, sephirah):
        """
        Get the specific harmonic ratios for a Sephiroth.

        Args:
            sephirah (str): Sephiroth name

        Returns:
            list: Harmonic ratios
        """
        sephirah = sephirah.lower()
        # Base harmonic ratios (used if sephirah name is unknown)
        base_harmonics = [1.0, PHI, 2.0, PHI * 2, 3.0]

        # Each Sephiroth has unique additional harmonics
        if sephirah == "kether":
            # Crown - highest divine connection
            return [1.0, 1.5, PHI, 2.0, 3.0, 5.0, PHI * 3]

        elif sephirah == "chokmah":
            # Wisdom - masculine energy
            return [1.0, 1.5, 2.0, 3.0, PHI, PHI * 2]

        elif sephirah == "binah":
            # Understanding - feminine energy
            return [1.0, 1.33, 1.66, 2.0, 2.5, PHI]

        elif sephirah == "chesed":
            # Mercy - expansive energy
            return [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

        elif sephirah == "geburah":
            # Severity - restrictive energy
            return [1.0, 1.1, 1.33, 1.66, 2.0, 2.5]

        elif sephirah == "tiphareth":
            # Beauty - balance and harmony
            return [1.0, PHI, 2.0, PHI * 2, 3.0, PHI * 3]

        elif sephirah == "netzach":
            # Victory - emotional energy
            return [1.0, 1.25, 1.5, 1.75, 2.0, PHI]

        elif sephirah == "hod":
            # Glory - intellectual energy
            return [1.0, 1.2, 1.33, 1.66, 2.0, 2.5]

        elif sephirah == "yesod":
            # Foundation - dream energy
            return [1.0, 1.125, 1.25, 1.5, 2.0, PHI]

        elif sephirah == "malkuth":
            # Kingdom - material energy
            return [1.0, 1.125, 1.25, 1.33, 1.5]

        # Default case
        logger.warning(f"Unknown sephirah '{sephirah}' in _get_sephirah_harmonic_ratios, using default.")
        return base_harmonics

    def _get_sephirah_amplitude_profile(self, sephirah, harmonic_count):
        """
        Get amplitude profile for a Sephiroth's harmonics.

        Args:
            sephirah (str): Sephiroth name
            harmonic_count (int): Number of harmonics

        Returns:
            list: Amplitude values
        """
        sephirah = sephirah.lower()
        # Start with the fundamental at significant amplitude
        amplitudes = [0.8]

        # Get falloff rate from frequency structure
        structure = self.frequencies.get_harmonic_structure(sephirah) # Handles unknown sephirah
        falloff = structure['falloff_rate']

        # Generate rest of amplitudes with appropriate falloff
        if harmonic_count > 1:
            for i in range(1, harmonic_count):
                 # Ensure denominator is not zero and falloff is reasonable
                denominator = 1 + i * abs(falloff)
                if denominator < 1e-6: denominator = 1e-6 # Prevent division by zero
                amp = 0.8 / denominator
                amplitudes.append(max(0, min(amp, 1.0))) # Clamp amplitude between 0 and 1

        return amplitudes

    def _generate_field_characteristics(self, sephirah, duration):
        """
        Generate field-specific characteristics for a Sephiroth.

        Args:
            sephirah (str): Sephiroth name
            duration (float): Duration in seconds

        Returns:
            numpy.ndarray: Field characteristics waveform or None
        """
        if self.sound_generator is None or not hasattr(self.sound_generator, 'sample_rate'):
            logger.error("Cannot generate field characteristics: SoundGenerator or sample_rate not available")
            return None

        sephirah = sephirah.lower()
        try:
            # Get frequency
            frequency = self.frequencies.get_frequency(sephirah) # Handles unknown sephirah

            # Field frequencies are slightly lower
            field_freq = frequency * 0.95
            if field_freq <= 0:
                 logger.warning(f"Calculated non-positive field frequency for {sephirah}, skipping.")
                 return None

            # Create time array
            sample_rate = self.sound_generator.sample_rate
            num_samples = int(duration * sample_rate)
            if num_samples <= 0:
                logger.warning("Duration results in zero samples, skipping field generation.")
                return None
            time = np.linspace(0, duration, num_samples, endpoint=False)

            # Base field tone
            field_tone = 0.5 * np.sin(2 * np.pi * field_freq * time)

            # Add field-specific modulations
            if sephirah in ["kether", "chokmah", "binah"]:
                # Higher Sephiroth have ethereal, high-frequency components
                high_freq = field_freq * 4
                field_tone += 0.2 * np.sin(2 * np.pi * high_freq * time)

                # Add subtle amplitude modulation
                mod_freq = 0.2  # 5-second cycle
                field_tone *= (0.2 * np.sin(2 * np.pi * mod_freq * time) + 0.8)

            elif sephirah in ["chesed", "geburah", "tiphareth"]:
                # Middle Sephiroth have balanced, resonant qualities
                field_tone += 0.3 * np.sin(2 * np.pi * field_freq * PHI * time)

                # Add rhythmic pulsing
                pulse_freq = 0.25  # 4-second cycle
                field_tone *= (0.3 * np.sin(2 * np.pi * pulse_freq * time) ** 2 + 0.7)

            else: # Includes Malkuth, Yesod, Hod, Netzach and unknowns
                # Lower Sephiroth have more grounded, stable qualities
                ground_freq = field_freq * 0.5
                if ground_freq > 0: # Ensure positive frequency
                    field_tone += 0.4 * np.sin(2 * np.pi * ground_freq * time)

                # Add gentle wave motion
                wave_freq = 0.1  # 10-second cycle
                field_tone *= (0.15 * np.sin(2 * np.pi * wave_freq * time) + 0.85)

            # Normalize
            max_amp = np.max(np.abs(field_tone))
            if max_amp > 1e-6:
                field_tone = field_tone / max_amp * 0.8
            else:
                 field_tone = np.zeros_like(field_tone)

            return field_tone

        except Exception as e:
            logger.error(f"Error generating field characteristics for {sephirah}: {str(e)}", exc_info=True)
            return None

    def generate_path_sound(self, sephirah1, sephirah2, duration=15.0):
        """
        Generate a sound for a path between two Sephiroth.

        These paths represent transitional sounds between Sephiroth dimensions.

        Args:
            sephirah1 (str): First Sephiroth name
            sephirah2 (str): Second Sephiroth name
            duration (float): Duration in seconds

        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None or not hasattr(self.sound_generator, 'sample_rate'):
            logger.error("Cannot generate path sound: SoundGenerator or sample_rate not available")
            return None

        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()

        try:
            # Get relationship frequency
            path_freq = self.frequencies.get_relationship_frequency(sephirah1, sephirah2)

            # Get individual frequencies
            freq1 = self.frequencies.get_frequency(sephirah1)
            freq2 = self.frequencies.get_frequency(sephirah2)

            if path_freq <= 0 or freq1 <= 0 or freq2 <= 0:
                 logger.warning(f"Non-positive frequency encountered for path {sephirah1}-{sephirah2}, skipping.")
                 return None

            # Create time array
            sample_rate = self.sound_generator.sample_rate
            num_samples = int(duration * sample_rate)
            if num_samples <= 0:
                logger.warning("Duration results in zero samples, skipping path generation.")
                return None
            time = np.linspace(0, duration, num_samples, endpoint=False)


            # --- Path Sound Components ---

            # 1. Dimensional Transition (if available)
            # This assumes generate_dimensional_transition produces a relevant effect
            transition = self.sound_generator.generate_dimensional_transition(
                start_dimension=0,  # Placeholder, actual meaning depends on SoundGenerator impl.
                end_dimension=0,    # Placeholder
                duration=duration,
                steps=50            # Example parameter
            )

            # 2. Morphing Frequency Tone
            t = np.linspace(0, 1, num_samples) # Morphing factor
            freq_morph = freq1 * (1 - t) + freq2 * t
            morph_tone = 0.5 * np.sin(2 * np.pi * freq_morph * time)

            # 3. Relationship Frequency Tone
            path_tone = 0.7 * np.sin(2 * np.pi * path_freq * time)


            # --- Mix Components ---
            path_sound = np.zeros(num_samples)

            if transition is not None and len(transition) > 0:
                # Ensure same length (simple resize)
                target_len = num_samples
                if len(transition) > target_len:
                    transition = transition[:target_len]
                elif len(transition) < target_len:
                    transition = np.pad(transition, (0, target_len - len(transition)))

                # Mix tones: 30% path, 30% morph, 40% transition
                path_sound = 0.3 * path_tone + 0.3 * morph_tone + 0.4 * transition
            else:
                # Without transition, just mix the tones
                logger.info("Dimensional transition component not available or empty, mixing tones only.")
                path_sound = 0.5 * path_tone + 0.5 * morph_tone

            # Normalize
            max_amp = np.max(np.abs(path_sound))
            if max_amp > 1e-6:
                path_sound = path_sound / max_amp * 0.8
            else:
                 path_sound = np.zeros_like(path_sound)


            logger.info(f"Generated path sound between {sephirah1} and {sephirah2}")
            return path_sound

        except Exception as e:
            logger.error(f"Error generating path sound between {sephirah1} and {sephirah2}: {str(e)}", exc_info=True)
            return None

    def generate_tree_of_life_progression(self, duration_per_sephirah=5.0,
                                        include_paths=True, direction="ascending"):
        """
        Generate a complete progression through the Tree of Life.

        This creates a sonic journey through all Sephiroth, either ascending
        (Malkuth to Kether) or descending (Kether to Malkuth).

        Args:
            duration_per_sephirah (float): Duration for each Sephiroth in seconds
            include_paths (bool): Whether to include transition paths
            direction (str): "ascending" or "descending"

        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None or not hasattr(self.sound_generator, 'sample_rate'):
            logger.error("Cannot generate Tree of Life progression: SoundGenerator or sample_rate not available")
            return None

        # Define the traditional Tree of Life path
        if direction.lower() == "ascending":
            # Bottom to top (material to divine)
            progression_order = [
                "malkuth", "yesod", "hod", "netzach", "tiphareth",
                "geburah", "chesed", "binah", "chokmah", "kether"
            ]
        elif direction.lower() == "descending":
            # Top to bottom (divine to material)
            progression_order = [
                "kether", "chokmah", "binah", "chesed", "geburah",
                "tiphareth", "netzach", "hod", "yesod", "malkuth"
            ]
        else:
            logger.error(f"Invalid direction '{direction}'. Use 'ascending' or 'descending'.")
            return None

        if duration_per_sephirah <= 0:
             logger.error("duration_per_sephirah must be positive.")
             return None

        try:
            sample_rate = self.sound_generator.sample_rate
            num_sephiroth = len(progression_order)
            num_paths = num_sephiroth - 1

            # Calculate path duration (shorter than sephirah duration)
            path_duration = duration_per_sephirah * 0.3 if include_paths else 0
            if path_duration < 0: path_duration = 0

            # Calculate total duration and samples
            total_duration = (num_sephiroth * duration_per_sephirah +
                              num_paths * path_duration)
            total_samples = int(total_duration * sample_rate)

            if total_samples <= 0:
                 logger.error("Calculated total duration is zero or negative.")
                 return None

            # Create an empty array for the progression
            progression_sound = np.zeros(total_samples)

            # Current position in samples
            current_pos = 0

            # Generate each Sephiroth tone and path, adding to progression
            for i, current_sephirah in enumerate(progression_order):
                # --- Generate Sephiroth Tone ---
                tone_duration = duration_per_sephirah
                tone_samples = int(tone_duration * sample_rate)
                if tone_samples <= 0: continue # Skip if duration is too small

                tone = self.generate_sephirah_tone(sephirah, tone_duration)
                if tone is None or len(tone) == 0:
                    logger.warning("Skipping tone for %s due to generation error or zero length.", sephirah)
                    # Still advance position by expected duration to avoid overlap issues
                    current_pos += tone_samples
                    continue

                # Calculate position in samples
                start_sample = current_pos
                end_sample = start_sample + len(tone)

                # Add to progression (handle potential overflow at the very end)
                if start_sample < total_samples:
                    samples_to_add = min(len(tone), total_samples - start_sample)
                    progression_sound[start_sample : start_sample + samples_to_add] += tone[:samples_to_add]

                # Update current position
                current_pos = end_sample

                # --- Generate Path Sound (if applicable) ---
                if include_paths and i < num_paths:
                    next_sephirah = progression_order[i + 1]
                    path_samples = int(path_duration * sample_rate)
                    if path_samples <=0: continue # Skip if path duration is too small

                    path_sound = self.generate_path_sound(
                        sephirah, next_sephirah, path_duration
                    )

                    if path_sound is not None and len(path_sound) > 0:
                        # Calculate position in samples
                        start_sample = current_pos
                        end_sample = start_sample + len(path_sound)

                         # Add to progression (handle potential overflow)
                        if start_sample < total_samples:
                            samples_to_add = min(len(path_sound), total_samples - start_sample)
                            progression_sound[start_sample : start_sample + samples_to_add] += path_sound[:samples_to_add]

                        # Update current position
                        current_pos = end_sample
                    else:
                         logger.warning(f"Skipping path sound between {sephirah} and {next_sephirah}.")
                         # Advance position even if path fails
                         current_pos += path_samples


            # Normalize the final progression
            max_amp = np.max(np.abs(progression_sound))
            if max_amp > 1e-6:
                progression_sound = progression_sound / max_amp * 0.8
            else:
                progression_sound = np.zeros_like(progression_sound) # Avoid NaNs

            logger.info(f"Generated Tree of Life progression in {direction} direction ({total_duration:.2f}s)")
            return progression_sound

        except Exception as e:
            logger.error(f"Error generating Tree of Life progression: {str(e)}", exc_info=True)
            return None

    def generate_gateway_sound(self, gateway_key, duration=20.0):
        """
        Generate a gateway sound for a specific platonic gateway key.

        Args:
            gateway_key (str): Name of the gateway key (e.g., "tetrahedron", "octahedron")
            duration (float): Duration in seconds

        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None or not hasattr(self.sound_generator, 'sample_rate'):
            logger.error("Cannot generate gateway sound: SoundGenerator or sample_rate not available")
            return None

        # Gateway keys map to specific Sephiroth combinations
        gateway_mappings = {
            "tetrahedron": ["tiphareth", "netzach", "hod"],
            "octahedron": ["binah", "kether", "chokmah", "chesed", "tiphareth", "geburah"],
            "hexahedron": ["hod", "netzach", "chesed", "chokmah", "binah", "geburah"],
            "icosahedron": ["kether", "chesed", "geburah"],
            "dodecahedron": ["hod", "netzach", "chesed", "binah", "geburah", "tiphareth", "kether"] # Example adjusted
        }

        gateway_key = gateway_key.lower()

        if gateway_key not in gateway_mappings:
            logger.warning(f"Unknown gateway key: {gateway_key}, available keys: {list(gateway_mappings.keys())}")
            return None

        if duration <= 0:
             logger.error("Gateway sound duration must be positive.")
             return None

        try:
            sample_rate = self.sound_generator.sample_rate
            num_samples = int(duration * sample_rate)
            if num_samples <= 0: return None # Already checked duration, but belt-and-suspenders

            # Get Sephiroth for this gateway
            sephiroth_list = gateway_mappings[gateway_key]
            if not sephiroth_list:
                logger.warning(f"No Sephiroth defined for gateway key {gateway_key}.")
                return None

            gateway_sound = np.zeros(num_samples)
            tones_added = 0

            # Generate each Sephiroth tone and combine
            for sephirah in sephiroth_list:
                # Generate tone with field characteristics for richer gateway sound
                tone = self.generate_sephirah_tone(sephirah, duration, include_field=True)

                if tone is not None and len(tone) > 0:
                    # Ensure same length (simple resize)
                    target_len = num_samples
                    if len(tone) > target_len:
                        tone = tone[:target_len]
                    elif len(tone) < target_len:
                        tone = np.pad(tone, (0, target_len - len(tone)))

                    # Add to gateway sound (average instead of sum to prevent clipping)
                    gateway_sound += tone
                    tones_added += 1
                else:
                     logger.warning(f"Failed to generate tone for {sephirah} in gateway {gateway_key}")

            if tones_added == 0:
                 logger.error(f"Failed to generate any tones for gateway {gateway_key}")
                 return None

            # Average the combined tones
            gateway_sound /= tones_added

            # --- Add gateway-specific modulation ---
            time = np.linspace(0, duration, num_samples, endpoint=False)

            if gateway_key == "tetrahedron":
                # Fire element - dynamic, transformative
                mod_freq = 2.0  # 0.5-second cycle (energetic)
                modulation = (0.2 * np.sin(2 * np.pi * mod_freq * time) ** 2 + 0.8)

            elif gateway_key == "octahedron":
                # Air element - flowing, mental
                mod_freq = 0.5  # 2-second cycle (flowing)
                modulation = (0.15 * np.sin(2 * np.pi * mod_freq * time) + 0.85)

            elif gateway_key == "hexahedron":
                # Earth element - stable, grounded
                mod_freq = 0.25  # 4-second cycle (stable)
                # Use cubic sine for a different feel
                modulation = (0.1 * np.sin(2 * np.pi * mod_freq * time) ** 3 + 0.9)

            elif gateway_key == "icosahedron":
                # Water element - flowing, emotional
                mod_freq = 0.33  # ~3-second cycle (wavelike)
                # Shifted sine wave (0 to 1) for smoother modulation
                modulation = (0.2 * (np.sin(2 * np.pi * mod_freq * time) + 1) / 2 + 0.8)

            elif gateway_key == "dodecahedron":
                # Aether/Spirit element - spiritual, transcendent
                mod_freq = 0.1  # 10-second cycle (slow, cosmic)

                # Create complex cosmic modulation using multiple sine waves
                mod = (np.sin(2 * np.pi * mod_freq * time) +
                       0.5 * np.sin(2 * np.pi * mod_freq * PHI * time) + # Golden ratio frequency
                       0.3 * np.sin(2 * np.pi * mod_freq * 2 * time))    # Octave frequency

                # Normalize modulation: estimate max value (1 + 0.5 + 0.3 = 1.8), scale to desired range
                mod_normalized = mod / 1.8 # Roughly scale to -1 to 1
                # Shift and scale to be mostly positive (e.g., 0.85 to 1.15 range around 1)
                modulation = 0.15 * mod_normalized + 1.0

            else:
                # Default modulation if key somehow slips through checks
                modulation = np.ones(num_samples)


            gateway_sound *= modulation

            # Normalize the final sound
            max_amp = np.max(np.abs(gateway_sound))
            if max_amp > 1e-6:
                gateway_sound = gateway_sound / max_amp * 0.8
            else:
                gateway_sound = np.zeros_like(gateway_sound)


            logger.info(f"Generated gateway sound for {gateway_key} using {tones_added} Sephiroth tones")
            return gateway_sound

        except Exception as e:
            logger.error(f"Error generating gateway sound for {gateway_key}: {str(e)}", exc_info=True)
            return None

    def save_sephiroth_tone(self, sephirah, duration=10.0, include_field=False, filename=None):
        """
        Generate and save a Sephiroth tone to a WAV file.

        Args:
            sephirah (str): Name of the Sephiroth
            duration (float): Duration in seconds
            include_field (bool): Whether to include field harmonics
            filename (str): Custom filename (optional, .wav extension added if missing)

        Returns:
            str: Path to the saved file, or None if generation/saving failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save Sephiroth tone: SoundGenerator not available")
            return None

        # Generate the tone
        tone = self.generate_sephirah_tone(sephirah, duration, include_field)
        if tone is None:
            logger.error(f"Failed to generate tone for {sephirah}, cannot save.")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            field_suffix = "_with_field" if include_field else ""
            filename = f"{sephirah.lower()}_tone{field_suffix}_{timestamp}.wav"
        elif not filename.lower().endswith(".wav"):
             filename += ".wav"


        # Save the tone using the base sound_generator's save method
        try:
            filepath = self.sound_generator.save_sound(
                tone,
                filename=filename,
                # Pass relevant metadata if save_sound supports it
                field_type="sephiroth",
                description=f"{sephirah.capitalize()} tone{' with field' if include_field else ''}"
            )
            if filepath:
                 logger.info(f"Saved Sephiroth tone to {filepath}")
                 return filepath
            else:
                 logger.error(f"SoundGenerator.save_sound failed for {filename}")
                 return None
        except Exception as e:
             logger.error(f"Error saving Sephiroth tone {filename}: {str(e)}", exc_info=True)
             return None

    def save_gateway_sound(self, gateway_key, duration=20.0, filename=None):
        """
        Generate and save a gateway sound to a WAV file.

        Args:
            gateway_key (str): Name of the gateway key
            duration (float): Duration in seconds
            filename (str): Custom filename (optional, .wav extension added if missing)

        Returns:
            str: Path to the saved file, or None if generation/saving failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save gateway sound: SoundGenerator not available")
            return None

        # Generate the gateway sound
        sound = self.generate_gateway_sound(gateway_key, duration)
        if sound is None:
            logger.error(f"Failed to generate sound for gateway {gateway_key}, cannot save.")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{gateway_key.lower()}_gateway_{timestamp}.wav"
        elif not filename.lower().endswith(".wav"):
             filename += ".wav"

        # Save the sound
        try:
            filepath = self.sound_generator.save_sound(
                sound,
                filename=filename,
                field_type="gateway",
                description=f"{gateway_key.capitalize()} gateway sound"
            )
            if filepath:
                 logger.info(f"Saved gateway sound to {filepath}")
                 return filepath
            else:
                 logger.error(f"SoundGenerator.save_sound failed for {filename}")
                 return None
        except Exception as e:
             logger.error(f"Error saving gateway sound {filename}: {str(e)}", exc_info=True)
             return None

    def save_tree_of_life_progression(self, duration_per_sephirah=5.0,
                                     include_paths=True, direction="ascending", filename=None):
        """
        Generate and save a Tree of Life progression to a WAV file.

        Args:
            duration_per_sephirah (float): Duration for each Sephiroth in seconds
            include_paths (bool): Whether to include transition paths
            direction (str): "ascending" or "descending"
            filename (str): Custom filename (optional, .wav extension added if missing)

        Returns:
            str: Path to the saved file, or None if generation/saving failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save Tree of Life progression: SoundGenerator not available")
            return None

        # Generate the progression
        progression = self.generate_tree_of_life_progression(
            duration_per_sephirah, include_paths, direction
        )
        if progression is None:
            logger.error(f"Failed to generate Tree of Life progression ({direction}), cannot save.")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            paths_suffix = "_with_paths" if include_paths else ""
            filename = f"tree_of_life_{direction}{paths_suffix}_{timestamp}.wav"
        elif not filename.lower().endswith(".wav"):
             filename += ".wav"

        # Save the progression
        try:
            filepath = self.sound_generator.save_sound(
                progression,
                filename=filename,
                field_type="tree_of_life",
                description=f"Tree of Life {direction} progression{' with paths' if include_paths else ''}"
            )
            if filepath:
                 logger.info(f"Saved Tree of Life progression to {filepath}")
                 return filepath
            else:
                 logger.error(f"SoundGenerator.save_sound failed for {filename}")
                 return None
        except Exception as e:
             logger.error(f"Error saving Tree of Life progression {filename}: {str(e)}", exc_info=True)
             return None

    def generate_sephiroth_field_activation(self, sephirah, duration=60.0):
        """
        Generate a complete field activation sequence for a specific Sephiroth.

        This creates a sound specifically designed to enhance Sephiroth field formation
        and resonance connection, incorporating build-up, stabilization, and sacred geometry.

        Args:
            sephirah (str): Name of the Sephiroth field to activate
            duration (float): Total duration in seconds

        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None or not hasattr(self.sound_generator, 'sample_rate'):
            logger.error("Cannot generate field activation: SoundGenerator or sample_rate not available")
            return None

        sephirah = sephirah.lower()
        if duration <= 0:
             logger.error("Field activation duration must be positive.")
             return None

        try:
            sample_rate = self.sound_generator.sample_rate
            num_samples = int(duration * sample_rate)
            if num_samples <= 0: return None

            # --- Base Tone ---
            # Get the base Sephiroth tone with field characteristics included
            base_tone = self.generate_sephirah_tone(sephirah, duration, include_field=True)
            if base_tone is None or len(base_tone) == 0:
                logger.error(f"Failed to generate base tone for {sephirah} activation.")
                return None
            # Ensure correct length
            if len(base_tone) != num_samples:
                 target_len = num_samples
                 if len(base_tone) > target_len: base_tone = base_tone[:target_len]
                 else: base_tone = np.pad(base_tone, (0, target_len - len(base_tone)))

            activation_sound = base_tone.copy() # Start with the base tone
            time = np.linspace(0, duration, num_samples, endpoint=False)

            # --- Add Sacred Geometry Frequencies (if available) ---
            sacred_chord = None
            # Check if the method exists and the base frequency is valid
            if hasattr(self.sound_generator, 'generate_sacred_chord'):
                base_freq = self.frequencies.get_frequency(sephirah)
                if base_freq > 0:
                    sacred_chord = self.sound_generator.generate_sacred_chord(
                        base_frequency=base_freq,
                        duration=duration
                    )

            if sacred_chord is not None and len(sacred_chord) > 0:
                # Ensure same length
                target_len = num_samples
                if len(sacred_chord) > target_len: sacred_chord = sacred_chord[:target_len]
                elif len(sacred_chord) < target_len: sacred_chord = np.pad(sacred_chord, (0, target_len - len(sacred_chord)))

                # Mix tones: 70% Sephiroth base, 30% sacred geometry
                activation_sound = 0.7 * activation_sound + 0.3 * sacred_chord
            else:
                 logger.info(f"Sacred chord component not available or failed for {sephirah} activation.")
                 # activation_sound remains just the base tone for now


            # --- Add Activation Phases & Patterns ---

            # 1. Initialization Phase (Build-up at the start)
            init_duration = min(15.0, duration / 4) # Up to 15s or 1/4 total duration
            init_samples = int(init_duration * sample_rate)
            if init_samples > 0 and init_samples < num_samples:
                # Create a rising amplitude envelope (e.g., sqrt curve)
                init_envelope = np.linspace(0, 1, init_samples) ** 0.5
                activation_sound[:init_samples] *= init_envelope

            # 2. Stabilization Phase (Pulsing near the end)
            stabilize_duration = min(15.0, duration / 4)
            stabilize_samples = int(stabilize_duration * sample_rate)
            if stabilize_samples > 0 and stabilize_samples < num_samples:
                # Create pulsing effect (e.g., sine wave modulation)
                pulse_freq = 0.2  # Hz (5-second cycle)
                # Apply only to the end portion of the time array
                stabilize_time = time[-stabilize_samples:] - time[-stabilize_samples] # Relative time for this section
                pulse_mod = 0.2 * np.sin(2 * np.pi * pulse_freq * stabilize_time) + 0.8
                activation_sound[-stabilize_samples:] *= pulse_mod

            # 3. Sephiroth-Specific Activation Patterns (applied over the whole duration or sections)
            # These add more character on top of the mixed tone and phases

            # Higher Sephiroth: Ethereal cosmic pulsing
            if sephirah in ["kether", "chokmah", "binah"]:
                cosmic_freq = 0.05  # Very slow 20-second cycle
                cosmic_pulse = 0.15 * np.sin(2 * np.pi * cosmic_freq * time) ** 2 + 0.85
                activation_sound *= cosmic_pulse # Apply over entire duration

            # Middle Sephiroth: Balanced harmonic enhancement in the middle
            elif sephirah in ["chesed", "geburah", "tiphareth"]:
                mid_start_time = duration * 0.3
                mid_end_time = duration * 0.7
                mid_start_sample = int(mid_start_time * sample_rate)
                mid_end_sample = int(mid_end_time * sample_rate)

                if mid_start_sample < mid_end_sample and mid_end_sample < num_samples:
                    # Calculate phi-based frequency relative to base
                    phi_freq = self.frequencies.get_frequency(sephirah) * PHI
                    if phi_freq > 0:
                         mid_time = time[mid_start_sample:mid_end_sample]
                         # Create harmonic enhancement tone (additively)
                         enhancement = 0.15 * np.sin(2 * np.pi * phi_freq * mid_time) # Lower amplitude
                         activation_sound[mid_start_sample:mid_end_sample] += enhancement

            # Lower Sephiroth: Grounding earthy thrumming (additive)
            else: # Malkuth, Yesod, Hod, Netzach, unknowns
                earth_freq = self.frequencies.get_frequency(sephirah) * 0.25 # Low frequency
                if earth_freq > 0:
                     earth_wave = 0.1 * np.sin(2 * np.pi * earth_freq * time) # Low amplitude thrum
                     activation_sound += earth_wave


            # --- Final Normalization ---
            max_amp = np.max(np.abs(activation_sound))
            if max_amp > 1e-6:
                activation_sound = activation_sound / max_amp * 0.8
            else:
                activation_sound = np.zeros_like(activation_sound)


            logger.info(f"Generated {sephirah} field activation sequence ({duration:.1f}s)")
            return activation_sound

        except Exception as e:
            logger.error(f"Error generating field activation for {sephirah}: {str(e)}", exc_info=True)
            return None

    def save_sephiroth_field_activation(self, sephirah, duration=60.0, filename=None):
        """
        Generate and save a Sephiroth field activation sequence to a WAV file.

        Args:
            sephirah (str): Name of the Sephiroth
            duration (float): Duration in seconds
            filename (str): Custom filename (optional, .wav extension added if missing)

        Returns:
            str: Path to the saved file, or None if generation/saving failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save field activation: SoundGenerator not available")
            return None

        # Generate the activation sequence
        activation = self.generate_sephiroth_field_activation(sephirah, duration)
        if activation is None:
            logger.error(f"Failed to generate field activation for {sephirah}, cannot save.")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{sephirah.lower()}_field_activation_{timestamp}.wav"
        elif not filename.lower().endswith(".wav"):
             filename += ".wav"

        # Save the activation sequence
        try:
            filepath = self.sound_generator.save_sound(
                activation,
                filename=filename,
                field_type="sephiroth_activation",
                description=f"{sephirah.capitalize()} field activation sequence"
            )
            if filepath:
                 logger.info(f"Saved Sephiroth field activation to {filepath}")
                 return filepath
            else:
                 logger.error(f"SoundGenerator.save_sound failed for {filename}")
                 return None
        except Exception as e:
             logger.error(f"Error saving field activation {filename}: {str(e)}", exc_info=True)
             return None


# Integration function to extend the field system with Sephiroth sounds
def extend_field_system_with_sephiroth_sounds(field_system):
    """
    Extend a field system object with Sephiroth sound generation capabilities.

    This function attempts to add a SephirothSoundGenerator instance and a
    helper method to the provided field_system object.

    Args:
        field_system: The field system object to extend. Needs to have
                      'sound_generator' and 'base_frequency' attributes.

    Returns:
        bool: True if extension was successful, False otherwise
    """
    if field_system is None:
        logger.error("Cannot extend None field system")
        return False

    try:
        # Check required attributes on the field_system object
        if not hasattr(field_system, 'sound_generator') or field_system.sound_generator is None:
            logger.error("Field system lacks a valid 'sound_generator' attribute, cannot extend.")
            return False
        if not hasattr(field_system, 'base_frequency'):
             logger.error("Field system lacks a 'base_frequency' attribute, cannot extend.")
             return False
        if not hasattr(field_system.sound_generator, 'output_dir'):
             logger.error("Field system's sound_generator lacks 'output_dir', cannot initialize Sephiroth generator.")
             return False


        # Create Sephiroth sound generator instance
        # Use properties from the existing field system
        sephiroth_generator = SephirothSoundGenerator(
            base_frequency=field_system.base_frequency,
            output_dir=os.path.join(field_system.sound_generator.output_dir, "sephiroth") # Subdirectory
        )

        # Check if the Sephiroth generator itself initialized correctly (found SoundGenerator)
        if sephiroth_generator.sound_generator is None:
             logger.error("Failed to initialize SephirothSoundGenerator (likely missing base SoundGenerator). Cannot extend.")
             # Clean up potentially created directory if needed (optional)
             # if os.path.exists(sephiroth_generator.output_dir):
             #     try: os.rmdir(sephiroth_generator.output_dir) except OSError: pass
             return False


        # Add the generator instance to the field system object
        field_system.sephiroth_sound_generator = sephiroth_generator
        logger.info("Added sephiroth_sound_generator instance to field system.")

        # --- Add helper method to generate Sephiroth sounds directly from field_system ---
        # Define the method we want to add
        def generate_sephiroth_field_sound(self, sephirah, duration=30.0):
            """
            Generate a sound for a Sephiroth field using the integrated generator.

            Args:
                sephirah (str): Name of the Sephiroth
                duration (float): Duration in seconds

            Returns:
                numpy.ndarray: Sound waveform or None if failed
            """
            # 'self' here refers to the field_system instance when called
            if not hasattr(self, 'sephiroth_sound_generator') or self.sephiroth_sound_generator is None:
                logger.error("Field system is missing the 'sephiroth_sound_generator'.")
                return None

            # Call the appropriate method on the attached generator
            return self.sephiroth_sound_generator.generate_sephiroth_field_activation(sephirah, duration)

        # Use types.MethodType to bind the function as a method to the instance
        field_system.generate_sephiroth_field_sound = types.MethodType(
            generate_sephiroth_field_sound, field_system
        )
        logger.info("Added generate_sephiroth_field_sound method to field system.")

        logger.info("Successfully extended field system with Sephiroth sounds.")
        return True

    except AttributeError as ae:
         logger.error(f"Attribute error during field system extension: {str(ae)}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"Unexpected error extending field system: {str(e)}", exc_info=True)
        # Clean up partially added attributes if desired
        if hasattr(field_system, 'sephiroth_sound_generator'): delattr(field_system, 'sephiroth_sound_generator')
        if hasattr(field_system, 'generate_sephiroth_field_sound'): delattr(field_system, 'generate_sephiroth_field_sound')
        return False


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    print("Running Sephiroth Sounds Module Example...")

    # Create the Sephiroth sound generator
    # This might fail if sound_generator.py is not found / working
    try:
        generator = SephirothSoundGenerator()
        generator_ok = generator.sound_generator is not None
    except Exception as main_e:
         print(f"Error initializing SephirothSoundGenerator: {main_e}")
         generator = None
         generator_ok = False

    if generator:
        # Print available Sephiroth frequencies
        print("\nSephiroth Frequencies:")
        if hasattr(generator, 'frequencies') and hasattr(generator.frequencies, 'frequencies'):
            for sephirah, freq in generator.frequencies.frequencies.items():
                print(f"  {sephirah.capitalize():<10}: {freq:.2f} Hz")
        else:
            print("  Could not retrieve frequencies.")

        # Check if sound generator component is available for actual sound creation
        if generator_ok:
            print("\nSound generator component is available. Attempting sound generation...")

            # Generate and save a Tiphareth tone
            print("\nGenerating Tiphareth tone...")
            tiphareth_path = generator.save_sephiroth_tone("tiphareth", duration=8.0, include_field=True)
            if tiphareth_path:
                print(f"  > Saved Tiphareth tone: {tiphareth_path}")
            else:
                print("  > Failed to generate/save Tiphareth tone.")

            # Generate and save a Tree of Life progression (descending)
            print("\nGenerating Tree of Life progression (descending)...")
            progression_path = generator.save_tree_of_life_progression(
                duration_per_sephirah=3.0,
                include_paths=True,
                direction="descending"
            )
            if progression_path:
                print(f"  > Saved Tree of Life progression: {progression_path}")
            else:
                print("  > Failed to generate/save Tree of Life progression.")

            # Generate and save a tetrahedron gateway sound
            print("\nGenerating Tetrahedron gateway sound...")
            gateway_path = generator.save_gateway_sound("tetrahedron", duration=12.0)
            if gateway_path:
                print(f"  > Saved Tetrahedron gateway sound: {gateway_path}")
            else:
                print("  > Failed to generate/save Tetrahedron gateway sound.")

            # Generate and save a Tiphareth field activation sequence
            print("\nGenerating Tiphareth field activation sequence...")
            activation_path = generator.save_sephiroth_field_activation("tiphareth", duration=25.0)
            if activation_path:
                print(f"  > Saved Tiphareth field activation: {activation_path}")
            else:
                print("  > Failed to generate/save Tiphareth field activation.")

        else:
            print("\nSound generator component (sound_generator.py) NOT available.")
            print("Cannot create actual sound files.")
            print("Frequency calculations and structures are still available.")

    else:
        print("\nFailed to initialize the main SephirothSoundGenerator.")

    print("\nSephiroth Sounds Module Example Finished.")