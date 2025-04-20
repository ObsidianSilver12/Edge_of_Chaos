{
  `path`: `sounds/sounds_of_sephiroth.py`,
  `content`: `\"\"\"
Sounds of Sephiroth Module

This module implements specialized sound generation for the Sephiroth dimensions.
It creates specific tones, frequencies, and harmonic structures for each Sephiroth
based on their metaphysical properties and position in the Tree of Life.

This module extends the base sound_generator.py with Sephiroth-specific
functionality, allowing for more detailed and accurate sonic representations.

Author: Soul Development Framework Team
\"\"\"

import numpy as np
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sound generator
try:
    from sounds.sound_generator import SoundGenerator, FUNDAMENTAL_FREQUENCY, PHI
except ImportError:
    logging.warning(\"sound_generator.py not available. Sephiroth sounds will not function correctly.\")
    SoundGenerator = None
    FUNDAMENTAL_FREQUENCY = 432.0
    PHI = 1.618033988749895

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_sounds.log'
)
logger = logging.getLogger('sephiroth_sounds')

class SephirothFrequencies:
    \"\"\"
    Class containing the frequency properties for each Sephiroth.
    
    These frequencies are derived from traditional correspondences and
    the spiritual properties of each Sephiroth, scaled from a base frequency.
    \"\"\"
    
    def __init__(self, base_frequency=FUNDAMENTAL_FREQUENCY):
        \"\"\"
        Initialize Sephiroth frequencies based on a given base frequency.
        
        Args:
            base_frequency (float): Base frequency to derive Sephiroth frequencies from
        \"\"\"
        self.base_frequency = base_frequency
        
        # Define frequency modifiers for each Sephiroth
        # These represent the relative frequency compared to the Creator/Kether
        # Higher Sephiroth have higher frequencies as they are closer to divine source
        self.frequency_modifiers = {
            # Higher triad
            \"kether\": 1.0,       # Crown - highest/purest frequency
            \"chokmah\": 0.95,     # Wisdom
            \"binah\": 0.9,        # Understanding
            
            # Middle triad
            \"chesed\": 0.85,      # Mercy
            \"geburah\": 0.8,      # Severity
            \"tiphareth\": 0.75,   # Beauty - central harmonizing Sephiroth
            
            # Lower triad
            \"netzach\": 0.7,      # Victory
            \"hod\": 0.65,         # Glory
            \"yesod\": 0.6,        # Foundation
            
            # Material realm
            \"malkuth\": 0.55      # Kingdom - lowest frequency, most dense
        }
        
        # Define harmonic structures for each Sephiroth
        self.harmonic_counts = {
            \"kether\": 12,      # Most complex harmonics (divine completeness)
            \"chokmah\": 10,
            \"binah\": 10,
            \"chesed\": 9,
            \"geburah\": 8,
            \"tiphareth\": 10,    # Rich harmonics (central balancing point)
            \"netzach\": 7,
            \"hod\": 7,
            \"yesod\": 6,
            \"malkuth\": 4       # Simplest harmonic structure (material simplicity)
        }
        
        # Define phi-harmonic counts (harmonics based on golden ratio)
        self.phi_harmonic_counts = {
            \"kether\": 7,       # Highest phi-connection (divine proportion)
            \"chokmah\": 6,
            \"binah\": 6,
            \"chesed\": 5,
            \"geburah\": 4,
            \"tiphareth\": 6,    # Strong phi-resonance (balanced beauty)
            \"netzach\": 3,
            \"hod\": 3,
            \"yesod\": 2,
            \"malkuth\": 1       # Minimal phi-connection (material density)
        }
        
        # Amplitude falloff rates for harmonics
        # Higher values mean harmonics diminish more quickly in amplitude
        self.harmonic_falloff = {
            \"kether\": 0.05,    # Sustained harmonics (divine persistence)
            \"chokmah\": 0.06, 
            \"binah\": 0.07,
            \"chesed\": 0.08,
            \"geburah\": 0.1,
            \"tiphareth\": 0.07, # Balanced falloff
            \"netzach\": 0.12,
            \"hod\": 0.12,
            \"yesod\": 0.15,
            \"malkuth\": 0.2     # Rapid falloff (material density absorbs)
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
            (\"chokmah\", \"chesed\"): 1.5,     # Wisdom to Mercy
            (\"chesed\", \"netzach\"): 1.2,     # Mercy to Victory
            
            # Pillar of Severity relationships
            (\"binah\", \"geburah\"): 1.5,     # Understanding to Severity
            (\"geburah\", \"hod\"): 1.2,       # Severity to Glory
            
            # Middle Pillar relationships
            (\"kether\", \"tiphareth\"): PHI,  # Crown to Beauty (phi)
            (\"tiphareth\", \"yesod\"): 1.25,           # Beauty to Foundation
            (\"yesod\", \"malkuth\"): 1.1,              # Foundation to Kingdom
            
            # Horizontal relationships
            (\"binah\", \"chokmah\"): 1.05,        # Understanding to Wisdom
            (\"geburah\", \"chesed\"): 1.05,       # Severity to Mercy
            (\"hod\", \"netzach\"): 1.05,          # Glory to Victory
        }
        
        logger.info(f\"Initialized Sephiroth frequencies with base {base_frequency} Hz\")
        
    def get_frequency(self, sephirah):
        \"\"\"
        Get the primary frequency for a specific Sephiroth.
        
        Args:
            sephirah (str): The name of the Sephiroth (lowercase)
            
        Returns:
            float: The primary frequency in Hz
        \"\"\"
        sephirah = sephirah.lower()
        if sephirah in self.frequencies:
            return self.frequencies[sephirah]
        else:
            logger.warning(f\"Unknown Sephirah: {sephirah}, returning base frequency\")
            return self.base_frequency
            
    def get_harmonic_structure(self, sephirah):
        \"\"\"
        Get the complete harmonic structure for a Sephiroth.
        
        Args:
            sephirah (str): The name of the Sephiroth (lowercase)
            
        Returns:
            dict: Dictionary with harmonic structure information
        \"\"\"
        sephirah = sephirah.lower()
        
        if sephirah not in self.frequencies:
            logger.warning(f\"Unknown Sephirah: {sephirah}, returning default structure\")
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
        \"\"\"
        Get the relationship frequency between two Sephiroth.
        
        Args:
            sephirah1 (str): First Sephiroth name
            sephirah2 (str): Second Sephiroth name
            
        Returns:
            float: Relationship frequency in Hz
        \"\"\"
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        # Check direct relationship
        if (sephirah1, sephirah2) in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[(sephirah1, sephirah2)]
            return self.frequencies[sephirah1] * multiplier
            
        # Check reverse relationship
        if (sephirah2, sephirah1) in self.relationship_harmonics:
            multiplier = self.relationship_harmonics[(sephirah2, sephirah1)]
            return self.frequencies[sephirah2] * multiplier
            
        # If no direct relationship, use geometric mean of frequencies
        if sephirah1 in self.frequencies and sephirah2 in self.frequencies:
            return np.sqrt(self.frequencies[sephirah1] * self.frequencies[sephirah2])
            
        logger.warning(f\"Unknown Sephiroth relationship: {sephirah1}-{sephirah2}\")
        return self.base_frequency


class SephirothSoundGenerator:
    \"\"\"
    Advanced generator for Sephiroth-specific sounds and tones.
    
    This extends the base SoundGenerator with specialized functionality
    for creating sounds that represent the Sephiroth dimensions and their
    unique metaphysical properties.
    \"\"\"
    
    def __init__(self, base_frequency=FUNDAMENTAL_FREQUENCY, output_dir=\"output/sounds/sephiroth\"):
        \"\"\"
        Initialize the Sephiroth sound generator.
        
        Args:
            base_frequency (float): Base frequency for sound generation
            output_dir (str): Directory for saving generated sounds
        \"\"\"
        self.frequencies = SephirothFrequencies(base_frequency)
        
        # Create sound generator if available
        if SoundGenerator is not None:
            self.sound_generator = SoundGenerator(output_dir=output_dir)
            logger.info(\"Sound generator initialized for Sephiroth sounds\")
        else:
            self.sound_generator = None
            logger.error(\"SoundGenerator not available. Sephiroth sounds will not function.\")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
    def generate_sephirah_tone(self, sephirah, duration=10.0, include_field=False):
        \"\"\"
        Generate a detailed audio tone representing a specific Sephiroth.
        
        Args:
            sephirah (str): The name of the Sephiroth
            duration (float): Duration in seconds
            include_field (bool): Whether to include field harmonics
            
        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        \"\"\"
        if self.sound_generator is None:
            logger.error(\"Cannot generate Sephiroth tone: SoundGenerator not available\")
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
                durations=[duration] * len(harmonics),
                amplitudes=amplitudes,
                fade_in_out=min(1.0, duration/10)  # 10% fade in/out
            )
            
            # If including field harmonics, add field characteristics
            if include_field:
                field_tone = self._generate_field_characteristics(sephirah, duration)
                if field_tone is not None:
                    # Mix 70% tone, 30% field
                    tone = 0.7 * tone + 0.3 * field_tone
                    
                    # Normalize
                    max_amp = np.max(np.abs(tone))
                    if max_amp > 0:
                        tone = tone / max_amp * 0.8  # Keep some headroom
            
            logger.info(f\"Generated {sephirah} tone at {frequency:.2f}Hz with {len(harmonics)} harmonics\")
            return tone
            
        except Exception as e:
            logger.error(f\"Error generating {sephirah} tone: {str(e)}\")
            return None
            
    def _get_sephirah_harmonic_ratios(self, sephirah):
        \"\"\"
        Get the specific harmonic ratios for a Sephiroth.
        
        Args:
            sephirah (str): Sephiroth name
            
        Returns:
            list: Harmonic ratios
        \"\"\"
        # Base harmonic ratios
        base_harmonics = [1.0, PHI, 2.0, PHI*2, 3.0]
        
        # Each Sephiroth has unique additional harmonics
        if sephirah == \"kether\":
            # Crown - highest divine connection
            return [1.0, 1.5, PHI, 2.0, 3.0, 5.0, PHI*3]
            
        elif sephirah == \"chokmah\":
            # Wisdom - masculine energy
            return [1.0, 1.5, 2.0, 3.0, PHI, PHI*2]
            
        elif sephirah == \"binah\":
            # Understanding - feminine energy
            return [1.0, 1.33, 1.66, 2.0, 2.5, PHI]
            
        elif sephirah == \"chesed\":
            # Mercy - expansive energy
            return [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
            
        elif sephirah == \"geburah\":
            # Severity - restrictive energy
            return [1.0, 1.1, 1.33, 1.66, 2.0, 2.5]
            
        elif sephirah == \"tiphareth\":
            # Beauty - balance and harmony
            return [1.0, PHI, 2.0, PHI*2, 3.0, PHI*3]
            
        elif sephirah == \"netzach\":
            # Victory - emotional energy
            return [1.0, 1.25, 1.5, 1.75, 2.0, PHI]
            
        elif sephirah == \"hod\":
            # Glory - intellectual energy
            return [1.0, 1.2, 1.33, 1.66, 2.0, 2.5]
            
        elif sephirah == \"yesod\":
            # Foundation - dream energy
            return [1.0, 1.125, 1.25, 1.5, 2.0, PHI]
            
        elif sephirah == \"malkuth\":
            # Kingdom - material energy
            return [1.0, 1.125, 1.25, 1.33, 1.5]
            
        # Default case
        return base_harmonics
        
    def _get_sephirah_amplitude_profile(self, sephirah, harmonic_count):
        \"\"\"
        Get amplitude profile for a Sephiroth's harmonics.
        
        Args:
            sephirah (str): Sephiroth name
            harmonic_count (int): Number of harmonics
            
        Returns:
            list: Amplitude values
        \"\"\"
        # Start with the fundamental at full amplitude
        amplitudes = [0.8]
        
        # Get falloff rate from frequency structure
        structure = self.frequencies.get_harmonic_structure(sephirah)
        falloff = structure['falloff_rate']
        
        # Generate rest of amplitudes with appropriate falloff
        for i in range(1, harmonic_count):
            amp = 0.8 / (1 + i * falloff)
            amplitudes.append(amp)
            
        return amplitudes
        
    def _generate_field_characteristics(self, sephirah, duration):
        \"\"\"
        Generate field-specific characteristics for a Sephiroth.
        
        Args:
            sephirah (str): Sephiroth name
            duration (float): Duration in seconds
            
        Returns:
            numpy.ndarray: Field characteristics waveform or None
        \"\"\"
        if self.sound_generator is None:
            return None
            
        try:
            # Get frequency
            frequency = self.frequencies.get_frequency(sephirah)
            
            # Field frequencies are slightly lower
            field_freq = frequency * 0.95
            
            # Create time array
            sample_rate = self.sound_generator.sample_rate
            time = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            
            # Base field tone
            field_tone = 0.5 * np.sin(2 * np.pi * field_freq * time)
            
            # Add field-specific modulations
            if sephirah in [\"kether\", \"chokmah\", \"binah\"]:
                # Higher Sephiroth have ethereal, high-frequency components
                high_freq = field_freq * 4
                field_tone += 0.2 * np.sin(2 * np.pi * high_freq * time)
                
                # Add subtle amplitude modulation
                mod_freq = 0.2  # 5-second cycle
                field_tone *= (0.2 * np.sin(2 * np.pi * mod_freq * time) + 0.8)
                
            elif sephirah in [\"chesed\", \"geburah\", \"tiphareth\"]:
                # Middle Sephiroth have balanced, resonant qualities
                field_tone += 0.3 * np.sin(2 * np.pi * field_freq * PHI * time)
                
                # Add rhythmic pulsing
                pulse_freq = 0.25  # 4-second cycle
                field_tone *= (0.3 * np.sin(2 * np.pi * pulse_freq * time) ** 2 + 0.7)
                
            else:
                # Lower Sephiroth have more grounded, stable qualities
                field_tone += 0.4 * np.sin(2 * np.pi * field_freq * 0.5 * time)
                
                # Add gentle wave motion
                wave_freq = 0.1  # 10-second cycle
                field_tone *= (0.15 * np.sin(2 * np.pi * wave_freq * time) + 0.85)
            
            # Normalize
            max_amp = np.max(np.abs(field_tone))
            if max_amp > 0:
                field_tone = field_tone / max_amp * 0.8
            
            return field_tone
            
        except Exception as e:
            logger.error(f\"Error generating field characteristics: {str(e)}\")
            return None
    
    def generate_path_sound(self, sephirah1, sephirah2, duration=15.0):
        \"\"\"
        Generate a sound for a path between two Sephiroth.
        
        These paths represent transitional sounds between Sephiroth dimensions.
        
        Args:
            sephirah1 (str): First Sephiroth name
            sephirah2 (str): Second Sephiroth name
            duration (float): Duration in seconds
            
        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        \"\"\"
        if self.sound_generator is None:
            logger.error(\"Cannot generate path sound: SoundGenerator not available\")
            return None
            
        sephirah1 = sephirah1.lower()
        sephirah2 = sephirah2.lower()
        
        try:
            # Get relationship frequency
            path_freq = self.frequencies.get_relationship_frequency(sephirah1, sephirah2)
            
            # Get individual frequencies
            freq1 = self.frequencies.get_frequency(sephirah1)
            freq2 = self.frequencies.get_frequency(sephirah2)
            
            # Create transition sound between the two frequencies
            transition = self.sound_generator.generate_dimensional_transition(
                start_dimension=0,  # Not using actual dimensions
                end_dimension=0,    # Just need the transition functionality
                duration=duration,
                steps=50
            )
            
            # Create custom transition frequencies
            sample_rate = self.sound_generator.sample_rate
            time = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            
            # Create a morph between the two frequencies
            t = np.linspace(0, 1, len(time))
            freq_morph = freq1 * (1-t) + freq2 * t
            
            # Add the morphing frequency
            morph_tone = 0.5 * np.sin(2 * np.pi * freq_morph * time)
            
            # Add the path frequency (represents the relationship)
            path_tone = 0.7 * np.sin(2 * np.pi * path_freq * time)
            
            # Mix with the dimensional transition
            if transition is not None:
                # Ensure same length
                if len(transition) > len(time):
                    transition = transition[:len(time)]
                elif len(transition) < len(time):
                    # Pad with zeros
                    transition = np.pad(transition, (0, len(time) - len(transition)))
                
                # Mix tones: 30% path, 30% morph, 40% transition
                path_sound = 0.3 * path_tone + 0.3 * morph_tone + 0.4 * transition
            else:
                # Without transition, just mix the tones
                path_sound = 0.5 * path_tone + 0.5 * morph_tone
            
            # Normalize
            max_amp = np.max(np.abs(path_sound))
            if max_amp > 0:
                path_sound = path_sound / max_amp * 0.8
            
            logger.info(f\"Generated path sound between {sephirah1} and {sephirah2}\")
            return path_sound
            
        except Exception as e:
            logger.error(f\"Error generating path sound: {str(e)}\")
            return None
    
    def generate_tree_of_life_progression(self, duration_per_sephirah=5.0,
                                        include_paths=True, direction=\"ascending\"):
        \"\"\"
        Generate a complete progression through the Tree of Life.
        
        This creates a sonic journey through all Sephiroth, either ascending
        (Malkuth to Kether) or descending (Kether to Malkuth).
        
        Args:
            duration_per_sephirah (float): Duration for each Sephiroth in seconds
            include_paths (bool): Whether to include transition paths
            direction (str): \"ascending\" or \"descending\"
            
        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        \"\"\"
        if self.sound_generator is None:
            logger.error(\"Cannot generate Tree of Life progression: SoundGenerator not available\")
            return None
        
        # Define the traditional Tree of Life path
        if direction.lower() == \"ascending\":
            # Bottom to top (material to divine)
            progression = [
                \"malkuth\", \"yesod\", \"hod\", \"netzach\", \"tiphareth\",
                \"geburah\", \"chesed\", \"binah\", \"chokmah\", \"kether\"
            ]
        else:
            # Top to bottom (divine to material)
            progression = [
                \"kether\", \"chokmah\", \"binah\", \"chesed\", \"geburah\",
                \"tiphareth\", \"netzach\", \"hod\", \"yesod\", \"malkuth\"
            ]
        
        try:
            # Calculate path duration
            path_duration = duration_per_sephirah * 0.3 if include_paths else 0
            
            # Calculate total duration
            total_duration = (len(progression) * duration_per_sephirah + 
                            path_duration * (len(progression) - 1))
            
            # Create an empty array for the progression
            sample_rate = self.sound_generator.sample_rate
            total_samples = int(total_duration * sample_rate)
            progression_sound = np.zeros(total_samples)
            
            # Current position in samples
            current_pos = 0
            
            # Generate each Sephiroth tone and add to progression
            for i, sephirah in enumerate(progression):
                # Generate Sephiroth tone
                tone = self.generate_sephirah_tone(sephirah, duration_per_sephirah)
                if tone is None:
                    continue
                
                # Calculate position in samples
                start_sample = current_pos
                end_sample = start_sample + len(tone)
                
                # Ensure we don't exceed array bounds
                if end_sample > total_samples:
                    end_sample = total_samples
                    tone = tone[:end_sample-start_sample]
                
                # Add to progression
                progression_sound[start_sample:end_sample] += tone
                
                # Update current position
                current_pos = end_sample
                
                # Add path to next Sephiroth if not the last one
                if include_paths and i < len(progression) - 1:
                    next_sephirah = progression[i + 1]
                    
                    # Generate path sound
                    path_sound = self.generate_path_sound(
                        sephirah, next_sephirah, path_duration
                    )
                    
                    if path_sound is not None:
                        # Calculate position in samples
                        start_sample = current_pos
                        end_sample = start_sample + len(path_sound)
                        
                        # Ensure we don't exceed array bounds
                        if end_sample > total_samples:
                            end_sample = total_samples
                            path_sound = path_sound[:end_sample-start_sample]
                        
                        # Add to progression
                        progression_sound[start_sample:end_sample] += path_sound
                        
                        # Update current position
                        current_pos = end_sample
            
            # Normalize the final progression
            max_amp = np.max(np.abs(progression_sound))
            if max_amp > 0:
                progression_sound = progression_sound / max_amp * 0.8
            
            logger.info(f\"Generated Tree of Life progression in {direction} direction\")
            return progression_sound
            
        except Exception as e:
            logger.error(f\"Error generating Tree of Life progression: {str(e)}\")
            return None
    
    def generate_gateway_sound(self, gateway_key, duration=20.0):
        \"\"\"
        Generate a gateway sound for a specific platonic gateway key.
        
        Args:
            gateway_key (str): Name of the gateway key (e.g., \"tetrahedron\", \"octahedron\")
            duration (float): Duration in seconds
            
        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        \"\"\"
        if self.sound_generator is None:
            logger.error(\"Cannot generate gateway sound: SoundGenerator not available\")
            return None
            
        # Gateway keys map to specific Sephiroth combinations
        gateway_mappings = {
            \"tetrahedron\": [\"tiphareth\", \"netzach\", \"hod\"],
            \"octahedron\": [\"binah\", \"kether\", \"chokmah\", \"chesed\", \"tiphareth\", \"geburah\"],
            \"hexahedron\": [\"hod\", \"netzach\", \"chesed\", \"chokmah\", \"binah\", \"geburah\"],
            \"icosahedron\": [\"kether\", \"chesed\", \"geburah\"],
            \"dodecahedron\": [\"hod\", \"netzach\", \"chesed\", \"binah\", \"geburah\"]
        }
        
        gateway_key = gateway_key.lower()
        
        if gateway_key not in gateway_mappings:
            logger.warning(f\"Unknown gateway key: {gateway_key}\")
            return None
            
        try:
            # Get Sephiroth for this gateway
            sephiroth = gateway_mappings[gateway_key]
            
            # Generate a combined tone from all Sephiroth in the gateway
            sample_rate = self.sound_generator.sample_rate
            gateway_sound = np.zeros(int(duration * sample_rate))
            
            # Generate each Sephiroth tone and combine
            for sephirah in sephiroth:
                tone = self.generate_sephirah_tone(sephirah, duration, include_field=True)
                if tone is not None:
                    # Ensure same length
                    if len(tone) > len(gateway_sound):
                        tone = tone[:len(gateway_sound)]
                    elif len(tone) < len(gateway_sound):
                        tone = np.pad(tone, (0, len(gateway_sound) - len(tone)))
                    
                    # Add to gateway sound
                    gateway_sound += tone
            
            # Add gateway-specific modulation
            time = np.linspace(0, duration, len(gateway_sound), endpoint=False)
            
            if gateway_key == \"tetrahedron\":
                # Fire element - dynamic, transformative
                mod_freq = 2.0  # 0.5-second cycle (energetic)
                gateway_sound *= (0.2 * np.sin(2 * np.pi * mod_freq * time) ** 2 + 0.8)
                
            elif gateway_key == \"octahedron\":
                # Air element - flowing, mental
                mod_freq = 0.5  # 2-second cycle (flowing)
                gateway_sound *= (0.15 * np.sin(2 * np.pi * mod_freq * time) + 0.85)
                
            elif gateway_key == \"hexahedron\":
                # Earth element - stable, grounded
                mod_freq = 0.25  # 4-second cycle (stable)
                gateway_sound *= (0.1 * np.sin(2 * np.pi * mod_freq * time) ** 3 + 0.9)
                
            elif gateway_key == \"icosahedron\":
                # Water element - flowing, emotional
                mod_freq = 0.33  # 3-second cycle (wavelike)
                gateway_sound *= (0.2 * (np.sin(2 * np.pi * mod_freq * time) + 1) / 2 + 0.8)
                
            elif gateway_key == \"dodecahedron\":
                # Aether element - spiritual, transcendent
                mod_freq = 0.1  # 10-second cycle (slow, cosmic)
                
                # Create complex cosmic modulation
                mod = (np.sin(2 * np.pi * mod_freq * time) + 
                      0.5 * np.sin(2 * np.pi * mod_freq *`
}
mod_freq = 0.1  # 10-second cycle (slow, cosmic)
                
                # Create complex cosmic modulation
                mod = (np.sin(2 * np.pi * mod_freq * time) + 
                      0.5 * np.sin(2 * np.pi * mod_freq * PHI * time) +
                      0.3 * np.sin(2 * np.pi * mod_freq * 2 * time))
                
                # Normalize modulation
                mod = 0.15 * ((mod / 1.8) + 1) + 0.85
                
                gateway_sound *= mod
            
            # Normalize the final sound
            max_amp = np.max(np.abs(gateway_sound))
            if max_amp > 0:
                gateway_sound = gateway_sound / max_amp * 0.8
            
            logger.info(f"Generated gateway sound for {gateway_key} using {len(sephiroth)} Sephiroth")
            return gateway_sound
            
        except Exception as e:
            logger.error(f"Error generating gateway sound: {str(e)}")
            return None
    
    def save_sephiroth_tone(self, sephirah, duration=10.0, include_field=False, filename=None):
        """
        Generate and save a Sephiroth tone to a WAV file.
        
        Args:
            sephirah (str): Name of the Sephiroth
            duration (float): Duration in seconds
            include_field (bool): Whether to include field harmonics
            filename (str): Custom filename (optional)
            
        Returns:
            str: Path to the saved file, or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save Sephiroth tone: SoundGenerator not available")
            return None
            
        # Generate the tone
        tone = self.generate_sephirah_tone(sephirah, duration, include_field)
        if tone is None:
            return None
            
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            field_suffix = "_with_field" if include_field else ""
            filename = f"{sephirah.lower()}_tone{field_suffix}_{timestamp}.wav"
        
        # Save the tone
        filepath = self.sound_generator.save_sound(
            tone, 
            filename=filename,
            field_type="sephiroth", 
            description=f"{sephirah.capitalize()} tone"
        )
        
        return filepath
    
    def save_gateway_sound(self, gateway_key, duration=20.0, filename=None):
        """
        Generate and save a gateway sound to a WAV file.
        
        Args:
            gateway_key (str): Name of the gateway key
            duration (float): Duration in seconds
            filename (str): Custom filename (optional)
            
        Returns:
            str: Path to the saved file, or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save gateway sound: SoundGenerator not available")
            return None
            
        # Generate the gateway sound
        sound = self.generate_gateway_sound(gateway_key, duration)
        if sound is None:
            return None
            
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{gateway_key.lower()}_gateway_{timestamp}.wav"
        
        # Save the sound
        filepath = self.sound_generator.save_sound(
            sound, 
            filename=filename,
            field_type="gateway", 
            description=f"{gateway_key.capitalize()} gateway"
        )
        
        return filepath
    
    def save_tree_of_life_progression(self, duration_per_sephirah=5.0, 
                                     include_paths=True, direction="ascending", filename=None):
        """
        Generate and save a Tree of Life progression to a WAV file.
        
        Args:
            duration_per_sephirah (float): Duration for each Sephiroth in seconds
            include_paths (bool): Whether to include transition paths
            direction (str): "ascending" or "descending"
            filename (str): Custom filename (optional)
            
        Returns:
            str: Path to the saved file, or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save Tree of Life progression: SoundGenerator not available")
            return None
            
        # Generate the progression
        progression = self.generate_tree_of_life_progression(
            duration_per_sephirah, include_paths, direction
        )
        if progression is None:
            return None
            
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            paths_suffix = "_with_paths" if include_paths else ""
            filename = f"tree_of_life_{direction}{paths_suffix}_{timestamp}.wav"
        
        # Save the progression
        filepath = self.sound_generator.save_sound(
            progression, 
            filename=filename,
            field_type="tree_of_life", 
            description=f"Tree of Life {direction} progression"
        )
        
        return filepath

    def generate_sephiroth_field_activation(self, sephirah, duration=60.0):
        """
        Generate a complete field activation sequence for a specific Sephiroth.
        
        This creates a sound specifically designed to enhance Sephiroth field formation
        and resonance connection.
        
        Args:
            sephirah (str): Name of the Sephiroth field to activate
            duration (float): Total duration in seconds
            
        Returns:
            numpy.ndarray: Audio waveform array or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot generate field activation: SoundGenerator not available")
            return None
            
        sephirah = sephirah.lower()
        
        try:
            # Get the base Sephiroth tone with field characteristics
            tone = self.generate_sephirah_tone(sephirah, duration, include_field=True)
            if tone is None:
                return None
                
            # Create time array
            sample_rate = self.sound_generator.sample_rate
            time = np.linspace(0, duration, len(tone), endpoint=False)
            
            # Add initialization phase (build-up)
            init_duration = min(15.0, duration / 4)
            init_samples = int(init_duration * sample_rate)
            
            if init_samples > 0 and init_samples < len(tone):
                # Create a rising amplitude envelope
                init_envelope = np.linspace(0, 1, init_samples) ** 0.5  # Square root for more natural build-up
                tone[:init_samples] *= init_envelope
            
            # Add stabilization phase near the end
            stabilize_duration = min(15.0, duration / 4)
            stabilize_samples = int(stabilize_duration * sample_rate)
            
            if stabilize_samples > 0 and stabilize_samples < len(tone):
                # Create pulsing effect
                pulse_freq = 0.2  # Hz (5-second cycle)
                stabilize_time = time[-stabilize_samples:]
                pulse = 0.2 * np.sin(2 * np.pi * pulse_freq * stabilize_time) + 0.8
                
                # Apply to the end portion of the sound
                tone[-stabilize_samples:] *= pulse
            
            # Add sacred geometry frequencies
            # Generate a sacred chord using the sound generator
            sacred_chord = self.sound_generator.generate_sacred_chord(
                base_frequency=self.frequencies.get_frequency(sephirah),
                duration=duration
            )
            
            # Ensure same length
            if sacred_chord is not None:
                if len(sacred_chord) > len(tone):
                    sacred_chord = sacred_chord[:len(tone)]
                elif len(sacred_chord) < len(tone):
                    sacred_chord = np.pad(sacred_chord, (0, len(tone) - len(sacred_chord)))
                
                # Mix tones: 70% Sephiroth, 30% sacred
                activation_sound = 0.7 * tone + 0.3 * sacred_chord
            else:
                activation_sound = tone
            
            # Add Sephiroth-specific activation patterns
            # Higher Sephiroth have more ethereal patterns
            if sephirah in ["kether", "chokmah", "binah"]:
                # Add cosmic pulsing
                cosmic_freq = 0.05  # Very slow 20-second cycle
                cosmic_pulse = 0.15 * np.sin(2 * np.pi * cosmic_freq * time) ** 2 + 0.85
                activation_sound *= cosmic_pulse
            
            # Middle Sephiroth have balanced patterns
            elif sephirah in ["chesed", "geburah", "tiphareth"]:
                # Add harmonic enhancement in middle section
                mid_start = int(duration * 0.3 * sample_rate)
                mid_end = int(duration * 0.7 * sample_rate)
                
                if mid_start < mid_end and mid_end < len(activation_sound):
                    # Calculate phi-based frequency
                    phi_freq = self.frequencies.get_frequency(sephirah) * PHI
                    mid_time = time[mid_start:mid_end]
                    
                    # Create harmonic enhancement
                    enhancement = 0.3 * np.sin(2 * np.pi * phi_freq * mid_time)
                    activation_sound[mid_start:mid_end] += enhancement
            
            # Lower Sephiroth have grounding patterns
            else:
                # Add earthy thrumming in bass range
                earth_freq = self.frequencies.get_frequency(sephirah) * 0.25  # Low frequency
                earth_wave = 0.2 * np.sin(2 * np.pi * earth_freq * time)
                activation_sound += earth_wave
            
            # Normalize final activation sound
            max_amp = np.max(np.abs(activation_sound))
            if max_amp > 0:
                activation_sound = activation_sound / max_amp * 0.8
            
            logger.info(f"Generated {sephirah} field activation sequence ({duration}s)")
            return activation_sound
            
        except Exception as e:
            logger.error(f"Error generating field activation: {str(e)}")
            return None
    
    def save_sephiroth_field_activation(self, sephirah, duration=60.0, filename=None):
        """
        Generate and save a Sephiroth field activation sequence to a WAV file.
        
        Args:
            sephirah (str): Name of the Sephiroth
            duration (float): Duration in seconds
            filename (str): Custom filename (optional)
            
        Returns:
            str: Path to the saved file, or None if generation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot save field activation: SoundGenerator not available")
            return None
            
        # Generate the activation sequence
        activation = self.generate_sephiroth_field_activation(sephirah, duration)
        if activation is None:
            return None
            
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{sephirah.lower()}_field_activation_{timestamp}.wav"
        
        # Save the activation sequence
        filepath = self.sound_generator.save_sound(
            activation, 
            filename=filename,
            field_type="sephiroth", 
            description=f"{sephirah.capitalize()} field activation"
        )
        
        return filepath


# Integration function to extend the field system with Sephiroth sounds
def extend_field_system_with_sephiroth_sounds(field_system):
    """
    Extend a field system with Sephiroth sound generation capabilities.
    
    Args:
        field_system: The field system to extend
        
    Returns:
        bool: True if extension was successful, False otherwise
    """
    if field_system is None:
        logger.error("Cannot extend None field system")
        return False
    
    try:
        # Check if sound_generator attribute exists
        if not hasattr(field_system, 'sound_generator') or field_system.sound_generator is None:
            logger.warning("Field system has no sound generator, cannot extend")
            return False
        
        # Create Sephiroth sound generator
        sephiroth_generator = SephirothSoundGenerator(
            base_frequency=field_system.base_frequency,
            output_dir=field_system.sound_generator.output_dir
        )
        
        # Add to field system
        field_system.sephiroth_sound_generator = sephiroth_generator
        
        # Add method to generate Sephiroth fields with appropriate sound
        def generate_sephiroth_field_sound(self, sephirah, duration=30.0):
            """
            Generate a sound for a Sephiroth field, enhancing field formation.
            
            Args:
                sephirah (str): Name of the Sephiroth
                duration (float): Duration in seconds
                
            Returns:
                numpy.ndarray: Sound waveform or None if failed
            """
            if not hasattr(self, 'sephiroth_sound_generator'):
                logger.error("Field system has no Sephiroth sound generator")
                return None
                
            return self.sephiroth_sound_generator.generate_sephiroth_field_activation(sephirah, duration)
        
        # Add the method to the field system
        import types
        field_system.generate_sephiroth_field_sound = types.MethodType(
            generate_sephiroth_field_sound, field_system
        )
        
        logger.info("Successfully extended field system with Sephiroth sounds")
        return True
        
    except Exception as e:
        logger.error(f"Error extending field system: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # Create the Sephiroth sound generator
    generator = SephirothSoundGenerator()
    
    # Print available Sephiroth frequencies
    print("Sephiroth Frequencies:")
    for sephirah, freq in generator.frequencies.frequencies.items():
        print(f"  {sephirah.capitalize()}: {freq:.2f} Hz")
    
    # Check if sound generator is available
    if generator.sound_generator is not None:
        # Generate and save a Tiphareth tone
        tiphareth_path = generator.save_sephiroth_tone("tiphareth", 10.0, include_field=True)
        print(f"Generated Tiphareth tone: {tiphareth_path}")
        
        # Generate and save a Tree of Life progression
        progression_path = generator.save_tree_of_life_progression(
            duration_per_sephirah=3.0,
            include_paths=True,
            direction="descending"
        )
        print(f"Generated Tree of Life progression: {progression_path}")
        
        # Generate and save a tetrahedron gateway sound
        gateway_path = generator.save_gateway_sound("tetrahedron", 15.0)
        print(f"Generated tetrahedron gateway sound: {gateway_path}")
        
        # Generate and save a Tiphareth field activation
        activation_path = generator.save_sephiroth_field_activation("tiphareth", 30.0)
        print(f"Generated Tiphareth field activation: {activation_path}")
    else:
        print("\nSound generator not available, cannot create sounds")
        print("This module requires the sound_generator.py module")