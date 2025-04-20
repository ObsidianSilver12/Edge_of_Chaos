"""
Sound Generator Module

This module implements the sound generation system that creates actual tones and sounds
during field generation. Sound is not just a theoretical construct but an active component
that improves energy coherence and field stability when generated during field operations.

Key functions:
- Generate real tones based on dimensional frequencies
- Create harmonic sound structures that enhance field formation
- Produce resonance patterns that strengthen quantum entanglement
- Output sound streams that can be played during field operations

Author: Soul Development Framework Team
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sound_generator.log'
)
logger = logging.getLogger('sound_generator')

# Constants for sound generation
SAMPLE_RATE = 44100  # Hz - Standard audio sample rate
MAX_AMPLITUDE = 0.8  # Maximum amplitude to prevent clipping
FUNDAMENTAL_FREQUENCY = 432.0  # Hz - Universal tuning frequency
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ~1.618
OUTPUT_DIR = "output/sounds"  # Directory for saving generated sounds


class SoundGenerator:
    """
    Class for generating actual sound files based on field frequencies.
    These sounds enhance field formation and stability when played during operations.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = OUTPUT_DIR):
        """
        Initialize a new sound generator.
        
        Args:
            sample_rate: Sample rate in Hz for audio generation
            output_dir: Directory to save generated sound files
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Sound Generator initialized with sample rate {sample_rate} Hz")
        logger.info(f"Output directory: {output_dir}")
    
    def generate_tone(self, frequency: float, duration: float = 5.0, 
                     amplitude: float = 0.8, fade_in_out: float = 0.1) -> np.ndarray:
        """
        Generate a pure sine tone.
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            amplitude: Amplitude (0-1)
            fade_in_out: Fade in/out duration in seconds
            
        Returns:
            NumPy array containing the audio samples
        """
        # Create time array
        num_samples = int(duration * self.sample_rate)
        time = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate tone
        tone = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Apply fade in/out
        if fade_in_out > 0:
            fade_samples = int(fade_in_out * self.sample_rate)
            if fade_samples > 0:
                # Ensure fade doesn't exceed half the signal length
                fade_samples = min(fade_samples, num_samples // 2)
                
                # Create linear fade-in/out envelopes
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                # Apply fades
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out
        
        logger.info(f"Generated {duration}s tone at {frequency}Hz")
        return tone
    
    def generate_harmonic_tone(self, base_frequency: float, harmonics: List[float] = None,
                             durations: List[float] = None, amplitudes: List[float] = None,
                             fade_in_out: float = 0.1) -> np.ndarray:
        """
        Generate a tone with harmonics.
        
        Args:
            base_frequency: Base frequency in Hz
            harmonics: List of harmonic ratios (e.g., [1.0, 2.0, 3.0])
            durations: List of durations for each harmonic
            amplitudes: List of amplitudes for each harmonic
            fade_in_out: Fade in/out duration in seconds
            
        Returns:
            NumPy array containing the audio samples
        """
        # Set default harmonics if not provided
        if harmonics is None:
            harmonics = [1.0, 2.0, 3.0, 4.0, 5.0]  # First 5 harmonics
        
        # Set default duration if not provided
        if durations is None:
            durations = [5.0] * len(harmonics)
        elif len(durations) < len(harmonics):
            # Extend with the last duration value
            durations.extend([durations[-1]] * (len(harmonics) - len(durations)))
        
        # Set default amplitudes if not provided
        if amplitudes is None:
            # Decreasing amplitudes for higher harmonics
            amplitudes = [0.8 / (i + 1) for i in range(len(harmonics))]
        elif len(amplitudes) < len(harmonics):
            # Extend with the last amplitude value
            amplitudes.extend([amplitudes[-1]] * (len(harmonics) - len(amplitudes)))
            
        # Find the maximum duration to determine array size
        max_duration = max(durations)
        num_samples = int(max_duration * self.sample_rate)
        
        # Create empty audio array
        combined_tone = np.zeros(num_samples)
        
        # Add each harmonic
        for i, ratio in enumerate(harmonics):
            # Calculate frequency
            freq = base_frequency * ratio
            
            # Generate individual harmonic tone
            harmonic_duration = durations[i]
            harmonic_samples = int(harmonic_duration * self.sample_rate)
            
            time = np.linspace(0, harmonic_duration, harmonic_samples, endpoint=False)
            harmonic_tone = amplitudes[i] * np.sin(2 * np.pi * freq * time)
            
            # Add to combined tone (pad with zeros if needed)
            combined_tone[:harmonic_samples] += harmonic_tone
        
        # Normalize to prevent clipping
        if np.max(np.abs(combined_tone)) > 0:
            combined_tone = combined_tone / np.max(np.abs(combined_tone)) * MAX_AMPLITUDE
        
        # Apply fade in/out
        if fade_in_out > 0:
            fade_samples = int(fade_in_out * self.sample_rate)
            if fade_samples > 0:
                # Ensure fade doesn't exceed half the signal length
                fade_samples = min(fade_samples, num_samples // 2)
                
                # Create linear fade-in/out envelopes
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                # Apply fades
                combined_tone[:fade_samples] *= fade_in
                combined_tone[-fade_samples:] *= fade_out
        
        logger.info(f"Generated harmonic tone with base frequency {base_frequency}Hz "
                   f"and {len(harmonics)} harmonics")
        return combined_tone
    
    def generate_sacred_chord(self, base_frequency: float = FUNDAMENTAL_FREQUENCY,
                           duration: float = 10.0, fade_in_out: float = 0.5) -> np.ndarray:
        """
        Generate a sacred geometry chord based on golden ratio and perfect fifths.
        
        Args:
            base_frequency: Base frequency in Hz
            duration: Duration in seconds
            fade_in_out: Fade in/out duration in seconds
            
        Returns:
            NumPy array containing the audio samples
        """
        # Sacred ratios
        ratios = [
            1.0,          # Unison
            1.5,          # Perfect fifth
            PHI,          # Golden ratio
            2.0,          # Octave
            2.0 * PHI,    # Octave + golden ratio
            3.0,          # Perfect fifth + octave
            3.0 * PHI/2   # Complex sacred ratio
        ]
        
        # Amplitudes (decreasing for higher frequencies)
        amplitudes = [0.7, 0.5, 0.6, 0.4, 0.3, 0.25, 0.2]
        
        # Generate the chord
        return self.generate_harmonic_tone(base_frequency, ratios, 
                                         [duration] * len(ratios), 
                                         amplitudes, fade_in_out)
    
    def generate_sephiroth_tone(self, sephiroth_name: str, duration: float = 10.0,
                              fade_in_out: float = 0.5) -> Optional[np.ndarray]:
        """
        Generate a tone specific to a Sephiroth based on its frequency and properties.
        
        Args:
            sephiroth_name: Name of the Sephiroth
            duration: Duration in seconds
            fade_in_out: Fade in/out duration in seconds
            
        Returns:
            NumPy array containing the audio samples, or None if Sephiroth not found
        """
        # Sephiroth frequency mapping
        sephiroth_frequencies = {
            'Kether': 963.0,      # Crown chakra
            'Chokmah': 963.0/PHI, # Derived through golden ratio
            'Binah': 852.0,       # Third eye chakra
            'Daath': 741.0,       # Throat chakra
            'Chesed': 639.0,      # Heart chakra
            'Geburah': 639.0*PHI, # Derived through golden ratio
            'Tiphareth': 528.0,   # Solar plexus / DNA repair
            'Netzach': 417.0,     # Sacral chakra
            'Hod': 417.0*PHI,     # Derived through golden ratio
            'Yesod': 396.0,       # Root chakra
            'Malkuth': 174.0      # Earth frequency
        }
        
        if sephiroth_name not in sephiroth_frequencies:
            logger.warning(f"Unknown Sephiroth: {sephiroth_name}")
            return None
        
        base_freq = sephiroth_frequencies[sephiroth_name]
        
        # Each Sephiroth has unique harmonic ratios
        harmonic_map = {
            'Kether': [1.0, PHI, 2.0, 3.0, 5.0],
            'Chokmah': [1.0, 1.5, 2.0, 3.0, PHI],
            'Binah': [1.0, 1.33, 2.0, 2.5, PHI],
            'Daath': [1.0, 1.2, 1.5, 2.0, PHI],
            'Chesed': [1.0, 1.25, 1.5, 2.0, 2.5],
            'Geburah': [1.0, 1.1, 1.33, 1.66, 2.0],
            'Tiphareth': [1.0, PHI, 2.0, PHI*2, 3.0],
            'Netzach': [1.0, 1.25, 1.5, 1.75, 2.0],
            'Hod': [1.0, 1.2, 1.33, 1.66, 2.0],
            'Yesod': [1.0, 1.125, 1.25, 1.5, 2.0],
            'Malkuth': [1.0, 1.125, 1.25, 1.33, 1.5]
        }
        
        harmonics = harmonic_map.get(sephiroth_name, [1.0, 1.5, 2.0, 2.5, 3.0])
        
        # Create specific amplitude profile for this Sephiroth
        amplitudes = [0.8]  # Start with base frequency at high amplitude
        
        # Set amplitude profile based on Sephiroth position in the Tree
        if sephiroth_name in ['Kether', 'Chokmah', 'Binah']:
            # Higher Sephiroth have stronger higher harmonics
            amplitudes.extend([0.7, 0.6, 0.5, 0.4])
        elif sephiroth_name in ['Chesed', 'Geburah', 'Tiphareth', 'Daath']:
            # Middle Sephiroth have balanced harmonics
            amplitudes.extend([0.6, 0.5, 0.4, 0.3])
        else:
            # Lower Sephiroth have weaker higher harmonics
            amplitudes.extend([0.5, 0.4, 0.3, 0.2])
        
        # Generate the tone
        tone = self.generate_harmonic_tone(base_freq, harmonics, 
                                         [duration] * len(harmonics),
                                         amplitudes, fade_in_out)
        
        logger.info(f"Generated {sephiroth_name} tone at {base_freq}Hz with {len(harmonics)} harmonics")
        return tone
    
    def generate_dimensional_transition(self, start_dimension: int, end_dimension: int,
                                     duration: float = 30.0, steps: int = 100) -> np.ndarray:
        """
        Generate a sound that facilitates transition between dimensions.
        
        Args:
            start_dimension: Starting dimension (e.g., 3 for physical reality)
            end_dimension: Ending dimension
            duration: Duration in seconds
            steps: Number of steps in the transition
            
        Returns:
            NumPy array containing the audio samples
        """
        # Calculate base frequencies for start and end dimensions
        # Using an exponential scaling where each dimension is phi times higher
        start_freq = FUNDAMENTAL_FREQUENCY * (PHI ** (start_dimension - 3))
        end_freq = FUNDAMENTAL_FREQUENCY * (PHI ** (end_dimension - 3))
        
        logger.info(f"Generating transition from {start_dimension}D ({start_freq:.2f}Hz) "
                   f"to {end_dimension}D ({end_freq:.2f}Hz)")
        
        # Create time array
        num_samples = int(duration * self.sample_rate)
        time = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Create frequency sweep array
        if start_dimension < end_dimension:
            # Ascending transition (logarithmic sweep sounds more natural)
            freq_array = np.geomspace(start_freq, end_freq, steps)
        else:
            # Descending transition
            freq_array = np.geomspace(start_freq, end_freq, steps)
        
        # Create empty audio array
        transition_sound = np.zeros(num_samples)
        
        # Calculate samples per step
        samples_per_step = num_samples // steps
        
        # Generate the transition sound
        for i, freq in enumerate(freq_array):
            start_sample = i * samples_per_step
            end_sample = min(start_sample + samples_per_step, num_samples)
            
            # Time array for this segment
            segment_time = time[start_sample:end_sample] - time[start_sample]
            
            # Generate tone for this segment
            segment_tone = 0.8 * np.sin(2 * np.pi * freq * segment_time)
            
            # Add harmonics for richer sound
            segment_tone += 0.4 * np.sin(2 * np.pi * freq * PHI * segment_time)
            segment_tone += 0.2 * np.sin(2 * np.pi * freq * 2 * segment_time)
            
            # Apply small fade at segment boundaries
            fade_samples = min(100, samples_per_step // 4)
            if fade_samples > 0:
                # Fade in
                if i > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    segment_tone[:fade_samples] *= fade_in
                
                # Fade out
                if i < steps - 1:
                    fade_out = np.linspace(1, 0, fade_samples)
                    segment_tone[-fade_samples:] *= fade_out
            
            # Add to transition sound
            transition_sound[start_sample:end_sample] += segment_tone
        
        # Normalize to prevent clipping
        if np.max(np.abs(transition_sound)) > 0:
            transition_sound = transition_sound / np.max(np.abs(transition_sound)) * MAX_AMPLITUDE
        
        # Apply overall fade in/out
        fade_samples = int(1.0 * self.sample_rate)  # 1-second fade
        if fade_samples > 0:
            # Ensure fade doesn't exceed half the signal length
            fade_samples = min(fade_samples, num_samples // 4)
            
            # Create fade envelopes
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fades
            transition_sound[:fade_samples] *= fade_in
            transition_sound[-fade_samples:] *= fade_out
        
        return transition_sound
    
    def generate_field_resonance(self, field_type: str, duration: float = 60.0) -> np.ndarray:
        """
        Generate a resonance sound specific to a field type to enhance field formation.
        
        Args:
            field_type: Type of field ('void', 'guff', 'sephiroth', 'earth')
            duration: Duration in seconds
            
        Returns:
            NumPy array containing the audio samples
        """
        # Field-specific base frequencies
        field_frequencies = {
            'void': 174.0,     # Lowest frequency - pure potential
            'guff': 285.0,     # Formation frequency
            'sephiroth': 396.0,  # Transformation frequency
            'earth': 7.83      # Schumann resonance
        }
        
        if field_type.lower() not in field_frequencies:
            logger.warning(f"Unknown field type: {field_type}. Using default frequency.")
            base_freq = FUNDAMENTAL_FREQUENCY
        else:
            base_freq = field_frequencies[field_type.lower()]
        
        logger.info(f"Generating {field_type} field resonance at base {base_freq}Hz")
        
        # Each field has specific harmonic structures
        if field_type.lower() == 'void':
            # Void has sparse, primordial harmonics
            harmonics = [1.0, PHI, 2.0, PHI*2, 3.0]
            amplitudes = [0.8, 0.6, 0.5, 0.4, 0.3]
            
            # Create a sound with quantum fluctuations
            sound = self.generate_harmonic_tone(base_freq, harmonics, 
                                              [duration] * len(harmonics),
                                              amplitudes, fade_in_out=1.0)
            
            # Add quantum noise (filtered white noise)
            noise = np.random.normal(0, 0.1, len(sound))
            
            # Filter noise to make it more pleasant
            b, a = signal.butter(3, 0.1, 'low')
            filtered_noise = signal.filtfilt(b, a, noise)
            
            # Combine with harmonic sound
            sound = sound + filtered_noise
            
        elif field_type.lower() == 'guff':
            # Guff has rich formation harmonics based on Fibonacci sequence
            fib = [1, 1, 2, 3, 5, 8, 13, 21]
            harmonics = [fib[i+1]/fib[i] for i in range(len(fib)-1)]  # Fibonacci ratios
            harmonics.insert(0, 1.0)  # Add fundamental
            
            amplitudes = [0.8]
            for i in range(1, len(harmonics)):
                amplitudes.append(0.8 / (i + 1))
            
            sound = self.generate_harmonic_tone(base_freq, harmonics, 
                                              [duration] * len(harmonics),
                                              amplitudes, fade_in_out=1.0)
            
            # Add gentle amplitude modulation for "breathing" effect
            mod_freq = 0.1  # Hz (10-second cycle)
            time = np.linspace(0, duration, len(sound), endpoint=False)
            modulation = 0.2 * np.sin(2 * np.pi * mod_freq * time) + 0.8
            sound = sound * modulation
            
        elif field_type.lower() == 'sephiroth':
            # Sephiroth has complex, multidimensional harmonics
            harmonics = [1.0, 1.5, PHI, 2.0, 2.5, 3.0, PHI*2]
            amplitudes = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
            
            sound = self.generate_harmonic_tone(base_freq, harmonics, 
                                              [duration] * len(harmonics),
                                              amplitudes, fade_in_out=1.0)
            
            # Add frequency modulation for dimensional shifting effect
            mod_depth = 5.0  # Hz
            mod_freq = 0.05  # Hz (20-second cycle)
            
            time = np.linspace(0, duration, len(sound), endpoint=False)
            fm_sound = np.zeros_like(sound)
            
            # Calculate frequency modulation
            phase = 2 * np.pi * mod_freq * time
            freq_mod = base_freq + mod_depth * np.sin(phase)
            
            # Reconstruct sound with frequency modulation
            phase_accumulator = np.cumsum(freq_mod) / self.sample_rate * 2 * np.pi
            fm_sound = 0.8 * np.sin(phase_accumulator)
            
            # Blend with original sound
            sound = 0.7 * sound + 0.3 * fm_sound
            
        elif field_type.lower() == 'earth':
            # Earth has natural cycles and Schumann resonances
            # Add first 5 Schumann resonances
            schumann = [7.83, 14.3, 20.8, 27.3, 33.8]
            amplitudes = [0.8, 0.6, 0.4, 0.3, 0.2]
            
            sound = np.zeros(int(duration * self.sample_rate))
            time = np.linspace(0, duration, len(sound), endpoint=False)
            
            for i, freq in enumerate(schumann):
                sound += amplitudes[i] * np.sin(2 * np.pi * freq * time)
            
            # Add natural rhythm (heartbeat-like)
            heartbeat_freq = 1.2  # ~72 BPM
            heartbeat = 0.3 * np.sin(2 * np.pi * heartbeat_freq * time) ** 3
            
            # Add earth resonances (deeper bass tones)
            earth_tones = 0.4 * np.sin(2 * np.pi * 33.0 * time) + 0.3 * np.sin(2 * np.pi * 55.0 * time)
            
            sound = 0.6 * sound + 0.2 * heartbeat + 0.2 * earth_tones
            
        else:
            # Default: create a simple harmonic tone
            sound = self.generate_sacred_chord(base_freq, duration)
        
        # Normalize
        if np.max(np.abs(sound)) > 0:
            sound = sound / np.max(np.abs(sound)) * MAX_AMPLITUDE
        
        return sound
    
    def save_sound(self, sound: np.ndarray, filename: str = None, 
                 field_type: str = None, description: str = None) -> str:
        """
        Save a generated sound to a WAV file.
        
        Args:
            sound: NumPy array containing the audio samples
            filename: Custom filename (optional)
            field_type: Type of field (for auto-naming)
            description: Additional description (for auto-naming)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            if field_type:
                filename = f"{field_type.lower()}_sound_{timestamp}.wav"
            elif description:
                filename = f"{description.lower().replace(' ', '_')}_{timestamp}.wav"
            else:
                filename = f"sound_{timestamp}.wav"
        
        # Add .wav extension if needed
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure sound is in correct format for WAV (int16)
        sound_int16 = np.int16(sound * 32767)
        
        # Save to WAV file
        wavfile.write(filepath, self.sample_rate, sound_int16)
        
        logger.info(f"Saved sound to {filepath}")
        return filepath
    
    def generate_field_activation_sequence(self, field_type: str, duration: float = 120.0) -> str:
        """
        Generate and save a complete field activation sound sequence.
        This creates a sound specifically designed to enhance field formation and stability.
        
        Args:
            field_type: Type of field ('void', 'guff', 'sephiroth', 'earth')
            duration: Total duration in seconds
            
        Returns:
            Path to the saved sound file
        """
        # Generate field resonance sound
        field_sound = self.generate_field_resonance(field_type, duration)
        
        # Add initialization phase (build-up)
        init_duration = min(30.0, duration / 4)
        init_samples = int(init_duration * self.sample_rate)
        
        if init_samples > 0:
            # Create a rising amplitude envelope
            init_envelope = np.linspace(0, 1, init_samples) ** 0.5  # Square root for more natural build-up
            field_sound[:init_samples] *= init_envelope
        
        # Add stabilization phase near the end
        stabilize_duration = min(30.0, duration / 4)
        stabilize_samples = int(stabilize_duration * self.sample_rate)
        
        if stabilize_samples > 0 and len(field_sound) > stabilize_samples:
            # Create pulsing effect
            pulse_freq = 0.2  # Hz (5-second cycle)
            time = np.linspace(0, stabilize_duration, stabilize_samples, endpoint=False)
            pulse = 0.2 * np.sin(2 * np.pi * pulse_freq * time) + 0.8
            
            # Apply to the end portion of the sound
            field_sound[-stabilize_samples:] *= pulse
        
        # Add sacred geometry frequencies
        sacred_chord = self.generate_sacred_chord(duration=duration)
        field_sound = 0.8 * field_sound + 0.2 * sacred_chord
        
        # Normalize final sound
        if np.max(np.abs(field_sound)) > 0:
            field_sound = field_sound / np.max(np.abs(field_sound)) * MAX_AMPLITUDE
        
        # Save the activation sequence
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{field_type.lower()}_activation_sequence_{timestamp}.wav"
        
        return self.save_sound(field_sound, filename, field_type, "activation sequence")

# Example usage
if __name__ == "__main__":
    generator = SoundGenerator()
    
    # Generate and save a sacred chord
    sacred_sound = generator.generate_sacred_chord()
    generator.save_sound(sacred_sound, "sacred_chord.wav")
    
    # Generate field sounds
    void_sound = generator.generate_field_resonance("void", 30.0)
    generator.save_sound(void_sound, "void_field.wav")
    
    guff_sound = generator.generate_field_resonance("guff", 30.0)
    generator.save_sound(guff_sound, "guff_field.wav")
    
    # Generate a dimensional transition sound (3D to 4D)
    transition_sound = generator.generate_dimensional_transition(3, 4, 60.0)
    generator.save_sound(transition_sound, "transition_3d_to_4d.wav")
    
    # Generate a complete field activation sequence
    activation_path = generator.generate_field_activation_sequence("void", 60.0)
    print(f"Created field activation sequence: {activation_path}")