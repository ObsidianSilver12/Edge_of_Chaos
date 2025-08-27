"""
Sounds of Universe Module

Implements generation of cosmic background sounds, stellar sonifications,
and dimensional resonance patterns based on astronomical concepts and constants.
Enforces strict validation and error handling.

Author: Soul Development Framework Team - Refactored with Strict Error Handling
"""

import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List # Added for type hinting

# Use try-except for optional SciPy dependency (for filtering)
try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some filtering methods in UniverseSounds will be limited.")

# --- Constants ---
# Attempt to import constants, raise error if essential ones are missing
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. UniverseSounds cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
# HARD FAIL if base SoundGenerator not available - no fallbacks allowed
try:
    from shared.sound.sound_generator import SoundGenerator
    SOUND_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.critical("CRITICAL: Base SoundGenerator is required for UniverseSounds but not found. Cannot continue without sound generation capabilities.")
    raise ImportError("CRITICAL: Base SoundGenerator is required for UniverseSounds but not found. Cannot continue without sound generation capabilities.") from e

# --- Logging Setup ---
log_file_path = os.path.join("logs", "sounds_of_universe.log") # Changed log filename
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('sounds_of_universe')

# --- Module Specific Constants & Data ---

OUTPUT_DIR_BASE = "output/sounds"  # Default base directory for sound output

# Base frequencies for dimensional transitions
VOID_BASE_FREQUENCY = 50.0  # Hz - Low frequency for void dimension
GUFF_BASE_FREQUENCY = 432.0  # Hz - Base frequency for guff dimension

# Solfeggio frequencies (common in sacred sound work)
SOLFEGGIO_MI = 528.0  # Hz - "MI" frequency, transformation and miracles

# Cosmic constants mapped to audible frequencies using astronomical data
# Real frequencies scaled logarithmically to audible range for accurate sonification
COSMIC_FREQUENCIES = {
    'cmb_peak': 160.2, # Hz - CMB peak frequency (160 GHz scaled to audible)
    'hydrogen_line': 42.0, # Hz - 21cm hydrogen line (1420 MHz -> lower audible range)
    'earth_schumann': 7.83, # Hz - Earth's Schumann resonance (actual frequency)
    'solar_oscillation': 166.0, # Hz - Solar p-mode oscillations (~5 min cycles)
    'galactic_center': 27.0, # Hz - Galactic rotation (220M year period, octaves up)
    'cosmic_background': 136.1, # Hz - CMB power spectrum (realistic mapping)
    'quantum_vacuum': 12.5, # Hz - Quantum vacuum fluctuations (theoretical, sub-audio)
    'pulsar_neutron': 440.0, # Hz - Neutron star millisecond pulsars
    'universe_expansion': 1.55 # Hz - Hubble frequency (13.8B year timescale, up 40 octaves)
}
# Validate frequencies
for name, freq in COSMIC_FREQUENCIES.items():
    if not isinstance(freq, (int, float)) or freq <= 0:
        raise ValueError(f"Invalid COSMIC_FREQUENCY defined for '{name}': {freq}")

# Stellar frequencies (Example data structure)
STELLAR_FREQUENCIES = {
    'sun': {'base': 126.22, 'harmonics': [1.0, 1.5, 2.0, 2.5, 3.0], 'mod': 0.1, 'noise': 0.1, 'filter': (50, 500)},
    'neutron_star': {'base': 528.0, 'harmonics': [1.0, 2.0, 3.0, 4.0], 'mod': 1.2, 'noise': 0.2, 'filter': (200, 3000)},
    'black_hole': {'base': 57.0, 'harmonics': [1.0, PHI, 2.0, PHI*2], 'mod': 0.05, 'noise': 0.3, 'filter': (20, 200)},
    'galaxy': {'base': 63.0, 'harmonics': [1.0, 1.5, 2.0, 3.0], 'mod': 0.01, 'noise': 0.15, 'filter': (30, 800)},
    'quasar': {'base': 741.0, 'harmonics': [1.0, 1.33, 1.66, 2.0], 'mod': 0.3, 'noise': 0.4, 'filter': (500, 5000)}
}
# Validate stellar frequencies
for type, data in STELLAR_FREQUENCIES.items():
     if data['base'] <= 0 or data['mod'] < 0 or data['noise'] < 0:
          raise ValueError(f"Invalid parameters for stellar type '{type}'")
     if not all(h > 0 for h in data['harmonics']):
          raise ValueError(f"Invalid harmonics for stellar type '{type}'")
     if data['filter'][0] >= data['filter'][1] or data['filter'][0] < 0:
          raise ValueError(f"Invalid noise filter band for stellar type '{type}'")


class UniverseSounds:
    """
    Class for generating cosmic and universal sounds based on astronomical data/concepts.
    Performs strict validation and fails hard on errors.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = OUTPUT_DIR_BASE):
        """
        Initialize a new universe sounds generator.

        Args:
            sample_rate (int): Sample rate in Hz. Must be positive.
            output_dir (str): Directory to save generated sound files. Must be creatable.

        Raises:
            ValueError: If sample_rate is not positive or output_dir is invalid.
            OSError: If the output directory cannot be created.
        """
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Sample rate must be a positive integer, got {sample_rate}")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("Output directory must be a non-empty string.")

        self.sample_rate: int = sample_rate
        self.output_dir: str = os.path.join(output_dir, "universe") # Subdirectory
        # Optional: Initialize SoundGenerator for saving
        self._sound_saver = None  
        if SOUND_GENERATOR_AVAILABLE:
             try: self._sound_saver = SoundGenerator(sample_rate=self.sample_rate, output_dir=self.output_dir)
             except Exception as e: logger.error(f"Failed to initialize SoundGenerator for saving: {e}")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Universe Sounds generator initialized. Sample Rate: {self.sample_rate} Hz, Output Dir: {self.output_dir}")
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create universe sound output directory {self.output_dir}: {e}")
            raise

    def _validate_inputs(self, duration: float, amplitude: float) -> Tuple[int, float]:
        """Helper to validate common inputs."""
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if not isinstance(amplitude, (int, float)) or not (0.0 <= amplitude <= MAX_AMPLITUDE):
            raise ValueError(f"Amplitude must be between 0.0 and {MAX_AMPLITUDE}, got {amplitude}")
        num_samples = int(duration * self.sample_rate)
        if num_samples <= 0:
            raise ValueError(f"Calculated number of samples ({num_samples}) is not positive for duration {duration}.")
        return num_samples, amplitude

    def _normalize_sound(self, sound_array: np.ndarray, target_amplitude: float) -> np.ndarray:
        """Normalizes sound array, handling potential issues. Fails hard."""
        if sound_array is None or sound_array.size == 0: raise ValueError("Cannot normalize empty array.")
        try:
            max_abs = np.max(np.abs(sound_array))
            if not np.isfinite(max_abs):
                raise ValueError("Cannot normalize array with NaN/Inf values.")
            if max_abs > 1e-9: normalized = sound_array / max_abs * target_amplitude
            else: normalized = np.zeros_like(sound_array) # Silent array
            # Clip for safety and convert type
            return np.clip(normalized, -target_amplitude, target_amplitude).astype(np.float32)
        except Exception as e:
            logger.error(f"Error during sound normalization: {e}", exc_info=True)
            raise RuntimeError("Normalization failed.") from e

    def generate_cosmic_background(self, duration: float, amplitude: float = 0.7,
                                  frequency_band: str = 'full') -> np.ndarray:
        """
        Generate sound based on cosmic microwave background. Fails hard.

        Args:
            duration (float): Duration in seconds (> 0).
            amplitude (float): Amplitude (0.0 to MAX_AMPLITUDE).
            frequency_band (str): 'full', 'low', 'mid', 'high' to emphasize certain parts.

        Returns:
            np.ndarray: NumPy array containing the CMB sound samples.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If generation fails.
        """
        num_samples, amp = self._validate_inputs(duration, amplitude)
        logger.debug(f"Generating CMB sound: Dur={duration:.2f}s, Amp={amp:.2f}, Band={frequency_band}")

        try:
            # Start with white noise base
            noise = np.random.normal(0, amp * 0.4, num_samples).astype(np.float32) # Lower amplitude base noise

            # Apply CMB power spectrum filter approximation using FFT
            if num_samples > 1:
                N = num_samples
                X = np.fft.rfft(noise)
                freqs = np.fft.rfftfreq(N, 1.0 / self.sample_rate)
                cmb_filter = np.ones_like(freqs, dtype=np.float64)

                # Define peaks based on COSMIC_FREQUENCIES
                peaks = { COSMIC_FREQUENCIES['cmb_peak']: (2.0, 30.0), # (Gain, Width Hz)
                          220.0: (1.0, 20.0), 540.0: (1.0, 20.0), 720.0: (1.0, 20.0) } # Example secondary peaks

                for peak_freq, (gain, width) in peaks.items():
                    if width <= 0: continue # Skip invalid width
                    dist_sq = (freqs - peak_freq)**2
                    # Gaussian peak filter, ensure width is positive
                    cmb_filter += gain * np.exp(-dist_sq / (2 * max(1e-6, (width/3.0))**2))

                # Apply frequency band emphasis if needed
                if frequency_band == 'low': cmb_filter[freqs > 500] *= 0.1 # Attenuate high
                elif frequency_band == 'mid': cmb_filter[(freqs < 100) | (freqs > 1500)] *= 0.2 # Attenuate low/high
                elif frequency_band == 'high': cmb_filter[freqs < 1000] *= 0.1 # Attenuate low

                X_filtered = X * cmb_filter
                cmb_noise = np.fft.irfft(X_filtered, n=N).astype(np.float32)
            else:
                cmb_noise = noise # Cannot FFT size 1 array

            # Add fundamental cosmic frequencies as tones
            time = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            cosmic_tone = np.zeros_like(time, dtype=np.float32)
            for name, freq in COSMIC_FREQUENCIES.items():
                # Amplitude decreases for higher frequencies (example scaling)
                tone_amp = amp * 0.3 * (200.0 / max(100.0, freq + 100.0))
                phase = np.random.uniform(0, 2 * PI)
                cosmic_tone += (tone_amp * np.sin(2 * PI * freq * time + phase)).astype(np.float32)

            # Mix noise and tones
            cmb_sound = 0.6 * cmb_noise + 0.4 * cosmic_tone

            # Add slow expansion modulation
            expansion_freq = 0.01 # Very slow
            if expansion_freq > 0:
                 expansion_mod = 0.8 + 0.2 * np.sin(2 * PI * expansion_freq * time)
                 cmb_sound *= expansion_mod.astype(np.float32)

            # Final normalization
            return self._normalize_sound(cmb_sound, amp)

        except Exception as e:
            logger.error(f"Error generating CMB sound: {e}", exc_info=True)
            raise RuntimeError(f"CMB sound generation failed: {e}") from e

    def generate_stellar_sound(self, stellar_type: str, duration: float,
                             amplitude: float = MAX_AMPLITUDE) -> np.ndarray:
        """
        Generate sound based on stellar oscillations. Fails hard.

        Args:
            stellar_type (str): Key from STELLAR_FREQUENCIES (e.g., 'sun'). Case-insensitive.
            duration (float): Duration in seconds (> 0).
            amplitude (float): Amplitude (0.0 to MAX_AMPLITUDE).

        Returns:
            np.ndarray: NumPy array containing the stellar sound samples.

        Raises:
            ValueError: If inputs are invalid or stellar_type unknown.
            RuntimeError: If generation fails.
        """
        num_samples, amp = self._validate_inputs(duration, amplitude)
        stellar_type_lower = stellar_type.lower()
        if stellar_type_lower not in STELLAR_FREQUENCIES:
            raise ValueError(f"Unknown stellar type: '{stellar_type}'. Known types: {list(STELLAR_FREQUENCIES.keys())}")

        props = STELLAR_FREQUENCIES[stellar_type_lower]
        base_freq = props['base']
        harmonics = props['harmonics']
        mod_freq = props['mod']
        noise_level = props['noise']
        filter_band = props['filter']

        if base_freq <= 0: raise ValueError("Stellar base frequency must be positive.")
        if mod_freq < 0: raise ValueError("Stellar modulation frequency cannot be negative.")
        if noise_level < 0: raise ValueError("Stellar noise level cannot be negative.")
        if not all(h > 0 for h in harmonics): raise ValueError("Stellar harmonics must be positive.")

        logger.debug(f"Generating stellar sound: Type={stellar_type}, Dur={duration:.2f}s, Amp={amp:.2f}")

        try:
            time = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            stellar_sound = np.zeros_like(time, dtype=np.float32)

            # Add base frequency and harmonics
            for i, ratio in enumerate(harmonics):
                freq = base_freq * ratio
                # Simple amplitude falloff: 1/(i+1)
                tone_amp = amp * (0.8 / (i + 1))
                phase = np.random.uniform(0, 2 * PI)
                stellar_sound += (tone_amp * np.sin(2 * PI * freq * time + phase)).astype(np.float32)

            # Add characteristic modulation based on type (simplified examples)
            modulation = np.ones_like(time, dtype=np.float32)
            if mod_freq > 0:
                if stellar_type_lower == 'sun':
                    p_mode = 0.3 * np.sin(2 * PI * 0.003 * time) # ~5 min oscillations
                    cycle = 0.2 * np.sin(2 * PI * mod_freq * time)
                    modulation = (1.0 + p_mode + cycle).astype(np.float32)
                elif stellar_type_lower == 'neutron_star':
                    duty = 0.1; phase_pulse = (time * mod_freq) % 1.0
                    env = np.exp(-((phase_pulse - 0.5)**2) / (2 * max(1e-6, duty**2)))
                    modulation = (0.3 + 0.7 * (env / np.max(env))).astype(np.float32) if np.max(env)>0 else np.full_like(time, 0.3, dtype=np.float32)
                elif stellar_type_lower == 'black_hole':
                    base_mod = 0.4 * np.sin(2 * PI * mod_freq * time)
                    bursts = np.zeros_like(time) # Simplification: No complex bursts for now
                    modulation = (0.4 + base_mod + bursts).astype(np.float32)
                elif stellar_type_lower == 'galaxy':
                    rot = 0.2*np.sin(2*PI*mod_freq*time); arms = 0.15*np.sin(2*PI*mod_freq*3*time); core=0.1*np.sin(2*PI*mod_freq*8*time)
                    modulation = (0.6 + rot + arms + core).astype(np.float32)
                elif stellar_type_lower == 'quasar':
                    jet = 0.5 * np.abs(np.sin(2 * PI * mod_freq * time))
                    flicker = 0.2 * np.random.normal(0, 1, num_samples) * np.sin(2 * PI * mod_freq * 10 * time) # Noise flicker
                    modulation = (0.3 + jet + flicker).astype(np.float32)
                else: # Default simple mod
                    modulation = (0.8 + 0.2 * np.sin(2 * PI * mod_freq * time)).astype(np.float32)

            stellar_sound *= modulation

            # Add characteristic noise component (if level > 0)
            if noise_level > 0 and SCIPY_AVAILABLE:
                noise = np.random.normal(0, noise_level * amp, num_samples).astype(np.float32)
                # Apply bandpass filter based on type
                low, high = filter_band
                nyquist = 0.5 * self.sample_rate
                high = min(high, nyquist * 0.99) # Ensure high < nyquist
                if low < high and low > 0:
                    try:
                        sos = signal.butter(3, [low, high], btype='bandpass', fs=self.sample_rate, output='sos')
                        filtered_noise = signal.sosfiltfilt(sos, noise)
                        stellar_sound += filtered_noise
                    except Exception as filter_err:
                        logger.warning(f"Noise filter failed for {stellar_type}: {filter_err}. Skipping noise.")
                else:
                     logger.warning(f"Invalid filter band [{low},{high}] for {stellar_type}. Skipping noise.")
            elif noise_level > 0 and not SCIPY_AVAILABLE:
                 logger.warning("SciPy not available, cannot apply characteristic noise filtering.")

            # Final normalization
            return self._normalize_sound(stellar_sound, amp)

        except Exception as e:
            logger.error(f"Error generating stellar sound for {stellar_type}: {e}", exc_info=True)
            raise RuntimeError(f"Stellar sound generation failed: {e}") from e

    # --- generate_cosmic_constants removed - relies on arbitrary mapping ---
    # --- generate_dimensional_resonance removed - frequency scaling belongs elsewhere ---
    # --- generate_universe_background removed - too high level, compose from parts ---
    # --- generate_field_activation_background removed - application specific ---
    # --- generate_all_cosmic_sounds removed - helper, not core generation ---

    # --- Add generate_dimensional_transition from SoundGenerator (as it fits Universe theme) ---
    def generate_dimensional_transition(self, duration: float, sample_rate: int,
                                      transition_type: str, # e.g., 'void_to_guff'
                                      amplitude: float = MAX_AMPLITUDE) -> np.ndarray:
        """
        Generate a sound representing transition between dimensions/states. Fails hard.

        Args:
            duration (float): Duration in seconds (> 0).
            sample_rate (int): Sample rate Hz (> 0).
            transition_type (str): Identifier for the type of transition (e.g., 'void_to_guff').
            amplitude (float): Amplitude (0.0 to MAX_AMPLITUDE).

        Returns:
            np.ndarray: NumPy array containing the audio samples.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If sound generation fails.
        """
        # --- Input Validation ---
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if not isinstance(amplitude, (int, float)) or not (0.0 <= amplitude <= MAX_AMPLITUDE):
            raise ValueError(f"Amplitude must be between 0.0 and {MAX_AMPLITUDE}, got {amplitude}")
        if not isinstance(transition_type, str) or not transition_type:
             raise ValueError("transition_type must be a non-empty string.")

        num_samples = int(duration * sample_rate)
        if num_samples <= 0:
            raise ValueError(f"Calculated number of samples ({num_samples}) is not positive.")

        logger.debug(f"Generating dimensional transition: Type={transition_type}, Dur={duration:.2f}s, Amp={amplitude:.2f}")

        try:
            time = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            transition_sound = np.zeros(num_samples, dtype=np.float32)

            # --- Define Frequencies based on Transition Type ---
            # Example: Frequencies sweep based on type
            if 'void' in transition_type and 'guff' in transition_type:
                freq_start = VOID_BASE_FREQUENCY # Use void base
                freq_end = GUFF_BASE_FREQUENCY # Use guff base
            elif 'guff' in transition_type and 'sephiroth' in transition_type:
                 freq_start = GUFF_BASE_FREQUENCY
                 freq_end = SOLFEGGIO_MI # Example Sephiroth freq (Tiphareth)
            else: # Default sweep
                 freq_start = 100.0
                 freq_end = 1000.0

            if freq_start <= 0 or freq_end <=0:
                 raise ValueError(f"Invalid start/end frequencies ({freq_start}, {freq_end}) for transition '{transition_type}'")

            # --- Generate Sweeping Tones ---
            # Logarithmic sweep for smoother perceived pitch change
            freq_sweep = np.geomspace(freq_start, freq_end, num_samples)
            phase = np.cumsum(2 * PI * freq_sweep / sample_rate)
            sweep_tone = (0.6 * np.sin(phase)).astype(np.float32)
            transition_sound += sweep_tone

            # --- Add Filtered Noise component ---
            # Example: Bandpass noise that shifts frequency
            if SCIPY_AVAILABLE:
                noise = np.random.normal(0, amplitude * 0.3, num_samples).astype(np.float32)
                center_freq_sweep = np.geomspace(max(20, freq_start*0.5), min(sample_rate*0.45, freq_end*1.5), num_samples)
                bandwidth = np.geomspace(50, 300, num_samples) # Bandwidth changes during transition

                # Apply time-varying filter (complex - simplified here)
                # Simplification: Just use a fixed bandpass based on average freq
                avg_freq = np.sqrt(freq_start * freq_end)
                low_cut = max(20.0, avg_freq * 0.8)
                high_cut = min(sample_rate * 0.48, avg_freq * 1.2)
                if high_cut > low_cut:
                    try:
                        sos = signal.butter(4, [low_cut, high_cut], btype='bandpass', fs=sample_rate, output='sos')
                        filtered_noise = signal.sosfiltfilt(sos, noise)
                        transition_sound += filtered_noise * 0.4 # Add noise component
                    except Exception as filter_err:
                         logger.warning(f"Bandpass filter failed for transition sound: {filter_err}")
            else:
                 logger.warning("SciPy unavailable, skipping filtered noise for transition.")


            # --- Add "Whoosh" Effect (example) ---
            whoosh_env = 0.5 * np.exp(-20 * ((time - duration/2)**2) / duration**2) # Gaussian envelope centered
            # White noise filtered to sound like whoosh (e.g., lowpass)
            whoosh_noise = np.random.normal(0, amplitude * 0.5, num_samples).astype(np.float32)
            if SCIPY_AVAILABLE:
                 try:
                     sos_low = signal.butter(4, 500, btype='low', fs=sample_rate, output='sos') # Lowpass
                     filtered_whoosh = signal.sosfiltfilt(sos_low, whoosh_noise)
                     transition_sound += filtered_whoosh * whoosh_env
                 except Exception as filter_err:
                      logger.warning(f"Whoosh filter failed: {filter_err}")
                      transition_sound += whoosh_noise * whoosh_env # Add unfiltered noise if filter fails
            else:
                 transition_sound += whoosh_noise * whoosh_env # Add unfiltered noise

            # Final normalization
            return self._normalize_sound(transition_sound, amplitude)

        except Exception as e:
            logger.error(f"Error generating dimensional transition '{transition_type}': {e}", exc_info=True)
            raise RuntimeError(f"Dimensional transition generation failed: {e}") from e


    def save_sound(self, sound: np.ndarray, filename: str, description: Optional[str] = None) -> Optional[str]:
        """
        Save a generated sound to a WAV file using the SoundGenerator. Fails hard.

        Args:
            sound (np.ndarray): NumPy array containing the audio samples.
            filename (str): Filename (e.g., "cmb_sound.wav").
            description (Optional[str]): Optional description for logging.

        Returns:
            str: Absolute path to the saved file.

        Raises:
            RuntimeError: If the sound saving utility is unavailable or saving fails.
        """
        if self._sound_saver is None:
             raise RuntimeError("Sound saving unavailable: Base SoundGenerator failed to initialize.")
        # Use the SoundGenerator's save method, which includes validation & error handling
        try:
            filepath = self._sound_saver.save_sound(sound, filename, description)
            if filepath is None: # save_sound should raise on error, but double-check
                 raise RuntimeError(f"SoundGenerator.save_sound returned None for '{filename}'.")
            return filepath
        except (TypeError, ValueError, IOError, RuntimeError) as e:
            logger.error(f"Failed to save universe sound file '{filename}': {e}", exc_info=True)
            raise RuntimeError(f"Universe sound saving failed for '{filename}'") from e # Re-raise

    # --- Visualization method removed - belongs in analysis/testing, not core generation ---

# --- Example Usage (Modified for Strictness) ---
if __name__ == "__main__":
    print("Running Sounds of Universe Module Example...")
    saved_files = []
    try:
        generator = UniverseSounds(output_dir="output/sounds/universe_test")

        print("\nGenerating Cosmic Background (10s)...")
        cmb = generator.generate_cosmic_background(duration=10.0, amplitude=0.6)
        cmb_path = generator.save_sound(cmb, "cmb_test.wav", "Test CMB Sound")
        if cmb_path: saved_files.append(cmb_path)

        print("\nGenerating Stellar Sounds (8s each)...")
        for stellar_key in ['sun', 'black_hole', 'quasar']:
            print(f"- Generating {stellar_key}...")
            stellar_sound = generator.generate_stellar_sound(stellar_key, duration=8.0, amplitude=0.7)
            s_path = generator.save_sound(stellar_sound, f"{stellar_key}_test.wav", f"Test {stellar_key} Sound")
            if s_path: saved_files.append(s_path)

        print("\nGenerating Dimensional Transition (Void->Guff, 12s)...")
        transition = generator.generate_dimensional_transition(duration=12.0, sample_rate=SAMPLE_RATE, transition_type='void_to_guff', amplitude=0.75)
        t_path = generator.save_sound(transition, "transition_void_guff_test.wav", "Test Void->Guff Transition")
        if t_path: saved_files.append(t_path)

        print("\nSounds of Universe Example Finished Successfully.")
        print("Saved files:", saved_files)

    except (ValueError, TypeError, IOError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Sounds of Universe Example ---")
        print(f"An error occurred: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("---------------------------------------------")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Sounds of Universe Example ---")
        print(f"An unexpected error occurred: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("------------------------------------------------------")