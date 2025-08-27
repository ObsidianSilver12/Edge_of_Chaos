"""
Noise Generator Module

Implements generators for various types of noise signals (white, pink, brown, etc.)
used as foundational elements or influences within the field system.
Enforces strict validation and error handling.

Author: Soul Development Framework Team - Refactored with Strict Error Handling
"""

import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Union, Tuple, Any # Added for type hinting

# Use try-except for optional SciPy dependency
try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some noise filtering methods (e.g., quantum) will be limited or unavailable.")

# --- Constants ---
# Attempt to import constants, raise error if essential ones are missing
try:
    from shared.constants.constants import *
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import essential constants: {e}. NoiseGenerator cannot function.")
    raise ImportError(f"Essential constants missing: {e}") from e

# --- Dependency Imports ---
# HARD FAIL if base SoundGenerator not available - no fallbacks allowed
try:
    from shared.sound.sound_generator import SoundGenerator
    SOUND_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.critical("CRITICAL: Base SoundGenerator is required for NoiseGenerator but not found. Cannot continue without sound generation capabilities.")
    raise ImportError("CRITICAL: Base SoundGenerator is required for NoiseGenerator but not found. Cannot continue without sound generation capabilities.") from e

# --- Logging Setup ---
log_file_path = os.path.join("logs", "noise_generator.log") # Changed log filename
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# Provide default values if not imported from constants
if 'LOG_LEVEL' not in globals():
    LOG_LEVEL = logging.INFO
if 'LOG_FORMAT' not in globals():
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('noise_generator')

# Define supported noise types explicitly
SUPPORTED_NOISE_TYPES = ['white', 'pink', 'brown', 'blue', 'violet', 'quantum', 'edge_of_chaos']

class NoiseGenerator:
    """
    Class for generating various types of noise waveforms.
    Different noise types have specific spectral properties and uses.
    Performs strict validation and fails hard on errors.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = DATA_DIR_BASE):
        """
        Initialize a new noise generator.

        Args:
            sample_rate (int): Sample rate in Hz. Must be positive.
            output_dir (str): Directory to save generated noise files (if saving is used).

        Raises:
            ValueError: If sample_rate is not positive or output_dir is invalid.
            OSError: If the output directory cannot be created.
        """
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Sample rate must be a positive integer, got {sample_rate}")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("Output directory must be a non-empty string.")

        self.sample_rate: int = sample_rate
        self.output_dir: str = os.path.join(output_dir, "noise") # Subdirectory for noise
        # Optional: Initialize base SoundGenerator for saving if needed
        self._sound_saver: Optional[SoundGenerator] = None
        if SOUND_GENERATOR_AVAILABLE:
            try:
                # Use the dedicated output dir for noise
                self._sound_saver = SoundGenerator(sample_rate=self.sample_rate, output_dir=self.output_dir)
            except Exception as e:
                 logger.error(f"Failed to initialize SoundGenerator for saving: {e}")
                 # Continue without saving capability

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Noise Generator initialized. Sample Rate: {self.sample_rate} Hz, Output Dir: {self.output_dir}")
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create noise output directory {self.output_dir}: {e}")
            raise # Fail hard

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

    def _normalize_noise(self, noise_array: np.ndarray, target_amplitude: float) -> np.ndarray:
        """Normalizes noise array to target amplitude, handling potential issues."""
        if noise_array is None or noise_array.size == 0:
             raise ValueError("Cannot normalize an empty or None array.")
        try:
            max_abs = np.max(np.abs(noise_array))
            # Check for NaN/Inf before scaling
            if not np.isfinite(max_abs):
                logger.warning("NaN/Inf detected in noise array before normalization. Cleaning up.")
                noise_array = np.nan_to_num(noise_array, nan=0.0, posinf=target_amplitude, neginf=-target_amplitude)
                max_abs = np.max(np.abs(noise_array)) # Recalculate max_abs

            if max_abs > 1e-9: # Avoid division by zero/small numbers
                # Scale to fit within [-target_amplitude, target_amplitude]
                normalized = noise_array / max_abs * target_amplitude
            else:
                # Array is essentially silent, return zeros
                normalized = np.zeros_like(noise_array)

            # Final clip for safety
            return np.clip(normalized, -target_amplitude, target_amplitude).astype(np.float32)
        except Exception as e:
            logger.error(f"Error during noise normalization: {e}", exc_info=True)
            raise RuntimeError("Normalization failed.") from e

    def generate_white_noise(self, duration: float, amplitude: float = MAX_AMPLITUDE) -> np.ndarray:
        """
        Generate pure white noise (Gaussian distribution). Fails hard.

        Args:
            duration (float): Duration in seconds (> 0).
            amplitude (float): Amplitude scaling factor (0.0 to MAX_AMPLITUDE).

        Returns:
            np.ndarray: NumPy array containing the noise samples.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If noise generation fails unexpectedly.
        """
        num_samples, amp = self._validate_inputs(duration, amplitude)
        logger.debug(f"Generating white noise: Dur={duration:.2f}s, Amp={amp:.2f}")
        try:
            # Generate noise using standard normal distribution, scale by amplitude/3 for better distribution within range
            # Scaling by amplitude directly can lead to frequent clipping if amplitude is high.
            # Using std dev related to amplitude aims for most values within range.
            std_dev = amp / 3.0 # Aim for 3 sigma within amplitude limits
            noise = np.random.normal(0, std_dev, num_samples).astype(np.float32)
            # Clip to ensure strict adherence to MAX_AMPLITUDE, although normalization is preferred
            # noise = np.clip(noise, -amp, amp)
            # Instead of clipping, normalize to the target amplitude
            noise = self._normalize_noise(noise, amp)
            return noise
        except Exception as e:
             logger.error(f"Error generating white noise: {e}", exc_info=True)
             raise RuntimeError(f"White noise generation failed: {e}") from e


    def _apply_filter(self, white_noise: np.ndarray, filter_func) -> np.ndarray:
        """Helper to apply a filter function and normalize."""
        if white_noise is None or white_noise.size == 0:
             raise ValueError("Input white_noise cannot be None or empty.")
        try:
            filtered = filter_func(white_noise)
            # Use the amplitude of the original white noise as the target
            target_amp = np.max(np.abs(white_noise))
            if target_amp < 1e-9: target_amp = MAX_AMPLITUDE # Fallback if input was silent
            return self._normalize_noise(filtered, target_amp)
        except Exception as e:
            logger.error(f"Error applying filter {filter_func.__name__}: {e}", exc_info=True)
            raise RuntimeError(f"Filter application failed: {e}") from e

    # --- Colored Noise Generation Methods ---
    def _generate_pink_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates pink noise (1/f power spectrum) using FFT."""
        N = len(white_noise)
        if N == 0: return white_noise # Return empty if input is empty
        X = np.fft.rfft(white_noise) # FFT of real signal
        freqs = np.fft.rfftfreq(N, 1.0 / self.sample_rate) # Frequencies for rfft output

        # Create 1/sqrt(f) filter (for power spectrum ~ 1/f)
        # Handle f=0 frequency (DC component) - set filter gain to 1 (or 0 if DC removal desired)
        sqrt_f = np.sqrt(np.maximum(freqs, 1e-9)) # Avoid sqrt(0) and zero division, use small epsilon
        filter_gain = 1.0 / sqrt_f
        filter_gain[0] = 1.0 # Keep DC component gain at 1

        X_filtered = X * filter_gain # Apply filter in frequency domain
        pink_noise = np.fft.irfft(X_filtered, n=N) # Inverse FFT back to time domain
        logger.debug("Generated pink noise (1/f spectrum)")
        return pink_noise # Normalization happens in _apply_filter

    def _generate_brown_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates brown noise (1/f^2 power spectrum) by integrating white noise."""
        if len(white_noise) == 0: return white_noise
        # Integrate white noise using cumulative sum
        brown_noise = np.cumsum(white_noise).astype(np.float32)
        # Remove DC offset by subtracting the mean
        brown_noise -= np.mean(brown_noise)
        logger.debug("Generated brown noise (1/f^2 spectrum)")
        return brown_noise # Normalization happens in _apply_filter

    def _generate_blue_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates blue noise (f power spectrum) using FFT."""
        N = len(white_noise)
        if N == 0: return white_noise
        X = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(N, 1.0 / self.sample_rate)

        # Create sqrt(f) filter (for power spectrum ~ f)
        # Handle f=0 frequency
        sqrt_f = np.sqrt(np.maximum(freqs, 0)) # Allow sqrt(0) = 0
        filter_gain = sqrt_f
        filter_gain[0] = 0.0 # Remove DC component for blue noise

        X_filtered = X * filter_gain
        blue_noise = np.fft.irfft(X_filtered, n=N)
        logger.debug("Generated blue noise (f spectrum)")
        return blue_noise

    def _generate_violet_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates violet noise (f^2 power spectrum) using FFT."""
        N = len(white_noise)
        if N == 0: return white_noise
        X = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(N, 1.0 / self.sample_rate)

        # Create f filter (for power spectrum ~ f^2)
        # Handle f=0 frequency
        filter_gain = np.maximum(freqs, 0) # Allow f=0 -> gain=0
        filter_gain[0] = 0.0 # Ensure DC removal

        X_filtered = X * filter_gain
        violet_noise = np.fft.irfft(X_filtered, n=N)
        logger.debug("Generated violet noise (f^2 spectrum)")
        return violet_noise

    def _generate_quantum_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates quantum-like noise. Requires SciPy."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Cannot generate quantum noise: SciPy library is not available.")
        if len(white_noise) == 0: return white_noise

        N = len(white_noise)
        # Bandpass filter (example: 20Hz to 2kHz)
        lowcut = 20.0; highcut = 2000.0
        nyquist = 0.5 * self.sample_rate
        if highcut >= nyquist: highcut = nyquist * 0.99 # Ensure highcut < nyquist
        if lowcut <= 0 or lowcut >= highcut: raise ValueError("Invalid bandpass frequencies for quantum noise.")
        try:
            # Use a Butterworth filter (adjust order as needed)
            sos = signal.butter(4, [lowcut, highcut], btype='bandpass', fs=self.sample_rate, output='sos')
            filtered_noise = signal.sosfiltfilt(sos, white_noise) # Zero-phase filtering
        except Exception as filter_err:
             logger.error(f"Butterworth filter failed for quantum noise: {filter_err}")
             raise RuntimeError("Filter application failed") from filter_err

        # Add structured interference patterns (example using phi)
        time = np.arange(N) / self.sample_rate
        quantum_pattern = np.zeros_like(filtered_noise, dtype=np.float32)
        base_f_q = 432.0 # Example base for quantum structure
        if base_f_q <= 0: base_f_q = 100.0 # Fallback positive freq
        for i in range(5):
            # Avoid potential large exponents
            freq = base_f_q * (PHI ** min(i, 5)) # Limit exponent growth
            if freq > self.sample_rate / 2: continue # Skip if above Nyquist
            phase = np.random.uniform(0, 2 * PI)
            quantum_pattern += 0.2 * np.sin(2 * PI * freq * time + phase)

        # Modulate filtered noise by pattern
        quantum_noise = filtered_noise * (1 + 0.3 * quantum_pattern)

        # Optional: Add edge of chaos component (requires the edge generator)
        # chaos_noise = self._generate_edge_of_chaos_noise(white_noise) # Call helper
        # quantum_noise = 0.7 * quantum_noise + 0.3 * self._normalize_noise(chaos_noise, np.max(np.abs(quantum_noise)) * 0.4) # Mix

        logger.debug("Generated quantum noise")
        return quantum_noise # Normalization happens in _apply_filter

    def _generate_edge_of_chaos_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """Generates edge-of-chaos noise. Requires SciPy."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Cannot generate edge of chaos noise: SciPy library is not available.")
        if len(white_noise) == 0: return white_noise

        N = len(white_noise)
        # Low-pass for ordered component
        lowcut_eoc = 200.0
        if lowcut_eoc <=0 or lowcut_eoc >= self.sample_rate/2: raise ValueError("Invalid lowcut freq for EOC noise.")
        try:
            sos_low = signal.butter(2, lowcut_eoc, btype='low', fs=self.sample_rate, output='sos')
            ordered_comp = signal.sosfiltfilt(sos_low, white_noise)
        except Exception as filter_err: raise RuntimeError("Lowpass filter failed for EOC noise") from filter_err

        # High-pass for chaotic component
        highcut_eoc = 1000.0
        if highcut_eoc <=0 or highcut_eoc >= self.sample_rate/2: raise ValueError("Invalid highcut freq for EOC noise.")
        try:
            sos_high = signal.butter(2, highcut_eoc, btype='high', fs=self.sample_rate, output='sos')
            chaotic_comp = signal.sosfiltfilt(sos_high, white_noise)
        except Exception as filter_err: raise RuntimeError("Highpass filter failed for EOC noise") from filter_err

        # Mix components using edge of chaos ratio
        eoc_noise = (EDGE_OF_CHAOS_RATIO * ordered_comp +
                     (1.0 - EDGE_OF_CHAOS_RATIO) * chaotic_comp)

        # Optional: Apply non-linear feedback (Logistic map like component)
        # This adds self-organized criticality but can be unstable
        # eoc_noise_soc = np.zeros_like(eoc_noise)
        # feedback_strength = EDGE_OF_CHAOS_RATIO * 3.8 # Example strength (can cause chaos)
        # x = eoc_noise[0] # Initial condition (use first noise sample)
        # for i in range(N):
        #     x = feedback_strength * x * (1 - abs(x)) + eoc_noise[i]*0.1 # Logistic map + noise input
        #     eoc_noise_soc[i] = np.clip(x, -1.0, 1.0) # Clip output
        # eoc_noise = eoc_noise_soc # Use SOC version if enabled

        logger.debug("Generated edge-of-chaos noise")
        return eoc_noise # Normalization happens in _apply_filter


    def generate_noise(self, noise_type: str, duration: float,
                      amplitude: float = MAX_AMPLITUDE, **kwargs) -> np.ndarray:
        """
        Generate noise of a specified type. Fails hard on unknown type or errors.

        Args:
            noise_type (str): Type of noise (e.g., 'white', 'pink', 'quantum').
            duration (float): Duration in seconds (> 0).
            amplitude (float): Amplitude (0.0 to MAX_AMPLITUDE).
            **kwargs: Optional additional parameters for specific noise types (e.g., filter_params).

        Returns:
            np.ndarray: NumPy array containing the noise waveform.

        Raises:
            ValueError: If inputs are invalid or noise_type is unsupported.
            RuntimeError: If noise generation fails for other reasons.
        """
        noise_type_lower = noise_type.lower()
        if noise_type_lower not in SUPPORTED_NOISE_TYPES:
            raise ValueError(f"Unsupported noise type: '{noise_type}'. Supported types are: {SUPPORTED_NOISE_TYPES}")

        # Generate base white noise first
        white_noise = self.generate_white_noise(duration, amplitude)
        if white_noise is None or len(white_noise) == 0:
            raise RuntimeError("Base white noise generation failed.")

        # Select appropriate filter function based on type
        filter_map = {
            'pink': self._generate_pink_noise,
            'brown': self._generate_brown_noise,
            'blue': self._generate_blue_noise,
            'violet': self._generate_violet_noise,
            'quantum': self._generate_quantum_noise,
            'edge_of_chaos': self._generate_edge_of_chaos_noise,
        }

        if noise_type_lower == 'white':
            return white_noise # No filtering needed
        elif noise_type_lower in filter_map:
            filter_func = filter_map[noise_type_lower]
            # Pass kwargs if filter function accepts them (currently they don't directly)
            # Consider modifying filter functions if kwargs are needed per type
            return self._apply_filter(white_noise, filter_func)
        else:
             # This case should not be reached due to initial validation, but acts as safeguard
             raise ValueError(f"Internal error: Filter function mapping missing for '{noise_type_lower}'.")

    def save_noise(self, noise_array: np.ndarray, filename: str, description: Optional[str] = None) -> Optional[str]:
        """
        Save a generated noise waveform to a WAV file using the SoundGenerator.

        Args:
            noise_array (np.ndarray): The noise waveform data.
            filename (str): The desired filename (e.g., "pink_noise_run1.wav").
            description (Optional[str]): Optional description for logging.

        Returns:
            str: The absolute path to the saved file, or None if saving failed.

        Raises:
            RuntimeError: If the sound saving utility is unavailable.
        """
        if self._sound_saver is None:
             raise RuntimeError("Sound saving unavailable: Base SoundGenerator failed to initialize.")
        # Use the SoundGenerator's save method, which includes validation
        try:
            return self._sound_saver.save_sound(noise_array, filename, description)
        except (TypeError, ValueError, IOError, RuntimeError) as e:
            logger.error(f"Failed to save noise file '{filename}': {e}", exc_info=True)
            # Re-raise as RuntimeError to indicate failure
            raise RuntimeError(f"Noise saving failed for '{filename}'") from e

# --- Example Usage ---
if __name__ == "__main__":
    print("Running Noise Generator Module Example...")
    try:
        noise_gen = NoiseGenerator(output_dir="output/sounds/noise_test") # Test specific dir

        print("\nGenerating different noise types (duration=5s)...")
        noise_types_to_test = SUPPORTED_NOISE_TYPES

        saved_files = {}
        for n_type in noise_types_to_test:
            print(f"- Generating {n_type} noise...")
            try:
                 noise_wave = noise_gen.generate_noise(noise_type=n_type, duration=5.0, amplitude=0.7)
                 # Save only if sound generator available
                 if noise_gen._sound_saver:
                      f_name = f"{n_type}_noise_5s_test.wav"
                      f_path = noise_gen.save_noise(noise_wave, f_name, description=f"Test {n_type} Noise")
                      if f_path:
                           saved_files[n_type] = f_path
                           print(f"  > Saved {n_type} noise to {f_path}")
                      else:
                           print(f"  > Failed to save {n_type} noise.")
                 else:
                      print(f"  > Generated {n_type} noise waveform (saving disabled).")

            except RuntimeError as e:
                 # Catch runtime errors specifically from generate_noise/save_noise
                 print(f"  > FAILED to generate/save {n_type} noise: {e}")
                 # Continue to next noise type
            except ValueError as e:
                 print(f"  > INVALID input for {n_type} noise generation: {e}")


        print("\nNoise Generation Example Finished.")
        if not noise_gen._sound_saver:
             print("NOTE: Saving was disabled as base SoundGenerator was not found.")

    except (ValueError, OSError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Noise Generator Example ---")
        print(f"An error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("------------------------------------------")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Noise Generator Example ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------")