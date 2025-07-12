"""
Sound Generator Module

Provides core functionality for generating basic sound waveforms like pure tones,
harmonic structures, and transitions. Used as a base by more specialized sound modules.
Enforces strict input validation and error handling.

Author: Soul Development Framework Team - Refactored with Strict Error Handling
"""

import numpy as np
import logging
import os
import scipy.io.wavfile as wavfile
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict, Any # Added for type hinting

# --- Constants ---
# Default constants in case import fails
OUTPUT_DIR_BASE = "output/sounds"  # Default output directory
SAMPLE_RATE = 44100  # Default sample rate in Hz

# Attempt to import constants, raise error if essential ones are missing
try:
    from constants.constants import *
except ImportError as e:
    # Basic logging setup if constants failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.warning(f"Failed to import constants: {e}. Using default values.")

# Configure logging
log_file_path = os.path.join("logs", "sound_generator.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# Default values in case constants are not available
LOG_LEVEL = getattr(logging, 'INFO') if not 'LOG_LEVEL' in globals() else LOG_LEVEL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if not 'LOG_FORMAT' in globals() else LOG_FORMAT
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=log_file_path, filemode='w')
logger = logging.getLogger('sound_generator')

# Define a small epsilon for float comparisons and division checks
FLOAT_EPSILON = 1e-9

class SoundGenerator:
    """
    Class for generating base sound waveforms (tones, harmonics).
    Performs strict validation and fails hard on errors.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = OUTPUT_DIR_BASE):
        """
        Initialize a new sound generator.

        Args:
            sample_rate (int): Sample rate in Hz for audio generation. Must be positive.
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
        self.output_dir: str = output_dir

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Sound Generator initialized. Sample Rate: {self.sample_rate} Hz, Output Dir: {self.output_dir}")
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to create output directory {self.output_dir}: {e}")
            raise # Fail hard

    def generate_tone(self, frequency: float, duration: float,
                     amplitude: float = MAX_AMPLITUDE, fade_in_out: float = 0.05) -> np.ndarray:
        """
        Generate a pure sine tone with optional fade. Fails hard on invalid input.

        Args:
            frequency (float): Frequency in Hz (> 0).
            duration (float): Duration in seconds (> 0).
            amplitude (float): Amplitude (0.0 to MAX_AMPLITUDE).
            fade_in_out (float): Fade in/out duration in seconds (>= 0).

        Returns:
            np.ndarray: NumPy array containing the audio samples.

        Raises:
            ValueError: If frequency, duration, amplitude, or fade_in_out are invalid.
        """
        # --- Input Validation ---
        if not isinstance(frequency, (int, float)) or frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if not isinstance(amplitude, (int, float)) or not (0.0 <= amplitude <= MAX_AMPLITUDE):
            raise ValueError(f"Amplitude must be between 0.0 and {MAX_AMPLITUDE}, got {amplitude}")
        if not isinstance(fade_in_out, (int, float)) or fade_in_out < 0:
            raise ValueError(f"Fade duration must be non-negative, got {fade_in_out}")

        num_samples = int(duration * self.sample_rate)
        if num_samples <= 0:
            raise ValueError(f"Calculated number of samples ({num_samples}) is not positive for duration {duration}.")

        logger.debug(f"Generating tone: Freq={frequency:.2f}Hz, Dur={duration:.2f}s, Amp={amplitude:.2f}")

        try:
            time = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            # Angular frequency * time
            angle = 2 * PI * frequency * time
            tone = (amplitude * np.sin(angle)).astype(np.float32) # Use float32 for audio

            # Apply fade in/out
            fade_samples = int(fade_in_out * self.sample_rate)
            if fade_samples > 0:
                # Ensure fade doesn't exceed half the signal length
                fade_samples = min(fade_samples, num_samples // 2)
                if fade_samples > 0: # Re-check after min
                    fade_in_env = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    fade_out_env = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                    tone[:fade_samples] *= fade_in_env
                    tone[-fade_samples:] *= fade_out_env
            return tone
        except Exception as e:
             logger.error(f"Error generating tone: {e}", exc_info=True)
             raise RuntimeError(f"Tone generation failed: {e}") from e # Fail hard


    def generate_harmonic_tone(self, base_frequency: float, harmonics: List[float],
                             amplitudes: List[float], duration: float,
                             fade_in_out: float = 0.05) -> np.ndarray:
        """
        Generate a tone composed of multiple harmonics. Fails hard on invalid input.

        Args:
            base_frequency (float): Base frequency in Hz (> 0).
            harmonics (List[float]): List of harmonic ratios (e.g., [1.0, 2.0, 3.0]). Must contain 1.0. All > 0.
            amplitudes (List[float]): List of amplitudes (0.0 to MAX_AMPLITUDE). Must match harmonics length.
            duration (float): Duration in seconds (> 0).
            fade_in_out (float): Fade in/out duration in seconds (>= 0).

        Returns:
            np.ndarray: NumPy array containing the audio samples.

        Raises:
            ValueError: If inputs are invalid (frequencies, duration, lengths mismatch, amplitudes out of range).
            TypeError: If inputs are not of the expected type.
        """
        # --- Input Validation ---
        if not isinstance(base_frequency, (int, float)) or base_frequency <= 0:
            raise ValueError(f"Base frequency must be positive, got {base_frequency}")
        if not isinstance(harmonics, list) or not harmonics:
            raise TypeError("Harmonics must be a non-empty list of ratios.")
        if 1.0 not in harmonics:
             logger.warning("Fundamental frequency (1.0) missing in harmonics list. Adding it.")
             harmonics.insert(0, 1.0)
             # Need corresponding amplitude - this part is tricky without fallbacks.
             # Let's require the user provides consistent lists.
             # raise ValueError("Harmonics list must contain the fundamental ratio 1.0") # Stricter
        if not all(isinstance(h, (int, float)) and h > 0 for h in harmonics):
            raise ValueError("All harmonic ratios must be positive numbers.")
        if not isinstance(amplitudes, list):
            raise TypeError("Amplitudes must be a list.")
        if len(harmonics) != len(amplitudes):
            raise ValueError(f"Harmonics list (len {len(harmonics)}) and Amplitudes list (len {len(amplitudes)}) must have the same length.")
        if not all(isinstance(a, (int, float)) and 0.0 <= a <= MAX_AMPLITUDE for a in amplitudes):
            raise ValueError(f"All amplitudes must be between 0.0 and {MAX_AMPLITUDE}.")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if not isinstance(fade_in_out, (int, float)) or fade_in_out < 0:
            raise ValueError(f"Fade duration must be non-negative, got {fade_in_out}")

        num_samples = int(duration * self.sample_rate)
        if num_samples <= 0:
            raise ValueError(f"Calculated number of samples ({num_samples}) is not positive for duration {duration}.")

        logger.debug(f"Generating harmonic tone: Base={base_frequency:.2f}Hz, {len(harmonics)} harmonics, Dur={duration:.2f}s")

        try:
            time = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            combined_tone = np.zeros(num_samples, dtype=np.float32)

            # Add each harmonic
            for ratio, amp in zip(harmonics, amplitudes):
                if amp == 0: continue # Skip zero amplitude harmonics
                freq = base_frequency * ratio
                angle = 2 * PI * freq * time
                combined_tone += (amp * np.sin(angle)).astype(np.float32)

            # Normalize ONLY if combined amplitude exceeds max (prevents clipping)
            max_abs_amp = np.max(np.abs(combined_tone))
            if max_abs_amp > MAX_AMPLITUDE:
                logger.warning(f"Harmonic tone amplitude exceeded limit ({max_abs_amp:.3f}). Normalizing.")
                combined_tone = (combined_tone / max_abs_amp) * MAX_AMPLITUDE

            # Apply fade in/out
            fade_samples = int(fade_in_out * self.sample_rate)
            if fade_samples > 0:
                fade_samples = min(fade_samples, num_samples // 2)
                if fade_samples > 0:
                    fade_in_env = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    fade_out_env = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                    combined_tone[:fade_samples] *= fade_in_env
                    combined_tone[-fade_samples:] *= fade_out_env

            return combined_tone
        except Exception as e:
             logger.error(f"Error generating harmonic tone: {e}", exc_info=True)
             raise RuntimeError(f"Harmonic tone generation failed: {e}") from e

    def generate_sacred_chord(self, base_frequency: float = FUNDAMENTAL_FREQUENCY_432,
                           duration: float = 10.0, fade_in_out: float = 0.5) -> np.ndarray:
        """
        Generate a sacred geometry chord based on golden ratio and perfect fifths.

        Args:
            base_frequency (float): Base frequency in Hz (> 0). Defaults to 432Hz constant.
            duration (float): Duration in seconds (> 0).
            fade_in_out (float): Fade in/out duration in seconds (>= 0).

        Returns:
            np.ndarray: NumPy array containing the audio samples.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If harmonic tone generation fails.
        """
        if not isinstance(base_frequency, (int, float)) or base_frequency <= 0:
            raise ValueError(f"Base frequency must be positive, got {base_frequency}")

        # Define sacred ratios relative to base_frequency
        ratios = [ 1.0, 1.5, PHI, 2.0, 2.0 * PHI, 3.0, 3.0 * PHI / 2.0 ]
        # Define amplitudes (ensure correct length and range)
        amplitudes = [0.7, 0.5, 0.6, 0.4, 0.3, 0.25, 0.2]
        # Ensure amplitudes do not exceed MAX_AMPLITUDE
        amplitudes = [min(a, MAX_AMPLITUDE) for a in amplitudes]

        logger.debug(f"Generating sacred chord: Base={base_frequency:.2f}Hz, Dur={duration:.2f}s")
        # Rely on generate_harmonic_tone for validation and generation
        return self.generate_harmonic_tone(base_frequency, ratios, amplitudes, duration, fade_in_out)


    def generate_dimensional_transition(self, start_dimension: float, end_dimension: float,
                                     duration: float, steps: int = 100) -> np.ndarray:
        """
        Generate a sound transitioning between dimensional frequencies. Fails hard.
        (Frequency based on PHI scaling from FUNDAMENTAL_FREQUENCY).

        Args:
            start_dimension (float): Starting dimension number (e.g., 3.0).
            end_dimension (float): Ending dimension number (e.g., 4.0).
            duration (float): Duration in seconds (> 0).
            steps (int): Number of discrete frequency steps (> 0).

        Returns:
            np.ndarray: NumPy array containing the audio samples.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If sound generation fails.
        """
        # --- Input Validation ---
        if not isinstance(start_dimension, (int, float)) or start_dimension < 0:
             raise ValueError(f"Start dimension must be non-negative, got {start_dimension}")
        if not isinstance(end_dimension, (int, float)) or end_dimension < 0:
             raise ValueError(f"End dimension must be non-negative, got {end_dimension}")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"Steps must be a positive integer, got {steps}")

        # Use a defined base frequency for dimensional scaling
        dim_base_freq = FUNDAMENTAL_FREQUENCY_432 # Or another suitable constant
        if dim_base_freq <= 0: raise ValueError("Dimensional base frequency must be positive.")

        # Calculate frequencies using exponential scaling (PHI based) from dimension 3 baseline
        try:
             start_freq = dim_base_freq * (PHI ** max(0, start_dimension - 3)) # Ensure exponent >= 0
             end_freq = dim_base_freq * (PHI ** max(0, end_dimension - 3))
             if start_freq <= 0 or end_freq <= 0:
                  raise ValueError("Calculated start/end frequencies must be positive.")
        except Exception as e:
             raise ValueError(f"Error calculating start/end frequencies: {e}") from e

        logger.info(f"Generating transition from {start_dimension}D ({start_freq:.2f}Hz) to {end_dimension}D ({end_freq:.2f}Hz)")

        num_samples = int(duration * self.sample_rate)
        if num_samples <= 0: raise ValueError("Duration results in zero samples.")

        try:
            time_full = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)
            transition_sound = np.zeros(num_samples, dtype=np.float32)

            # Create frequency sweep array (logarithmic for perceived smoothness)
            if abs(start_freq - end_freq) < FLOAT_EPSILON: # No transition if frequencies are the same
                freq_steps = np.full(steps, start_freq)
            else:
                freq_steps = np.geomspace(start_freq, end_freq, steps)

            samples_per_step = num_samples // steps
            if samples_per_step == 0:
                 raise ValueError(f"Too many steps ({steps}) for the number of samples ({num_samples}). Reduce steps or increase duration.")

            # Generate sound step-by-step
            for i, freq in enumerate(freq_steps):
                start_sample = i * samples_per_step
                # Ensure end_sample doesn't exceed total samples, especially for the last step
                end_sample = min((i + 1) * samples_per_step, num_samples) if i < steps - 1 else num_samples
                step_len = end_sample - start_sample
                if step_len <= 0: continue # Skip empty steps

                step_time = np.linspace(0, step_len / self.sample_rate, step_len, endpoint=False, dtype=np.float64)

                # Generate tone for this segment with simple harmonics
                angle = 2 * PI * freq * step_time
                segment_tone = (0.7 * np.sin(angle) +
                                0.2 * np.sin(2 * angle) + # Octave
                                0.1 * np.sin(1.5 * angle) # Perfect fifth (approx)
                               ).astype(np.float32)

                # Apply short crossfade between segments to reduce clicks
                fade_len = min(50, step_len // 10) # Short fade (e.g., 50 samples)
                if fade_len > 1:
                    segment_fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)**2 # Smoother fade in
                    if i > 0: # Fade in (except first step)
                        segment_tone[:fade_len] *= segment_fade
                    if i < steps - 1: # Fade out (except last step)
                        segment_tone[-fade_len:] *= segment_fade[::-1] # Reversed fade out

                transition_sound[start_sample:end_sample] += segment_tone[:step_len] # Ensure length match

            # Normalize final sound
            max_abs_amp = np.max(np.abs(transition_sound))
            if max_abs_amp > FLOAT_EPSILON:
                transition_sound = (transition_sound / max_abs_amp) * MAX_AMPLITUDE
            else:
                 transition_sound.fill(0.0) # Sound is silent

            # Apply overall fade in/out to the whole transition
            overall_fade_sec = 0.5 # Shorter overall fade for transition
            fade_samples = int(overall_fade_sec * self.sample_rate)
            if fade_samples > 0:
                fade_samples = min(fade_samples, num_samples // 2)
                if fade_samples > 0:
                    fade_in_env = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    fade_out_env = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                    transition_sound[:fade_samples] *= fade_in_env
                    transition_sound[-fade_samples:] *= fade_out_env

            return transition_sound
        except Exception as e:
             logger.error(f"Error generating dimensional transition: {e}", exc_info=True)
             raise RuntimeError(f"Dimensional transition generation failed: {e}") from e


    def save_sound(self, sound: np.ndarray, filename: str, description: Optional[str] = None) -> Optional[str]:
        """
        Save a generated sound to a WAV file. Fails hard on error.

        Args:
            sound (np.ndarray): NumPy array containing the audio samples. Must be non-empty.
            filename (str): Filename (e.g., "mysound.wav"). Must be non-empty.
            description (Optional[str]): Optional description for logging.

        Returns:
            str: Absolute path to the saved file.

        Raises:
            TypeError: If 'sound' is not a NumPy array.
            ValueError: If 'sound' is empty or 'filename' is invalid.
            IOError: If saving to disk fails.
            RuntimeError: For other unexpected errors during saving.
        """
        # --- Input Validation ---
        if not isinstance(sound, np.ndarray):
            raise TypeError(f"Sound data must be a NumPy array, got {type(sound)}")
        if sound.size == 0:
            raise ValueError("Cannot save empty sound array.")
        if not isinstance(filename, str) or not filename:
            raise ValueError("Filename must be a non-empty string.")

        # Ensure filename ends with .wav
        if not filename.lower().endswith('.wav'):
            filename += '.wav'

        # Create safe filename (remove potentially invalid characters)
        # Allow alphanumeric, underscore, hyphen, dot
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        safe_filename = "".join(c for c in filename if c in safe_chars)
        if not safe_filename or safe_filename == ".wav": # Handle cases where all chars were invalid
            safe_filename = f"sound_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
            logger.warning(f"Original filename '{filename}' was invalid, using safe name: '{safe_filename}'")

        # Create full path
        filepath = os.path.abspath(os.path.join(self.output_dir, safe_filename))
        log_desc = description if description else safe_filename

        try:
            # Ensure directory exists (might be redundant but safe)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # --- Data Validation and Conversion ---
            # Check for NaN or Inf values - wavfile write might fail or create corrupt file
            if np.any(np.isnan(sound)) or np.any(np.isinf(sound)):
                 logger.error(f"Sound data for '{log_desc}' contains NaN or Inf values. Attempting cleanup.")
                 sound = np.nan_to_num(sound, nan=0.0, posinf=MAX_AMPLITUDE, neginf=-MAX_AMPLITUDE) # Replace bad values

            # Check amplitude range again before converting to int16
            max_abs_val = np.max(np.abs(sound))
            if max_abs_val > MAX_AMPLITUDE * 1.01: # Allow slight tolerance
                logger.warning(f"Amplitude ({max_abs_val:.3f}) exceeds max ({MAX_AMPLITUDE}) for '{log_desc}'. Clipping.")
                sound = np.clip(sound, -MAX_AMPLITUDE, MAX_AMPLITUDE)

            # Convert to int16 for WAV format
            # 32767 is the maximum value for a signed 16-bit integer
            scaled_sound = sound * 32767.0
            sound_int16 = scaled_sound.astype(np.int16)

            # --- Write WAV file ---
            wavfile.write(filepath, self.sample_rate, sound_int16)

            logger.info(f"Saved sound '{log_desc}' to {filepath}")
            return filepath
        except IOError as ioe:
            logger.error(f"IOError saving sound '{log_desc}' to {filepath}: {ioe}", exc_info=True)
            raise IOError(f"Failed to write WAV file to {filepath}: {ioe}") from ioe # Fail hard
        except Exception as e:
            logger.error(f"Unexpected error saving sound '{log_desc}' to {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error saving sound: {e}") from e # Fail hard

    def create_sound(self, name: str = "default_sound", fundamental_frequency: float = FUNDAMENTAL_FREQUENCY_432,
                      duration: float = 1.0) -> Optional[Dict[str, Any]]:
         """
         Placeholder or simplified method to create a 'sound object' representation
         if needed by other modules like SephirothSoundIntegration.

         Args:
            name (str): Name of the sound.
            fundamental_frequency (float): Base frequency.
            duration (float): Sound duration.

         Returns:
            Dict: A dictionary representing the sound, or None on error.
         """
         logger.warning("create_sound is a basic placeholder. Generate specific sounds directly.")
         if fundamental_frequency <= 0 or duration <= 0:
              logger.error(f"Invalid frequency or duration for create_sound ({fundamental_frequency}, {duration})")
              return None
         # Return a dictionary representation, not a waveform
         return {
              "name": name,
              "fundamental_frequency": fundamental_frequency,
              "duration": duration,
              # Example: Define basic harmonics based on frequency
              "harmonics": [
                   {"frequency": fundamental_frequency, "amplitude": 0.8, "phase": 0.0},
                   {"frequency": fundamental_frequency*2, "amplitude": 0.4, "phase": 0.0},
                   {"frequency": fundamental_frequency*3, "amplitude": 0.2, "phase": 0.0},
              ]
         }

# --- Example Usage (Modified for Strictness) ---
if __name__ == "__main__":
    print("Running Sound Generator Module Example...")
    try:
        generator = SoundGenerator(output_dir="output/sounds/generator_test") # Test specific dir

        # Generate and save a sacred chord
        print("Generating sacred chord...")
        sacred_sound = generator.generate_sacred_chord(duration=8.0) # Shorter duration for testing
        sacred_path = generator.save_sound(sacred_sound, "sacred_chord_test.wav", description="Test Sacred Chord")
        if sacred_path:
            print(f"  > Saved sacred chord: {sacred_path}")
        # No else needed, save_sound raises error on failure

        # Generate a dimensional transition sound (3D to 5D)
        print("\nGenerating dimensional transition (3D -> 5D)...")
        transition_sound = generator.generate_dimensional_transition(3.0, 5.0, duration=10.0, steps=80)
        transition_path = generator.save_sound(transition_sound, "transition_3d_to_5d_test.wav", description="Test 3D-5D Transition")
        if transition_path:
            print(f"  > Saved 3D->5D transition: {transition_path}")

        # Example of generating a harmonic tone
        print("\nGenerating C4 harmonic tone...")
        c4_freq = 261.63
        harmonics_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        amplitudes_list = [0.7, 0.35, 0.2, 0.15, 0.1] # Ensure sum doesn't greatly exceed MAX_AMPLITUDE implicitly
        c4_tone = generator.generate_harmonic_tone(c4_freq, harmonics_list, amplitudes_list, duration=5.0)
        c4_path = generator.save_sound(c4_tone, "c4_harmonic_test.wav", description="Test C4 Harmonic Tone")
        if c4_path:
            print(f"  > Saved C4 harmonic tone: {c4_path}")

        print("\nSound Generator Module Example Finished Successfully.")

    except (ValueError, TypeError, IOError, RuntimeError, ImportError) as e:
        print(f"\n--- ERROR during Sound Generator Example ---")
        print(f"An error occurred: {type(e).__name__}: {e}")
        # Potentially log the full traceback here for debugging
        import traceback
        traceback.print_exc()
        print("--------------------------------------------")

    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR during Sound Generator Example ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------")