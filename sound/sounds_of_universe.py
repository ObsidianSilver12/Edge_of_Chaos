"""
Sounds of Universe Module

This module implements cosmic background sounds and fundamental universal frequencies.
These sounds represent the primordial vibrations of the universe, cosmic background radiation,
and other universal constants translated into audible frequencies.

Key functions:
- Generate cosmic microwave background (CMB) radiation sound
- Create fundamental frequency patterns based on cosmic constants
- Produce stellar and galactic sounds based on actual astronomical data
- Generate dimensional resonance patterns for interdimensional travel

Author: Soul Development Framework Team
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sounds_of_universe.log'
)
logger = logging.getLogger('sounds_of_universe')

# Constants
SAMPLE_RATE = 44100  # Hz - Standard audio sample rate
MAX_AMPLITUDE = 0.8  # Maximum amplitude to prevent clipping
OUTPUT_DIR = "output/sounds"  # Directory for saving generated sounds
DEFAULT_DURATION = 60.0  # Default duration in seconds
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # ~1.618
SPEED_OF_LIGHT = 299792458  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg

# Cosmic constants mapped to audible frequencies
COSMIC_FREQUENCIES = {
    'cmb_peak': 160.2,  # Hz - CMB peak frequency scaled to audible range
    'hydrogen_line': 420.0,  # Hz - Based on 21cm hydrogen line
    'earth_schumann': 7.83,  # Hz - Earth's resonant frequency
    'solar_oscillation': 432.0,  # Hz - Sun's core resonant frequency (normalized)
    'galactic_center': 528.0,  # Hz - Galactic core frequency (scaled)
    'cosmic_background': 136.1,  # Hz - CMB power spectrum transformed
    'quantum_vacuum': 369.0,  # Hz - Quantum vacuum fluctuations (theoretical)
    'planck_frequency': 288.0,  # Hz - Derived from Planck time/length
    'universe_expansion': 126.22  # Hz - Based on Hubble constant
}

# Stellar frequencies for various astronomical objects
# (These are conversions of actual astronomical data to the audible range)
STELLAR_FREQUENCIES = {
    'sun': {
        'base_frequency': 126.22,  # Hz
        'harmonics': [1.0, 1.5, 2.0, 2.5, 3.0],
        'modulation': 0.1,  # Hz (solar cycle approximation)
        'type': 'G-type main-sequence'
    },
    'neutron_star': {
        'base_frequency': 528.0,  # Hz
        'harmonics': [1.0, 2.0, 3.0, 4.0],
        'modulation': 1.2,  # Hz (pulsar-like)
        'type': 'Pulsar'
    },
    'black_hole': {
        'base_frequency': 57.0,  # Hz
        'harmonics': [1.0, GOLDEN_RATIO, 2.0, GOLDEN_RATIO*2],
        'modulation': 0.05,  # Hz (event horizon oscillations)
        'type': 'Stellar black hole'
    },
    'galaxy': {
        'base_frequency': 63.0,  # Hz
        'harmonics': [1.0, 1.5, 2.0, 3.0],
        'modulation': 0.01,  # Hz (galactic rotation period scaled)
        'type': 'Spiral galaxy'
    },
    'quasar': {
        'base_frequency': 741.0,  # Hz
        'harmonics': [1.0, 1.33, 1.66, 2.0],
        'modulation': 0.3,  # Hz (quasar jet oscillations)
        'type': 'Active galactic nucleus'
    }
}

class UniverseSounds:
    """
    Class for generating cosmic and universal sounds based on actual astronomical data.
    These sounds represent the fundamental frequencies and vibrations of the universe.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = OUTPUT_DIR):
        """
        Initialize a new universe sounds generator.
        
        Args:
            sample_rate: Sample rate in Hz for audio generation
            output_dir: Directory to save generated sound files
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Universe Sounds generator initialized with sample rate {sample_rate} Hz")
        logger.info(f"Output directory: {output_dir}")
    
    def generate_cosmic_background(self, duration: float = DEFAULT_DURATION,
                                amplitude: float = 0.7) -> np.ndarray:
        """
        Generate sound based on cosmic microwave background radiation.
        
        Args:
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the CMB sound samples
        """
        # Number of samples
        num_samples = int(duration * self.sample_rate)
        
        # Start with filtered noise representing the CMB static
        # Generate white noise
        noise = np.random.normal(0, amplitude * 0.5, num_samples)
        
        # Apply CMB power spectrum filter
        # (This approximates the actual CMB power spectrum)
        N = len(noise)
        X = np.fft.rfft(noise)
        f = np.fft.rfftfreq(N, 1/self.sample_rate)
        
        # Create CMB-like filter
        # The actual CMB has peaks at specific angular scales
        # We're translating this to audible frequencies
        cmb_filter = np.ones_like(f)
        
        # Add the primary peak
        peak_freq = COSMIC_FREQUENCIES['cmb_peak']
        peak_idx = np.argmin(np.abs(f - peak_freq))
        
        # Create a broader peak in the spectrum
        peak_width = 30  # Hz
        for i in range(len(f)):
            dist = abs(f[i] - peak_freq)
            if dist < peak_width:
                # Gaussian peak
                cmb_filter[i] = 1.0 + 2.0 * np.exp(-(dist**2) / (2 * (peak_width/3)**2))
        
        # Add secondary peaks
        secondary_peaks = [220.0, 540.0, 720.0]
        for peak in secondary_peaks:
            peak_idx = np.argmin(np.abs(f - peak))
            
            # Create narrower secondary peaks
            peak_width = 20  # Hz
            for i in range(len(f)):
                dist = abs(f[i] - peak)
                if dist < peak_width:
                    # Gaussian peak
                    cmb_filter[i] = 1.0 + 1.0 * np.exp(-(dist**2) / (2 * (peak_width/3)**2))
        
        # Apply the CMB-like filter
        X_filtered = X * cmb_filter
        
        # Convert back to time domain
        cmb_noise = np.fft.irfft(X_filtered, n=N)
        
        # Add the fundamental cosmic frequencies
        time = np.arange(num_samples) / self.sample_rate
        
        # Add base cosmic frequencies
        cosmic_tone = np.zeros_like(time)
        
        for name, freq in COSMIC_FREQUENCIES.items():
            # Randomize phase
            phase = np.random.random() * 2 * np.pi
            
            # Amplitude decreases for higher frequencies
            amp = amplitude * 0.3 * (200.0 / (freq + 100.0))
            
            cosmic_tone += amp * np.sin(2 * np.pi * freq * time + phase)
        
        # Mix noise and cosmic tones
        cmb_sound = 0.6 * cmb_noise + 0.4 * cosmic_tone
        
        # Add slow amplitude modulation based on universe expansion
        expansion_freq = 0.01  # Very slow modulation
        expansion_mod = 0.2 * np.sin(2 * np.pi * expansion_freq * time) + 0.8
        
        cmb_sound = cmb_sound * expansion_mod
        
        # Normalize
        if np.max(np.abs(cmb_sound)) > 0:
            cmb_sound = cmb_sound / np.max(np.abs(cmb_sound)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {duration}s of cosmic microwave background sound")
        return cmb_sound
    
    def generate_stellar_sound(self, stellar_type: str, duration: float = DEFAULT_DURATION,
                             amplitude: float = 0.8) -> np.ndarray:
        """
        Generate sound based on stellar oscillations and frequencies.
        
        Args:
            stellar_type: Type of stellar object ('sun', 'neutron_star', 'black_hole', 'galaxy', 'quasar')
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the stellar sound samples
        """
        if stellar_type.lower() not in STELLAR_FREQUENCIES:
            logger.warning(f"Unknown stellar type: {stellar_type}. Using sun.")
            stellar_type = 'sun'
        
        # Get stellar properties
        stellar_props = STELLAR_FREQUENCIES[stellar_type.lower()]
        base_freq = stellar_props['base_frequency']
        harmonics = stellar_props['harmonics']
        mod_freq = stellar_props['modulation']
        
        # Number of samples
        num_samples = int(duration * self.sample_rate)
        time = np.arange(num_samples) / self.sample_rate
        
        # Initialize sound array
        stellar_sound = np.zeros_like(time)
        
        # Add base frequency and harmonics
        for i, harmonic in enumerate(harmonics):
            freq = base_freq * harmonic
            
            # Decreasing amplitude for higher harmonics
            amp = amplitude * (0.8 / (i + 1))
            
            # Randomize phase
            phase = np.random.random() * 2 * np.pi
            
            # Add to sound
            stellar_sound += amp * np.sin(2 * np.pi * freq * time + phase)
        
        # Add characteristic modulation
        # Different stellar objects have different modulation patterns
        
        if stellar_type.lower() == 'sun':
            # Sun has solar oscillations and cycles
            # Add main p-mode oscillations
            p_mode_freq = 0.003  # ~5-minute oscillations
            p_mode = 0.3 * np.sin(2 * np.pi * p_mode_freq * time)
            
            # Add slower solar cycle modulation
            solar_cycle = 0.2 * np.sin(2 * np.pi * mod_freq * time)
            
            modulation = 1.0 + p_mode + solar_cycle
            
        elif stellar_type.lower() == 'neutron_star':
            # Neutron stars/pulsars have rapid, regular pulses
            # Create pulsar-like rapid modulation
            pulse_duty = 0.1  # short pulse width
            pulse_phase = (time * mod_freq) % 1.0
            pulse_envelope = np.exp(-((pulse_phase - 0.5) ** 2) / (2 * pulse_duty ** 2))
            
            # Normalize pulse envelope
            pulse_envelope = pulse_envelope / np.max(pulse_envelope)
            
            modulation = 0.3 + 0.7 * pulse_envelope
            
        elif stellar_type.lower() == 'black_hole':
            # Black holes have ripples in spacetime and event horizon oscillations
            # Create deep, rhythmic modulation with occasional "bursts"
            
            # Base modulation
            base_mod = 0.4 * np.sin(2 * np.pi * mod_freq * time)
            
            # Add occasional bursts (accretion events)
            bursts = np.zeros_like(time)
            num_bursts = int(duration * 0.2)  # Number of bursts
            
            for _ in range(num_bursts):
                # Random burst time
                burst_time = np.random.random() * duration
                burst_idx = int(burst_time * self.sample_rate)
                
                # Burst width
                burst_width = int(0.5 * self.sample_rate)  # 0.5 second burst
                
                # Create exponential decay envelope
                if burst_idx < num_samples:
                    end_idx = min(burst_idx + burst_width, num_samples)
                    t = np.arange(end_idx - burst_idx)
                    envelope = np.exp(-t / (burst_width * 0.2))
                    
                    # Apply envelope
                    bursts[burst_idx:end_idx] += envelope[:end_idx-burst_idx]
            
            # Normalize bursts
            if np.max(bursts) > 0:
                bursts = bursts / np.max(bursts) * 0.6
            
            modulation = 0.4 + base_mod + bursts
            
        elif stellar_type.lower() == 'galaxy':
            # Galaxies have slow rotational patterns and harmonics
            # Create complex layered modulation
            
            # Slow rotation
            rotation = 0.2 * np.sin(2 * np.pi * mod_freq * time)
            
            # Density wave oscillations (spiral arms)
            arm_freq = mod_freq * 3
            arms = 0.15 * np.sin(2 * np.pi * arm_freq * time)
            
            # Central core pulsations
            core_freq = mod_freq * 8
            core = 0.1 * np.sin(2 * np.pi * core_freq * time)
            
            modulation = 0.6 + rotation + arms + core
            
        elif stellar_type.lower() == 'quasar':
            # Quasars have energetic jets and intense radiation
            # Create pulsating modulation with high energy
            
            # Jet oscillations
            jet_freq = mod_freq
            jet = 0.5 * np.abs(np.sin(2 * np.pi * jet_freq * time))
            
            # Add high frequency flicker
            flicker_freq = mod_freq * 10
            flicker = 0.2 * np.random.random(num_samples) * np.sin(2 * np.pi * flicker_freq * time)
            
            modulation = 0.3 + jet + flicker
            
        else:
            # Default modulation
            modulation = 1.0 + 0.2 * np.sin(2 * np.pi * mod_freq * time)
        
        # Apply modulation
        stellar_sound = stellar_sound * modulation
        
        # Add characteristic noise component
        noise_level = 0.0
        
        if stellar_type.lower() == 'sun':
            noise_level = 0.1
        elif stellar_type.lower() == 'neutron_star':
            noise_level = 0.2
        elif stellar_type.lower() == 'black_hole':
            noise_level = 0.3
        elif stellar_type.lower() == 'galaxy':
            noise_level = 0.15
        elif stellar_type.lower() == 'quasar':
            noise_level = 0.4
        
        if noise_level > 0:
            # Generate and filter noise
            noise = np.random.normal(0, noise_level, num_samples)
            
            # Filter based on stellar type
            if stellar_type.lower() in ['sun', 'galaxy']:
                # Lower frequency noise
                b, a = signal.butter(3, 200, 'low', fs=self.sample_rate)
                noise = signal.filtfilt(b, a, noise)
            elif stellar_type.lower() in ['neutron_star', 'quasar']:
                # Higher frequency noise
                b, a = signal.butter(3, [100, 2000], 'band', fs=self.sample_rate)
                noise = signal.filtfilt(b, a, noise)
            
            # Add noise
            stellar_sound = stellar_sound + noise
        
        # Normalize
        if np.max(np.abs(stellar_sound)) > 0:
            stellar_sound = stellar_sound / np.max(np.abs(stellar_sound)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {duration}s of {stellar_type} sound")
        return stellar_sound
    
    def generate_cosmic_constants(self, duration: float = DEFAULT_DURATION,
                               amplitude: float = 0.8) -> np.ndarray:
        """
        Generate sound based on fundamental cosmic constants translated to frequencies.
        
        Args:
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the cosmic constants sound
        """
        # Number of samples
        num_samples = int(duration * self.sample_rate)
        time = np.arange(num_samples) / self.sample_rate
        
        # Initialize sound array
        constants_sound = np.zeros_like(time)
        
        # Calculate frequencies from fundamental constants
        # (scaled to audible range using logarithmic mapping)
        
        # Frequency from speed of light
        # log(c) = 8.47 -> scale to range 100-800 Hz
        c_freq = 100 + 700 * (np.log10(SPEED_OF_LIGHT) - 8) / 0.5
        
        # Frequency from gravitational constant
        # log(G) = -10.18 -> scale to range 200-600 Hz
        g_freq = 200 + 400 * (np.log10(GRAVITATIONAL_CONSTANT) + 10.2) / 0.2
        
        # Frequency from Planck constant
        # log(h) = -33.18 -> scale to range 300-700 Hz
        h_freq = 300 + 400 * (np.log10(PLANCK_CONSTANT) + 33.2) / 0.2
        
        # Frequency from electron/proton mass ratio
        # log(m_e/m_p) = 3.26 -> scale to range 400-800 Hz
        m_ratio = ELECTRON_MASS / PROTON_MASS
        m_freq = 400 + 400 * (np.log10(m_ratio) + 3.3) / 0.1
        
        # Add each constant as a pure tone
        constants = [
            {'name': 'speed_of_light', 'freq': c_freq, 'amp': 0.7},
            {'name': 'gravitational_constant', 'freq': g_freq, 'amp': 0.6},
            {'name': 'planck_constant', 'freq': h_freq, 'amp': 0.65},
            {'name': 'mass_ratio', 'freq': m_freq, 'amp': 0.5}
        ]
        
        for constant in constants:
            freq = constant['freq']
            amp = constant['amp'] * amplitude
            
            # Randomize phase
            phase = np.random.random() * 2 * np.pi
            
            # Add harmonic overtones
            for i in range(3):
                harmonic_freq = freq * (i + 1)
                harmonic_amp = amp / (i + 1)
                
                constants_sound += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * time + phase)
        
        # Add phi-related frequency
        phi_freq = 432.0  # Base frequency
        phi_amp = 0.7 * amplitude
        
        for i in range(5):
            # Frequencies based on powers of phi
            f = phi_freq * (GOLDEN_RATIO ** i)
            a = phi_amp / (i + 1)
            
            constants_sound += a * np.sin(2 * np.pi * f * time)
        
        # Apply gentle modulation
        mod_freq = 0.1  # Hz
        modulation = 0.2 * np.sin(2 * np.pi * mod_freq * time) + 0.8
        
        constants_sound = constants_sound * modulation
        
        # Normalize
        if np.max(np.abs(constants_sound)) > 0:
            constants_sound = constants_sound / np.max(np.abs(constants_sound)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {duration}s of cosmic constants sound")
        return constants_sound
    
    def generate_dimensional_resonance(self, dimension: float, duration: float = DEFAULT_DURATION,
                                    amplitude: float = 0.8) -> np.ndarray:
        """
        Generate resonance frequencies for a specific dimension.
        
        Args:
            dimension: Target dimension (3-10, can be fractional for transitions)
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the dimensional resonance
        """
        # Number of samples
        num_samples = int(duration * self.sample_rate)
        time = np.arange(num_samples) / self.sample_rate
        
        # Initialize sound array
        dim_sound = np.zeros_like(time)
        
        # Base frequency scales with dimension (higher dimensions = higher frequencies)
        base_freq = 72.0 * (GOLDEN_RATIO ** (dimension - 3))
        
        # Each dimension has unique frequency ratios
        if dimension <= 3.5:
            # 3D (physical) - simpler ratios
            ratios = [1.0, 5/4, 3/2, 2.0, 7/3]
            mod_freq = 0.2  # Hz
        elif dimension <= 5.5:
            # 4D-5D - golden ratio becomes important
            ratios = [1.0, GOLDEN_RATIO, 2.0, GOLDEN_RATIO*2, 3.0]
            mod_freq = 0.3  # Hz
        elif dimension <= 7.5:
            # 6D-7D - more complex ratios
            ratios = [1.0, np.sqrt(2), GOLDEN_RATIO, np.sqrt(5), 2.0, np.pi/1.5]
            mod_freq = 0.4  # Hz
        else:
            # 8D-10D - most complex harmonic structures
            ratios = [1.0, GOLDEN_RATIO, np.sqrt(3), np.pi/2, GOLDEN_RATIO*2, 3.0, np.sqrt(7)]
            mod_freq = 0.5  # Hz
        
        # Add base frequency and harmonics
        for i, ratio in enumerate(ratios):
            freq = base_freq * ratio
            
            # Amplitude decreases for higher harmonics
            if i == 0:
                amp = amplitude  # Base frequency at full amplitude
            else:
                amp = amplitude * (0.8 / i)
            
            # Randomize phase
            phase = np.random.random() * 2 * np.pi
            
            # Add to dimensional sound
            dim_sound += amp * np.sin(2 * np.pi * freq * time + phase)
        
        # Add dimension-specific modulation
        if dimension % 1 != 0:
            # Transitional dimension (e.g., 3.5, 4.7)
            # Create beat frequency between adjacent dimensions
            lower_dim = int(dimension)
            upper_dim = lower_dim + 1
            
            lower_freq = 72.0 * (GOLDEN_RATIO ** (lower_dim - 3))
            upper_freq = 72.0 * (GOLDEN_RATIO ** (upper_dim - 3))
            
            # Calculate transition progress
            progress = dimension - lower_dim
            
            # Add transition frequencies with crossfade
            lower_amp = amplitude * (1 - progress)
            upper_amp = amplitude * progress
            
            transition_mod = lower_amp * np.sin(2 * np.pi * lower_freq * time) + upper_amp * np.sin(2 * np.pi * upper_freq * time)
            
            # Mix with dimensional sound
            dim_sound = 0.7 * dim_sound + 0.3 * transition_mod
        
        # Add dimension-specific modulation
        modulation = 0.3 * np.sin(2 * np.pi * mod_freq * time) + 0.7
        
        # Higher dimensions have faster modulation
        if dimension > 7:
            fast_mod_freq = mod_freq * 3
            fast_mod = 0.2 * np.sin(2 * np.pi * fast_mod_freq * time)
            modulation += fast_mod
        
        # Apply modulation
        dim_sound = dim_sound * modulation
        
        # Add quantum noise for higher dimensions
        if dimension > 5:
            # More quantum noise for higher dimensions
            noise_level = 0.1 + 0.05 * (dimension - 5)
            noise_level = min(0.3, noise_level)  # Cap at 0.3
            
            # Generate and filter noise
            noise = np.random.normal(0, noise_level, num_samples)
            
            # Filter based on dimension
            if dimension < 7:
                b, a = signal.butter(3, [100, 1000], 'band', fs=self.sample_rate)
            else:
                b, a = signal.butter(3, [200, 2000], 'band', fs=self.sample_rate)
                
            filtered_noise = signal.filtfilt(b, a, noise)
            
            # Add filtered noise
            dim_sound = dim_sound + filtered_noise
        
        # Normalize
        if np.max(np.abs(dim_sound)) > 0:
            dim_sound = dim_sound / np.max(np.abs(dim_sound)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {duration}s of {dimension}D dimensional resonance")
        return dim_sound
    
    def generate_universe_background(self, duration: float = DEFAULT_DURATION, 
                                  layered: bool = True) -> np.ndarray:
        """
        Generate a complete universe background sound with multiple cosmic elements.
        
        Args:
            duration: Duration in seconds
            layered: Whether to layer multiple cosmic sounds (True) or use a simpler version (False)
            
        Returns:
            NumPy array containing the universe background sound
        """
        if layered:
            # Generate component sounds
            cmb = self.generate_cosmic_background(duration, 0.6)
            constants = self.generate_cosmic_constants(duration, 0.5)
            
            # Generate stellar components
            galaxy = self.generate_stellar_sound('galaxy', duration, 0.4)
            
            # Mix components
            universe_sound = 0.5 * cmb + 0.3 * constants + 0.2 * galaxy
            
        else:
            # Generate simplified cosmic background
            universe_sound = self.generate_cosmic_background(duration, 0.8)
        
        # Apply subtle amplitude variation
        num_samples = len(universe_sound)
        time = np.arange(num_samples) / self.sample_rate
        
        # Very slow modulation (cosmic breathing)
        mod_freq = 0.02  # Hz
        modulation = 0.1 * np.sin(2 * np.pi * mod_freq * time) + 0.9
        
        universe_sound = universe_sound * modulation
        
        # Normalize
        if np.max(np.abs(universe_sound)) > 0:
            universe_sound = universe_sound / np.max(np.abs(universe_sound)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {duration}s of universe background sound")
        return universe_sound

def save_sound(self, sound: np.ndarray, filename: str = None, sound_type: str = None) -> str:
        """
        Save a generated sound to a WAV file.
        
        Args:
            sound: NumPy array containing the sound samples
            filename: Custom filename (optional)
            sound_type: Type of sound (for auto-naming)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            if sound_type:
                filename = f"{sound_type.lower().replace(' ', '_')}_{timestamp}.wav"
            else:
                filename = f"universe_sound_{timestamp}.wav"
        
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
    
    def visualize_sound_spectrum(self, sound: np.ndarray, title: str = "Sound Spectrum",
                              show: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the spectrum of a cosmic sound.
        
        Args:
            sound: Sound array to visualize
            title: Title for the plot
            show: Whether to display the plot
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot time domain (just a sample)
        max_plot_samples = min(10000, len(sound))
        time = np.arange(max_plot_samples) / self.sample_rate
        ax1.plot(time, sound[:max_plot_samples])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{title} - Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # Calculate and plot spectrum
        n_fft = min(8192, len(sound))
        freqs, psd = signal.welch(sound, self.sample_rate, nperseg=n_fft)
        
        ax2.semilogy(freqs, psd)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power/Frequency (dB/Hz)')
        ax2.set_title(f'{title} - Power Spectral Density')
        ax2.set_xlim([20, 2000])  # Focus on most important frequencies
        ax2.grid(True, alpha=0.3)
        
        # Add cosmic frequency markers
        for name, freq in COSMIC_FREQUENCIES.items():
            if 20 <= freq <= 2000:  # Only within visible range
                ax2.axvline(x=freq, color='r', linestyle='--', alpha=0.3)
                ax2.text(freq, ax2.get_ylim()[0] * 10, name.replace('_', ' '),
                        rotation=90, verticalalignment='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved spectrum visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_field_activation_background(self, field_type: str, 
                                         duration: float = DEFAULT_DURATION) -> str:
        """
        Generate and save background universe sounds for field activation.
        These sounds enhance the formation and stability of dimensional fields.
        
        Args:
            field_type: Type of field ('void', 'guff', 'sephiroth', 'earth')
            duration: Duration in seconds
            
        Returns:
            Path to the saved sound file
        """
        if field_type.lower() == 'void':
            # Void field uses cosmic background and quantum harmonics
            cmb = self.generate_cosmic_background(duration, 0.6)
            constants = self.generate_cosmic_constants(duration, 0.5)
            
            # Higher dimensions for spark formation
            dim_resonance = self.generate_dimensional_resonance(8.5, duration, 0.4)
            
            # Mix components
            activation_sound = 0.4 * cmb + 0.3 * constants + 0.3 * dim_resonance
            
            # Add low-frequency pulses for formation potential
            time = np.arange(len(activation_sound)) / self.sample_rate
            pulse_freq = 0.1  # Hz
            pulses = 0.2 * np.sin(2 * np.pi * pulse_freq * time) ** 2
            
            activation_sound = activation_sound * (1 + pulses)
            
        elif field_type.lower() == 'guff':
            # Guff field uses creator resonance and cosmic constants
            constants = self.generate_cosmic_constants(duration, 0.7)
            
            # Creator dimension (highest)
            dim_resonance = self.generate_dimensional_resonance(10.0, duration, 0.6)
            
            # Sun as creator representation
            stellar = self.generate_stellar_sound('sun', duration, 0.5)
            
            # Mix components
            activation_sound = 0.4 * constants + 0.4 * dim_resonance + 0.2 * stellar
            
            # Add phi-based modulation for sacred geometry
            time = np.arange(len(activation_sound)) / self.sample_rate
            phi_mod_freq = 1.0 / GOLDEN_RATIO  # Hz
            phi_mod = 0.3 * np.sin(2 * np.pi * phi_mod_freq * time) ** 3
            
            activation_sound = activation_sound * (1 + phi_mod)
            
        elif field_type.lower() == 'sephiroth':
            # Sephiroth uses combination of dimensional resonances
            dim_base = 7.0  # Middle Sephiroth dimension
            dim_resonance = self.generate_dimensional_resonance(dim_base, duration, 0.5)
            
            # Add resonance from adjacent dimensions
            dim_higher = self.generate_dimensional_resonance(dim_base + 1.0, duration, 0.3)
            dim_lower = self.generate_dimensional_resonance(dim_base - 1.0, duration, 0.3)
            
            # Mix dimensional resonances
            activation_sound = 0.6 * dim_resonance + 0.2 * dim_higher + 0.2 * dim_lower
            
            # Add galactic center frequency (spiritual center)
            time = np.arange(len(activation_sound)) / self.sample_rate
            galactic_freq = COSMIC_FREQUENCIES['galactic_center']
            galactic_tone = 0.3 * np.sin(2 * np.pi * galactic_freq * time)
            
            activation_sound = 0.8 * activation_sound + 0.2 * galactic_tone
            
        elif field_type.lower() == 'earth':
            # Earth uses Schumann resonance and earth-specific frequencies
            # Generate earth-specific background
            earth_resonance = np.zeros(int(duration * self.sample_rate))
            time = np.arange(len(earth_resonance)) / self.sample_rate
            
            # Add Schumann resonance frequencies
            schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]
            for i, freq in enumerate(schumann_freqs):
                amp = 0.8 / (i + 1)  # Decreasing amplitude
                earth_resonance += amp * np.sin(2 * np.pi * freq * time)
            
            # Add earth dimensional resonance
            dim_resonance = self.generate_dimensional_resonance(3.0, duration, 0.4)
            
            # Add natural cycles
            day_cycle_freq = 1.0 / 86400  # 1 day in Hz
            day_cycle = 0.2 * np.sin(2 * np.pi * day_cycle_freq * time)
            
            # Mix components
            activation_sound = 0.5 * earth_resonance + 0.3 * dim_resonance + 0.2 * day_cycle
            
            # Add subtle pulses at heart rate frequency
            heart_freq = 1.2  # ~72 BPM
            heart_pulse = 0.15 * np.sin(2 * np.pi * heart_freq * time) ** 2
            
            activation_sound = activation_sound * (1 + heart_pulse)
            
        else:
            # Default to simple cosmic background
            activation_sound = self.generate_cosmic_background(duration, 0.8)
        
        # Normalize
        if np.max(np.abs(activation_sound)) > 0:
            activation_sound = activation_sound / np.max(np.abs(activation_sound)) * MAX_AMPLITUDE
        
        # Save the activation background
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{field_type.lower()}_background_{timestamp}.wav"
        
        return self.save_sound(activation_sound, filename, f"{field_type} background")
    
    def generate_all_cosmic_sounds(self, duration: float = 30.0, 
                               save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate and save all types of cosmic sounds as a sample library.
        
        Args:
            duration: Duration of each sound in seconds
            save_dir: Optional custom directory to save sounds
            
        Returns:
            Dictionary mapping sound types to file paths
        """
        # Use provided directory or default
        output_dir = save_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        sound_files = {}
        
        # Generate cosmic background
        cmb_sound = self.generate_cosmic_background(duration)
        cmb_path = os.path.join(output_dir, "cosmic_background.wav")
        sound_files['cosmic_background'] = self.save_sound(cmb_sound, cmb_path)
        
        # Generate cosmic constants
        constants_sound = self.generate_cosmic_constants(duration)
        constants_path = os.path.join(output_dir, "cosmic_constants.wav")
        sound_files['cosmic_constants'] = self.save_sound(constants_sound, constants_path)
        
        # Generate stellar sounds
        for stellar_type in STELLAR_FREQUENCIES.keys():
            stellar_sound = self.generate_stellar_sound(stellar_type, duration)
            stellar_path = os.path.join(output_dir, f"{stellar_type}.wav")
            sound_files[stellar_type] = self.save_sound(stellar_sound, stellar_path)
        
        # Generate dimensional resonances
        for dim in [3.0, 4.0, 5.0, 7.0, 10.0]:
            dim_sound = self.generate_dimensional_resonance(dim, duration)
            dim_path = os.path.join(output_dir, f"dimension_{int(dim)}d.wav")
            sound_files[f'dimension_{int(dim)}d'] = self.save_sound(dim_sound, dim_path)
        
        # Generate complete universe background
        universe_sound = self.generate_universe_background(duration)
        universe_path = os.path.join(output_dir, "universe_background.wav")
        sound_files['universe_background'] = self.save_sound(universe_sound, universe_path)
        
        logger.info(f"Generated {len(sound_files)} cosmic sounds in {output_dir}")
        return sound_files

# Example usage
if __name__ == "__main__":
    generator = UniverseSounds()
    
    # Generate and save cosmic background radiation sound
    cmb = generator.generate_cosmic_background(10.0)
    generator.save_sound(cmb, "cosmic_background.wav")
    
    # Generate stellar sounds
    sun = generator.generate_stellar_sound('sun', 10.0)
    generator.save_sound(sun, "sun.wav")
    
    black_hole = generator.generate_stellar_sound('black_hole', 10.0)
    generator.save_sound(black_hole, "black_hole.wav")
    
    # Generate dimensional resonance
    dim7 = generator.generate_dimensional_resonance(7.0, 10.0)
    generator.save_sound(dim7, "dimension_7d.wav")
    
    # Generate field activation background
    void_bg = generator.generate_field_activation_background('void', 30.0)
    print(f"Generated void field background: {void_bg}")
    
    # Visualize spectra
    generator.visualize_sound_spectrum(cmb, "Cosmic Background Radiation", 
                                     show=False, save_path="cmb_spectrum.png")
    generator.visualize_sound_spectrum(black_hole, "Black Hole Sonification", 
                                     show=False, save_path="black_hole_spectrum.png")