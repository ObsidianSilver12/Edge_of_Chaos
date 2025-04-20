"""
White Noise Module

This module implements generators for various types of noise used in the soul development framework.
White noise represents primordial randomness and chaos from which order emerges.
Different colored noise variants (white, pink, brown, etc.) have different frequency distributions
that affect field formation in specific ways.

Key functions:
- Generate quantum white noise for primordial field formation
- Create filtered noise variants with specific spectral properties
- Produce structured noise patterns that enhance specific field characteristics
- Generate edge-of-chaos noise that facilitates emergence

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
    filename='white_noise.log'
)
logger = logging.getLogger('white_noise')

# Constants
SAMPLE_RATE = 44100  # Hz - Standard audio sample rate
MAX_AMPLITUDE = 0.8  # Maximum amplitude to prevent clipping
OUTPUT_DIR = "output/sounds"  # Directory for saving generated sounds
DEFAULT_DURATION = 60.0  # Default duration in seconds
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # ~1.618
EDGE_OF_CHAOS_RATIO = 1 / GOLDEN_RATIO  # ~0.618 - Critical for emergence

# Noise type specifications
NOISE_TYPES = {
    'white': {
        'description': 'Equal power across all frequencies, pure randomness',
        'filter_type': None,
        'filter_params': None,
        'spectral_density': 'flat',
        'use_case': 'Primordial chaos, void field initialization'
    },
    'pink': {
        'description': '1/f power spectrum, natural balance',
        'filter_type': 'pink',
        'filter_params': None,
        'spectral_density': '1/f',
        'use_case': 'Natural field formations, harmonization'
    },
    'brown': {
        'description': '1/f^2 power spectrum, integration over time',
        'filter_type': 'brown',
        'filter_params': None,
        'spectral_density': '1/f^2',
        'use_case': 'Stabilization, structural support'
    },
    'blue': {
        'description': 'f power spectrum, differentiation',
        'filter_type': 'blue',
        'filter_params': None,
        'spectral_density': 'f',
        'use_case': 'Edge detection, boundary formation'
    },
    'violet': {
        'description': 'f^2 power spectrum, acceleration',
        'filter_type': 'violet',
        'filter_params': None,
        'spectral_density': 'f^2',
        'use_case': 'Rapid transitions, frequency shifting'
    },
    'quantum': {
        'description': 'Filtered white noise with quantum-like properties',
        'filter_type': 'quantum',
        'filter_params': {'edge_of_chaos': EDGE_OF_CHAOS_RATIO},
        'spectral_density': 'structured white',
        'use_case': 'Void field quantum fluctuations, spark formation'
    },
    'edge_of_chaos': {
        'description': 'Noise at the boundary of order and chaos',
        'filter_type': 'edge_of_chaos',
        'filter_params': {'ratio': EDGE_OF_CHAOS_RATIO},
        'spectral_density': 'critical',
        'use_case': 'Emergence points, dimensional transitions'
    }
}

class NoiseGenerator:
    """
    Class for generating various types of noise used in field formation.
    Different noise types have specific effects on field properties and formation processes.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, output_dir: str = OUTPUT_DIR):
        """
        Initialize a new noise generator.
        
        Args:
            sample_rate: Sample rate in Hz for audio generation
            output_dir: Directory to save generated noise files
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Noise Generator initialized with sample rate {sample_rate} Hz")
        logger.info(f"Output directory: {output_dir}")
    
    def generate_white_noise(self, duration: float = DEFAULT_DURATION, 
                           amplitude: float = 0.8) -> np.ndarray:
        """
        Generate pure white noise (equal power at all frequencies).
        
        Args:
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the noise samples
        """
        # Calculate number of samples
        num_samples = int(duration * self.sample_rate)
        
        # Generate white noise
        noise = np.random.normal(0, amplitude, num_samples)
        
        # Ensure amplitude doesn't exceed limits
        if np.max(np.abs(noise)) > 0:
            noise = noise / np.max(np.abs(noise)) * amplitude
        
        logger.info(f"Generated {duration}s of white noise at amplitude {amplitude}")
        return noise
    
    def generate_colored_noise(self, noise_type: str, duration: float = DEFAULT_DURATION,
                             amplitude: float = 0.8) -> np.ndarray:
        """
        Generate colored noise with specified spectral characteristics.
        
        Args:
            noise_type: Type of noise ('pink', 'brown', 'blue', 'violet')
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the colored noise samples
        """
        if noise_type.lower() not in NOISE_TYPES:
            logger.warning(f"Unknown noise type: {noise_type}. Using white noise instead.")
            return self.generate_white_noise(duration, amplitude)
        
        # Start with white noise
        white_noise = self.generate_white_noise(duration, amplitude)
        
        # Apply the appropriate filter based on noise type
        if noise_type.lower() == 'pink':
            # 1/f power spectrum
            return self._generate_pink_noise(white_noise)
        elif noise_type.lower() == 'brown' or noise_type.lower() == 'brownian':
            # 1/f^2 power spectrum (integrate white noise)
            return self._generate_brown_noise(white_noise)
        elif noise_type.lower() == 'blue':
            # f power spectrum (differentiate white noise)
            return self._generate_blue_noise(white_noise)
        elif noise_type.lower() == 'violet':
            # f^2 power spectrum (differentiate twice)
            return self._generate_violet_noise(white_noise)
        elif noise_type.lower() == 'quantum':
            # Quantum-like noise (filtered and structured)
            return self._generate_quantum_noise(white_noise)
        elif noise_type.lower() == 'edge_of_chaos':
            # Edge of chaos noise (critical state)
            return self._generate_edge_of_chaos_noise(white_noise)
        else:
            # Default to white noise
            return white_noise
    
    def _generate_pink_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate pink noise (1/f power spectrum) from white noise.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Pink noise array
        """
        # Generate in frequency domain
        N = len(white_noise)
        X = np.fft.rfft(white_noise)
        
        # Create 1/f filter
        f = np.fft.rfftfreq(N, 1/self.sample_rate)
        f[0] = 1  # Avoid division by zero
        
        # Apply 1/f filter
        X = X / np.sqrt(f)
        
        # Convert back to time domain
        pink_noise = np.fft.irfft(X, n=N)
        
        # Normalize
        if np.max(np.abs(pink_noise)) > 0:
            pink_noise = pink_noise / np.max(np.abs(pink_noise)) * MAX_AMPLITUDE
        
        logger.info("Generated pink noise (1/f spectrum)")
        return pink_noise
    
    def _generate_brown_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate brown noise (1/f^2 power spectrum) from white noise.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Brown noise array
        """
        # Generate through cumulative sum (integration)
        brown_noise = np.cumsum(white_noise)
        
        # Remove DC offset (mean)
        brown_noise = brown_noise - np.mean(brown_noise)
        
        # Normalize
        if np.max(np.abs(brown_noise)) > 0:
            brown_noise = brown_noise / np.max(np.abs(brown_noise)) * MAX_AMPLITUDE
        
        logger.info("Generated brown noise (1/f^2 spectrum)")
        return brown_noise
    
    def _generate_blue_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate blue noise (f power spectrum) from white noise.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Blue noise array
        """
        # Generate in frequency domain
        N = len(white_noise)
        X = np.fft.rfft(white_noise)
        
        # Create f filter
        f = np.fft.rfftfreq(N, 1/self.sample_rate)
        f[0] = 1  # Avoid division by zero
        
        # Apply f filter
        X = X * np.sqrt(f)
        
        # Convert back to time domain
        blue_noise = np.fft.irfft(X, n=N)
        
        # Normalize
        if np.max(np.abs(blue_noise)) > 0:
            blue_noise = blue_noise / np.max(np.abs(blue_noise)) * MAX_AMPLITUDE
        
        logger.info("Generated blue noise (f spectrum)")
        return blue_noise
    
    def _generate_violet_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate violet noise (f^2 power spectrum) from white noise.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Violet noise array
        """
        # Generate in frequency domain
        N = len(white_noise)
        X = np.fft.rfft(white_noise)
        
        # Create f^2 filter
        f = np.fft.rfftfreq(N, 1/self.sample_rate)
        f[0] = 1  # Avoid division by zero
        
        # Apply f^2 filter
        X = X * f
        
        # Convert back to time domain
        violet_noise = np.fft.irfft(X, n=N)
        
        # Normalize
        if np.max(np.abs(violet_noise)) > 0:
            violet_noise = violet_noise / np.max(np.abs(violet_noise)) * MAX_AMPLITUDE
        
        logger.info("Generated violet noise (f^2 spectrum)")
        return violet_noise
    
    def _generate_quantum_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate quantum-like noise with structured fluctuations.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Quantum noise array
        """
        # Create quantum fluctuations by filtering and structuring white noise
        
        # First, apply a bandpass filter to focus energy in useful frequency ranges
        # For quantum effects, we want a mix of scales
        sos = signal.butter(4, [20, 2000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_noise = signal.sosfilt(sos, white_noise)
        
        # Create quantum-like structure with interference patterns
        N = len(white_noise)
        time = np.arange(N) / self.sample_rate
        
        # Add interference patterns at key frequencies
        quantum_pattern = np.zeros_like(time)
        
        # Add several oscillating components with phi-related frequencies
        base_freq = 432.0  # Base frequency (Hz)
        
        for i in range(5):
            # Use Fibonacci sequence derived frequencies
            freq = base_freq * (1 + i/GOLDEN_RATIO)
            phase = np.random.random() * 2 * np.pi  # Random phase
            quantum_pattern += 0.2 * np.sin(2 * np.pi * freq * time + phase)
        
        # Apply modulation to create structured quantum effects
        quantum_noise = filtered_noise * (1 + 0.3 * quantum_pattern)
        
        # Add small amount of edge-of-chaos noise
        chaos_noise = self._generate_edge_of_chaos_noise(white_noise)
        quantum_noise = 0.7 * quantum_noise + 0.3 * chaos_noise
        
        # Normalize
        if np.max(np.abs(quantum_noise)) > 0:
            quantum_noise = quantum_noise / np.max(np.abs(quantum_noise)) * MAX_AMPLITUDE
        
        logger.info("Generated quantum noise with structured fluctuations")
        return quantum_noise
    
    def _generate_edge_of_chaos_noise(self, white_noise: np.ndarray) -> np.ndarray:
        """
        Generate edge-of-chaos noise with critical state properties.
        
        Args:
            white_noise: White noise input array
            
        Returns:
            Edge-of-chaos noise array
        """
        # Edge of chaos noise sits between order and disorder
        # We'll create this by blending filtered and structured noise
        
        # Create ordered component (filtered, structured)
        sos = signal.butter(2, 200, 'low', fs=self.sample_rate, output='sos')
        ordered_component = signal.sosfilt(sos, white_noise)
        
        # Create chaotic component (high-frequency, less structured)
        sos_high = signal.butter(2, 1000, 'high', fs=self.sample_rate, output='sos')
        chaotic_component = signal.sosfilt(sos_high, white_noise)
        
        # Mix with the golden ratio inverse (edge of chaos ratio ~0.618)
        edge_chaos_noise = (EDGE_OF_CHAOS_RATIO * ordered_component + 
                          (1 - EDGE_OF_CHAOS_RATIO) * chaotic_component)
        
        # Add self-organized criticality by applying non-linear feedback
        N = len(white_noise)
        
        # Create self-organized critical filter
        edge_chaos_noise_soc = np.zeros_like(edge_chaos_noise)
        edge_chaos_noise_soc[0] = edge_chaos_noise[0]
        
        for i in range(1, N):
            # Add neighbor-dependent non-linear feedback
            edge_chaos_noise_soc[i] = edge_chaos_noise[i] + EDGE_OF_CHAOS_RATIO * edge_chaos_noise_soc[i-1] * (1 - abs(edge_chaos_noise_soc[i-1]))
        
        # Normalize
        if np.max(np.abs(edge_chaos_noise_soc)) > 0:
            edge_chaos_noise_soc = edge_chaos_noise_soc / np.max(np.abs(edge_chaos_noise_soc)) * MAX_AMPLITUDE
        
        logger.info("Generated edge-of-chaos noise with critical state properties")
        return edge_chaos_noise_soc
    
    def generate_structured_noise(self, structure_type: str, duration: float = DEFAULT_DURATION,
                                base_amplitude: float = 0.8) -> np.ndarray:
        """
        Generate noise with sacred geometry or dimensional structures embedded.
        
        Args:
            structure_type: Type of structure ('fibonacci', 'phi', 'fractal', 'sacred')
            duration: Duration in seconds
            base_amplitude: Base amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the structured noise samples
        """
        # Generate base noise (pink noise provides a good foundation)
        base_noise = self.generate_colored_noise('pink', duration, base_amplitude)
        N = len(base_noise)
        
        if structure_type.lower() == 'fibonacci':
            # Embed Fibonacci sequence structure in the noise
            fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
            
            # Create Fibonacci-structured amplitude modulation
            time = np.arange(N) / self.sample_rate
            
            # Create modulation signal
            mod_signal = np.zeros_like(time)
            for i in range(len(fib) - 1):
                # Use ratios between consecutive Fibonacci numbers
                freq = 0.1 * fib[i]  # Lower frequencies for audible effects
                amp = 0.2 / (i + 1)  # Decreasing amplitude for higher frequencies
                mod_signal += amp * np.sin(2 * np.pi * freq * time)
            
            # Apply modulation
            structured_noise = base_noise * (1 + 0.5 * mod_signal)
            
        elif structure_type.lower() == 'phi' or structure_type.lower() == 'golden':
            # Embed golden ratio (phi) structures
            time = np.arange(N) / self.sample_rate
            
            # Create modulation with phi-based frequencies
            mod_signal = np.zeros_like(time)
            for i in range(7):
                freq = 0.1 * (GOLDEN_RATIO ** i)  # Phi-based frequency scaling
                amp = 0.3 / (i + 1)  # Decreasing amplitude
                mod_signal += amp * np.sin(2 * np.pi * freq * time)
            
            # Apply modulation
            structured_noise = base_noise * (1 + 0.5 * mod_signal)
            
        elif structure_type.lower() == 'fractal':
            # Embed fractal structure using 1/f^beta noise with varying beta
            
            # Create fractal process with varying spectral slope
            X = np.fft.rfft(base_noise)
            f = np.fft.rfftfreq(N, 1/self.sample_rate)
            f[0] = 1  # Avoid division by zero
            
            # Create fractal filter with self-similar characteristics
            fractal_filter = 1.0 / (f ** (0.5 + 0.5 * np.sin(np.log(f + 1))))
            
            # Apply fractal filter
            X_fractal = X * fractal_filter
            
            # Convert back to time domain
            structured_noise = np.fft.irfft(X_fractal, n=N)
            
        elif structure_type.lower() == 'sacred':
            # Embed sacred geometry ratios and harmonics
            sacred_ratios = [1.0, GOLDEN_RATIO, np.sqrt(2), np.sqrt(3), np.sqrt(5), 3/2, 4/3, 5/3]
            
            time = np.arange(N) / self.sample_rate
            
            # Create modulation with sacred geometry ratios
            mod_signal = np.zeros_like(time)
            base_freq = 1.0  # Very low base frequency
            
            for i, ratio in enumerate(sacred_ratios):
                freq = base_freq * ratio
                amp = 0.3 / (i + 1)  # Decreasing amplitude
                mod_signal += amp * np.sin(2 * np.pi * freq * time)
            
            # Apply modulation
            structured_noise = base_noise * (1 + 0.6 * mod_signal)
            
        else:
            logger.warning(f"Unknown structure type: {structure_type}. Using base noise.")
            structured_noise = base_noise
        
        # Normalize
        if np.max(np.abs(structured_noise)) > 0:
            structured_noise = structured_noise / np.max(np.abs(structured_noise)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {structure_type}-structured noise")
        return structured_noise
    
    def generate_dimensional_noise(self, dimension: int, duration: float = DEFAULT_DURATION,
                                 amplitude: float = 0.8) -> np.ndarray:
        """
        Generate noise with characteristics of a specific dimension.
        
        Args:
            dimension: Target dimension (3-10)
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            NumPy array containing the dimensional noise samples
        """
        # Each dimension has different noise characteristics
        if dimension < 3 or dimension > 10:
            logger.warning(f"Dimension {dimension} out of range (3-10). Using middle properties.")
            dimension = 6
        
        # Start with different base noise types based on dimension
        if dimension <= 4:
            # Lower dimensions use more structured, brown-like noise
            base_noise = self.generate_colored_noise('brown', duration, amplitude)
        elif dimension <= 7:
            # Middle dimensions use pink noise (balance of order and chaos)
            base_noise = self.generate_colored_noise('pink', duration, amplitude)
        else:
            # Higher dimensions use more chaotic, white or blue noise
            base_noise = self.generate_colored_noise('blue', duration, amplitude)
        
        # Add dimension-specific modulations
        N = len(base_noise)
        time = np.arange(N) / self.sample_rate
        
        # Create dimension-specific modulation
        # Higher dimensions have higher frequency components
        mod_signal = np.zeros_like(time)
        
        # Base modulation frequency scales with dimension
        mod_freq = 0.1 * dimension  # Hz
        
        # Add dimension-specific oscillations
        for i in range(dimension - 2):
            freq = mod_freq * (1 + i/3)
            amp = 0.3 / (i + 1)  # Decreasing amplitude
            mod_signal += amp * np.sin(2 * np.pi * freq * time)
        
        # Higher dimensions have more quantum fluctuations
        if dimension >= 8:
            quantum_component = self.generate_colored_noise('quantum', duration, amplitude * 0.5)
            base_noise = 0.7 * base_noise + 0.3 * quantum_component
        
        # Apply dimension-specific modulation
        dimensional_noise = base_noise * (1 + 0.4 * mod_signal)
        
        # Add more edge-of-chaos components for transitional dimensions (4.5, 5.5, etc.)
        if dimension % 1 != 0:
            edge_chaos_component = self.generate_colored_noise('edge_of_chaos', duration, amplitude * 0.5)
            dimensional_noise = 0.7 * dimensional_noise + 0.3 * edge_chaos_component
        
        # Normalize
        if np.max(np.abs(dimensional_noise)) > 0:
            dimensional_noise = dimensional_noise / np.max(np.abs(dimensional_noise)) * MAX_AMPLITUDE
        
        logger.info(f"Generated {dimension}D dimensional noise")
        return dimensional_noise
    
    def visualize_noise_spectrum(self, noise_array: np.ndarray, noise_type: str = "noise",
                              show: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the spectrum of a noise signal.
        
        Args:
            noise_array: Noise array to visualize
            noise_type: Type of noise (for title)
            show: Whether to display the plot
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot time domain (just a sample)
        max_plot_samples = min(10000, len(noise_array))
        time = np.arange(max_plot_samples) / self.sample_rate
        ax1.plot(time, noise_array[:max_plot_samples])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{noise_type.capitalize()} - Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # Calculate and plot spectrum
        n_fft = min(8192, len(noise_array))
        freqs, psd = signal.welch(noise_array, self.sample_rate, nperseg=n_fft)
        
        ax2.loglog(freqs, psd)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power/Frequency (dB/Hz)')
        ax2.set_title(f'{noise_type.capitalize()} - Power Spectral Density')
        ax2.set_xlim([20, self.sample_rate/2])  # Focus on audible range
        ax2.grid(True, alpha=0.3)
        
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
    
    def save_noise(self, noise: np.ndarray, filename: str = None, 
                 noise_type: str = None) -> str:
        """
        Save a generated noise to a WAV file.
        
        Args:
            noise: NumPy array containing the noise samples
            filename: Custom filename (optional)
            noise_type: Type of noise (for auto-naming)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            if noise_type:
                filename = f"{noise_type.lower()}_noise_{timestamp}.wav"
            else:
                filename = f"noise_{timestamp}.wav"
        
        # Add .wav extension if needed
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure noise is in correct format for WAV (int16)
        noise_int16 = np.int16(noise * 32767)
        
        # Save to WAV file
        wavfile.write(filepath, self.sample_rate, noise_int16)
        
        logger.info(f"Saved noise to {filepath}")
        return filepath
    
    def generate_field_noise(self, field_type: str, duration: float = DEFAULT_DURATION,
                          amplitude: float = 0.8) -> str:
        """
        Generate and save noise specifically designed for a field type.
        
        Args:
            field_type: Type of field ('void', 'guff', 'sephiroth', 'earth')
            duration: Duration in seconds
            amplitude: Amplitude scaling factor (0-1)
            
        Returns:
            Path to the saved noise file
        """
        if field_type.lower() == 'void':
            # Void uses quantum noise with edge-of-chaos components
            quantum_noise = self.generate_colored_noise('quantum', duration, amplitude)
            edge_chaos_noise = self.generate_colored_noise('edge_of_chaos', duration, amplitude)
            
            # Mix with more quantum noise
            noise = 0.7 * quantum_noise + 0.3 * edge_chaos_noise
            
            # Add some very low frequency modulation
            N = len(noise)
            time = np.arange(N) / self.sample_rate
            mod = 0.3 * np.sin(2 * np.pi * 0.05 * time)  # 0.05 Hz modulation
            
            noise = noise * (1 + mod)
            
        elif field_type.lower() == 'guff':
            # Guff uses structured noise with Fibonacci and golden ratio components
            fibonacci_noise = self.generate_structured_noise('fibonacci', duration, amplitude)
            phi_noise = self.generate_structured_noise('phi', duration, amplitude)
            
            # Mix structured noise components
            noise = 0.5 * fibonacci_noise + 0.5 * phi_noise
            
        elif field_type.lower() == 'sephiroth':
            # Sephiroth uses dimensional noise appropriate for higher dimensions
            dim_noise = self.generate_dimensional_noise(7, duration, amplitude)  # Typical Sephiroth dimension
            sacred_noise = self.generate_structured_noise('sacred', duration, amplitude)
            
            # Mix dimensional and sacred components
            noise = 0.6 * dim_noise + 0.4 * sacred_noise
            
        elif field_type.lower() == 'earth':
            # Earth uses brown noise with natural cycles
            brown_noise = self.generate_colored_noise('brown', duration, amplitude)
            
            # Add natural cycles (day/night, etc.)
            N = len(brown_noise)
            time = np.arange(N) / self.sample_rate
            
            # Day cycle (24 hours = 86400 seconds)
            day_cycle = 0.2 * np.sin(2 * np.pi * time / 86400)
            
            # Schumann resonance frequency (7.83 Hz)
            schumann = 0.15 * np.sin(2 * np.pi * 7.83 * time)
            
            # Apply modulation
            noise = brown_noise * (1 + day_cycle + schumann)
            
        else:
            # Default to pink noise
            logger.warning(f"Unknown field type: {field_type}. Using pink noise.")
            noise = self.generate_colored_noise('pink', duration, amplitude)
        
        # Normalize
        if np.max(np.abs(noise)) > 0:
            noise = noise / np.max(np.abs(noise)) * MAX_AMPLITUDE
        
        # Save noise file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{field_type.lower()}_field_noise_{timestamp}.wav"
        
        return self.save_noise(noise, filename, f"{field_type} field")

# Example usage
if __name__ == "__main__":
    generator = NoiseGenerator()
    
    # Generate and save different types of noise
    white = generator.generate_white_noise(5.0)
    generator.save_noise(white, "white_noise_sample.wav", "white")
    
    pink = generator.generate_colored_noise("pink", 5.0)
    generator.save_noise(pink, "pink_noise_sample.wav", "pink")
    
    brown = generator.generate_colored_noise("brown", 5.0)
    generator.save_noise(brown, "brown_noise_sample.wav", "brown")
    
    quantum = generator.generate_colored_noise("quantum", 5.0)
    generator.save_noise(quantum, "quantum_noise_sample.wav", "quantum")
    
    # Generate structured noise
    fib = generator.generate_structured_noise("fibonacci", 10.0)
    generator.save_noise(fib, "fibonacci_noise_sample.wav", "fibonacci")
    
    # Generate field-specific noise
    void_path = generator.generate_field_noise("void", 30.0)
    print(f"Generated void field noise: {void_path}")
    
    # Visualize noise spectra
    generator.visualize_noise_spectrum(white, "white noise", show=False, 
                                     save_path="white_noise_spectrum.png")
    generator.visualize_noise_spectrum(pink, "pink noise", show=False, 
                                     save_path="pink_noise_spectrum.png")
    generator.visualize_noise_spectrum(quantum, "quantum noise", show=False, 
                                     save_path="quantum_noise_spectrum.png")