"""
Field Creation Module

This module handles the creation and basic manipulation of multidimensional quantum fields
that serve as the foundation for various dimensional spaces in the soul development framework.

Key functions:
- Create field with specific properties
- Initialize field with quantum potential
- Set base parameters for field stability and coherence
- Manipulate field energy distribution
- Generate and apply sound frequencies to fields
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import uuid
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sound system
try:
    from sounds.sound_system import SoundSystem, Sound, SoundType
except ImportError:
    # Log warning but continue - field will work without sound
    logging.warning("Sound system not available. Field will function without sound capabilities.")
    SoundSystem = None
    Sound = None
    SoundType = None

# Constants
GOLDEN_RATIO = 1.618033988749895
CHAOS_ORDER_RATIO = 0.618033988749895  # Inverse of golden ratio, edge of chaos
PLANCK_LENGTH = 1.616255e-35  # Planck length in meters
PLANCK_TIME = 5.39124e-44     # Planck time in seconds
VACUUM_PERMITTIVITY = 8.85418782e-12  # Vacuum permittivity
CREATOR_FREQUENCY = 432.0     # Base universal frequency

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='field_system.log'
)
logger = logging.getLogger('field_system')

@dataclass
class Harmonic:
    """Class to represent a harmonic frequency component."""
    frequency: float
    amplitude: float
    phase: float
    
    def __post_init__(self):
        # Normalize phase to [0, 2π)
        self.phase = self.phase % (2 * np.pi)
    
    def get_value_at_time(self, time: float) -> float:
        """Calculate the harmonic's value at a given time."""
        return self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase)


@dataclass
class ResonanceNode:
    """Class to represent a high resonance point in the field."""
    position: Tuple[int, int, int]
    frequency: float
    strength: float


@dataclass
class Field:
    """
    Class to represent a multidimensional quantum field with specific properties.
    This serves as the foundation for all dimensional spaces including Void, Sephiroth, and Earth.
    """
    name: str
    dimensions: Tuple[int, int, int]
    base_frequency: float
    dimension_count: int = 3  # Default is 3D
    
    # Field properties initialized with default values
    energy_field: np.ndarray = field(init=False)
    harmonics: List[Harmonic] = field(default_factory=list)
    resonance_nodes: List[ResonanceNode] = field(default_factory=list)
    stability: float = 0.0
    coherence: float = 0.0
    chaos_ratio: float = CHAOS_ORDER_RATIO
    
    # Sound-related properties
    sound_system: Optional[Any] = None
    field_sound: Optional[Any] = None
    sound_harmonics: List[Dict] = field(default_factory=list)
    sound_enabled: bool = False
    
    def __post_init__(self):
        """Initialize the energy field after the object is created."""
        # Create the basic energy field filled with quantum potential
        self.energy_field = self._initialize_quantum_field()
        
        # Initialize sound system if available
        self._initialize_sound_system()
        
        # Calculate initial stability and coherence
        self.update_field_metrics()
    
    def _initialize_quantum_field(self) -> np.ndarray:
        """
        Initialize a quantum field with base energy potential.
        The field contains quantum fluctuations with a specific chaos-order ratio.
        """
        # Create base field with small random fluctuations around zero
        field = np.random.normal(0, 0.01, self.dimensions)
        
        # Add quantum noise based on chaos ratio
        quantum_noise = np.random.normal(0, self.chaos_ratio, self.dimensions)
        field += quantum_noise
        
        # Apply smoothing to create coherent regions within the chaos
        # This creates a field at the "edge of chaos" which enables emergence
        sigma = [dim // 10 for dim in self.dimensions]  # Smoothing factor
        field = signal.gaussian_filter(field, sigma)
        
        # Normalize field values to range [-1, 1]
        field = field / np.max(np.abs(field))
        
        return field
    
    def _initialize_sound_system(self):
        """Initialize the sound system for this field if available."""
        if SoundSystem is not None:
            try:
                # Create sound system with base frequency
                self.sound_system = SoundSystem(creator_frequency=self.base_frequency)
                
                # Create primary field sound
                self.field_sound = self.sound_system.create_sound(
                    name=f"Field {self.name}",
                    fundamental_frequency=self.base_frequency,
                    sound_type=SoundType.DIMENSIONAL,
                    description=f"Primary sound for {self.name} field"
                )
                
                # Add natural harmonics
                self.field_sound.add_natural_harmonics(count=5)
                
                # Add phi harmonics
                self.field_sound.add_phi_harmonics(count=3)
                
                # Store sound harmonics data
                self.sound_harmonics = self.field_sound.get_harmonic_data()
                
                # Enable sound
                self.sound_enabled = True
                
                logger.info(f"Sound system initialized for field {self.name}")
                logger.info(f"Primary frequency: {self.base_frequency} Hz with {len(self.sound_harmonics)} harmonics")
                
            except Exception as e:
                logger.error(f"Error initializing sound system: {str(e)}")
                self.sound_enabled = False
        else:
            self.sound_enabled = False
    
    def add_harmonic(self, frequency: float, amplitude: float, phase: float = 0.0) -> None:
        """Add a harmonic frequency component to the field."""
        harmonic = Harmonic(frequency=frequency, amplitude=amplitude, phase=phase)
        self.harmonics.append(harmonic)
        
        # Apply the harmonic to the energy field
        self._apply_harmonic_to_field(harmonic)
        
        # Add to sound if enabled
        if self.sound_enabled and self.field_sound is not None:
            try:
                self.field_sound.add_harmonic(
                    frequency=frequency,
                    amplitude=amplitude,
                    phase=phase,
                    description=f"Added field harmonic {frequency:.2f} Hz"
                )
                
                # Update sound harmonics data
                self.sound_harmonics = self.field_sound.get_harmonic_data()
            except Exception as e:
                logger.warning(f"Could not add harmonic to sound: {str(e)}")
        
        # Update field metrics after modification
        self.update_field_metrics()
    
    def _apply_harmonic_to_field(self, harmonic: Harmonic) -> None:
        """Apply a harmonic frequency component to the energy field."""
        # Create meshgrid for 3D coordinates
        x, y, z = np.meshgrid(
            np.linspace(0, 1, self.dimensions[0]),
            np.linspace(0, 1, self.dimensions[1]),
            np.linspace(0, 1, self.dimensions[2]),
            indexing='ij'
        )
        
        # Calculate distance from center
        center = np.array([d / 2 for d in self.dimensions])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Create harmonic wave pattern
        wave = harmonic.amplitude * np.sin(2 * np.pi * harmonic.frequency * r + harmonic.phase)
        
        # Apply wave to energy field with attenuation based on distance
        attenuation = np.exp(-r / np.max(self.dimensions))
        self.energy_field += wave * attenuation
        
        # Normalize field again after modification
        self.energy_field = self.energy_field / np.max(np.abs(self.energy_field))
    
    def calculate_energy_at_point(self, position: Tuple[int, int, int]) -> float:
        """Calculate the energy potential at a specific point in the field."""
        return self.energy_field[position]
    
    def detect_resonance_nodes(self, threshold: float = 0.7) -> List[ResonanceNode]:
        """
        Detect high resonance points in the field.
        These are points where energy concentrates due to the interaction
        of harmonics and field patterns.
        """
        # Calculate gradient magnitude to identify high-energy nodes
        gradient = np.gradient(self.energy_field)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Find local maxima in gradient magnitude
        maxima = signal.peak_local_max(
            gradient_magnitude, 
            min_distance=max(d // 20 for d in self.dimensions),
            threshold_abs=threshold
        )
        
        # Create resonance nodes list
        self.resonance_nodes = []
        for position in maxima:
            # Calculate local frequency by measuring oscillation rate
            # in surrounding area (simplified version)
            x, y, z = position
            local_region = self.energy_field[
                max(0, x-5):min(self.dimensions[0], x+6),
                max(0, y-5):min(self.dimensions[1], y+6),
                max(0, z-5):min(self.dimensions[2], z+6)
            ]
            
            # Count zero crossings to estimate frequency
            zero_crossings = np.sum(np.diff(np.signbit(local_region.flatten())))
            estimated_frequency = zero_crossings * self.base_frequency / local_region.size
            
            # Calculate strength based on energy and gradient
            strength = self.energy_field[tuple(position)] * gradient_magnitude[tuple(position)]
            
            # Create resonance node
            node = ResonanceNode(
                position=tuple(position),
                frequency=estimated_frequency,
                strength=strength
            )
            self.resonance_nodes.append(node)
        
        # Update sound system with resonance nodes if enabled
        if self.sound_enabled and self.sound_system is not None:
            try:
                # Create resonance sound for each node
                for i, node in enumerate(self.resonance_nodes[:5]):  # Limit to first 5 nodes
                    resonance_sound = self.sound_system.create_sound(
                        name=f"Resonance {self.name} Node {i}",
                        fundamental_frequency=node.frequency,
                        sound_type=SoundType.RESONANT,
                        description=f"Resonance node sound for {self.name} field"
                    )
                    
                    # Add harmonics suited for resonance
                    resonance_sound.add_natural_harmonics(count=3)
                    resonance_sound.add_harmonic(
                        ratio=GOLDEN_RATIO,
                        amplitude=0.7,
                        description="Golden ratio resonance"
                    )
            except Exception as e:
                logger.warning(f"Could not create resonance sounds: {str(e)}")
        
        return self.resonance_nodes
    
    def update_field_metrics(self) -> None:
        """Update the stability and coherence metrics for the field."""
        # Calculate stability based on energy distribution
        # Higher stability means less chaotic, more organized energy patterns
        gradient = np.gradient(self.energy_field)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Stability is inverse of average gradient magnitude
        # More uniform fields have higher stability
        self.stability = 1.0 - np.mean(gradient_magnitude)
        
        # Calculate coherence based on spatial correlation
        # Higher coherence means more organized patterns
        # Using autocorrelation as a measure of self-similarity
        flat_field = self.energy_field.flatten() - np.mean(self.energy_field)
        autocorr = np.correlate(flat_field, flat_field, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        autocorr = autocorr / autocorr[0]
        
        # Coherence is the sum of autocorrelation values
        # (indicating how well the field correlates with itself)
        self.coherence = np.sum(autocorr[:min(100, len(autocorr))]) / min(100, len(autocorr))
        
        # Update sound properties based on metrics if enabled
        if self.sound_enabled and self.field_sound is not None:
            try:
                # Adjust sound amplitude based on field stability
                for harmonic in self.field_sound.harmonics:
                    harmonic.amplitude = min(1.0, harmonic.amplitude * (1.0 + 0.1 * self.stability))
                    
                # Add coherence-based harmonic if field is highly coherent
                if self.coherence > 0.8 and len(self.field_sound.harmonics) < 10:
                    coherence_freq = self.base_frequency * self.coherence * GOLDEN_RATIO
                    self.field_sound.add_harmonic(
                        frequency=coherence_freq,
                        amplitude=self.coherence * 0.8,
                        phase=0.0,
                        description="Coherence harmonic"
                    )
                    
                # Update sound harmonics data
                self.sound_harmonics = self.field_sound.get_harmonic_data()
            except Exception as e:
                logger.warning(f"Could not update sound properties: {str(e)}")
    
    def visualize_field_slice(self, z_slice: int = None) -> plt.Figure:
        """
        Visualize a 2D slice of the 3D energy field.
        
        Args:
            z_slice: The z-index for the slice. If None, uses the middle of the field.
            
        Returns:
            matplotlib Figure object
        """
        if z_slice is None:
            z_slice = self.dimensions[2] // 2
            
        fig, ax = plt.subplots(figsize=(10, 8))
        slice_data = self.energy_field[:, :, z_slice]
        
        im = ax.imshow(slice_data, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax, label='Energy Potential')
        
        # Mark resonance nodes that appear in this slice
        nodes_in_slice = [node for node in self.resonance_nodes 
                          if node.position[2] == z_slice]
        
        if nodes_in_slice:
            node_x = [node.position[0] for node in nodes_in_slice]
            node_y = [node.position[1] for node in nodes_in_slice]
            node_s = [node.strength * 100 for node in nodes_in_slice]
            
            ax.scatter(node_y, node_x, s=node_s, c='red', marker='o', 
                      edgecolor='white', alpha=0.7)
        
        ax.set_title(f'Energy Field Slice (z={z_slice})')
        ax.set_xlabel('Y Dimension')
        ax.set_ylabel('X Dimension')
        
        # Add field metrics as text
        sound_info = f"Sound Harmonics: {len(self.sound_harmonics)}" if self.sound_enabled else "Sound: Disabled"
        
        info_text = (
            f"Field: {self.name}\n"
            f"Base Frequency: {self.base_frequency:.4f}\n"
            f"Stability: {self.stability:.4f}\n"
            f"Coherence: {self.coherence:.4f}\n"
            f"Harmonics: {len(self.harmonics)}\n"
            f"Resonance Nodes: {len(self.resonance_nodes)}\n"
            f"{sound_info}"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        return fig
        
    def apply_potential_shift(self, center: Tuple[int, int, int], magnitude: float, 
                             radius: float, shape: str = 'gaussian') -> None:
        """
        Apply a potential shift to the field, creating a localized change in energy.
        
        Args:
            center: The center point (x, y, z) of the shift
            magnitude: The strength of the shift (positive or negative)
            radius: The effective radius of the shift
            shape: The shape of the potential ('gaussian', 'linear', 'quadratic')
        """
        # Create meshgrid for 3D coordinates
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        z = np.arange(self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center
        r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # Create potential based on shape
        if shape == 'gaussian':
            potential = magnitude * np.exp(-(r**2) / (2 * radius**2))
        elif shape == 'linear':
            potential = magnitude * np.maximum(0, 1 - r / radius)
        elif shape == 'quadratic':
            potential = magnitude * np.maximum(0, 1 - (r / radius)**2)
        else:
            raise ValueError(f"Unknown potential shape: {shape}")
        
        # Apply potential to energy field
        self.energy_field += potential
        
        # Normalize field after modification
        self.energy_field = self.energy_field / np.max(np.abs(self.energy_field))
        
        # Update field metrics
        self.update_field_metrics()
        
        # Create sound effect for potential shift if enabled
        if self.sound_enabled and self.sound_system is not None:
            try:
                # Create potential shift sound
                shift_sound = self.sound_system.create_sound(
                    name=f"Potential Shift {self.name}",
                    fundamental_frequency=self.base_frequency * (1.0 + 0.1 * magnitude),
                    sound_type=SoundType.TRANSITIONAL,
                    description=f"Potential shift sound for {self.name} field"
                )
                
                # Add special harmonics for the shift
                shift_sound.add_harmonic(
                    frequency=self.base_frequency * (1.0 + 0.2 * magnitude),
                    amplitude=0.8,
                    description="Shift primary harmonic"
                )
                
                shift_sound.add_harmonic(
                    frequency=self.base_frequency * GOLDEN_RATIO * (1.0 + 0.05 * magnitude),
                    amplitude=0.6,
                    description="Shift phi harmonic"
                )
            except Exception as e:
                logger.warning(f"Could not create potential shift sound: {str(e)}")
    
    def apply_sound_to_field(self, sound_obj=None, intensity=1.0):
        """
        Apply a sound's frequency patterns to influence the field energy.
        
        This allows sounds to directly impact the field structure.
        
        Args:
            sound_obj: The Sound object to apply (uses field_sound if None)
            intensity: The intensity of the sound's influence (0-1)
            
        Returns:
            bool: True if sound was successfully applied
        """
        if not self.sound_enabled:
            logger.warning("Sound is not enabled for this field")
            return False
            
        # Use field's primary sound if none provided
        if sound_obj is None:
            if self.field_sound is None:
                logger.warning("No sound object available to apply")
                return False
            sound_obj = self.field_sound
            
        try:
            # Extract harmonics from the sound
            harmonics = sound_obj.harmonics
            
            # Apply each harmonic to the field
            for harmonic in harmonics:
                # Convert to field harmonic with scaled amplitude
                field_harmonic = Harmonic(
                    frequency=harmonic.frequency,
                    amplitude=harmonic.amplitude * intensity,
                    phase=harmonic.phase
                )
                
                # Apply to field without adding to field's harmonic list
                self._apply_harmonic_to_field(field_harmonic)
                
            # Update field metrics
            self.update_field_metrics()
            
            logger.info(f"Applied sound '{sound_obj.name}' to field with {len(harmonics)} harmonics")
            return True
            
        except Exception as e:
            logger.error(f"Error applying sound to field: {str(e)}")
            return False
    
    def generate_field_soundscape(self, duration=5.0, sample_rate=44100):
        """
        Generate an audio representation of the field's current state.
        
        This creates a soundscape that represents the field's energy distribution,
        harmonics, and resonance patterns.
        
        Args:
            duration: Duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            tuple: (time_array, waveform_array) or None if sound not enabled
        """
        if not self.sound_enabled or self.field_sound is None:
            logger.warning("Sound is not enabled or no field sound available")
            return None
            
        try:
            # Get base soundscape from field sound
            time_array, base_waveform = self.field_sound.generate_waveform(
                duration=duration,
                sample_rate=sample_rate
            )
            
            # Add resonance node contributions if available
            if self.resonance_nodes:
                resonance_waveform = np.zeros_like(base_waveform)
                
                # Add the first few resonance nodes (limit to 5 to avoid overcomplexity)
                for i, node in enumerate(self.resonance_nodes[:5]):
                    # Create simple sine wave for this node
                    node_amplitude = node.strength * 0.3  # Scale down to avoid overwhelming
                    node_frequency = node.frequency
                    node_wave = node_amplitude * np.sin(2 * np.pi * node_frequency * time_array)
                    
                    # Add to resonance waveform
                    resonance_waveform += node_wave
                
                # Normalize resonance waveform
                if np.max(np.abs(resonance_waveform)) > 0:
                    resonance_waveform = resonance_waveform / np.max(np.abs(resonance_waveform))
                    
                # Mix with base waveform (70% base, 30% resonance)
                combined_waveform = 0.7 * base_waveform + 0.3 * resonance_waveform
                
                # Final normalization
                if np.max(np.abs(combined_waveform)) > 1.0:
                    combined_waveform = combined_waveform / np.max(np.abs(combined_waveform))
                    
                return time_array, combined_waveform
            else:
                return time_array, base_waveform
                
        except Exception as e:
            logger.error(f"Error generating field soundscape: {str(e)}")
            return None
    
    def save_field_state(self, filename: str) -> None:
        """Save the current field state to a file."""
        field_data = {
            'name': self.name,
            'dimensions': self.dimensions,
            'base_frequency': self.base_frequency,
            'energy_field': self.energy_field,
            'stability': self.stability,
            'coherence': self.coherence,
            'chaos_ratio': self.chaos_ratio,
            'harmonics': [(h.frequency, h.amplitude, h.phase) for h in self.harmonics],
            'resonance_nodes': [(n.position, n.frequency, n.strength) for n in self.resonance_nodes],
            'sound_enabled': self.sound_enabled,
            'sound_harmonics': self.sound_harmonics
        }
        np.savez_compressed(filename, **field_data)
        
    @classmethod
    def load_field_state(cls, filename: str) -> 'Field':
        """Load a field state from a file."""
        data = np.load(filename, allow_pickle=True)
        
        # Create field instance
        field = cls(
            name=str(data['name']),
            dimensions=tuple(data['dimensions']),
            base_frequency=float(data['base_frequency'])
        )
        
        # Restore field properties
        field.energy_field = data['energy_field']
        field.stability = float(data['stability'])
        field.coherence = float(data['coherence'])
        field.chaos_ratio = float(data['chaos_ratio'])
        
        # Restore harmonics
        field.harmonics = []
        for freq, amp, phase in data['harmonics']:
            field.harmonics.append(Harmonic(
frequency=float(freq),
                amplitude=float(amp),
                phase=float(phase)
            ))
        
        # Restore resonance nodes
        field.resonance_nodes = []
        for pos, freq, strength in data['resonance_nodes']:
            field.resonance_nodes.append(ResonanceNode(
                position=tuple(pos),
                frequency=float(freq),
                strength=float(strength)
            ))
        
        # Restore sound properties if available
        if 'sound_enabled' in data:
            field.sound_enabled = bool(data['sound_enabled'])
            
            if field.sound_enabled:
                # Reinitialize sound system
                field._initialize_sound_system()
                
                # Restore sound harmonics if available
                if 'sound_harmonics' in data and field.field_sound is not None:
                    field.sound_harmonics = data['sound_harmonics']
                    
                    # Recreate harmonics in field sound
                    for harmonic_data in field.sound_harmonics:
                        if isinstance(harmonic_data, dict) and 'frequency' in harmonic_data:
                            try:
                                field.field_sound.add_harmonic(
                                    frequency=float(harmonic_data['frequency']),
                                    amplitude=float(harmonic_data['amplitude']),
                                    phase=float(harmonic_data['phase']),
                                    description=harmonic_data.get('description', '')
                                )
                            except Exception as e:
                                logger.warning(f"Could not restore sound harmonic: {str(e)}")
            
        return field


# Field factory functions for specific dimensional spaces
def create_void_field(dimensions: Tuple[int, int, int] = (100, 100, 100), 
                      base_frequency: float = 432.0) -> Field:
    """
    Create a void field with specific properties suitable for soul spark formation.
    
    Args:
        dimensions: The dimensions of the field as (x, y, z)
        base_frequency: The base frequency of the field in Hz
        
    Returns:
        A Field object representing the void
    """
    void_field = Field(
        name="Void",
        dimensions=dimensions,
        base_frequency=base_frequency
    )
    
    # Add fundamental harmonics based on Schumann resonances and cosmic frequencies
    void_field.add_harmonic(frequency=7.83, amplitude=0.5, phase=0.0)  # Earth's Schumann resonance
    void_field.add_harmonic(frequency=base_frequency/2, amplitude=0.3, phase=np.pi/4)
    void_field.add_harmonic(frequency=base_frequency*GOLDEN_RATIO, amplitude=0.4, phase=np.pi/3)
    
    # Create a field at the edge of chaos for optimal emergence conditions
    void_field.chaos_ratio = CHAOS_ORDER_RATIO
    void_field.update_field_metrics()
    
    # Add void-specific sounds if sound system is enabled
    if void_field.sound_enabled and void_field.sound_system is not None:
        try:
            # Create void-specific sound with primordial qualities
            void_sound = void_field.sound_system.create_sound(
                name="Void Primordial",
                fundamental_frequency=base_frequency * CHAOS_ORDER_RATIO,
                sound_type=SoundType.DIMENSIONAL,
                description="Primordial void sound at the edge of chaos"
            )
            
            # Add natural harmonics with void-specific properties
            void_sound.add_natural_harmonics(count=7, amplitude_falloff=0.6)
            
            # Add phi harmonics for sacred geometry resonance
            void_sound.add_phi_harmonics(count=5, amplitude_falloff=0.7)
            
            # Add special void harmonic at the edge of chaos frequency
            void_sound.add_harmonic(
                frequency=base_frequency * CHAOS_ORDER_RATIO,
                amplitude=0.9,
                phase=np.pi/4,
                description="Edge of chaos void harmonic"
            )
            
            # Apply void sound to field
            void_field.apply_sound_to_field(void_sound, intensity=0.8)
            
        except Exception as e:
            logger.warning(f"Could not create void-specific sounds: {str(e)}")
    
    return void_field


def create_sephiroth_field(name: str, dimensions: Tuple[int, int, int] = (100, 100, 100),
                          base_frequency: float = 432.0) -> Field:
    """
    Create a Sephiroth field with specific properties.
    
    Args:
        name: The name of the Sephiroth (e.g., "Kether", "Binah")
        dimensions: The dimensions of the field as (x, y, z)
        base_frequency: The base frequency of the field in Hz
        
    Returns:
        A Field object representing the Sephiroth dimension
    """
    # Each Sephiroth has specific frequency modifications based on its nature
    frequency_modifiers = {
        "Kether": 1.0,       # Crown - highest frequency
        "Chokmah": 0.95,     # Wisdom
        "Binah": 0.9,        # Understanding
        "Chesed": 0.85,      # Mercy
        "Geburah": 0.8,      # Severity
        "Tiphareth": 0.75,   # Beauty
        "Netzach": 0.7,      # Victory
        "Hod": 0.65,         # Glory
        "Yesod": 0.6,        # Foundation
        "Malkuth": 0.55      # Kingdom - lowest frequency
    }
    
    if name not in frequency_modifiers:
        raise ValueError(f"Unknown Sephiroth name: {name}")
    
    modified_frequency = base_frequency * frequency_modifiers[name]
    
    sephiroth_field = Field(
        name=name,
        dimensions=dimensions,
        base_frequency=modified_frequency
    )
    
    # Add harmonics specific to this Sephiroth
    # The harmonics are related to the position in the Tree of Life
    sephiroth_field.add_harmonic(frequency=modified_frequency, amplitude=0.7, phase=0.0)
    sephiroth_field.add_harmonic(frequency=modified_frequency*GOLDEN_RATIO, amplitude=0.5, phase=np.pi/4)
    
    # Higher Sephiroth have more order, lower have more chaos
    order_level = frequency_modifiers[name]
    sephiroth_field.chaos_ratio = CHAOS_ORDER_RATIO * (1.5 - order_level)
    sephiroth_field.update_field_metrics()
    
    # Add Sephiroth-specific sounds if sound system is enabled
    if sephiroth_field.sound_enabled and sephiroth_field.sound_system is not None:
        try:
            # Create Sephiroth-specific sound
            sephiroth_sound = sephiroth_field.sound_system.create_sephiroth_sound(
                sephiroth_name=name.lower()
            )
            
            # Higher Sephiroth get more phi harmonics
            if name in ["Kether", "Chokmah", "Binah"]:
                sephiroth_sound.add_phi_harmonics(count=7, amplitude_falloff=0.8)
                
            # Apply Sephiroth sound to field
            sephiroth_field.apply_sound_to_field(sephiroth_sound, intensity=0.9)
            
        except Exception as e:
            logger.warning(f"Could not create Sephiroth-specific sounds: {str(e)}")
    
    return sephiroth_field


def create_earth_field(dimensions: Tuple[int, int, int] = (100, 100, 100),
                      base_frequency: float = 7.83) -> Field:
    """
    Create an Earth field with specific properties related to Earth frequencies.
    
    Args:
        dimensions: The dimensions of the field as (x, y, z)
        base_frequency: The base frequency of the field in Hz (default: Schumann resonance)
        
    Returns:
        A Field object representing the Earth dimension
    """
    earth_field = Field(
        name="Earth",
        dimensions=dimensions,
        base_frequency=base_frequency  # Schumann resonance
    )
    
    # Add Earth's fundamental harmonics
    earth_field.add_harmonic(frequency=base_frequency, amplitude=0.8, phase=0.0)  # Schumann
    earth_field.add_harmonic(frequency=14.3, amplitude=0.5, phase=np.pi/6)  # Second Schumann
    earth_field.add_harmonic(frequency=20.8, amplitude=0.3, phase=np.pi/4)  # Third Schumann
    
    # Add diurnal cycle (24 hours) as a harmonic
    # Converted to Hz (1/86400 Hz for daily cycle)
    diurnal_freq = 1.0 / 86400.0
    earth_field.add_harmonic(frequency=diurnal_freq, amplitude=0.6, phase=0.0)
    
    # Add lunar cycle (29.5 days) as a harmonic
    # Converted to Hz (1/(29.5*86400) Hz for monthly cycle)
    lunar_freq = 1.0 / (29.5 * 86400.0)
    earth_field.add_harmonic(frequency=lunar_freq, amplitude=0.4, phase=np.pi/3)
    
    # Earth has a balance of chaos and order
    earth_field.chaos_ratio = CHAOS_ORDER_RATIO * 1.2
    earth_field.update_field_metrics()
    
    # Add Earth-specific sounds if sound system is enabled
    if earth_field.sound_enabled and earth_field.sound_system is not None:
        try:
            # Create Earth-specific sound with natural frequencies
            earth_sound = earth_field.sound_system.create_sound(
                name="Earth Resonance",
                fundamental_frequency=base_frequency,
                sound_type=SoundType.DIMENSIONAL,
                description="Earth's natural resonance frequencies"
            )
            
            # Add Schumann resonance harmonics
            earth_sound.add_harmonic(frequency=base_frequency, amplitude=1.0, description="First Schumann")
            earth_sound.add_harmonic(frequency=14.3, amplitude=0.7, description="Second Schumann")
            earth_sound.add_harmonic(frequency=20.8, amplitude=0.5, description="Third Schumann")
            earth_sound.add_harmonic(frequency=27.3, amplitude=0.3, description="Fourth Schumann")
            earth_sound.add_harmonic(frequency=33.8, amplitude=0.2, description="Fifth Schumann")
            
            # Add natural cycles
            earth_sound.add_harmonic(frequency=diurnal_freq, amplitude=0.8, description="Diurnal cycle")
            earth_sound.add_harmonic(frequency=lunar_freq, amplitude=0.6, description="Lunar cycle")
            earth_sound.add_harmonic(frequency=1.0/31536000.0, amplitude=0.5, description="Annual cycle")
            
            # Apply Earth sound to field
            earth_field.apply_sound_to_field(earth_sound, intensity=0.9)
            
        except Exception as e:
            logger.warning(f"Could not create Earth-specific sounds: {str(e)}")
    
    return earth_field


def create_guff_field(dimensions: Tuple[int, int, int] = (100, 100, 100),
                     base_frequency: float = 528.0) -> Field:
    """
    Create a Guff field with specific properties for soul harmonization.
    
    Args:
        dimensions: The dimensions of the field as (x, y, z)
        base_frequency: The base frequency of the field in Hz (default: 528Hz - "miracle tone")
        
    Returns:
        A Field object representing the Guff dimension
    """
    guff_field = Field(
        name="Guff",
        dimensions=dimensions,
        base_frequency=base_frequency  # "Miracle frequency" for healing
    )
    
    # Add harmonics based on Fibonacci sequence and golden ratio
    guff_field.add_harmonic(frequency=base_frequency, amplitude=0.9, phase=0.0)
    
    # Add Fibonacci-based harmonics
    fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for i, fib in enumerate(fibonacci[2:8]):  # Use middle Fibonacci numbers
        freq = base_frequency * (fib / 89.0)  # Normalize by a larger Fibonacci number
        guff_field.add_harmonic(
            frequency=freq, 
            amplitude=0.7 * (1.0 - i * 0.1),  # Decreasing amplitude
            phase=i * np.pi / 8
        )
    
    # Guff has very high order for stabilization
    guff_field.chaos_ratio = CHAOS_ORDER_RATIO * 0.7
    guff_field.update_field_metrics()
    
    # Add Guff-specific sounds if sound system is enabled
    if guff_field.sound_enabled and guff_field.sound_system is not None:
        try:
            # Create Guff-specific sound with harmonizing frequencies
            guff_sound = guff_field.sound_system.create_sound(
                name="Guff Harmonization",
                fundamental_frequency=base_frequency,
                sound_type=SoundType.DIMENSIONAL,
                description="Harmonizing frequencies for soul formation"
            )
            
            # Add solfeggio frequency harmonics
            guff_sound.add_harmonic(frequency=396.0, amplitude=0.7, description="UT - Liberating guilt")
            guff_sound.add_harmonic(frequency=417.0, amplitude=0.7, description="RE - Undoing situations")
            guff_sound.add_harmonic(frequency=528.0, amplitude=1.0, description="MI - Transformation")
            guff_sound.add_harmonic(frequency=639.0, amplitude=0.8, description="FA - Connecting")
            guff_sound.add_harmonic(frequency=741.0, amplitude=0.7, description="SOL - Expression")
            guff_sound.add_harmonic(frequency=852.0, amplitude=0.6, description="LA - Intuition")
            guff_sound.add_harmonic(frequency=963.0, amplitude=0.5, description="SI - Spiritual order")
            
            # Add Fibonacci harmonics
            for i, fib in enumerate(fibonacci[2:7]):
                freq = base_frequency * (fib / fibonacci[6])
                guff_sound.add_harmonic(
                    frequency=freq,
                    amplitude=0.8 - (i * 0.1),
                    description=f"Fibonacci {fib} harmonic"
                )
            
            # Apply Guff sound to field
            guff_field.apply_sound_to_field(guff_sound, intensity=0.9)
            
        except Exception as e:
            logger.warning(f"Could not create Guff-specific sounds: {str(e)}")
    
    return guff_field


# Extended FieldSystem class implementing all base functionality
class FieldSystem:
    """
    Base class for all field implementations.
    
    This provides common functionality for field creation, manipulation,
    and evolution across different dimensional spaces.
    """
    
    def __init__(self, dimensions=(64, 64, 64), field_name="Generic Field", 
                edge_of_chaos_ratio=CHAOS_ORDER_RATIO, base_frequency=CREATOR_FREQUENCY):
        """
        Initialize a new field system.
        
        Args:
            dimensions (tuple): 3D dimensions of the field (x, y, z)
            field_name (str): Name identifier for the field
            edge_of_chaos_ratio (float): The edge of chaos parameter
            base_frequency (float): Base frequency for the field in Hz
        """
        self.field_id = str(uuid.uuid4())
        self.dimensions = dimensions
        self.field_name = field_name
        self.edge_of_chaos_ratio = edge_of_chaos_ratio
        self.base_frequency = base_frequency
        
        # Create energy potential field
        self.energy_potential = np.zeros(dimensions, dtype=np.float64)
        
        # Quantum state representation (complex values)
        self.quantum_state = np.zeros(dimensions, dtype=np.complex128)
        
        # Wave function (normalized quantum state)
        self.wave_function = np.zeros(dimensions, dtype=np.complex128)
        
        # Resonance matrix for frequency effects
        self.resonance_matrix = np.zeros(dimensions, dtype=np.float64)
        
        # Resonance frequencies
        self.resonance_frequencies = []
        
        # Field metrics
        self.stability = 0.0
        self.coherence = 0.0
        self.resonance_quality = 0.0
        
        # Sound system integration
        self._initialize_sound_system()
        
        logger.info(f"Initialized field system: {field_name} with ID {self.field_id}")
        logger.info(f"Dimensions: {dimensions}, Base frequency: {base_frequency} Hz")
    
    def _initialize_sound_system(self):
        """Initialize sound system for this field."""
        try:
            if SoundSystem is not None:
                # Create sound system with base frequency
                self.sound_system = SoundSystem(creator_frequency=self.base_frequency)
                
                # Create field sound
                self.field_sound = self.sound_system.create_sound(
                    name=f"{self.field_name} Base Sound",
                    fundamental_frequency=self.base_frequency,
                    sound_type=SoundType.DIMENSIONAL,
                    description=f"Base dimensional sound for {self.field_name}"
                )
                
                # Add basic harmonics
                self.field_sound.add_natural_harmonics(count=3)
                
                self.sound_enabled = True
                logger.info(f"Sound system initialized for {self.field_name}")
            else:
                self.sound_system = None
                self.field_sound = None
                self.sound_enabled = False
                logger.info(f"Sound system not available for {self.field_name}")
        except Exception as e:
            logger.error(f"Error initializing sound system: {str(e)}")
            self.sound_system = None
            self.field_sound = None
            self.sound_enabled = False
    
    def initialize_quantum_field(self, base_amplitude=0.01):
        """
        Initialize the quantum field with random fluctuations.
        
        Args:
            base_amplitude (float): Base amplitude for quantum fluctuations
        """
        # Create random quantum state
        real_part = np.random.normal(0, base_amplitude, self.dimensions)
        imag_part = np.random.normal(0, base_amplitude, self.dimensions)
        
        self.quantum_state = real_part + 1j * imag_part
        
        # Normalize to create wave function
        self.wave_function = self.normalize_wave_function(self.quantum_state)
        
        # Initialize energy potential from wave function
        self.energy_potential = np.abs(self.wave_function) ** 2
        
        # Apply edge of chaos adjustment
        self._apply_edge_of_chaos()
        
        # Initialize resonance matrix
        self.resonance_matrix = np.ones(self.dimensions) * 0.1
        
        logger.info(f"Initialized quantum field for {self.field_name}")
    
    def normalize_wave_function(self, wave_function):
        """
        Normalize a wave function to ensure it has unit probability.
        
        Args:
            wave_function: Complex ndarray representing quantum state
            
        Returns:
            Normalized wave function
        """
        # Calculate total probability
        probability = np.sum(np.abs(wave_function) ** 2)
        
        # Avoid division by zero
        if probability > 0:
            # Normalize
            normalized = wave_function / np.sqrt(probability)
            return normalized
        else:
            # If zero probability, return original
            return wave_function
    
    def _apply_edge_of_chaos(self):
        """
        Apply edge of chaos adjustments to the quantum field.
        
        This creates a field that balances between order and chaos
        to enable emergence of complex patterns.
        """
        # Apply smoothing to create order
        order_component = signal.gaussian_filter(self.energy_potential, 
                                              sigma=[d//10 for d in self.dimensions])
        
        # Create chaos component
        chaos_component = np.random.normal(0, 0.1, self.dimensions)
        
        # Combine using edge of chaos ratio
        # Higher ratio means more chaos, lower means more order
        self.energy_potential = ((1 - self.edge_of_chaos_ratio) * order_component + 
                                self.edge_of_chaos_ratio * chaos_component)
        
        # Normalize energy potential
        max_energy = np.max(np.abs(self.energy_potential))
        if max_energy > 0:
            self.energy_potential = self.energy_potential / max_energy
    
    def add_resonance_frequency(self, frequency, amplitude=1.0, phase=0.0, is_harmonic=False):
        """
        Add a resonance frequency to the field.
        
        Args:
            frequency (float): Frequency in Hz
            amplitude (float): Amplitude factor (0-1)
            phase (float): Phase angle in radians
            is_harmonic (bool): Whether this is a harmonic of the base frequency
        """
        # Add to resonance frequencies list
        self.resonance_frequencies.append({
            'frequency': frequency,
            'amplitude': amplitude,
            'phase': phase,
            'is_harmonic': is_harmonic
        })
        
        # Update resonance matrix
        self._update_resonance_matrix()
        
        # Update sound system if enabled
        if self.sound_enabled and self.field_sound is not None:
            try:
                # Add to field sound
                self.field_sound.add_harmonic(
                    frequency=frequency,
                    amplitude=amplitude,
                    phase=phase,
                    description=f"{'Harmonic' if is_harmonic else 'Primary'} resonance frequency"
                )
            except Exception as e:
                logger.warning(f"Could not add frequency to sound: {str(e)}")
        
        logger.info(f"Added resonance frequency {frequency} Hz to field {self.field_name}")
    
    def _update_resonance_matrix(self):
        """
        Update the resonance matrix based on current frequencies.
        
        This creates a 3D resonance pattern throughout the field.
        """
        # Reset resonance matrix
        self.resonance_matrix = np.zeros(self.dimensions, dtype=np.float64)
        
        # Create coordinate grid
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        z = np.arange(self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate center
        center = np.array([d // 2 for d in self.dimensions])
        
        # Add contribution from each frequency
        for freq_data in self.resonance_frequencies:
            frequency = freq_data['frequency']
            amplitude = freq_data['amplitude']
            phase = freq_data['phase']
            
            # Calculate distance from center
            distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            
            # Create standing wave pattern
            # Use distance as wavelength factor
            pattern = amplitude * np.cos(2 * np.pi * frequency * distance / 100 + phase)
            
            # Add to resonance matrix
            self.resonance_matrix += pattern
        
        # Normalize resonance matrix
        max_resonance = np.max(np.abs(self.resonance_matrix))
        if max_resonance > 0:
            self.resonance_matrix = self.resonance_matrix / max_resonance
    
    def apply_resonance_to_field(self):
        """
        Apply resonance effects to the energy field.
        
        This modulates the energy potential based on resonance patterns.
        """
        # Calculate resonance influence
        resonance_factor = 0.3  # Strength of resonance influence
        
        # Apply resonance modulation to energy potential
        modulated_energy = self.energy_potential * (1 + resonance_factor * self.resonance_matrix)
        
        # Normalize
        max_energy = np.max(modulated_energy)
        if max_energy > 0:
            modulated_energy = modulated_energy / max_energy
            
        self.energy_potential = modulated_energy
        
        # Update quantum state to reflect new energy potential
        # This preserves phase but adjusts amplitude
        amplitude = np.sqrt(self.energy_potential)
        phase = np.angle(self.quantum_state)
        self.quantum_state = amplitude * np.exp(1j * phase)
        
        # Re-normalize wave function
        self.wave_function = self.normalize_wave_function(self.quantum_state)
        
        logger.info(f"Applied resonance to field {self.field_name}")
    
    def embed_pattern(self, pattern_name, pattern_array, position=None, strength=1.0):
        """
        Embed a pattern into the energy field.
        
        Args:
            pattern_name (str): Name identifier for pattern
            pattern_array (ndarray): The pattern to embed
            position (tuple): Center position (default: field center)
            strength (float): Strength factor for embedding
            
        Returns:
            bool: True if successful
        """
        try:
            # Default to center if position not specified
            if position is None:
                position = tuple(d // 2 for d in self.dimensions)
                
            # Check pattern dimensions
            if pattern_array.shape != self.dimensions:
                logger.warning(f"Pattern dimensions {pattern_array.shape} don't match field {self.dimensions}")
                logger.warning(f"Pattern will be applied at center with potential cropping/padding")
                
                # Create a matching array
                embedded_pattern = np.zeros(self.dimensions)
                
                # Calculate placement bounds
                x_start = position[0] - pattern_array.shape[0] // 2
                y_start = position[1] - pattern_array.shape[1] // 2
                z_start = position[2] - pattern_array.shape[2] // 2
                
                # Ensure within bounds
                x_start = max(0, min(self.dimensions[0] - 1, x_start))
                y_start = max(0, min(self.dimensions[1] - 1, y_start))
                z_start = max(0, min(self.dimensions[2] - 1, z_start))
                
                # Calculate end positions and dimensions to copy
                x_end = min(self.dimensions[0], x_start + pattern_array.shape[0])
                y_end = min(self.dimensions[1], y_start + pattern_array.shape[1])
                z_end = min(self.dimensions[2], z_start + pattern_array.shape[2])
                
                x_dim = x_end - x_start
                y_dim = y_end - y_start
                z_dim = z_end - z_start
                
                # Copy pattern portion
                embedded_pattern[x_start:x_end, y_start:y_end, z_start:z_end] = (
                    pattern_array[:x_dim, :y_dim, :z_dim]
                )
                
                pattern_array = embedded_pattern
            
            # Scale pattern
            scaled_pattern = pattern_array * strength
            
            # Combine with energy potential
            self.energy_potential = self.energy_potential * 0.7 + scaled_pattern * 0.3
            
            # Normalize
            max_energy = np.max(self.energy_potential)
            if max_energy > 0:
                self.energy_potential = self.energy_potential / max_energy
                
            # Update quantum state to match
            amplitude = np.sqrt(self.energy_potential)
            phase = np.angle(self.quantum_state)
            self.quantum_state = amplitude * np.exp(1j * phase)
            
            # Re-normalize wave function
            self.wave_function = self.normalize_wave_function(self.quantum_state)
            
            # Apply sound pattern if enabled
            if self.sound_enabled and self.field_sound is not None:
                try:
                    # Create pattern sound
                    pattern_sound = self.sound_system.create_sound(
                        name=f"Pattern {pattern_name}",
                        fundamental_frequency=self.base_frequency * GOLDEN_RATIO,
                        sound_type=SoundType.COMPOSITE,
                        description=f"Sound for {pattern_name} pattern"
                    )
                    
                    # Add harmonics based on pattern
                    pattern_sound.add_natural_harmonics(count=3)
                    pattern_sound.add_phi_harmonics(count=3)
                    
                    # Apply pattern sound to field
                    self.apply_sound_to_field(pattern_sound, intensity=strength * 0.7)
                    
                except Exception as e:
                    logger.warning(f"Could not create pattern sound: {str(e)}")
            
            logger.info(f"Embedded pattern {pattern_name} into field {self.field_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding pattern {pattern_name}: {str(e)}")
            return False
    
    def evolve_wave_function(self, time_step=0.01, iterations=10):
        """
        Evolve the quantum wave function over time.
        
        This simulates the time evolution of the field according to
        quantum mechanical principles.
        
        Args:
            time_step (float): Time step for evolution
            iterations (int): Number of evolution steps
        """
        for i in range(iterations):
            # Simplified Schrödinger equation evolution
            # This is a basic approximation that creates wave-like behavior
            
            # Calculate Laplacian (∇²ψ) using finite difference
            laplacian = np.zeros(self.dimensions, dtype=np.complex128)
            
            # Calculate second derivatives
            for axis in range(3):
                # Central difference
                laplacian += np.gradient(np.gradient(self.wave_function, axis=axis), axis=axis)
            
            # Apply evolution operator
            # i * ħ * ∂ψ/∂t = -ħ²/(2m) * ∇²ψ + V * ψ
            # Where we set ħ = 1, m = 1 for simplicity
            evolution = -0.5j * laplacian + 1j * self.energy_potential * self.wave_function
            
            # Update wave function
            self.wave_function = self.wave_function + time_step * evolution
            
            # Normalize
            self.wave_function = self.normalize_wave_function(self.wave_function)
            
            # Update quantum state
            self.quantum_state = self.wave_function.copy()
            
            # Update energy potential
            new_energy = 0.9 * self.energy_potential + 0.1 * np.abs(self.wave_function) ** 2
            max_energy = np.max(new_energy)
            if max_energy > 0:
                self.energy_potential = new_energy / max_energy
        
        # Update resonance matrix after evolution
        self._update_resonance_matrix()
        
        # Generate evolution sound if enabled
        if self.sound_enabled and self.sound_system is not None and iterations > 5:
            try:
                # Create evolution sound
                evolution_sound = self.sound_system.create_sound(
                    name=f"Evolution {self.field_name}",
                    fundamental_frequency=self.base_frequency * (1.0 + 0.05 * np.random.random()),
                    sound_type=SoundType.TRANSITIONAL,
                    description=f"Evolution sound for {self.field_name}"
                )
                
                # Add harmonics specific to this evolution
                evolution_sound.add_harmonic(
                    frequency=self.base_frequency * GOLDEN_RATIO,
                    amplitude=0.7,
                    description="Evolution harmonic"
                )
                
                # Apply to field (with low intensity to avoid disrupting the evolution)
                self.apply_sound_to_field(evolution_sound, intensity=0.3)
                
            except Exception as e:
                logger.warning(f"Could not create evolution sound: {str(e)}")
        
        logger.info(f"Evolved wave function for {self.field_name}: {iterations} iterations")
    
    def find_energy_peaks(self, threshold=0.7, min_distance=5):
        """
        Find peaks in the energy potential field.
        
        Args:
            threshold (float): Minimum energy for peak detection
            min_distance (int): Minimum distance between peaks
            
        Returns:
            list: List of detected peaks and their properties
        """
        # Use peak_local_max to find local maxima
        peak_positions = signal.peak_local_max(
            self.energy_potential,
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False
        )
        
        # Create peak information
        peaks = []
        for position in peak_positions:
            # Extract energy value
            energy = self.energy_potential[tuple(position)]
            
            # Get resonance value
            resonance = self.resonance_matrix[tuple(position)]
            
            # Record peak
            peaks.append({
                'position': tuple(position),
                'energy': float(energy),
                'resonance': float(resonance)
            })
            
        # Sort by energy (highest first)
        peaks.sort(key=lambda x: x['energy'], reverse=True)
        
        # Generate peak sounds if enabled
        if self.sound_enabled and self.sound_system is not None and len(peaks) > 0:
            try:
                # Create peak detection sound
                peak_sound = self.sound_system.create_sound(
                    name=f"Peaks {self.field_name}",
                    fundamental_frequency=self.base_frequency * 1.2,
                    sound_type=SoundType.RESONANT,
                    description=f"Energy peak sound for {self.field_name}"
                )
                
                # Add harmonics based on highest peaks
                for i, peak in enumerate(peaks[:3]):  # Use top 3 peaks
                    peak_freq = self.base_frequency * (1.0 + 0.2 * peak['energy'])
                    peak_sound.add_harmonic(
                        frequency=peak_freq,
                        amplitude=peak['energy'] * 0.8,
                        description=f"Peak {i+1} harmonic"
                    )
                    
                # Add phi harmonic
                peak_sound.add_harmonic(
                    ratio=GOLDEN_RATIO,
                    amplitude=0.7,
                    description="Peak phi harmonic"
                )
                
            except Exception as e:
                logger.warning(f"Could not create peak detection sound: {str(e)}")
        
        logger.info(f"Found {len(peaks)} energy peaks in field {self.field_name}")
        return peaks
    
    def calculate_stability_metrics(self):
        """
        Calculate various stability and coherence metrics for the field.
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        # Calculate gradient for stability
        gradient = np.gradient(self.energy_potential)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Stability is inverse of average gradient magnitude
        stability = 1.0 - np.mean(gradient_magnitude)
        
        # Calculate coherence using spatial autocorrelation
        flat_field = self.energy_potential.flatten() - np.mean(self.energy_potential)
        autocorr = np.correlate(flat_field, flat_field, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            
        # Coherence is the sum of autocorrelation values
        coherence = np.sum(autocorr[:min(100, len(autocorr))]) / min(100, len(autocorr))
        
        # Calculate resonance quality
        resonance_quality = np.mean(self.resonance_matrix)
        
        # Calculate phi-alignment (golden ratio harmony)
        # This measures how well the energy pattern follows phi proportions
        phi_alignment = self._calculate_phi_alignment()
        
        # Calculate the edge of chaos metric
        # This measures how close the field is to the ideal edge of chaos
        edge_of_chaos_metric = 1.0 - abs(self.edge_of_chaos_ratio - 
                                       CHAOS_ORDER_RATIO) / CHAOS_ORDER_RATIO
        
        # Calculate sound harmony if enabled
        if self.sound_enabled and self.field_sound is not None:
            try:
                # Count harmonics in phi ratios
                harmonics = self.field_sound.harmonics
                phi_harmonics = 0
                
                for i, h1 in enumerate(harmonics):
                    for h2 in harmonics[i+1:]:
                        # Check if ratio is close to phi
                        if h2.frequency > h1.frequency:
                            ratio = h2.frequency / h1.frequency
                        else:
                            ratio = h1.frequency / h2.frequency
                            
                        if abs(ratio - GOLDEN_RATIO) < 0.05:
                            phi_harmonics += 1
                            
                sound_harmony = phi_harmonics / max(1, len(harmonics))
            except Exception:
                sound_harmony = 0.5  # Default if calculation fails
        else:
            sound_harmony = 0.0
            
        # Update stored metrics
        self.stability = stability
        self.coherence = coherence
        self.resonance_quality = resonance_quality
        
        # Combine all metrics
        metrics = {
            'stability': stability,
            'coherence': coherence,
            'resonance_quality': resonance_quality,
            'phi_alignment': phi_alignment,
            'edge_of_chaos': edge_of_chaos_metric,
            'sound_harmony': sound_harmony
        }
        
        return metrics
    
    def _calculate_phi_alignment(self):
        """
        Calculate how well the field energy distribution follows golden ratio proportions.
        
        Returns:
            float: Phi alignment metric (0-1)
        """
        # Calculate energy distribution in concentric shells
        center = tuple(d // 2 for d in self.dimensions)
        
        # Create coordinate grid
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        z = np.arange(self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center
        distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # Maximum distance
        max_dist = np.sqrt(sum((d/2)**2 for d in self.dimensions))
        
        # Create shells based on phi ratios
        shell_radii = [0]
        radius = max_dist * 0.2  # Start at 20% of max radius
        
        # Create 5 shells with phi-based spacing
        for i in range(5):
            shell_radii.append(radius)
            radius *= GOLDEN_RATIO
            
        # Calculate energy in each shell
        shell_energies = []
        
        for i in range(len(shell_radii) - 1):
            inner_radius = shell_radii[i]
            outer_radius = shell_radii[i+1]
            
            shell_mask = (distance >= inner_radius) & (distance < outer_radius)
            shell_energy = np.sum(self.energy_potential[shell_mask])
            
            shell_energies.append(shell_energy)
            
        # Calculate ratios between adjacent shells
        ratios = []
        
        for i in range(len(shell_energies) - 1):
            if shell_energies[i+1] > 0:
                ratio = shell_energies[i] / shell_energies[i+1]
                ratios.append(ratio)
                
        # Calculate how close these ratios are to phi
        if ratios:
            phi_deviations = [abs(ratio - GOLDEN_RATIO) / GOLDEN_RATIO for ratio in ratios]
            phi_alignment = 1.0 - sum(phi_deviations) / len(phi_deviations)
        else:
            phi_alignment = 0.5  # Default if calculation fails
            
        return phi_alignment
    
    def visualize_field_slice(self, axis=2, index=None, show_peaks=True, save_path=None):
        """
        Visualize a 2D slice of the 3D field.
        
        Args:
            axis (int): Axis to slice (0=x, 1=y, 2=z)
            index (int): Slice index (default: middle of the axis)
            show_peaks (bool): Whether to mark energy peaks
            save_path (str): Optional path to save visualization
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Default to middle if index not specified
            if index is None:
                index = self.dimensions[axis] // 2
                
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Extract slice
            if axis == 0:
                slice_data = self.energy_potential[index, :, :]
                xlabel, ylabel = 'Y', 'Z'
            elif axis == 1:
                slice_data = self.energy_potential[:, index, :]
                xlabel, ylabel = 'X', 'Z'
            else:  # axis == 2
                slice_data = self.energy_potential[:, :, index]
                xlabel, ylabel = 'X', 'Y'
                
            # Plot slice
            plt.imshow(slice_data, cmap='viridis', origin='lower')
            plt.colorbar(label='Energy Potential')
            
            # Show peaks if requested
            if show_peaks:
                # Find peaks
                peaks = self.find_energy_peaks(threshold=0.5)
                
                # Filter peaks in this slice
                slice_peaks = []
                for peak in peaks:
                    pos = peak['position']
                    if pos[axis] == index:
                        if axis == 0:
                            # Z-Y plane
                            peak_x = pos[1]  # Y
                            peak_y = pos[2]  # Z
                        elif axis == 1:
                            # X-Z plane
                            peak_x = pos[0]  # X
                            peak_y = pos[2]  # Z
                        else:
                            # X-Y plane
                            peak_x = pos[0]  # X
                            peak_y = pos[1]  # Y
                            
                        slice_peaks.append((peak_x, peak_y, peak['energy']))
                
                # Plot peaks
                for peak_x, peak_y, energy in slice_peaks:
                    marker_size = energy * 100
                    plt.scatter(peak_x, peak_y, s=marker_size, c='red', 
                              edgecolor='white', alpha=0.7)
                    
                    # Add energy label
                    plt.annotate(f"{energy:.2f}", (peak_x, peak_y), 
                               xytext=(5, 5), textcoords='offset points')
            
            # Add field metrics
            metrics = self.calculate_stability_metrics()
            
            # Add sound info if enabled
            if self.sound_enabled and self.field_sound is not None:
                sound_info = f"Sound: {len(self.field_sound.harmonics)} harmonics"
            else:
                sound_info = "Sound: Disabled"
                
            metrics_text = (
                f"Field: {self.field_name}\n"
                f"Frequency: {self.base_frequency:.1f} Hz\n"
                f"Stability: {metrics['stability']:.3f}\n"
                f"Coherence: {metrics['coherence']:.3f}\n"
                f"Resonance: {metrics['resonance_quality']:.3f}\n"
                f"Phi Align: {metrics['phi_alignment']:.3f}\n"
                f"{sound_info}"
            )
            
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Set labels and title
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"{self.field_name} - {['X', 'Y', 'Z'][axis]}={index} Slice")
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Field visualization saved to {save_path}")
                
            # Show the plot (can be disabled if running in non-interactive mode)
            plt.tight_layout()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing field slice: {str(e)}")
            return False
    
    def visualize_sound_waveform(self, duration=3.0, save_path=None):
        """
        Visualize the sound waveform for this field.
        
        Args:
            duration (float): Duration in seconds
            save_path (str): Optional path to save visualization
            
        Returns:
            bool: True if visualization was successful
        """
        if not self.sound_enabled or self.field_sound is None:
            logger.warning("Sound is not enabled or no field sound available")
            return False
            
        try:
            # Generate waveform
            time_array, waveform = self.field_sound.generate_waveform(duration=duration)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot waveform
            plt.plot(time_array, waveform)
            
            # Set labels and title
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f"{self.field_name} Sound Waveform")
            
            # Add harmonics info
            harmonics_text = f"Base Frequency: {self.base_frequency:.1f} Hz\n"
            harmonics_text += f"Harmonics: {len(self.field_sound.harmonics)}\n\n"
            
            # Show top 5 harmonics
            harmonics_text += "Top Harmonics:\n"
            
            sorted_harmonics = sorted(
                self.field_sound.harmonics, 
                key=lambda h: h.amplitude, 
                reverse=True
            )
            
            for i, harmonic in enumerate(sorted_harmonics[:5]):
                harmonics_text += f"{harmonic.frequency:.1f} Hz @ {harmonic.amplitude:.2f}\n"
                
            plt.text(0.02, 0.98, harmonics_text, transform=plt.gca().transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Sound visualization saved to {save_path}")
                
            # Show the plot
            plt.tight_layout()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing sound waveform: {str(e)}")
            return False
    
    def save_field_data(self, output_dir="output", filename=None):
        """
        Save field data to file.
        
        Args:
            output_dir (str): Directory to save output files
            filename (str): Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"field_{self.field_name.lower().replace(' ', '_')}_{self.field_id[:8]}.npz"
            
        save_path = os.path.join(output_dir, filename)
        
        # Compile field data
        field_data = {
            'field_id': self.field_id,
            'field_name': self.field_name,
            'dimensions': self.dimensions,
            'base_frequency': self.base_frequency,
            'edge_of_chaos_ratio': self.edge_of_chaos_ratio,
            'energy_potential': self.energy_potential,
            'resonance_matrix': self.resonance_matrix,
            'stability': self.stability,
            'coherence': self.coherence,
            'resonance_quality': self.resonance_quality,
            'resonance_frequencies': self.resonance_frequencies,
            'sound_enabled': self.sound_enabled
        }
        
        # Save sound data if enabled
        if self.sound_enabled and self.field_sound is not None:
            try:
                # Save sound harmonics data
                field_data['sound_harmonics'] = self.field_sound.get_harmonic_data()
                
                # Also save sound to separate JSON file
                sound_filename = f"sound_{self.field_name.lower().replace(' ', '_')}_{self.field_id[:8]}.json"
                sound_path = os.path.join(output_dir, sound_filename)
                
                self.field_sound.save_sound_data(output_dir, sound_filename)
                field_data['sound_file'] = sound_filename
                
            except Exception as e:
                logger.warning(f"Could not save sound data: {str(e)}")
        
        # Save to file
        try:
            np.savez_compressed(save_path, **field_data)
            logger.info(f"Field data saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving field data: {str(e)}")
            return None
    
    @classmethod
    def load_field_data(cls, filepath):
        """
        Load field data from file.
        
        Args:
            filepath (str): Path to the saved field data file
            
        Returns:
            FieldSystem: Loaded field system
        """
        try:
            # Load data
            data = np.load(filepath, allow_pickle=True)
            
            # Create field system
            field = cls(
                dimensions=tuple(data['dimensions']),
                field_name=str(data['field_name']),
                edge_of_chaos_ratio=float(data['edge_of_chaos_ratio']),
                base_frequency=float(data['base_frequency'])
            )
            
            # Restore field ID
            field.field_id = str(data['field_id'])
            
            # Restore field data
            field.energy_potential = data['energy_potential']
            field.resonance_matrix = data['resonance_matrix']
            field.stability = float(data['stability'])
            field.coherence = float(data['coherence'])
            field.resonance_quality = float(data['resonance_quality'])
            
            # Restore resonance frequencies
            if 'resonance_frequencies' in data:
                field.resonance_frequencies = data['resonance_frequencies']
                
            # Restore sound if enabled
            if 'sound_enabled' in data and bool(data['sound_enabled']):
                field.sound_enabled = True
                
                # Initialize sound system
                field._initialize_sound_system()
                
                # Restore sound harmonics if available
                if 'sound_harmonics' in data and field.field_sound is not None:
                    harmonics_data = data['sound_harmonics']
                    
                    # Add harmonics to field sound
                    for harmonic_data in harmonics_data:
                        if isinstance(harmonic_data, dict) and 'frequency' in harmonic_data:
                            try:
                                field.field_sound.add_harmonic(
                                    frequency=float(harmonic_data['frequency']),
                                    amplitude=float(harmonic_data['amplitude']),
                                    phase=float(harmonic_data['phase']),
                                    description=harmonic_data.get('description', '')
                                )
                            except Exception as e:
                                logger.warning(f"Could not restore sound harmonic: {str(e)}")
            
            logger.info(f"Loaded field data from {filepath}")
            return field
            
        except Exception as e:
            logger.error(f"Error loading field data: {str(e)}")
            return None
    
    def __str__(self):
        """String representation of the field system."""
        metrics = self.calculate_stability_metrics()
        
        # Add sound info if enabled
        if self.sound_enabled and self.field_sound is not None:
            sound_info = f"Sound Harmonics: {len(self.field_sound.harmonics)}"
        else:
            sound_info = "Sound: Disabled"
            
        return (f"{self.field_name} (ID: {self.field_id[:8]})\n"
                f"Dimensions: {self.dimensions}\n"
                f"Base Frequency: {self.base_frequency:.2f} Hz\n"
                f"Edge of Chaos: {self.edge_of_chaos_ratio:.4f}\n"
                f"Stability: {metrics['stability']:.4f}\n"
                f"Coherence: {metrics['coherence']:.4f}\n"
                f"Resonance: {metrics['resonance_quality']:.4f}\n"
                f"Phi Alignment: {metrics['phi_alignment']:.4f}\n"
                f"{sound_info}")


# Example usage
if __name__ == "__main__":
    # Create and test a basic field
    field = FieldSystem(dimensions=(50, 50, 50), 
                       field_name="Test Field", 
                       base_frequency=432.0)
    
    # Initialize quantum field
    field.initialize_quantum_field()
    
    # Add some resonance frequencies
    field.add_resonance_frequency(432.0, amplitude=1.0)
    field.add_resonance_frequency(432.0 * GOLDEN_RATIO, amplitude=0.8)
    field.add_resonance_frequency(432.0 * 2, amplitude=0.6, is_harmonic=True)
    
    # Apply resonance
    field.apply_resonance_to_field()
    
    # Evolve the field
    field.evolve_wave_function(time_step=0.01, iterations=10)
    
    # Calculate metrics
    metrics = field.calculate_stability_metrics()
    print(f"Field metrics: {metrics}")
    
    # Visualize
    field.visualize_field_slice(show_peaks=True)
    
    # Visualize sound if enabled
    if field.sound_enabled:
        field.visualize_sound_waveform()
        
    plt.show()
    
    # Print field info
    print(field)
    
    # Test sound generation if enabled
    if field.sound_enabled and field.field_sound is not None:
        time_array, waveform = field.generate_field_soundscape(duration=3.0)
        print(f"Generated sound: {len(waveform)} samples")
        
        # Save field and sound data
        field.save_field_data()