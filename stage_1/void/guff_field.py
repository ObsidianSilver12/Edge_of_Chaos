"""
Guff Field Implementation

This module implements the Guff dimension - the realm where soul sparks are
strengthened and prepared for their journey through the Sephiroth.

The Guff represents a transitional dimension between the Void and the Sephiroth,
where souls receive their initial structure and establish resonance with the creator.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base field system
from field_system import FieldSystem

# Import sacred geometry patterns
from shared.flower_of_life import FlowerOfLife
from shared.merkaba import Merkaba
from shared.seed_of_life import SeedOfLife

# Import aspects
from stage_1.sephiroth.kether_aspects import KetherAspects

# Import sound system if available
try:
    from sounds.sound_system import SoundSystem, SoundType
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    logging.warning("Sound system not available. Guff field will function without sound capabilities.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='guff_field.log'
)
logger = logging.getLogger('guff_field')

class GuffField(FieldSystem):
    """
    Implementation of the Guff dimension.
    
    The Guff is a transitional dimension where soul sparks are strengthened and
    prepared for their journey through the Sephiroth. It establishes the initial
    soul structure and creates resonance with the creator.
    """
    
    def __init__(self, dimensions=(64, 64, 64), edge_of_chaos_ratio=0.618, 
                creator_resonance=0.7, field_name="Guff Field"):
        """
        Initialize a new Guff field.
        
        Args:
            dimensions (tuple): 3D dimensions of the field (x, y, z)
            edge_of_chaos_ratio (float): The edge of chaos parameter (default: 0.618 - golden ratio)
            creator_resonance (float): Strength of the creator's resonance
            field_name (str): Name identifier for the field
        """
        # Initialize base field with harmonizing frequency (528Hz - "miracle tone")
        super().__init__(dimensions=dimensions, field_name=field_name, 
                        edge_of_chaos_ratio=edge_of_chaos_ratio,
                        base_frequency=528.0)
        
        # Guff-specific properties
        self.creator_resonance = creator_resonance
        self.spark_formation_points = []
        self.sacred_patterns = {}
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        self.golden_ratio = 1.618
        
        # Load Kether aspects
        self.kether_aspects = KetherAspects()
        
        # Guff field frequencies
        self.primary_frequencies = []
        self.harmonic_frequencies = []
        self._initialize_frequencies()
        
        # Initialize quantum field
        self.initialize_quantum_field(base_amplitude=0.015)
        
        # Initialize Guff-specific sound properties
        self._initialize_guff_sound()
        
        logger.info(f"Guff Field initialized with creator resonance {creator_resonance}")
    
    def _initialize_frequencies(self):
        """
        Initialize the frequencies for the Guff field.
        
        The Guff incorporates frequencies from Kether (the Creator) and adds
        its own unique harmonics based on the golden ratio and Fibonacci sequence.
        """
        # Get frequencies from Kether aspects
        kether_freq_data = self.kether_aspects.export_resonant_frequencies()
        
        # Add primary frequencies from Kether
        primary_aspects = self.kether_aspects.get_primary_aspects()
        for aspect_name, aspect in primary_aspects.items():
            self.add_resonance_frequency(aspect['frequency'], 
                                       amplitude=aspect['strength'] * self.creator_resonance)
            self.primary_frequencies.append({
                'name': aspect_name,
                'frequency': aspect['frequency'],
                'amplitude': aspect['strength'] * self.creator_resonance
            })
        
        # Add Guff-specific frequencies based on golden ratio and Fibonacci
        # These frequencies create the formation template for soul structure
        base_freq = 432.0  # Base universal frequency
        
        # Create harmonics using Fibonacci sequence
        for i, fib in enumerate(self.fibonacci_sequence[2:]):  # Skip first two 1's
            harmonic_freq = base_freq * (fib / 10.0)
            amplitude = 0.9 - (i * 0.05)  # Decreasing amplitude
            amplitude = max(0.4, amplitude)  # Minimum amplitude of 0.4
            
            self.add_resonance_frequency(harmonic_freq, amplitude=amplitude, is_harmonic=True)
            self.harmonic_frequencies.append({
                'name': f'fibonacci_{fib}',
                'frequency': harmonic_freq,
                'amplitude': amplitude
            })
        
        # Create frequencies based on golden ratio
        for i in range(1, 6):
            phi_freq = base_freq * (self.golden_ratio ** i)
            amplitude = 0.85 - (i * 0.05)  # Decreasing amplitude
            amplitude = max(0.4, amplitude)  # Minimum amplitude of 0.4
            
            self.add_resonance_frequency(phi_freq, amplitude=amplitude, is_harmonic=True)
            self.harmonic_frequencies.append({
                'name': f'golden_ratio_{i}',
                'frequency': phi_freq,
                'amplitude': amplitude
            })
        
        logger.info(f"Initialized {len(self.primary_frequencies)} primary frequencies")
        logger.info(f"Initialized {len(self.harmonic_frequencies)} harmonic frequencies")
    
    def _initialize_guff_sound(self):
        """Initialize Guff-specific sound properties if sound is available."""
        if not self.sound_enabled:
            return
            
        try:
            if self.sound_system and self.field_sound:
                # Clear existing harmonics to replace with Guff-specific ones
                self.field_sound.harmonics = []
                
                # Add Solfeggio frequency harmonics - healing frequencies
                solfeggio_freqs = [
                    (396.0, 0.6, "UT - Liberating guilt"),
                    (417.0, 0.65, "RE - Undoing situations"),
                    (528.0, 1.0, "MI - Transformation (primary)"),
                    (639.0, 0.8, "FA - Connecting"),
                    (741.0, 0.7, "SOL - Expression"),
                    (852.0, 0.6, "LA - Intuition"),
                    (963.0, 0.55, "SI - Spiritual order")
                ]
                
                for freq, amp, desc in solfeggio_freqs:
                    self.field_sound.add_harmonic(
                        frequency=freq,
                        amplitude=amp,
                        description=desc
                    )
                
                # Add Fibonacci harmonics for soul formation
                for i, fib in enumerate(self.fibonacci_sequence[2:7]):  # Skip first two 1's
                    freq = self.base_frequency * (fib / 21.0)  # Normalize by Fibonacci 21
                    amplitude = 0.9 - (i * 0.1)  # Decreasing amplitude
                    
                    self.field_sound.add_harmonic(
                        frequency=freq,
                        amplitude=max(0.3, amplitude),
                        description=f"Fibonacci {fib} formation harmonic"
                    )
                
                # Add Phi (golden ratio) harmonics
                for i in range(1, 5):
                    freq = self.base_frequency * (self.golden_ratio ** (i/2))
                    amplitude = 0.85 - (i * 0.1)  # Decreasing amplitude
                    
                    self.field_sound.add_harmonic(
                        frequency=freq,
                        amplitude=max(0.3, amplitude),
                        description=f"Phi^{i/2} formation harmonic"
                    )
                
                # Add creator resonance frequencies
                for freq_data in self.primary_frequencies[:3]:  # Top 3 creator frequencies
                    self.field_sound.add_harmonic(
                        frequency=freq_data['frequency'],
                        amplitude=freq_data['amplitude'] * 0.9,
                        description=f"Creator '{freq_data['name']}' resonance"
                    )
                
                logger.info(f"Initialized Guff-specific sound with {len(self.field_sound.harmonics)} harmonics")
        except Exception as e:
            logger.warning(f"Could not initialize Guff sound: {str(e)}")
    
    def embed_sacred_geometry(self, center_position=None):
        """
        Embed sacred geometry patterns into the Guff field.
        
        These patterns create the template for initial soul structure formation.
        
        Args:
            center_position (tuple): Center position for the patterns (default: field center)
            
        Returns:
            dict: Dictionary of embedded pattern information
        """
        # Default to field center if not specified
        if center_position is None:
            center_position = (self.dimensions[0] // 2, 
                              self.dimensions[1] // 2,
                              self.dimensions[2] // 2)
        
        # Create and embed Seed of Life pattern (primary Guff pattern)
        seed = SeedOfLife(radius=0.6, resolution=self.dimensions[0])
        pattern_2d = seed.get_2d_pattern()
        
        # Convert 2D pattern to 3D with stronger central axis
        pattern_3d = np.zeros(self.dimensions, dtype=np.float64)
        for i in range(self.dimensions[2]):
            # Calculate distance from center z-plane
            z_distance = abs(i - center_position[2]) / (self.dimensions[2] // 2)
            
            # Create stronger presence along the central axis
            if z_distance < 0.2:
                falloff = 1.0 - z_distance / 0.2
            else:
                falloff = np.exp(-3 * z_distance**2)  # Gaussian falloff
                
            pattern_3d[:, :, i] = pattern_2d * falloff
            
        # Embed the pattern
        success = self.embed_pattern("seed_of_life", pattern_3d, 
                                   position=center_position, strength=1.2)
        
        # Store pattern information
        if success:
            self.sacred_patterns["seed_of_life"] = {
                "pattern_object": seed,
                "position": center_position,
                "strength": 1.2
            }
            
        # Create and embed Merkaba pattern
        merkaba = Merkaba(radius=0.7, resolution=self.dimensions[0], phi_ratio=self.golden_ratio)
        pattern_3d = merkaba.get_3d_pattern()
        
        # Scale pattern to field dimensions if needed
        if pattern_3d.shape != self.dimensions:
            from scipy.ndimage import zoom
            zoom_factors = (self.dimensions[0] / pattern_3d.shape[0],
                           self.dimensions[1] / pattern_3d.shape[1],
                           self.dimensions[2] / pattern_3d.shape[2])
            pattern_3d = zoom(pattern_3d, zoom_factors, order=1)
            
        # Embed the pattern
        success = self.embed_pattern("merkaba", pattern_3d, 
                                   position=center_position, strength=1.0)
        
        # Store pattern information
        if success:
            self.sacred_patterns["merkaba"] = {
                "pattern_object": merkaba,
                "position": center_position,
                "strength": 1.0
            }
            
        # Apply resonance to integrate patterns
        self.apply_resonance_to_field()
        
        # Update energy potential after embedding
        self.evolve_wave_function(time_step=0.01, iterations=5)
        
        # Generate sacred geometry sounds if sound is enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create a composite pattern sound for sacred geometry
                pattern_sound = self.sound_system.create_sound(
                    name="Guff Sacred Geometry",
                    fundamental_frequency=self.base_frequency,
                    sound_type=SoundType.COMPOSITE,
                    description="Sacred geometry patterns in Guff field"
)
                
                # Add specific harmonics for each pattern
                if "seed_of_life" in self.sacred_patterns:
                    # Add 7-based harmonics for Seed of Life (7 circles)
                    for i in range(1, 4):
                        pattern_sound.add_harmonic(
                            frequency=self.base_frequency * (7 / (7 + i)),
                            amplitude=0.8 - (i * 0.1),
                            description=f"Seed of Life harmonic {i}"
                        )
                
                if "merkaba" in self.sacred_patterns:
                    # Add tetrahedron-based harmonics (star tetrahedron/merkaba)
                    pattern_sound.add_harmonic(
                        frequency=self.base_frequency * self.golden_ratio,
                        amplitude=0.85,
                        description="Merkaba phi harmonic"
                    )
                    
                    pattern_sound.add_harmonic(
                        frequency=self.base_frequency * (4 / 3),  # Perfect fourth
                        amplitude=0.7,
                        description="Merkaba tetrahedral harmonic"
                    )
                
                # Apply pattern sound with moderate intensity
                self.apply_sound_to_field(pattern_sound, intensity=0.5)
            except Exception as e:
                logger.warning(f"Could not create sacred geometry sounds: {str(e)}")
        
        logger.info(f"Sacred geometry patterns embedded in Guff Field")
        
        return self.sacred_patterns
    
    def create_fibonacci_spirals(self, count=3):
        """
        Create Fibonacci spirals in the energy field.
        
        These spirals establish the growth pattern for soul development.
        
        Args:
            count (int): Number of spirals to create
            
        Returns:
            list: List of created spiral information
        """
        spirals = []
        
        # Center of the field
        center = (self.dimensions[0] // 2, 
                 self.dimensions[1] // 2,
                 self.dimensions[2] // 2)
        
        # Create multiple spirals with different orientations
        for i in range(count):
            # Calculate orientation angle based on golden ratio
            phi_angle = 2 * np.pi * i / count
            
            # Create the spiral
            spiral_info = self._create_single_spiral(center, phi_angle)
            spirals.append(spiral_info)
            
            logger.info(f"Created Fibonacci spiral {i+1}/{count} at angle {phi_angle:.2f}")
        
        # Create spiral sounds if sound is enabled
        if self.sound_enabled and self.sound_system and spirals:
            try:
                # Create a spiral sound
                spiral_sound = self.sound_system.create_sound(
                    name="Fibonacci Spirals",
                    fundamental_frequency=self.base_frequency * (1/self.golden_ratio),
                    sound_type=SoundType.COMPOSITE,
                    description="Fibonacci spirals in Guff field"
                )
                
                # Add Fibonacci sequence based harmonics
                for i, fib in enumerate(self.fibonacci_sequence[2:8]):  # Skip first two 1's
                    if i+2 < len(self.fibonacci_sequence):
                        ratio = self.fibonacci_sequence[i+3] / self.fibonacci_sequence[i+2]
                        spiral_sound.add_harmonic(
                            frequency=self.base_frequency * ratio,
                            amplitude=0.9 - (i * 0.1),
                            description=f"Fibonacci spiral ratio {fib}/{self.fibonacci_sequence[i+1]}"
                        )
                
                # Add golden ratio harmonics
                for i in range(1, 4):
                    spiral_sound.add_harmonic(
                        frequency=self.base_frequency * (self.golden_ratio ** i),
                        amplitude=0.85 - (i * 0.15),
                        description=f"Golden spiral φ^{i}"
                    )
                
                # Apply spiral sound with moderate intensity
                self.apply_sound_to_field(spiral_sound, intensity=0.6)
            except Exception as e:
                logger.warning(f"Could not create spiral sounds: {str(e)}")
        
        return spirals
    
    def _create_single_spiral(self, center, orientation_angle=0.0):
        """
        Create a single Fibonacci spiral in the energy field.
        
        Args:
            center (tuple): Center position for the spiral
            orientation_angle (float): Base orientation angle in radians
            
        Returns:
            dict: Information about the created spiral
        """
        # Constants for the Fibonacci spiral
        a = 0.1  # Spiral tightness
        n_points = 100  # Number of points in the spiral
        
        # Generate spiral points
        points = []
        energy_values = []
        
        for i in range(n_points):
            # Angle increases by golden ratio each step
            angle = orientation_angle + i * 2 * np.pi / self.golden_ratio
            
            # Radius increases with square root to maintain golden ratio proportions
            radius = a * np.sqrt(i)
            
            # Calculate 3D position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + radius * 0.3  # Gentle rise in z-direction
            
            # Ensure coordinates are within field dimensions
            x = min(max(0, x), self.dimensions[0]-1)
            y = min(max(0, y), self.dimensions[1]-1)
            z = min(max(0, z), self.dimensions[2]-1)
            
            point = (int(x), int(y), int(z))
            points.append(point)
            
            # Add energy at this point
            energy_value = 0.5 + 0.5 * np.exp(-i / 50)  # Decreasing energy along spiral
            energy_values.append(energy_value)
            
            # Create energy well at this point
            self.energy_potential[point] += energy_value
        
        # Apply smoothing to create continuous spiral
        self._smooth_energy_field()
        
        # Store spiral information
        spiral_info = {
            'center': center,
            'orientation': orientation_angle,
            'points': points,
            'energy_values': energy_values
        }
        
        return spiral_info
    
    def _smooth_energy_field(self, iterations=2):
        """
        Apply smoothing to the energy field.
        
        This creates more natural energy gradients and flow patterns.
        
        Args:
            iterations (int): Number of smoothing iterations
        """
        from scipy.ndimage import gaussian_filter
        
        # Apply Gaussian smoothing
        sigma = 1.0  # Smoothing strength
        for _ in range(iterations):
            self.energy_potential = gaussian_filter(self.energy_potential, sigma=sigma)
    
    def create_soul_formation_template(self):
        """
        Create a complete template for soul formation.
        
        This combines sacred geometry patterns, Fibonacci spirals, and resonant
        frequencies to establish the formation template for soul structure.
        
        Returns:
            dict: Information about the created template
        """
        # Embed sacred geometry
        self.embed_sacred_geometry()
        
        # Create Fibonacci spirals
        spirals = self.create_fibonacci_spirals(count=3)
        
        # Apply resonance frequencies
        self.apply_resonance_to_field()
        
        # Evolve the field to stabilize
        self.evolve_wave_function(time_step=0.01, iterations=10)
        
        # Create formation template sound if sound is enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create a complete template sound
                template_sound = self.sound_system.create_sound(
                    name="Soul Formation Template",
                    fundamental_frequency=self.base_frequency,
                    sound_type=SoundType.DIMENSIONAL,
                    description="Complete soul formation template sound"
                )
                
                # Add creator frequencies with high amplitude
                creator_freqs = [
                    (963.0, 0.85, "Creator Crown"),
                    (852.0, 0.75, "Creator Wisdom"),
                    (741.0, 0.7, "Creator Understanding")
                ]
                
                for freq, amp, desc in creator_freqs:
                    template_sound.add_harmonic(
                        frequency=freq,
                        amplitude=amp * self.creator_resonance,
                        description=desc
                    )
                
                # Add Solfeggio transformation frequency (528 Hz)
                template_sound.add_harmonic(
                    frequency=528.0,
                    amplitude=1.0,
                    description="Transformation frequency (MI)"
                )
                
                # Add formation frequencies
                template_sound.add_harmonic(
                    frequency=self.base_frequency * self.golden_ratio,
                    amplitude=0.9,
                    description="Golden ratio formation"
                )
                
                # Add phi-based frequencies
                template_sound.add_phi_harmonics(count=5, amplitude_falloff=0.9)
                
                # Apply template sound with high intensity
                self.apply_sound_to_field(template_sound, intensity=0.8)
                
                # Save this sound for later use
                self.template_sound = template_sound
                
            except Exception as e:
                logger.warning(f"Could not create formation template sound: {str(e)}")
        
        # Template information
        template_info = {
            'sacred_patterns': list(self.sacred_patterns.keys()),
            'spirals': len(spirals),
            'primary_frequencies': len(self.primary_frequencies),
            'harmonic_frequencies': len(self.harmonic_frequencies),
            'formation_quality': self.calculate_formation_quality()
        }
        
        logger.info(f"Soul formation template created with quality {template_info['formation_quality']:.4f}")
        return template_info
    
    def calculate_formation_quality(self):
        """
        Calculate the quality of the soul formation template.
        
        This evaluates how well the field can support soul development.
        
        Returns:
            float: Formation quality metric (0-1)
        """
        # Get field metrics
        metrics = self.calculate_stability_metrics()
        
        # Formation quality based on:
        # - Field stability
        # - Pattern coherence
        # - Resonance quality
        # - Golden ratio alignment
        
        # Calculate golden ratio alignment
        phi_alignment = self._calculate_phi_alignment()
        
        # Calculate overall formation quality
        formation_quality = (
            0.3 * metrics['stability'] +
            0.25 * metrics['coherence'] +
            0.25 * metrics['resonance_quality'] +
            0.2 * phi_alignment
        )
        
        logger.info(f"Formation quality components: Stability={metrics['stability']:.4f}, "
                   f"Coherence={metrics['coherence']:.4f}, Resonance={metrics['resonance_quality']:.4f}, "
                   f"Phi={phi_alignment:.4f}")
        
        return formation_quality
    
    def _calculate_phi_alignment(self):
        """
        Calculate how well the field aligns with golden ratio proportions.
        
        Returns:
            float: Phi alignment metric (0-1)
        """
        # Calculate energy distribution in different regions
        # and check if they follow golden ratio proportions
        
        # Divide the field into concentric spherical shells
        center = (self.dimensions[0] // 2, 
                 self.dimensions[1] // 2,
                 self.dimensions[2] // 2)
        
        # Create coordinate grid
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        z = np.arange(self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center
        distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        
        # Define shells based on Fibonacci sequence
        max_dist = np.sqrt(sum((d//2)**2 for d in self.dimensions))
        shell_radii = [fib * max_dist / 30 for fib in self.fibonacci_sequence]
        
        # Calculate energy in each shell
        shell_energies = []
        for i in range(len(shell_radii) - 1):
            inner_radius = shell_radii[i]
            outer_radius = shell_radii[i+1]
            
            shell_mask = (distance >= inner_radius) & (distance < outer_radius)
            shell_energy = np.sum(self.energy_potential[shell_mask])
            shell_energies.append(shell_energy)
        
        # Calculate ratio between consecutive shells
        energy_ratios = []
        for i in range(len(shell_energies) - 1):
            if shell_energies[i+1] > 0:
                ratio = shell_energies[i] / shell_energies[i+1]
                energy_ratios.append(ratio)
        
        # Calculate how close these ratios are to the golden ratio
        phi_deviations = [abs(ratio - self.golden_ratio) / self.golden_ratio 
                         for ratio in energy_ratios]
        
        # Convert to alignment metric (1 = perfect alignment)
        if phi_deviations:
            phi_alignment = 1.0 - (sum(phi_deviations) / len(phi_deviations))
            phi_alignment = max(0.0, min(1.0, phi_alignment))
        else:
            phi_alignment = 0.5  # Default if calculation fails
        
        return phi_alignment
    
    def strengthen_spark(self, spark, iteration_count=10):
        """
        Strengthen a soul spark in the Guff field.
        
        This process enhances the spark's stability, harmonic richness, and
        creator alignment through resonance coupling.
        
        Args:
            spark: The SoulSpark object to strengthen
            iteration_count (int): Number of strengthening iterations
            
        Returns:
            dict: Strengthening results with metrics
        """
        # Record initial metrics
        initial_metrics = spark.get_spark_metrics()
        
        # Prepare field for strengthening
        if not hasattr(self, 'template_created') or not self.template_created:
            self.create_soul_formation_template()
            self.template_created = True
        
        # Position the spark in the field center
        center = (self.dimensions[0] // 2, 
                 self.dimensions[1] // 2,
                 self.dimensions[2] // 2)
        
        # Create strengthening sound if sound is enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create strengthening sound
                strengthen_sound = self.sound_system.create_sound(
                    name="Soul Strengthening",
                    fundamental_frequency=self.base_frequency,
                    sound_type=SoundType.ENTANGLEMENT,
                    description="Soul strengthening process sound"
                )
                
                # Add resonant harmonics for strengthening
                strengthen_sound.add_harmonic(
                    frequency=528.0,  # Transformation frequency
                    amplitude=1.0,
                    description="Transformation harmonic"
                )
                
                strengthen_sound.add_harmonic(
                    frequency=639.0,  # Connection frequency
                    amplitude=0.8,
                    description="Connection harmonic"
                )
                
                # Add creator resonance
                strengthen_sound.add_harmonic(
                    frequency=963.0,
                    amplitude=0.9 * self.creator_resonance,
                    description="Creator resonance"
                )
                
                # Apply strengthening sound
                self.apply_sound_to_field(strengthen_sound, intensity=0.7)
            except Exception as e:
                logger.warning(f"Could not create strengthening sound: {str(e)}")
        
        # Strengthening process
        for i in range(iteration_count):
            # Calculate field resonance at this position
            field_resonance = self.resonance_matrix[center]
            
            # Extract quantum state properties
            field_energy = self.energy_potential[center]
            field_phase = np.angle(self.quantum_state[center])
            
            # Extract primary frequencies
            primary_freq = self.primary_frequencies[0]['frequency'] if self.primary_frequencies else 432.0
            
            # Strengthen the spark through resonance coupling
            spark.stability *= (1.0 + 0.02 * field_energy)
            spark.resonance *= (1.0 + 0.02 * field_resonance)
            spark.creator_alignment *= (1.0 + 0.02 * self.creator_resonance)
            
            # Cap values at 1.0
            spark.stability = min(1.0, spark.stability)
            spark.resonance = min(1.0, spark.resonance)
            spark.creator_alignment = min(1.0, spark.creator_alignment)
            
            # Enhance frequency signature
            if i % 3 == 0 and hasattr(spark, 'frequency_signature'):
                # Add some of Guff's frequencies to the spark
                if len(self.harmonic_frequencies) > i//3:
                    harmonic = self.harmonic_frequencies[i//3]
                    if 'frequencies' in spark.frequency_signature:
                        spark.frequency_signature['frequencies'].append(harmonic['frequency'])
                        spark.frequency_signature['amplitudes'].append(harmonic['amplitude'] * 0.8)
                        spark.frequency_signature['phases'].append(field_phase)
                        spark.frequency_signature['num_frequencies'] = len(spark.frequency_signature['frequencies'])
            
            # Evolve the spark's quantum state
            if hasattr(spark, 'evolve_quantum_state'):
                spark.evolve_quantum_state(time_step=0.02, iterations=2)
            
            # Recalculate harmonic structure and dimensional stability
            if hasattr(spark, 'generate_harmonic_structure'):
                spark.generate_harmonic_structure()
            
            if hasattr(spark, 'calculate_dimensional_stability'):
                spark.calculate_dimensional_stability()
            
            # Evolve the field to respond to the spark
            self.evolve_wave_function(time_step=0.01, iterations=2)
            
            # Update sound for each iteration if enabled
            if self.sound_enabled and self.sound_system and i % 3 == 0:
                try:
                    # Create an iteration-specific harmonic
                    progress = (i + 1) / iteration_count
                    
                    # Add to field sound - increasing frequency as we progress
                    self.field_sound.add_harmonic(
                        frequency=self.base_frequency * (1.0 + 0.1 * progress),
                        amplitude=0.5 + 0.3 * progress,
                        description=f"Strengthening progress {int(progress*100)}%"
                    )
                except Exception:
                    pass  # Silently continue if sound update fails
            
            logger.info(f"Strengthening iteration {i+1}/{iteration_count}: "
                       f"Stability={spark.stability:.4f}, Resonance={spark.resonance:.4f}")
        
        # Final metrics
        final_metrics = spark.get_spark_metrics()
        
        # Calculate improvement
        improvement = {
            'stability': final_metrics['formation']['stability'] - initial_metrics['formation']['stability'],
            'resonance': final_metrics['formation']['resonance'] - initial_metrics['formation']['resonance'],
            'creator_alignment': final_metrics['formation']['creator_alignment'] - 
                               initial_metrics['formation']['creator_alignment'],
            'harmonic_richness': final_metrics['harmonic']['richness'] - 
                               initial_metrics['harmonic']['richness'],
            'dimensional_stability': final_metrics['stability']['overall'] - 
                                  initial_metrics['stability']['overall']
        }
        
        # Create completion sound if sound enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create a completion sound based on final metrics
                completion_sound = self.sound_system.create_sound(
                    name="Strengthening Complete",
                    fundamental_frequency=self.base_frequency * (1.0 + 0.1 * improvement['stability']),
                    sound_type=SoundType.COMPOSITE,
                    description="Soul strengthening completion"
                )
                
                # Add harmonics reflecting improvements
                # Higher improvements = stronger harmonics
                
                # Stability harmonic
                completion_sound.add_harmonic(
                    frequency=self.base_frequency * 1.2,
                    amplitude=min(0.9, 0.5 + improvement['stability'] * 2),
                    description="Stability improvement"
                )
                
                # Resonance harmonic
                completion_sound.add_harmonic(
                    frequency=self.base_frequency * 1.5,  # Perfect fifth
                    amplitude=min(0.9, 0.5 + improvement['resonance'] * 2),
                    description="Resonance improvement"
                )
                
                # Creator alignment harmonic
                completion_sound.add_harmonic(
                    frequency=963.0,  # Creator frequency
                    amplitude=min(0.9, 0.5 + improvement['creator_alignment'] * 2),
                    description="Creator alignment improvement"
                )
                
                # Apply completion sound
                self.apply_sound_to_field(completion_sound, intensity=0.8)
            except Exception as e:
                logger.warning(f"Could not create completion sound: {str(e)}")
        
        # Overall strengthening result
        strengthening_result = {
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': improvement,
            'iterations': iteration_count,
            'success': True
        }
        
        logger.info(f"Spark strengthening complete with overall stability improvement of "
                   f"{improvement['stability']:.4f}")
        
        return strengthening_result
    
    def visualize_guff_field(self, show_template=True, slice_axis=2, 
                           slice_index=None, save_path=None):
        """
        Visualize the Guff field, showing the soul formation template.
        
        Args:
            show_template (bool): Whether to show template elements
            slice_axis (int): Axis to slice (0=x, 1=y, 2=z)
            slice_index (int): Slice index (default: middle of the axis)
            save_path (str): Path to save the visualization
            
        Returns:
            bool: True if visualization was successful
        """
        # Delegate to base class visualization with additional elements
        result = self.visualize_field_slice(axis=slice_axis, index=slice_index, 
                                          show_peaks=False, save_path=None)
        
        # Add template markers if requested
        if show_template and self.sacred_patterns:
            plt.figure(plt.gcf().number)  # Get current figure
            
            # Set default slice index to middle if not specified
            if slice_index is None:
                slice_index = self.dimensions[slice_axis] // 2
                
            # Highlight pattern centers
            for pattern_name, pattern_info in self.sacred_patterns.items():
                pos = pattern_info['position']
                
                # Only show if close to the current slice
                if abs(pos[slice_axis] - slice_index) <= 2:  # Within 2 units of the slice
                    # Convert 3D position to 2D slice coordinates
                    if slice_axis == 0:  # X-slice, showing Y-Z
                        marker_x = pos[1]  # Y
                        marker_y = pos[2]  # Z
                    elif slice_axis == 1:  # Y-slice, showing X-Z
                        marker_x = pos[0]  # X
                        marker_y = pos[2]  # Z
                    else:  # Z-slice, showing X-Y
                        marker_x = pos[0]  # X
                        marker_y = pos[1]  # Y
                        
                    # Plot marker with pattern name
                    plt.plot(marker_x, marker_y, 'wo', markersize=8)
                    plt.annotate(pattern_name, (marker_x, marker_y),
                               xytext=(5, 5), textcoords='offset points', color='white')
        
        # Add sound visualization if sound enabled
        if self.sound_enabled and hasattr(self, 'field_sound') and self.field_sound:
            plt.figure(plt.gcf().number)  # Get current figure
            
            # Add sound info text in corner
            sound_info = (
                f"Sound: Enabled\n"
                f"Base Frequency: {self.base_frequency:.1f} Hz\n"
                f"Harmonics: {len(self.field_sound.harmonics)}"
            )
            
            # Add to bottom left
            plt.text(0.02, 0.02, sound_info, transform=plt.gca().transAxes,
                   verticalalignment='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white')
        
        # Update title
        formation_quality = self.calculate_formation_quality()
        plt.title(f"Guff Field - {['X', 'Y', 'Z'][slice_axis]}={slice_index} " +
                f"(Formation Quality: {formation_quality:.4f})")
        
        # Add colorbar label
        cbar = plt.gcf().axes[1]  # Get the colorbar axis
        cbar.set_ylabel('Energy Potential')
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Guff field visualization saved to {save_path}")
            
        return True
    
    def visualize_guff_sound(self, duration=3.0, save_path=None):
        """
        Visualize the sound waveform for this Guff field.
        
        Args:
            duration (float): Duration in seconds
            save_path (str): Optional path to save visualization
            
        Returns:
            bool: True if visualization was successful
        """
        if not self.sound_enabled or not hasattr(self, 'field_sound') or not self.field_sound:
            logger.warning("Sound is not enabled or no field sound available")
            return False
        
        # Generate waveform
        soundscape = self.generate_field_soundscape(duration=duration)
        if not soundscape:
            return False
            
        time_array, waveform = soundscape
        
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
            description = harmonic.description if hasattr(harmonic, 'description') else ""
            harmonics_text += f"{harmonic.frequency:.1f} Hz @ {harmonic.amplitude:.2f} - {description}\n"
            
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
    
    def get_guff_metrics(self):
        """
        Get metrics about the Guff field state.
        
        Returns:
            dict: Dictionary of field metrics
        """
        # Get base field metrics
        base_metrics = self.calculate_stability_metrics()
        
        # Add Guff-specific metrics
        guff_metrics = {
            'formation_quality': self.calculate_formation_quality(),
            'phi_alignment': self._calculate_phi_alignment(),
            'sacred_patterns': list(self.sacred_patterns.keys()),
            'creator_resonance': self.creator_resonance,
            'primary_frequencies': [f['frequency'] for f in self.primary_frequencies],
            'harmonic_frequencies': [f['frequency'] for f in self.harmonic_frequencies]
        }
        
        # Add sound metrics if enabled
        if self.sound_enabled and hasattr(self, 'field_sound') and self.field_sound:
            sound_metrics = {
                'sound_harmonics': len(self.field_sound.harmonics),
                'sound_harmony': base_metrics.get('sound_harmony', 0),
                'primary_sound_frequency': self.field_sound.fundamental_frequency
            }
            guff_metrics.update(sound_metrics)
        
        # Combine metrics
        metrics = {**base_metrics, **guff_metrics}
        
        return metrics
    
    def save_guff_data(self, output_dir="output", filename=None):
        """
        Save Guff field data including sound information.
        
        Args:
            output_dir (str): Directory to save output files
            filename (str): Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        # First save using the base field system method
        save_path = self.save_field_data(output_dir, filename)
        
        if not save_path:
            return None
            
        # Save additional Guff-specific data
        if filename is None:
            filename = f"guff_specific_{self.field_id[:8]}.npz"
        else:
            # Add prefix to avoid name collision
            filename = f"guff_specific_{filename}"
            
        guff_save_path = os.path.join(output_dir, filename)
        
        # Compile Guff-specific data
        guff_data = {
            'field_id': self.field_id,
            'sacred_patterns': list(self.sacred_patterns.keys()),
            'primary_frequencies': [(f['name'], f['frequency'], f['amplitude']) 
                                  for f in self.primary_frequencies],
            'harmonic_frequencies': [(f['name'], f['frequency'], f['amplitude']) 
                                   for f in self.harmonic_frequencies],
            'fibonacci_sequence': self.fibonacci_sequence,
            'golden_ratio': self.golden_ratio,
            'creator_resonance': self.creator_resonance
        }
        
        # Save
        # Save Guff-specific data
        try:
            np.savez_compressed(guff_save_path, **guff_data)
            logger.info(f"Guff-specific data saved to {guff_save_path}")
            return save_path  # Return the main file path
            
        except Exception as e:
            logger.error(f"Error saving Guff-specific data: {str(e)}")
            return save_path  # Still return the main file path
    
    @classmethod
    def load_guff_data(cls, field_path, guff_specific_path=None):
        """
        Load Guff field data including sound information.
        
        Args:
            field_path (str): Path to the main field data file
            guff_specific_path (str): Optional path to Guff-specific data
            
        Returns:
            GuffField: Loaded Guff field
        """
        # First load base field data
        field = cls.load_field_data(field_path)
        
        if not field:
            return None
            
        # Now load Guff-specific data if available
        if guff_specific_path and os.path.exists(guff_specific_path):
            try:
                guff_data = np.load(guff_specific_path, allow_pickle=True)
                
                # Restore sacred patterns if available
                if 'sacred_patterns' in guff_data:
                    field.sacred_patterns = {
                        pattern: {"pattern_object": pattern, "position": None, "strength": 1.0}
                        for pattern in guff_data['sacred_patterns']
                    }
                    
                # Restore primary frequencies if available
                if 'primary_frequencies' in guff_data:
                    field.primary_frequencies = []
                    for name, freq, amp in guff_data['primary_frequencies']:
                        field.primary_frequencies.append({
                            'name': str(name),
                            'frequency': float(freq),
                            'amplitude': float(amp)
                        })
                        
                # Restore harmonic frequencies if available
                if 'harmonic_frequencies' in guff_data:
                    field.harmonic_frequencies = []
                    for name, freq, amp in guff_data['harmonic_frequencies']:
                        field.harmonic_frequencies.append({
                            'name': str(name),
                            'frequency': float(freq),
                            'amplitude': float(amp)
                        })
                        
                # Restore other Guff-specific properties
                if 'fibonacci_sequence' in guff_data:
                    field.fibonacci_sequence = list(guff_data['fibonacci_sequence'])
                    
                if 'golden_ratio' in guff_data:
                    field.golden_ratio = float(guff_data['golden_ratio'])
                    
                if 'creator_resonance' in guff_data:
                    field.creator_resonance = float(guff_data['creator_resonance'])
                    
                logger.info(f"Loaded Guff-specific data from {guff_specific_path}")
                
            except Exception as e:
                logger.error(f"Error loading Guff-specific data: {str(e)}")
        
        return field
    
    def __str__(self):
        """String representation of the Guff field."""
        metrics = self.get_guff_metrics()
        
        # Add sound info if enabled
        sound_info = ""
        if self.sound_enabled and hasattr(self, 'field_sound') and self.field_sound:
            sound_info = f"\nSound Harmonics: {len(self.field_sound.harmonics)}"
            sound_info += f"\nSound Harmony: {metrics.get('sound_harmony', 0):.4f}"
        
        return (f"Guff Field (ID: {self.field_id[:8]})\n"
                f"Dimensions: {self.dimensions}\n"
                f"Edge of Chaos Ratio: {self.edge_of_chaos_ratio}\n"
                f"Creator Resonance: {self.creator_resonance}\n"
                f"Base Frequency: {self.base_frequency} Hz\n"
                f"Formation Quality: {metrics['formation_quality']:.4f}\n"
                f"Phi Alignment: {metrics['phi_alignment']:.4f}\n"
                f"Stability: {metrics['stability']:.4f}\n"
                f"Coherence: {metrics['coherence']:.4f}\n"
                f"Sacred Patterns: {', '.join(metrics['sacred_patterns'])}"
                f"{sound_info}")


if __name__ == "__main__":
    # Example usage
    guff = GuffField(dimensions=(64, 64, 64), edge_of_chaos_ratio=0.618, creator_resonance=0.7)
    
    # Create soul formation template
    template_info = guff.create_soul_formation_template()
    
    # Visualize field
    guff.visualize_guff_field(save_path="guff_field_visualization.png")
    
    # Visualize sound if enabled
    if guff.sound_enabled:
        guff.visualize_guff_sound(save_path="guff_field_sound.png")
    
    # Print metrics
    print(guff)
    
    # Save data
    guff.save_guff_data()
    
    # Import soul spark to test strengthening
    try:
        from soul_formation.soul_spark import SoulSpark
        
        # Create a test spark
        spark = SoulSpark(creator_resonance=0.7)
        
        # Strengthen the spark
        result = guff.strengthen_spark(spark, iteration_count=10)
        
        print(f"\nStrengthening Results:")
        for key, value in result['improvement'].items():
            print(f"- {key}: {value:+.4f}")
    except ImportError:
        print("\nSoul Spark module not available, skipping strengthening test")
