"""
Void Field Implementation

This module implements the Void dimension - the primordial quantum field where
soul sparks form through quantum fluctuations at the edge of chaos.

The Void represents the primal reality before manifestation, where pure potentiality
exists in a quantum state of superposition. It contains embedded sacred geometry patterns
that create potential wells where soul sparks can coalesce from quantum fluctuations.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base field system
from field_system import FieldSystem

# Import sacred geometry patterns
from shared.flower_of_life import FlowerOfLife
from shared.vesica_piscis import VesicaPiscis
from shared.seed_of_life import SeedOfLife

# Import sound system if available
try:
    from sounds.sound_system import SoundSystem, SoundType
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    logging.warning("Sound system not available. Void field will function without sound capabilities.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='void_field.log'
)
logger = logging.getLogger('void_field')

class VoidField(FieldSystem):
    """
    Implementation of the Void dimension.
    
    The Void is the primordial quantum field where soul sparks emerge through
    quantum fluctuations at sacred geometry intersection points. It represents
    pure potentiality and the edge of chaos where order can emerge.
    """
    
    def __init__(self, dimensions=(64, 64, 64), edge_of_chaos_ratio=0.618, 
                creator_resonance=0.5, field_name="Void Field"):
        """
        Initialize a new Void field.
        
        Args:
            dimensions (tuple): 3D dimensions of the field (x, y, z)
            edge_of_chaos_ratio (float): The edge of chaos parameter (default: 0.618 - golden ratio)
            creator_resonance (float): Strength of the creator's (Kether) resonance
            field_name (str): Name identifier for the field
        """
        # Initialize base field
        super().__init__(dimensions=dimensions, field_name=field_name, 
                        edge_of_chaos_ratio=edge_of_chaos_ratio,
                        base_frequency=963.0)  # Crown/Kether resonance
        
        # Void-specific properties
        self.creator_resonance = creator_resonance
        self.potential_wells = []
        self.spark_formation_points = []
        self.sacred_patterns = {}
        
        # Void field frequencies
        self.primary_void_frequency = 963.0  # Hz - Crown/Kether resonance
        self.secondary_void_frequencies = [
            432.0,  # Cosmic/natural resonance
            528.0,  # Creation/DNA frequency
            639.0,  # Connection frequency
            741.0,  # Expression frequency
            852.0   # Intuition frequency
        ]
        
        # Add void frequencies to the field
        self.add_resonance_frequency(self.primary_void_frequency, amplitude=1.0)
        for freq in self.secondary_void_frequencies:
            self.add_resonance_frequency(freq, amplitude=0.5, is_harmonic=True)
            
        # Initialize quantum fluctuations
        self.initialize_quantum_field(base_amplitude=0.02)
        
        # Initialize void-specific sound properties if sound available
        self._initialize_void_sound()
        
        logger.info(f"Void Field initialized with creator resonance {creator_resonance}")

    def _initialize_void_sound(self):
        """Initialize void-specific sound properties if sound is available."""
        if not self.sound_enabled:
            return
            
        try:
            if self.sound_system and self.field_sound:
                # Add specific void harmonics to the existing field sound
                
                # Add primordial white noise component (low amplitude)
                noise_freq = self.primary_void_frequency * 0.618  # Edge of chaos frequency
                self.field_sound.add_harmonic(
                    frequency=noise_freq,
                    amplitude=0.3,
                    phase=np.random.random() * 2 * np.pi,
                    description="Void primordial noise"
                )
                
                # Add creator frequencies with varying amplitudes
                creator_freqs = [
                    (self.primary_void_frequency, 0.9, "Creator primary"),
                    (432.0, 0.7, "Creator harmonic 1"),
                    (528.0, 0.6, "Creator harmonic 2"),
                    (self.primary_void_frequency * 0.618, 0.8, "Creator edge of chaos")
                ]
                
                for freq, amp, desc in creator_freqs:
                    self.field_sound.add_harmonic(
                        frequency=freq, 
                        amplitude=amp * self.creator_resonance,  # Scale by creator resonance
                        description=desc
                    )
                
                logger.info(f"Initialized void-specific sound properties for {self.field_name}")
        except Exception as e:
            logger.warning(f"Could not initialize void sound: {str(e)}")
    
    def embed_sacred_geometry(self, center_position=None):
        """
        Embed sacred geometry patterns into the void field.
        
        These patterns create the potential wells where soul sparks can form.
        
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
        
        # Create and embed Flower of Life pattern
        flower = FlowerOfLife(radius=0.5, iterations=3, resolution=self.dimensions[0])
        pattern_2d = flower.get_2d_pattern()
        
        # Convert 2D pattern to 3D
        pattern_3d = np.zeros(self.dimensions, dtype=np.float64)
        for i in range(self.dimensions[2]):
            # Add pattern with falloff from center z-plane
            z_distance = abs(i - center_position[2]) / (self.dimensions[2] // 2)
            falloff = np.exp(-5 * z_distance**2)  # Gaussian falloff
            pattern_3d[:, :, i] = pattern_2d * falloff
            
        # Embed the pattern
        success = self.embed_pattern("flower_of_life", pattern_3d, 
                                    position=center_position, strength=1.0)
        
        # Store pattern information
        if success:
            self.sacred_patterns["flower_of_life"] = {
                "pattern_object": flower,
                "position": center_position,
                "strength": 1.0
            }
            
        # Embed Seed of Life pattern (subset of Flower of Life)
        seed_pattern_2d = flower.extract_seed_of_life()
        
        # Convert 2D pattern to 3D
        seed_pattern_3d = np.zeros(self.dimensions, dtype=np.float64)
        for i in range(self.dimensions[2]):
            # Add pattern with falloff from center z-plane
            z_distance = abs(i - center_position[2]) / (self.dimensions[2] // 2)
            falloff = np.exp(-5 * z_distance**2)  # Gaussian falloff
            seed_pattern_3d[:, :, i] = seed_pattern_2d * falloff
            
        # Embed the pattern
        success = self.embed_pattern("seed_of_life", seed_pattern_3d, 
                                    position=center_position, strength=1.2)

        # Store pattern information
        if success:
            self.sacred_patterns["seed_of_life"] = {
                "pattern_object": "seed_of_life",
                "position": center_position,
                "strength": 1.2
            }
                                    
        # Embed Vesica Piscis pattern in multiple orientations
        vesica = VesicaPiscis(radius=0.6, resolution=self.dimensions[0])
        pattern_3d = vesica.get_3d_pattern()
        
        # Scale pattern to field dimensions
        if pattern_3d.shape != self.dimensions:
            from scipy.ndimage import zoom
            zoom_factors = (self.dimensions[0] / pattern_3d.shape[0],
                           self.dimensions[1] / pattern_3d.shape[1],
                           self.dimensions[2] / pattern_3d.shape[2])
            pattern_3d = zoom(pattern_3d, zoom_factors, order=1)
            
        # Embed in original orientation (XY plane)
        success = self.embed_pattern("vesica_piscis_xy", pattern_3d, 
                                   position=center_position, strength=1.0)

        # Store pattern information
        if success:
            self.sacred_patterns["vesica_piscis_xy"] = {
                "pattern_object": "vesica_piscis_xy",
                "position": center_position,
                "strength": 1.0
            }
                                   
        # Create rotated versions for XZ and YZ planes
        pattern_3d_xz = np.transpose(pattern_3d, (0, 2, 1))
        pattern_3d_yz = np.transpose(pattern_3d, (2, 1, 0))
        
        # Embed rotated patterns
        success_xz = self.embed_pattern("vesica_piscis_xz", pattern_3d_xz, 
                                      position=center_position, strength=1.0)
        success_yz = self.embed_pattern("vesica_piscis_yz", pattern_3d_yz, 
                                      position=center_position, strength=1.0)

        # Store pattern information for successful embeddings
        if success_xz:
            self.sacred_patterns["vesica_piscis_xz"] = {
                "pattern_object": "vesica_piscis_xz",
                "position": center_position,
                "strength": 1.0
            }
        
        if success_yz:
            self.sacred_patterns["vesica_piscis_yz"] = {
                "pattern_object": "vesica_piscis_yz",
                "position": center_position,
                "strength": 1.0
            }
        
        # Apply resonance to integrate patterns
        self.apply_resonance_to_field()
        
        # Update energy potential after embedding
        self.evolve_wave_function(time_step=0.01, iterations=5)
        
        # Identify potential wells at pattern intersections
        self.identify_potential_wells()
        
        # Create sacred geometry sound effects if sound enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create sounds specific to each pattern
                pattern_names = list(self.sacred_patterns.keys())
                
                # Only create new sounds if we have patterns
                if pattern_names:
                    # Create a composite pattern sound
                    pattern_sound = self.sound_system.create_sound(
                        name=f"Sacred Geometry {self.field_name}",
                        fundamental_frequency=self.base_frequency * 0.618,  # Edge of chaos frequency
                        sound_type=SoundType.COMPOSITE,
                        description=f"Sacred geometry patterns for {self.field_name}"
                    )
                    
                    # Add pattern-specific harmonics
                    if "flower_of_life" in pattern_names:
                        pattern_sound.add_harmonic(
                            frequency=self.base_frequency * 0.5,
                            amplitude=0.8,
                            description="Flower of Life harmonic"
                        )
                        
                    if "seed_of_life" in pattern_names:
                        pattern_sound.add_harmonic(
                            frequency=self.base_frequency * 0.7,
                            amplitude=0.7,
                            description="Seed of Life harmonic"
                        )
                        
                    if "vesica_piscis_xy" in pattern_names:
                        pattern_sound.add_harmonic(
                            frequency=self.base_frequency * 0.6,
                            amplitude=0.6,
                            description="Vesica Piscis harmonic"
                        )
                    
                    # Apply this composite sound to the field
                    # Using lower intensity to blend with existing field sounds
                    self.apply_sound_to_field(pattern_sound, intensity=0.6)
            except Exception as e:
                logger.warning(f"Could not create sacred geometry sounds: {str(e)}")
        
        logger.info(f"Sacred geometry patterns embedded in Void Field")
        logger.info(f"Found {len(self.potential_wells)} potential wells for spark formation")
        
        return self.sacred_patterns
        
    def identify_potential_wells(self, threshold=1.5, min_distance=3):
        """
        Identify potential wells at sacred geometry pattern intersections.
        
        These wells are points where soul sparks can potentially form from
        quantum fluctuations.
        
        Args:
            threshold (float): Energy threshold for well detection
            min_distance (int): Minimum distance between wells
            
        Returns:
            list: List of potential well coordinates and properties
        """
        # Find peaks in the energy potential
        peaks = self.find_energy_peaks(threshold=threshold, min_distance=min_distance)
        
        # Process peaks to identify potential wells
        self.potential_wells = []
        
        for peak in peaks:
            position = peak['position']
            energy = peak['energy']
            
            # Calculate well properties
            stability = self.calculate_well_stability(position)
            resonance = self.calculate_well_resonance(position)
            creator_alignment = self.calculate_creator_alignment(position)
            
            # Combined well quality metric
            quality = 0.4 * stability + 0.3 * resonance + 0.3 * creator_alignment
            
            self.potential_wells.append({
                'position': position,
                'energy': energy,
                'stability': stability,
                'resonance': resonance,
                'creator_alignment': creator_alignment,
                'quality': quality,
                'active': True  # Well is available for spark formation
            })
        
        # Sort wells by quality (highest first)
        self.potential_wells.sort(key=lambda x: x['quality'], reverse=True)
        
        # Create well identification sounds if sound enabled
        if self.sound_enabled and self.sound_system and self.potential_wells:
            try:
                # Only create sounds if we have high-quality wells
                high_quality_wells = [w for w in self.potential_wells if w['quality'] > 0.7]
                
                if high_quality_wells:
                    # Create a wells sound
                    wells_sound = self.sound_system.create_sound(
                        name=f"Potential Wells {self.field_name}",
                        fundamental_frequency=self.base_frequency * 0.75,
                        sound_type=SoundType.RESONANT,
                        description=f"Potential wells for {self.field_name}"
                    )
                    
                    # Add harmonics based on well qualities
                    for i, well in enumerate(high_quality_wells[:3]):  # Use top 3 wells
                        wells_sound.add_harmonic(
                            frequency=self.base_frequency * (0.8 + 0.1 * well['quality']),
                            amplitude=well['quality'] * 0.8,
                            description=f"Well {i+1} harmonic"
                        )
                    
                    # Apply this wells sound to the field
                    # Using lower intensity to avoid disrupting field
                    self.apply_sound_to_field(wells_sound, intensity=0.4)
            except Exception as e:
                logger.warning(f"Could not create potential wells sounds: {str(e)}")
        
        logger.info(f"Identified {len(self.potential_wells)} potential wells in Void Field")
        return self.potential_wells
        
    def calculate_well_stability(self, position):
        """
        Calculate the stability of a potential well.
        
        Args:
            position (tuple): (x, y, z) position of the well
            
        Returns:
            float: Stability metric (0-1)
        """
        # Extract local 5x5x5 region around the position
        x, y, z = position
        x_min = max(0, x - 2)
        y_min = max(0, y - 2)
        z_min = max(0, z - 2)
        x_max = min(self.dimensions[0], x + 3)
        y_max = min(self.dimensions[1], y + 3)
        z_max = min(self.dimensions[2], z + 3)
        
        region = self.energy_potential[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Calculate metrics
        if region.size > 0:
            # Stability based on local gradient and curvature
            gradient_x = np.gradient(region, axis=0)
            gradient_y = np.gradient(region, axis=1)
            gradient_z = np.gradient(region, axis=2) if region.shape[2] > 1 else np.zeros_like(region)
            
            # Gradient magnitude
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
            
            # Curvature (approximate through Laplacian)
            laplacian = np.gradient(gradient_x, axis=0) + np.gradient(gradient_y, axis=1)
            if region.shape[2] > 1:
                laplacian += np.gradient(gradient_z, axis=2)
                
            # Well stability metrics
            avg_gradient = np.mean(gradient_mag)
            avg_curvature = np.mean(np.abs(laplacian))
            
            # Combined stability metric (1 = stable, 0 = unstable)
            # Good wells have low average gradient and high curvature
            stability = (1.0 / (1.0 + avg_gradient)) * (avg_curvature / (1.0 + avg_curvature))
            
            return min(1.0, max(0.0, stability))
        else:
            return 0.0
    
    def calculate_well_resonance(self, position):
        """
        Calculate the resonance quality of a potential well.
        
        Args:
            position (tuple): (x, y, z) position of the well
            
        Returns:
            float: Resonance metric (0-1)
        """
        # Extract the resonance value at this position
        x, y, z = position
        if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]:
            # Raw resonance value at this point
            resonance_value = self.resonance_matrix[x, y, z]
            
            # Normalize to 0-1 range
            if resonance_value > 0:
                # Check local neighborhood for resonance patterns
                x_min = max(0, x - 1)
                y_min = max(0, y - 1)
                z_min = max(0, z - 1)
                x_max = min(self.dimensions[0], x + 2)
                y_max = min(self.dimensions[1], y + 2)
                z_max = min(self.dimensions[2], z + 2)
                
                neighborhood = self.resonance_matrix[x_min:x_max, y_min:y_max, z_min:z_max]
                
                # Calculate resonance quality based on local properties
                local_max = np.max(neighborhood)
                local_mean = np.mean(neighborhood)
                local_std = np.std(neighborhood)
                
                # Ideal resonance has high value, dominates neighborhood, but has structure (not uniform)
                resonance_metric = (resonance_value / local_max) * (local_std / (local_mean + 0.001))
                
                # Scale with golden ratio for optimal resonance
                phi_alignment = 1.0 - abs((local_max / (local_mean + 0.001)) - 1.618) / 1.618
                
                # Combined metric
                resonance_quality = 0.6 * resonance_metric + 0.4 * phi_alignment
                
                return min(1.0, max(0.0, resonance_quality))
            else:
                return 0.0
        else:
            return 0.0
    
    def calculate_creator_alignment(self, position):
        """
        Calculate the alignment with creator resonance at a potential well.
        
        Args:
            position (tuple): (x, y, z) position of the well
            
        Returns:
            float: Creator alignment metric (0-1)
        """
        # The creator alignment is based on:
        # 1. Distance from the field center (creator influence is strongest at center)
        # 2. Alignment with primary void frequency
        # 3. Golden ratio proportions
        
        # Distance from field center
        center = (self.dimensions[0] // 2, self.dimensions[1] // 2, self.dimensions[2] // 2)
        x, y, z = position
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        max_distance = np.sqrt(sum((d // 2)**2 for d in self.dimensions))
        
        # Distance factor (1 at center, decreasing outward)
        distance_factor = 1.0 - (distance / max_distance)
        
        # Frequency alignment
        # Extract local wave properties
        if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]:
            # Get the quantum state at this position
            quantum_val = self.quantum_state[x, y, z]
            
            # Extract phase - correlated with frequency alignment
            phase = np.angle(quantum_val) if quantum_val != 0 else 0
            
            # Calculate alignment with primary frequency (normalized to 0-1)
            primary_phase = 2 * np.pi * self.primary_void_frequency * 0.01  # Reference phase
            phase_alignment = 1.0 - min(1.0, abs(phase - primary_phase) / np.pi)
        else:
            phase_alignment = 0.0
            
        # Golden ratio alignment - checks if energy proportions follow golden ratio
        x_min = max(0, x - 1)
        y_min = max(0, y - 1)
        z_min = max(0, z - 1)
        x_max = min(self.dimensions[0], x + 2)
        y_max = min(self.dimensions[1], y + 2)
        z_max = min(self.dimensions[2], z + 2)
        
        neighborhood = self.energy_potential[x_min:x_max, y_min:y_max, z_min:z_max]
        
        if neighborhood.size > 0:
            sorted_energies = np.sort(neighborhood.flatten())
            if len(sorted_energies) > 2:
                # Check ratio of consecutive energy levels
                ratios = sorted_energies[1:] / np.maximum(sorted_energies[:-1], 1e-10)
                
                # Find ratio closest to golden ratio
                phi = 1.618
                golden_alignment = 1.0 - min(1.0, np.min(np.abs(ratios - phi) / phi))
            else:
                golden_alignment = 0.5  # Default if not enough points
        else:
            golden_alignment = 0.0
            
        # Combined alignment metric
        creator_alignment = (0.4 * distance_factor + 
                            0.3 * phase_alignment + 
                            0.3 * golden_alignment)
        
        # Apply creator resonance factor
        creator_alignment *= self.creator_resonance
        
        return min(1.0, max(0.0, creator_alignment))
    
    def simulate_quantum_fluctuations(self, iterations=10, fluctuation_strength=0.02):
        """
        Simulate quantum fluctuations in the void field.
        
        These fluctuations can lead to spark formation if they occur at potential wells.
        
        Args:
            iterations (int): Number of simulation iterations
            fluctuation_strength (float): Strength of quantum fluctuations
            
        Returns:
            list: List of any spark formation events that occurred
        """
        spark_formations = []
        
        # Generate a fluctuation sound if sound is enabled
        if self.sound_enabled and self.sound_system:
            try:
                # Create a fluctuation sound
                fluctuation_sound = self.sound_system.create_sound(
                    name=f"Quantum Fluctuations {self.field_name}",
                    fundamental_frequency=self.base_frequency * self.edge_of_chaos_ratio,
                    sound_type=SoundType.TRANSITIONAL,
                    description=f"Quantum fluctuations in {self.field_name}"
                )
                
                # Add chaotic but structured harmonics
                for i in range(5):
                    # Random frequency with some structure
                    freq_factor = 0.5 + np.random.random()
                    fluctuation_sound.add_harmonic(
                        frequency=self.base_frequency * freq_factor,
                        amplitude=0.3 + 0.2 * np.random.random(),
                        phase=np.random.random() * 2 * np.pi,
                        description=f"Fluctuation harmonic {i+1}"
                    )
                
                # Apply fluctuation sound with moderate intensity
                self.apply_sound_to_field(fluctuation_sound, intensity=0.5)
            except Exception as e:
                logger.warning(f"Could not create fluctuation sound: {str(e)}")
        
        for i in range(iterations):
            # Add random quantum fluctuations
            real_fluctuations = np.random.normal(0, fluctuation_strength, self.dimensions)
            imag_fluctuations = np.random.normal(0, fluctuation_strength, self.dimensions)
            
            quantum_fluctuations = real_fluctuations + 1j * imag_fluctuations
            
            # Apply fluctuations with edge of chaos factor
            # The edge of chaos is where quantum fluctuations are most likely to result in emergent behavior
            chaos_factor = np.random.random(self.dimensions) < self.edge_of_chaos_ratio
            self.quantum_state = self.quantum_state + quantum_fluctuations * chaos_factor
            
            # Normalize the wave function
            self.wave_function = self.normalize_wave_function(self.quantum_state)
            
            # Update energy potential
            self.energy_potential = 0.9 * self.energy_potential + 0.1 * np.abs(self.wave_function) ** 2
            
            # Apply resonance effects
            self.apply_resonance_to_field()
            
            # Evolve the system
            self.evolve_wave_function(time_step=0.02, iterations=1)
            
            # Check for spark formation at potential wells
            sparks = self.check_spark_formation()
            if sparks:
                spark_formations.extend(sparks)
                
                # Generate spark formation sounds if enabled
                if self.sound_enabled and self.sound_system:
                    try:
                        # Create a spark formation sound
                        formation_sound = self.sound_system.create_sound(
                            name=f"Spark Formation {self.field_name}",
                            fundamental_frequency=self.base_frequency * 1.2,  # Higher frequency for spark
                            sound_type=SoundType.ENTANGLEMENT,
                            description=f"Soul spark formation in {self.field_name}"
                        )
                        
                        # Add bright, crystalline harmonics
                        formation_sound.add_harmonic(
                            frequency=self.base_frequency * 1.618,  # Golden ratio
                            amplitude=0.9,
                            description="Spark phi harmonic"
                        )
                        
                        formation_sound.add_harmonic(
                            frequency=self.base_frequency * 2.0,  # Octave
                            amplitude=0.8,
                            description="Spark octave"
                        )
                        
                        formation_sound.add_harmonic(
                            frequency=self.base_frequency * 3.0/2.0,  # Perfect fifth
                            amplitude=0.7,
                            description="Spark fifth"
                        )
                        
                        # Apply formation sound with higher intensity
                        self.apply_sound_to_field(formation_sound, intensity=0.8)
                    except Exception as e:
                        logger.warning(f"Could not create spark formation sound: {str(e)}")
                
        logger.info(f"Simulated {iterations} quantum fluctuation cycles")
        logger.info(f"Detected {len(spark_formations)} spark formation events")
        
        return spark_formations
    
    def check_spark_formation(self, threshold=0.85):
        """
        Check if any potential wells have developed soul sparks.
        
        Args:
            threshold (float): Energy threshold for spark formation
            
        Returns:
            list: List of formed sparks and their properties
        """
        sparks = []
        
        # Check each potential well
        for i, well in enumerate(self.potential_wells):
            if not well['active']:
                continue  # Skip wells that are already used
                
            position = well['position']
            x, y, z = position
            
            # Skip if position is out of bounds
            if not (0 <= x < self.dimensions[0] and 
                   0 <= y < self.dimensions[1] and 
                   0 <= z < self.dimensions[2]):
                continue
                
            # Calculate current metrics
            current_energy = self.energy_potential[x, y, z]
            current_stability = self.calculate_well_stability(position)
            current_resonance = self.calculate_well_resonance(position)
            current_alignment = self.calculate_creator_alignment(position)
            
            # Combined formation potential
            formation_potential = (0.3 * current_energy + 
                                  0.3 * current_stability + 
                                  0.2 * current_resonance + 
                                  0.2 * current_alignment)
            
            # Check if spark forms
            if formation_potential > threshold:
                # Create spark
                spark_id = str(uuid.uuid4())
                
                spark = {
                    'spark_id': spark_id,
                    'position': position,
                    'formation_time': 0,  # Placeholder - would be actual timestamp/iteration
                    'energy': current_energy,
                    'stability': current_stability,
                    'resonance': current_resonance,
                    'creator_alignment': current_alignment,
                    'formation_potential': formation_potential,
                    'well_index': i
                }
                
                sparks.append(spark)
                
                # Mark this well as used
                self.potential_wells[i]['active'] = False
                
                # Add to spark formation points
                self.spark_formation_points.append(spark)
                
                logger.info(f"Soul spark formed at position {position} with potential {formation_potential:.4f}")
        
        return sparks
    
    def visualize_void_field(self, show_wells=True, show_sparks=True, slice_axis=2, 
                           slice_index=None, save_path=None):
        """
        Visualize the void field, showing potential wells and formed sparks.
        
        Args:
            show_wells (bool): Whether to show potential wells
            show_sparks (bool): Whether to show formed sparks
            slice_axis (int): Axis to slice (0=x, 1=y, 2=z)
            slice_index (int): Slice index (default: middle of the axis)
            save_path (str): Path to save the visualization
            
        Returns:
            bool: True if visualization was successful
        """
        # Delegate to base class visualization with additional elements
        result = self.visualize_field_slice(axis=slice_axis, index=slice_index, 
                                          show_peaks=False, save_path=None)
        
        # Add well markers if requested
        if show_wells and self.potential_wells:
            plt.figure(plt.gcf().number)  # Get current figure
            
            # Set default slice index to middle if not specified
            if slice_index is None:
                slice_index = self.dimensions[slice_axis] // 2
                
            # Filter wells that are in this slice (or close to it)
            visible_wells = []
            for well in self.potential_wells:
                pos = well['position']
                if abs(pos[slice_axis] - slice_index) <= 1:  # Within 1 unit of the slice
                    visible_wells.append(well)
            
            # Plot well markers
            for well in visible_wells:
                pos = well['position']
                quality = well['quality']
                active = well['active']
                
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
                    
                # Plot with color based on quality and marker based on active status
                marker = 'o' if active else 'x'
                plt.plot(marker_x, marker_y, marker, 
                       color=plt.cm.plasma(quality), 
                       markersize=10, markeredgewidth=2)
                
                # Add quality label
                plt.annotate(f"{quality:.2f}", (marker_x, marker_y),
                           xytext=(5, 5), textcoords='offset points')
        
        # Add spark markers if requested
        if show_sparks and self.spark_formation_points:
            plt.figure(plt.gcf().number)  # Get current figure
            
            # Filter sparks that are in this slice (or close to it)
            visible_sparks = []
            for spark in self.spark_formation_points:
                pos = spark['position']
                if abs(pos[slice_axis] - slice_index) <= 1:  # Within 1 unit of the slice
                    visible_sparks.append(spark)
            
            # Plot spark markers
            for spark in visible_sparks:
                pos = spark['position']
                potential = spark['formation_potential']
                
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
                    
                # Plot with star marker
                plt.plot(marker_x, marker_y, '*', 
                       color='yellow', 
                       markersize=15, markeredgewidth=2)
                
                # Add spark label
                plt.annotate(f"Spark: {potential:.2f}", (marker_x, marker_y),
                           xytext=(5, -15), textcoords='offset points', color='yellow')
        
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
        plt.title(f"Void Field - {['X', 'Y', 'Z'][slice_axis]}={slice_index} " +
                f"(Wells: {len(self.potential_wells)}, Sparks: {len(self.spark_formation_points)})")
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Void field visualization saved to {save_path}")
            
        return True
    
    def visualize_field_sound(self, duration=3.0, save_path=None):
        """
        Visualize the sound waveform for this void field.
        
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
    
    def get_void_metrics(self):
        """
        Get metrics about the void field state.
        
        Returns:
            dict: Dictionary of field metrics
        """
        # Get base field metrics
        base_metrics = self.calculate_stability_metrics()
        
        # Add void-specific metrics
        void_metrics = {
            'num_potential_wells': len(self.potential_wells),
            'average_well_quality': np.mean([w['quality'] for w in self.potential_wells]) if self.potential_wells else 0,
            'num_sparks_formed': len(self.spark_formation_points),
            'creator_resonance': self.creator_resonance,
            'primary_frequency': self.primary_void_frequency,
            'edge_of_chaos': self.edge_of_chaos_ratio,
            'field_dimensions': self.dimensions
        }
        
        # Add sound metrics if enabled
        if self.sound_enabled and hasattr(self, 'field_sound') and self.field_sound:
            sound_metrics = {
                'sound_harmonics': len(self.field_sound.harmonics),
                'sound_harmony': base_metrics.get('sound_harmony', 0),
                'primary_sound_frequency': self.field_sound.fundamental_frequency
            }
            void_metrics.update(sound_metrics)
        
        # Combine metrics
        metrics = {**base_metrics, **void_metrics}
        
        return metrics
    
    def save_void_data(self, output_dir="output", filename=None):
        """
        Save void field data including sound information.
        
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
            
        # Save additional void-specific data
        if filename is None:
            filename = f"void_specific_{self.field_id[:8]}.npz"
        else:
            # Add prefix to avoid name collision
            filename = f"void_specific_{filename}"
            
        void_save_path = os.path.join(output_dir, filename)
        
        # Compile void-specific data
        void_data = {
            'field_id': self.field_id,
            'potential_wells': [(w['position'], w['quality'], w['active']) for w in self.potential_wells],
            'spark_formation_points': [(s['position'], s['spark_id'], s['formation_potential']) 
                                     for s in self.spark_formation_points],
            'sacred_patterns': list(self.sacred_patterns.keys()),
            'primary_void_frequency': self.primary_void_frequency,
            'secondary_void_frequencies': self.secondary_void_frequencies,
            'creator_resonance': self.creator_resonance
        }
        
        # Save void-specific data
        try:
            np.savez_compressed(void_save_path, **void_data)
            logger.info(f"Void-specific data saved to {void_save_path}")
            return save_path  # Return the main file path
            
        except Exception as e:
            logger.error(f"Error saving void-specific data: {str(e)}")
            return save_path  # Still return the main file path
    
    @classmethod
    def load_void_data(cls, field_path, void_specific_path=None):
        """
        Load void field data including sound information.
        
        Args:
            field_path (str): Path to the main field data file
            void_specific_path (str): Optional path to void-specific data
            
        Returns:
            VoidField: Loaded void field
        """
        # First load base field data
        field = cls.load_field_data(field_path)
        
        if not field:
            return None
            
        # Now load void-specific data if available
        if void_specific_path and os.path.exists(void_specific_path):
            try:
                void_data = np.load(void_specific_path, allow_pickle=True)
                
                # Restore potential wells if available
                if 'potential_wells' in void_data:
                    field.potential_wells = []
                    for pos, quality, active in void_data['potential_wells']:
                        # Create basic well info (missing some fields, but has essentials)
                        field.potential_wells.append({
                            'position': tuple(pos),
                            'quality': float(quality),
                            'active': bool(active),
                            'energy': 0.0,  # Default
                            'stability': 0.0,  # Default
                            'resonance': 0.0,  # Default
                            'creator_alignment': 0.0  # Default
                        })
                        
                # Restore spark formation points if available
                if 'spark_formation_points' in void_data:
                    field.spark_formation_points = []
                    for pos, spark_id, potential in void_data['spark_formation_points']:
                        # Create basic spark info (missing some fields, but has essentials)
                        field.spark_formation_points.append({
                            'position': tuple(pos),
                            'spark_id': str(spark_id),
                            'formation_potential': float(potential),
                            'formation_time': 0,  # Default
                            'energy': 0.0,  # Default
                            'stability': 0.0,  # Default
                            'resonance': 0.0,  # Default
                            'creator_alignment': 0.0  # Default
                        })
                        
                # Restore other void-specific properties
                if 'primary_void_frequency' in void_data:
                    field.primary_void_frequency = float(void_data['primary_void_frequency'])
                    
                if 'secondary_void_frequencies' in void_data:
                    field.secondary_void_frequencies = list(void_data['secondary_void_frequencies'])
                    
                if 'creator_resonance' in void_data:
                    field.creator_resonance = float(void_data['creator_resonance'])
                    
                if 'sacred_patterns' in void_data:
                    field.sacred_patterns = {
                        pattern: {"pattern_object": pattern, "position": None, "strength": 1.0}
                        for pattern in void_data['sacred_patterns']
                    }
                    
                logger.info(f"Loaded void-specific data from {void_specific_path}")
                
            except Exception as e:
                logger.error(f"Error loading void-specific data: {str(e)}")
        
        return field
    
    def __str__(self):
        """String representation of the void field."""
        metrics = self.get_void_metrics()
        
        # Add sound info if enabled
        sound_info = ""
        if self.sound_enabled and hasattr(self, 'field_sound') and self.field_sound:
            sound_info = f"\nSound Harmonics: {len(self.field_sound.harmonics)}"
            sound_info += f"\nSound Harmony: {metrics.get('sound_harmony', 0):.4f}"
        
        return (f"Void Field (ID: {self.field_id[:8]})\n"
                f"Dimensions: {self.dimensions}\n"
                f"Edge of Chaos Ratio: {self.edge_of_chaos_ratio}\n"
                f"Creator Resonance: {self.creator_resonance}\n"
                f"Primary Frequency: {self.primary_void_frequency} Hz\n"
                f"Stability: {metrics['stability']:.4f}\n"
                f"Coherence: {metrics['coherence']:.4f}\n"
                f"Potential Wells: {metrics['num_potential_wells']}\n"
                f"Average Well Quality: {metrics['average_well_quality']:.4f}\n"
                f"Sparks Formed: {metrics['num_sparks_formed']}"
                f"{sound_info}")


if __name__ == "__main__":
    # Example usage
    void = VoidField(dimensions=(64, 64, 64), edge_of_chaos_ratio=0.618, creator_resonance=0.7)
    
    # Embed sacred geometry patterns
    void.embed_sacred_geometry()
    
    # Simulate quantum fluctuations
    sparks = void.simulate_quantum_fluctuations(iterations=20, fluctuation_strength=0.03)
    
    # Visualize field
    void.visualize_void_field(save_path="void_field_visualization.png")
    
    # Visualize sound if enabled
    if void.sound_enabled:
        void.visualize_field_sound(save_path="void_field_sound.png")
    
    # Print metrics
    print(void)
    
    # Save data
    void.save_void_data()  