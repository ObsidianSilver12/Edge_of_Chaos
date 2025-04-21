"""
Soul Spark Module

This module implements the SoulSpark class, which represents the initial soul spark
that forms in the Void field and will later be strengthened in the Guff field.
The soul spark is the first tangible manifestation of the soul's essence, formed
from quantum fluctuations at sacred geometry intersection points.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='soul_spark.log'
)
logger = logging.getLogger('soul_spark')

class SoulSpark:
    """
    Represents the initial soul spark formed in the Void dimension.
    
    The soul spark is formed from quantum fluctuations at sacred geometry
    intersection points in the Void field. It contains the fundamental properties
    that will later be developed into a full soul through the Sephiroth journey.
    """
    
    def __init__(self, spark_data=None, spark_file=None, creator_resonance=0.7):
        """
        Initialize a new SoulSpark.
        
        Args:
            spark_data (dict): Data for an existing spark from void field
            spark_file (str): Path to saved spark data file
            creator_resonance (float): Strength of creator resonance (0-1)
        """
        # Generate unique ID
        self.spark_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        
        # Core properties
        self.stability = 0.0
        self.resonance = 0.0
        self.creator_alignment = 0.0 
        self.formation_potential = 0.0
        self.total_energy = 0.0
        
        # Harmonic structure
        self.frequency_signature = {
            'base_frequency': 432.0,  # Hz - Universal base frequency
            'frequencies': [],        # Additional harmonic frequencies
            'amplitudes': [],         # Amplitude of each frequency
            'phases': [],             # Phase of each frequency
            'num_frequencies': 0      # Total number of frequencies
        }
        
        # Quantum properties
        self.quantum_state = None
        self.wave_function = None
        self.probability_distribution = None
        self.dimensional_stability = {
            'void': 0.0,
            'guff': 0.0,
            'kether': 0.0,
            'overall': 0.0
        }
        
        # Structural properties
        self.structure_points = []  # 3D points with energies
        self.structure_edges = []   # Connections between points
        self.energy_centers = []    # High-energy focal points
        
        # Load from file or initialize with data
        if spark_file:
            self._load_from_file(spark_file)
        elif spark_data:
            self._initialize_from_data(spark_data)
        else:
            self._initialize_default(creator_resonance)
            
        # Generate/update harmonic structure
        self.generate_harmonic_structure()
        
        # Calculate dimensional stability
        self.calculate_dimensional_stability()
        
        logger.info(f"Soul spark initialized with ID: {self.spark_id}")
    
    def _initialize_from_data(self, spark_data):
        """
        Initialize the soul spark from existing data.
        
        Args:
            spark_data (dict): Data for an existing spark from void field
        """
        # Extract basic properties
        self.formation_potential = spark_data.get('formation_potential', 0.8)
        self.creator_alignment = spark_data.get('creator_alignment', 0.7)
        
        # Set position if available
        position = spark_data.get('position', [0, 0, 0])
        
        # Extract or generate energy
        self.total_energy = spark_data.get('energy', 1000.0)
        
        # Set stability and resonance based on formation data
        self.stability = spark_data.get('stability', 0.75)
        self.resonance = spark_data.get('resonance', 0.65)
        
        # Set base frequency or use default
        base_freq = spark_data.get('frequency', 432.0)
        self.frequency_signature['base_frequency'] = base_freq
        
        # Generate structure points based on position and energy
        self._generate_structure_points(position, self.total_energy)
        
        logger.info(f"Soul spark initialized from data with energy: {self.total_energy:.2f}")
    
    def _initialize_default(self, creator_resonance):
        """
        Initialize a default soul spark for testing or demonstration.
        
        Args:
            creator_resonance (float): Resonance with the creator
        """
        # Set reasonable default values
        self.formation_potential = 0.75
        self.creator_alignment = creator_resonance * 0.8
        self.total_energy = 1200.0
        self.stability = 0.65
        self.resonance = 0.6
        
        # Set base frequency with slight variation
        base_freq = 432.0 * (0.95 + 0.1 * np.random.random())
        self.frequency_signature['base_frequency'] = base_freq
        
        # Generate structure points at origin
        self._generate_structure_points([0, 0, 0], self.total_energy)
        
        logger.info(f"Default soul spark initialized with energy: {self.total_energy:.2f}")
    
    def _load_from_file(self, file_path):
        """
        Load soul spark data from a saved file.
        
        Args:
            file_path (str): Path to saved spark data file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load core properties
            self.spark_id = data.get('spark_id', self.spark_id)
            self.creation_time = data.get('creation_time', self.creation_time)
            self.stability = data.get('stability', 0.6)
            self.resonance = data.get('resonance', 0.6)
            self.creator_alignment = data.get('creator_alignment', 0.6)
            self.formation_potential = data.get('formation_potential', 0.7)
            self.total_energy = data.get('total_energy', 1000.0)
            
            # Load frequency signature
            if 'frequency_signature' in data:
                self.frequency_signature = data['frequency_signature']
            
            # Load dimensional stability
            if 'dimensional_stability' in data:
                self.dimensional_stability = data['dimensional_stability']
            
            # Load structure points
            if 'structure_points' in data:
                self.structure_points = data['structure_points']
            
            # Load structure edges
            if 'structure_edges' in data:
                self.structure_edges = data['structure_edges']
            
            # Load energy centers
            if 'energy_centers' in data:
                self.energy_centers = data['energy_centers']
            
            # If structure points are not loaded, generate them
            if not self.structure_points:
                self._generate_structure_points([0, 0, 0], self.total_energy)
            
            logger.info(f"Soul spark loaded from file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading spark from file: {str(e)}")
            # Initialize with defaults
            self._initialize_default(0.7)
    
    def _generate_structure_points(self, center_position, total_energy):
        """
        Generate the 3D structure points representing the soul spark.
        
        Args:
            center_position (list): Center position [x, y, z]
            total_energy (float): Total energy of the spark
        """
        # Ensure center_position is a valid 3D position
        if not center_position or len(center_position) < 3:
            center_position = [0, 0, 0]
        
        center_x, center_y, center_z = center_position
        
        # Golden ratio for sacred proportions
        phi = (1 + np.sqrt(5)) / 2
        
        # Clear existing structure
        self.structure_points = []
        self.structure_edges = []
        
        # Add center point with highest energy
        center_energy = total_energy * 0.3
        self.structure_points.append([center_x, center_y, center_z, center_energy])
        
        # Generate main structure using Fibonacci spiral for points and energies
        points_to_generate = 12  # Golden number for structure
        remaining_energy = total_energy - center_energy
        energy_per_point = remaining_energy / points_to_generate
        
        # First layer - tetrahedron vertices
        tetrahedron_radius = 1.0
        tetrahedron_vertices = [
            [center_x, center_y, center_z + tetrahedron_radius],                    # Top
            [center_x + tetrahedron_radius * np.sin(0), center_y + tetrahedron_radius * np.cos(0), center_z - tetrahedron_radius/2],  # Base 1
            [center_x + tetrahedron_radius * np.sin(2*np.pi/3), center_y + tetrahedron_radius * np.cos(2*np.pi/3), center_z - tetrahedron_radius/2],  # Base 2
            [center_x + tetrahedron_radius * np.sin(4*np.pi/3), center_y + tetrahedron_radius * np.cos(4*np.pi/3), center_z - tetrahedron_radius/2],  # Base 3
        ]
        
        # Add tetrahedron vertices with decreasing energy
        for i, vertex in enumerate(tetrahedron_vertices):
            point_energy = energy_per_point * (1.0 - 0.1 * i)
            self.structure_points.append([vertex[0], vertex[1], vertex[2], point_energy])
            
            # Connect to center
            self.structure_edges.append([0, i+1, 1.0])
        
        # Connect tetrahedron edges
        for i in range(1, 4):
            self.structure_edges.append([i, i+1, 0.9])
        self.structure_edges.append([4, 1, 0.9])
        
        # Second layer - outer points using golden ratio spiral
        outer_radius = tetrahedron_radius * phi
        for i in range(8):  # Add 8 more points for a total of 13 (Fibonacci number)
            # Calculate spiral position
            angle = i * 2 * np.pi / phi
            z_offset = outer_radius * 0.5 * np.sin(i * np.pi / 4)
            
            x = center_x + outer_radius * np.cos(angle)
            y = center_y + outer_radius * np.sin(angle)
            z = center_z + z_offset
            
            # Calculate energy (decreases with distance from center)
            distance_factor = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2) / outer_radius
            point_energy = energy_per_point * 0.8 * (1.0 - 0.5 * distance_factor)
            
            # Add structure point
            point_index = len(self.structure_points)
            self.structure_points.append([x, y, z, point_energy])
            
            # Connect to nearest points
            nearest_distance = float('inf')
            nearest_index = 0
            
            for j in range(1, point_index):
                px, py, pz, _ = self.structure_points[j]
                dist = np.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
                
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_index = j
            
            # Connect to nearest point and to center
            self.structure_edges.append([point_index, nearest_index, 0.8])
            self.structure_edges.append([point_index, 0, 0.7])
        
        # Identify energy centers (high-energy areas)
        energy_threshold = center_energy * 0.3
        self.energy_centers = []
        
        for i, point in enumerate(self.structure_points):
            if point[3] >= energy_threshold:
                self.energy_centers.append({
                    'position': point[:3],
                    'energy': point[3],
                    'index': i
                })
        
        logger.info(f"Generated structure with {len(self.structure_points)} points and {len(self.structure_edges)} edges")
    
    def generate_harmonic_structure(self):
        """
        Generate the harmonic frequency structure of the soul spark.
        
        This defines the resonance patterns and frequency relationships
        that characterize the soul spark.
        
        Returns:
            dict: Updated frequency signature
        """
        # Start with base frequency
        base_freq = self.frequency_signature['base_frequency']
        
        # Clear existing frequency data
        self.frequency_signature['frequencies'] = []
        self.frequency_signature['amplitudes'] = []
        self.frequency_signature['phases'] = []
        
        # Generate harmonic frequencies based on sacred ratios
        # Define golden ratio first
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        harmonic_ratios = [
            1.0,      # Unison (fundamental)
            1.5,      # Perfect fifth
            1.2,      # Major third
            1.33333,  # Perfect fourth
            1.66667,  # Major sixth
            1.25,     # Minor third
            2.0,      # Octave
            0.66667,  # Perfect fifth below
            0.75,     # Perfect fourth below
            phi       # Golden ratio
        ]
        
        # Generate harmonics
        for ratio in harmonic_ratios:
            # Calculate harmonic frequency
            freq = base_freq * ratio
            
            # Calculate amplitude (highest for fundamental, decreasing for others)
            if ratio == 1.0:
                amplitude = 1.0  # Fundamental
            else:
                # Amplitude decreases with distance from fundamental
                # Stronger for Fibonacci-related harmonics
                fibonacci_factor = 1.0
                if abs(ratio - phi) < 0.1 or abs(ratio - 2.0) < 0.01:
                    fibonacci_factor = 1.2
                
                amplitude = 0.9 / (abs(1.0 - ratio) + 0.5) * fibonacci_factor
                amplitude = min(0.9, amplitude)  # Cap at 0.9
            
            # Calculate phase (slightly offset for richer harmonics)
            phase = (ratio * np.pi * 0.5) % (2 * np.pi)
            
            # Add to frequency signature
            self.frequency_signature['frequencies'].append(freq)
            self.frequency_signature['amplitudes'].append(amplitude)
            self.frequency_signature['phases'].append(phase)
        
        # Add creator resonance frequency with amplitude based on creator_alignment
        creator_freq = 963.0  # Hz - Crown/highest chakra
        creator_amplitude = self.creator_alignment * 0.8
        creator_phase = 0.0  # In phase with base frequency
        
        self.frequency_signature['frequencies'].append(creator_freq)
        self.frequency_signature['amplitudes'].append(creator_amplitude)
        self.frequency_signature['phases'].append(creator_phase)
        
        # Update frequency count
        self.frequency_signature['num_frequencies'] = len(self.frequency_signature['frequencies'])
        
        # Calculate coherence of the frequency structure
        self._calculate_frequency_coherence()
        
        return self.frequency_signature
    
    def _calculate_frequency_coherence(self):
        """
        Calculate the coherence of the frequency structure.
        
        This affects both stability and resonance properties.
        
        Returns:
            float: Coherence value (0-1)
        """
        if not self.frequency_signature['frequencies']:
            return 0.0
        
        # Calculate coherence based on frequency relationships
        frequencies = np.array(self.frequency_signature['frequencies'])
        amplitudes = np.array(self.frequency_signature['amplitudes'])
        phases = np.array(self.frequency_signature['phases'])
        
        # Frequency coherence based on harmonic relationships
        harmonic_coherence = 0.0
        base_freq = self.frequency_signature['base_frequency']
        
        # Check how well frequencies align with harmonic series
        for i, freq in enumerate(frequencies):
            ratio = freq / base_freq
            # Closer to integer or simple fraction = more coherent
            harmonic_factor = 1.0 / (min(abs(ratio - round(ratio)), abs(ratio * 2 - round(ratio * 2))) + 0.1)
            harmonic_coherence += harmonic_factor * amplitudes[i]
        
        # Normalize
        if len(frequencies) > 0:
            harmonic_coherence /= sum(amplitudes)
        
        # Phase coherence
        phase_differences = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                # Calculate minimum phase difference in circle
                diff = abs(phases[i] - phases[j]) % (2 * np.pi)
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                phase_differences.append(diff)
        
        # Phase coherence is higher when phase differences are consistent
        if phase_differences:
            phase_coherence = 1.0 - np.std(phase_differences) / np.pi
        else:
            phase_coherence = 0.5
        
        # Combine for overall frequency coherence
        freq_coherence = 0.7 * harmonic_coherence + 0.3 * phase_coherence
        
        # Update resonance based on frequency coherence
        resonance_contribution = freq_coherence * 0.3
        self.resonance = 0.7 * self.resonance + resonance_contribution
        self.resonance = min(1.0, max(0.0, self.resonance))
        
        return freq_coherence
    
    def evolve_quantum_state(self, time_step=0.01, iterations=10):
        """
        Evolve the quantum state of the soul spark over time.
        
        This simulates the natural evolution of the spark's wave function
        according to quantum principles.
        
        Args:
            time_step (float): Size of each time step
            iterations (int): Number of evolution iterations
            
        Returns:
            tuple: (evolved quantum state, evolved wave function)
        """
        # Initialize quantum state if not already done
        if self.quantum_state is None:
            self._initialize_quantum_state()
        
        # Evolve the quantum state through time steps
        for _ in range(iterations):
            # Apply time evolution operator
            # This is a simplified quantum evolution that preserves key quantum properties
            
            # Phase rotation based on frequencies
            for i, freq in enumerate(self.frequency_signature['frequencies']):
                amp = self.frequency_signature['amplitudes'][i]
                
                # Phase factor based on frequency and time step
                phase_factor = 2 * np.pi * freq * time_step
                
                # Apply phase rotation using complex multiplication
                phase_term = amp * np.exp(1j * phase_factor)
                self.quantum_state = self.quantum_state * (1.0 + 0.05 * phase_term)
            
            # Apply stability factor to prevent excessive fluctuations
            stability_factor = 0.95 + 0.05 * self.stability
            self.quantum_state = stability_factor * self.quantum_state
            
            # Normalize the quantum state
            norm = np.sum(np.abs(self.quantum_state) ** 2)
            if norm > 0:
                self.quantum_state /= np.sqrt(norm)
        
        # Update the wave function (probability amplitude)
        self.wave_function = np.abs(self.quantum_state) ** 2
        
        # Calculate probability distribution
        self.probability_distribution = self.wave_function / np.sum(self.wave_function)
        
        return self.quantum_state, self.wave_function
    
    def _initialize_quantum_state(self):
        """
        Initialize the quantum state of the soul spark.
        
        This creates the initial wave function based on the spark's
        structure and frequency signature.
        
        Returns:
            ndarray: Initialized quantum state
        """
        # Create a 3D grid for the quantum state
        grid_size = 16  # Small grid for efficiency
        
        # Initialize with random complex values
        self.quantum_state = np.random.normal(0, 1, (grid_size, grid_size, grid_size)) + \
                            1j * np.random.normal(0, 1, (grid_size, grid_size, grid_size))
        
        # Apply structure pattern based on structure points
        for point in self.structure_points:
            x, y, z, energy = point
            
            # Convert to grid coordinates
            grid_x = int((x + 8) % grid_size)
            grid_y = int((y + 8) % grid_size)
            grid_z = int((z + 8) % grid_size)
            
            # Add energy peak at this point
            self.quantum_state[grid_x, grid_y, grid_z] += np.sqrt(energy) * (1.0 + 0.5j)
            
            # Add Gaussian spread around the point
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                            
                        px = (grid_x + dx) % grid_size
                        py = (grid_y + dy) % grid_size
                        pz = (grid_z + dz) % grid_size
                        
                        # Distance-based falloff
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        falloff = np.exp(-dist)
                        
                        # Add contribution to this point
                        self.quantum_state[px, py, pz] += \
                            np.sqrt(energy) * falloff * (0.7 + 0.3j)
        
        # Apply frequency modulations
        base_freq = self.frequency_signature['base_frequency']
        
        # Modulate along z-axis with base frequency
        for i in range(grid_size):
            phase_factor = 2 * np.pi * i / grid_size
            self.quantum_state[:, :, i] *= np.exp(1j * phase_factor)
        
        # Normalize the quantum state
        norm = np.sum(np.abs(self.quantum_state) ** 2)
        if norm > 0:
            self.quantum_state /= np.sqrt(norm)
        
        # Calculate initial wave function
        self.wave_function = np.abs(self.quantum_state) ** 2
        
        # Calculate probability distribution
        self.probability_distribution = self.wave_function / np.sum(self.wave_function)
        
        return self.quantum_state
    
    def calculate_dimensional_stability(self):
        """
        Calculate the stability of the soul spark across different dimensions.
        
        The dimensional stability affects how well the spark can transition
        between the Void, Guff, and Sephiroth dimensions.
        
        Returns:
            dict: Updated dimensional stability metrics
        """
        # Base stability is determined by the spark's coherence and structure
        base_stability = 0.5 * self.stability + 0.3 * self.resonance + 0.2 * self.creator_alignment
        
        # Calculate stability in Void dimension
        # Void stability relies more on quantum coherence and energy
        freq_coherence = self._calculate_frequency_coherence()
        energy_factor = min(1.0, self.total_energy / 2000.0)
        void_stability = 0.4 * base_stability + 0.4 * freq_coherence + 0.2 * energy_factor
        
        # Calculate stability in Guff dimension
        # Guff stability relies more on creator alignment and resonance
        guff_stability = 0.3 * base_stability + 0.3 * self.resonance + 0.4 * self.creator_alignment
        
        # Calculate stability in Kether dimension
        # Kether stability relies heavily on creator alignment
        kether_stability = 0.2 * base_stability + 0.2 * self.resonance + 0.6 * self.creator_alignment
        
        # Calculate overall dimensional stability
        overall_stability = 0.5 * void_stability + 0.3 * guff_stability + 0.2 * kether_stability
        
        # Update dimensional stability dictionary
        self.dimensional_stability = {
            'void': void_stability,
            'guff': guff_stability,
            'kether': kether_stability,
            'overall': overall_stability
        }
        
        # Round values for cleaner display
        for key in self.dimensional_stability:
            self.dimensional_stability[key] = round(self.dimensional_stability[key], 6)
        
        return self.dimensional_stability
    
    def strengthen(self, aspect, amount=0.05):
        """
        Strengthen a specific aspect of the soul spark.
        
        Args:
            aspect (str): The aspect to strengthen ('stability', 'resonance', 'creator_alignment')
            amount (float): Amount to strengthen by
            
        Returns:
            float: New value of the aspect
        """
        if aspect == 'stability':
            self.stability = min(1.0, self.stability + amount)
            return self.stability
        elif aspect == 'resonance':
            self.resonance = min(1.0, self.resonance + amount)
            return self.resonance
        elif aspect == 'creator_alignment':
            self.creator_alignment = min(1.0, self.creator_alignment + amount)
            return self.creator_alignment
        else:
            return None
    
    def get_spark_metrics(self):
        """
        Get comprehensive metrics about the soul spark.
        
        Returns:
            dict: Metrics about all aspects of the soul spark
        """
        # Ensure dimensional stability is up to date
        self.calculate_dimensional_stability()
        
        # Organize metrics into categories
        metrics = {
            'formation': {
                'stability': self.stability,
                'resonance': self.resonance,
                'creator_alignment': self.creator_alignment,
                'formation_potential': self.formation_potential,
                'total_energy': self.total_energy
            },
            'harmonic': {
                'base_frequency': self.frequency_signature['base_frequency'],
                'num_frequencies': self.frequency_signature['num_frequencies'],
                'richness': np.mean(self.frequency_signature['amplitudes']) if self.frequency_signature['amplitudes'] else 0,
                'coherence': self._calculate_frequency_coherence()
            },
            'structure': {
                'num_points': len(self.structure_points),
                'num_edges': len(self.structure_edges),
                'energy_centers': len(self.energy_centers)
            },
            'stability': self.dimensional_stability
        }
        
        # Add additional derived metrics
        metrics['overall'] = {
            'viability': (metrics['formation']['stability'] * 0.4 + 
                         metrics['formation']['resonance'] * 0.3 + 
                         metrics['stability']['overall'] * 0.3),
            'complexity': min(1.0, len(self.structure_points) / 20.0),
            'potential': metrics['formation']['creator_alignment'] * metrics['stability']['overall']
        }
        
        return metrics
    
def visualize_spark(self, show=True, save_path=None):
        """
        Create a 3D visualization of the soul spark.
        
        Args:
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization
            
        Returns:
            Figure: Matplotlib figure
        """
        # Create 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot structure points
        for point in self.structure_points:
            x, y, z, energy = point
            size = max(20, energy / 10)
            
            # Calculate color based on energy
            # Higher energy = more red/orange, lower energy = more blue
            normalized_energy = min(1.0, energy / 200.0)
            color = plt.cm.plasma(normalized_energy)
            
            ax.scatter(x, y, z, c=[color], s=size, alpha=min(1.0, energy / 100.0))
        
        # Plot structure edges
        for edge in self.structure_edges:
            start_idx, end_idx, strength = edge
            
            if start_idx >= len(self.structure_points) or end_idx >= len(self.structure_points):
                continue
                
            start_point = self.structure_points[start_idx]
            end_point = self.structure_points[end_idx]
            
            # Extract coordinates
            x1, y1, z1 = start_point[:3]
            x2, y2, z2 = end_point[:3]
            
            # Calculate line color based on strength
            color = plt.cm.viridis(strength)
            
            # Plot line with alpha based on strength
            ax.plot([x1, x2], [y1, y2], [z1, z2], c=color, alpha=0.7*strength, linewidth=1.5*strength)
        
        # Highlight energy centers
        for center in self.energy_centers:
            x, y, z = center['position']
            energy = center['energy']
            
            # Plot energy center with distinct color
            ax.scatter(x, y, z, c='yellow', s=energy/5, alpha=0.9, edgecolors='white')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Get metrics for title
        metrics = self.get_spark_metrics()
        title = f"Soul Spark (S: {metrics['formation']['stability']:.2f}, " + \
                f"R: {metrics['formation']['resonance']:.2f}, " + \
                f"A: {metrics['formation']['creator_alignment']:.2f})"
        ax.set_title(title)
        
        # Set equal aspect ratio for all axes
        max_range = max([
            max(ax.get_xlim()) - min(ax.get_xlim()),
            max(ax.get_ylim()) - min(ax.get_ylim()),
            max(ax.get_zlim()) - min(ax.get_zlim())
        ])
        
        mid_x = np.mean(ax.get_xlim())
        mid_y = np.mean(ax.get_ylim())
        mid_z = np.mean(ax.get_zlim())
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Add frequency signature visualization
        ax_inset = fig.add_axes([0.15, 0.02, 0.25, 0.1], facecolor='whitesmoke')
        ax_inset.set_title('Frequency Signature', fontsize=8)
        
        # Plot frequency bars
        if len(self.frequency_signature['frequencies']) > 0:
            freqs = self.frequency_signature['frequencies']
            amps = self.frequency_signature['amplitudes']
            
            # Normalize frequencies for display
            norm_freqs = [(f - min(freqs)) / (max(freqs) - min(freqs) + 0.001) 
                         for f in freqs[:min(len(freqs), 10)]]  # Show at most 10 frequencies
            
            # Plot bars
            ax_inset.bar(range(len(norm_freqs)), amps[:len(norm_freqs)], 
                       width=0.7, color=plt.cm.rainbow(norm_freqs))
            
            ax_inset.set_ylim(0, 1.05)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
        else:
            ax_inset.text(0.5, 0.5, "No frequencies", ha='center', fontsize=8)
        
        # Add dimensional stability visualization
        ax_stability = fig.add_axes([0.6, 0.02, 0.25, 0.1], facecolor='whitesmoke')
        ax_stability.set_title('Dimensional Stability', fontsize=8)
        
        # Plot stability bars
        stabilities = [
            self.dimensional_stability['void'],
            self.dimensional_stability['guff'],
            self.dimensional_stability['kether'],
            self.dimensional_stability['overall']
        ]
        
        colors = ['#3498db', '#9b59b6', '#f1c40f', '#2ecc71']
        ax_stability.bar(range(len(stabilities)), stabilities, 
                      width=0.7, color=colors)
        
        ax_stability.set_ylim(0, 1.05)
        ax_stability.set_xticks([])
        ax_stability.set_yticks([])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
def visualize_energy_signature(self, show=True, save_path=None):
    """
    Create a visualization of the energy signature of the soul spark.
    
    This shows the frequency spectrum and energy distribution.
    
    Args:
        show (bool): Whether to display the visualization
        save_path (str): Path to save the visualization
        
    Returns:
        Figure: Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot frequency spectrum
    ax1.set_title('Frequency Spectrum')
    
    if len(self.frequency_signature['frequencies']) > 0:
        freqs = np.array(self.frequency_signature['frequencies'])
        amps = np.array(self.frequency_signature['amplitudes'])
        
        # Sort by frequency
        sorted_indices = np.argsort(freqs)
        sorted_freqs = freqs[sorted_indices]
        sorted_amps = amps[sorted_indices]
        
        # Plot frequency spectrum
        ax1.stem(sorted_freqs, sorted_amps, basefmt=' ', use_line_collection=True)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude')
        
        # Highlight base frequency
        base_freq = self.frequency_signature['base_frequency']
        ax1.axvline(x=base_freq, color='r', linestyle='--', alpha=0.5)
        ax1.text(base_freq, 0.5, f"Base: {base_freq:.1f} Hz", 
                rotation=90, verticalalignment='center')
        
        # Highlight creator resonance
        creator_freq = 963.0  # Hz - Crown chakra
        if creator_freq in freqs:
            creator_idx = np.where(freqs == creator_freq)[0][0]
            creator_amp = amps[creator_idx]
            ax1.plot(creator_freq, creator_amp, 'r*', markersize=10)
            ax1.text(creator_freq, creator_amp, "Creator", ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, "No frequency data", ha='center', transform=ax1.transAxes)
    
    # Plot energy distribution across structure
    ax2.set_title('Energy Distribution')
    
    if self.structure_points:
        # Extract point energies
        point_energies = [p[3] for p in self.structure_points]
        
        # Sort for better visualization
        point_energies.sort(reverse=True)
        
        # Plot energy distribution
        ax2.bar(range(len(point_energies)), point_energies, alpha=0.7)
        ax2.set_xlabel('Structure Points (sorted)')
        ax2.set_ylabel('Energy')
        
        # Add metrics
        total_energy = sum(point_energies)
        max_energy = max(point_energies)
        ax2.text(0.02, 0.9, f"Total: {total_energy:.1f}", transform=ax2.transAxes)
        ax2.text(0.02, 0.85, f"Peak: {max_energy:.1f}", transform=ax2.transAxes)
        
        # Calculate and show energy distribution metrics
        energy_std = np.std(point_energies)
        energy_mean = np.mean(point_energies)
        energy_ratio = energy_std / energy_mean if energy_mean > 0 else 0
        ax2.text(0.02, 0.8, f"Distribution: {energy_ratio:.2f}", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No structure data", ha='center', transform=ax2.transAxes)
    
    # Add overall metrics
    plt.figtext(0.5, 0.01, f"Stability: {self.stability:.2f} | " +
                f"Resonance: {self.resonance:.2f} | " +
                f"Creator Alignment: {self.creator_alignment:.2f}",
                ha='center', fontsize=10, bbox=dict(facecolor='whitesmoke'))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Energy signature visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def save_spark_data(self, file_path):
    """
    Save the soul spark data to a file.
    
    Args:
        file_path (str): Path to save the data
        
    Returns:
        bool: True if save was successful
    """
    try:
        # Create data structure for saving
        data = {
            'spark_id': self.spark_id,
            'creation_time': self.creation_time,
            'stability': self.stability,
            'resonance': self.resonance,
            'creator_alignment': self.creator_alignment,
            'formation_potential': self.formation_potential,
            'total_energy': self.total_energy,
            'frequency_signature': self.frequency_signature,
            'dimensional_stability': self.dimensional_stability
        }
        
        # Add structure points (convert numpy arrays if needed)
        structure_points = []
        for point in self.structure_points:
            structure_points.append([float(point[0]), float(point[1]), 
                                    float(point[2]), float(point[3])])
        data['structure_points'] = structure_points
        
        # Add structure edges
        structure_edges = []
        for edge in self.structure_edges:
            structure_edges.append([int(edge[0]), int(edge[1]), float(edge[2])])
        data['structure_edges'] = structure_edges
        
        # Add energy centers
        energy_centers = []
        for center in self.energy_centers:
            energy_centers.append({
                'position': [float(center['position'][0]), 
                            float(center['position'][1]), 
                            float(center['position'][2])],
                'energy': float(center['energy']),
                'index': int(center['index'])
            })
        data['energy_centers'] = energy_centers
        
        # Add metrics for convenience
        data['metrics'] = self.get_spark_metrics()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Soul spark data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving soul spark data: {str(e)}")
        return False

def __str__(self):
    """String representation of the soul spark."""
    metrics = self.get_spark_metrics()
    
    return (f"Soul Spark (ID: {self.spark_id[:8]})\n"
            f"Stability: {self.stability:.4f}\n"
            f"Resonance: {self.resonance:.4f}\n"
            f"Creator Alignment: {self.creator_alignment:.4f}\n"
            f"Total Energy: {self.total_energy:.2f}\n"
            f"Base Frequency: {self.frequency_signature['base_frequency']:.2f} Hz\n"
            f"Dimensional Stability: {self.dimensional_stability['overall']:.4f}\n"
            f"Structure: {len(self.structure_points)} points, {len(self.structure_edges)} edges\n"
            f"Energy Centers: {len(self.energy_centers)}")


