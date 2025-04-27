"""
brain_seed.py - Core module for Brain Seed definition and structure.

This module defines the BrainSeed class which creates the foundational
structure for brain development with strong energetic processes.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any

# Import constants when available
try:
    from soul.constants import (
        SACRED_GEOMETRY,
        PLATONIC_SOLIDS,
        FIBONACCI_SEQUENCE,
        GOLDEN_RATIO,
        BRAIN_FREQUENCIES
    )
except ImportError:
    # Default constants if module not available
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    BRAIN_FREQUENCIES = {
        'delta': (0.5, 4),      # Deep sleep
        'theta': (4, 8),        # Drowsy, meditation
        'alpha': (8, 13),       # Relaxed, calm
        'beta': (13, 30),       # Alert, active
        'gamma': (30, 100),     # High cognition
        'lambda': (100, 400)    # Higher spiritual states
    }
    SACRED_GEOMETRY = {
        'vesica_piscis': {'complexity': 2, 'dimensions': 2},
        'seed_of_life': {'complexity': 7, 'dimensions': 2},
        'flower_of_life': {'complexity': 19, 'dimensions': 2},
        'tree_of_life': {'complexity': 10, 'dimensions': 2},
        'metatrons_cube': {'complexity': 13, 'dimensions': 3},
        'sri_yantra': {'complexity': 9, 'dimensions': 2},
        'torus': {'complexity': 1, 'dimensions': 3}
    }
    PLATONIC_SOLIDS = {
        'tetrahedron': {'vertices': 4, 'faces': 4, 'element': 'fire'},
        'hexahedron': {'vertices': 8, 'faces': 6, 'element': 'earth'},
        'octahedron': {'vertices': 6, 'faces': 8, 'element': 'air'},
        'dodecahedron': {'vertices': 20, 'faces': 12, 'element': 'aether'},
        'icosahedron': {'vertices': 12, 'faces': 20, 'element': 'water'}
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainSeed')

class BrainSeed:
    """
    Represents the brain seed with natural energy generators.
    
    The brain seed is the energetic core that will develop into
    the complete brain structure with hemispheres and regions.
    """
    
    def __init__(self, resonant_soul, complexity=9):
        """
        Initialize the brain seed structure.
        
        Parameters:
            resonant_soul: The earth resonant soul
            complexity (int): Complexity level of the brain seed (1-10)
        """
        self.resonant_soul = resonant_soul
        self.complexity = min(10, max(1, complexity))  # Ensure 1-10 range
        
        # Energy metrics
        self.base_energy_level = 0.0
        self.energy_capacity = 0.0
        self.energy_generation_rate = 0.0
        self.energy_stability = 0.0
        
        # Structure components
        self.seed_core = {}
        self.energy_generators = []
        self.resonance_patterns = {}
        self.frequency_map = {}
        self.sacred_geometry = {}
        self.platonic_structures = {}
        
        # Hemisphere foundation
        self.hemisphere_structure = {
            "left": {"developed": False, "complexity": 0, "energy": 0.0},
            "right": {"developed": False, "complexity": 0, "energy": 0.0}
        }
        
        # Region placeholders
        self.region_structure = {}
        
        # Formation metrics
        self.formation_progress = 0.0
        self.structural_integrity = 0.0
        self.resonance_coherence = 0.0
        self.stability = 0.0
        
        # Randomized components for uniqueness
        self.uniqueness_factor = np.random.random()
        self.variation_pattern = self._generate_variation_pattern()
        
        # Initialize the seed
        self._initialize_seed_core()
        
    def _initialize_seed_core(self):
        """Initialize the core structure of the brain seed."""
        logger.info("Initializing brain seed core structure")
        
        # Set up base energy level based on complexity
        self.base_energy_level = 30 + (self.complexity * 10)
        
        # Set up energy capacity
        self.energy_capacity = self.base_energy_level * 3
        
        # Set up energy generation rate
        self.energy_generation_rate = self.complexity * 0.4
        
        # Initialize resonance patterns
        self._initialize_resonance_patterns()
        
        # Create seed core
        self.seed_core = {
            'position': np.array([0, 0, 0]),  # Center position
            'radius': 0.1 + (0.02 * self.complexity),  # Seed core size
            'energy_density': self.base_energy_level / (4/3 * np.pi * (0.1 + (0.02 * self.complexity))**3),
            'frequency': 7.83,  # Start with Earth's Schumann resonance
            'stability': 0.3 + (0.05 * self.complexity),  # Initial stability
            'growth_potential': 0.5 + (0.05 * self.complexity)  # Growth potential
        }
        
        # Create energy generators
        self._create_energy_generators()
        
        # Initialize sacred geometry patterns
        self._initialize_sacred_geometry()
        
        # Initialize platonic structures
        self._initialize_platonic_structures()
        
        # Calculate initial formation metrics
        self.formation_progress = 0.1  # Just started
        self.structural_integrity = self.seed_core['stability']
        self.resonance_coherence = 0.3 + (0.05 * self.complexity)
        self.stability = self.seed_core['stability']
        
        logger.info(f"Brain seed core initialized with energy level {self.base_energy_level}")
    
    def _generate_variation_pattern(self):
        """Generate a unique variation pattern for this brain seed."""
        # Create a unique pattern based on randomness that will affect development
        pattern = {}
        
        # Random growth tendencies
        pattern['growth_bias'] = np.random.random()  # 0-1, affects growth direction preference
        
        # Random energy distribution
        pattern['energy_distribution'] = []
        for _ in range(7):  # 7 major energy distribution points
            pattern['energy_distribution'].append(np.random.random())
        
        # Random frequency preference
        pattern['frequency_preference'] = {}
        for band in BRAIN_FREQUENCIES:
            pattern['frequency_preference'][band] = 0.3 + (0.7 * np.random.random())
        
        # Random geometric preference
        pattern['geometry_preference'] = {}
        for geom in SACRED_GEOMETRY:
            pattern['geometry_preference'][geom] = np.random.random()
        
        # Random platonic solid preference
        pattern['platonic_preference'] = {}
        for solid in PLATONIC_SOLIDS:
            pattern['platonic_preference'][solid] = np.random.random()
        
        return pattern
    
    def _initialize_resonance_patterns(self):
        """Initialize the resonance patterns for the brain seed."""
        # Create resonance patterns for different brain states
        self.resonance_patterns = {
            'resting': {
                'primary_frequency': self._generate_brain_frequency('alpha'),
                'secondary_frequencies': [
                    self._generate_brain_frequency('theta'),
                    self._generate_brain_frequency('delta')
                ],
                'amplitude': 0.7,
                'coherence': 0.6
            },
            'active': {
                'primary_frequency': self._generate_brain_frequency('beta'),
                'secondary_frequencies': [
                    self._generate_brain_frequency('alpha'),
                    self._generate_brain_frequency('gamma')
                ],
                'amplitude': 0.9,
                'coherence': 0.7
            },
            'meditative': {
                'primary_frequency': self._generate_brain_frequency('theta'),
                'secondary_frequencies': [
                    self._generate_brain_frequency('alpha'),
                    self._generate_brain_frequency('delta')
                ],
                'amplitude': 0.5,
                'coherence': 0.9
            },
            'dream': {
                'primary_frequency': self._generate_brain_frequency('theta'),
                'secondary_frequencies': [
                    self._generate_brain_frequency('delta'),
                    self._generate_brain_frequency('gamma', low_power=True)
                ],
                'amplitude': 0.6,
                'coherence': 0.8
            },
            'liminal': {
                'primary_frequency': self._generate_brain_frequency('gamma'),
                'secondary_frequencies': [
                    self._generate_brain_frequency('theta'),
                    self._generate_brain_frequency('lambda', low_power=True)
                ],
                'amplitude': 0.4,
                'coherence': 0.95
            }
        }
        
        # Create a frequency map for the brain seed
        self._create_frequency_map()
    
    def _generate_brain_frequency(self, band, low_power=False):
        """Generate a specific frequency within a brain wave band."""
        low, high = BRAIN_FREQUENCIES[band]
        
        # Add some variation for uniqueness
        preferred_position = self.variation_pattern['frequency_preference'].get(band, 0.5)
        
        # Calculate frequency within the band range
        position = 0.2 + (0.6 * preferred_position)  # Keep within 20-80% of range to avoid extremes
        frequency = low + position * (high - low)
        
        # If low_power is requested, create a subdued version
        power = 0.3 if low_power else 0.7 + (0.3 * np.random.random())
        
        return {
            'frequency': frequency,
            'band': band,
            'power': power,
            'phase': 2 * np.pi * np.random.random()  # Random phase
        }
    
    def _create_frequency_map(self):
        """Create a spatial frequency map for the brain seed."""
        # Initialize frequency map with base theta frequency at center
        self.frequency_map = {
            'center': self._generate_brain_frequency('theta'),
            'surface': {}
        }
        
        # Create frequency points on the surface
        directions = ['anterior', 'posterior', 'left', 'right', 'dorsal', 'ventral']
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        for direction in directions:
            # Assign a predominant frequency type to each direction
            predominant_band = random.choice(bands)
            
            # Create a frequency profile for this direction
            self.frequency_map['surface'][direction] = {
                'primary': self._generate_brain_frequency(predominant_band),
                'secondary': self._generate_brain_frequency(random.choice(bands)),
                'intensity': 0.5 + (0.5 * np.random.random())
            }
    
    def _create_energy_generators(self):
        """Create natural energy generators for the brain seed."""
        # Create generators based on complexity
        generator_count = 3 + self.complexity
        
        # Different types of energy generators
        generator_types = [
            'resonant_field',
            'vortex_node',
            'scalar_amplifier',
            'harmonic_oscillator',
            'quantum_field_stabilizer'
        ]
        
        # Create generators
        for i in range(generator_count):
            # Select generator type
            g_type = generator_types[i % len(generator_types)]
            
            # Calculate position (on the surface of an imaginary sphere)
            phi = np.pi * (1 + np.sqrt(5)) * i  # Golden angle increment
            y = 1 - (i / (generator_count - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)  # Radius at this y
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            
            position = np.array([x, y, z]) * self.seed_core['radius'] * 1.5
            
            # Calculate energy output
            base_output = 5 + (self.complexity * 2)
            variation = 0.7 + (0.6 * np.random.random())
            output = base_output * variation
            
            # Create generator
            generator = {
                'type': g_type,
                'position': position,
                'output': output,
                'frequency': self._generate_brain_frequency('theta')['frequency'],
                'efficiency': 0.7 + (0.3 * np.random.random()),
                'stability': 0.6 + (0.4 * np.random.random()),
                'growth_factor': 0.5 + (0.5 * np.random.random())
            }
            
            self.energy_generators.append(generator)
        
        # Calculate total energy generation rate
        self.energy_generation_rate = sum(g['output'] * g['efficiency'] for g in self.energy_generators)
        
        logger.info(f"Created {len(self.energy_generators)} energy generators with total output {self.energy_generation_rate:.2f}")
    
    def _initialize_sacred_geometry(self):
        """Initialize sacred geometry patterns for hemispheres."""
        # Select appropriate sacred geometry patterns based on complexity
        available_patterns = list(SACRED_GEOMETRY.keys())
        
        # Left hemisphere - more logical/structured patterns
        left_pattern_count = 1 + self.complexity // 3
        left_patterns = []
        
        # Right hemisphere - more creative/fluid patterns
        right_pattern_count = 1 + self.complexity // 3
        right_patterns = []
        
        # Select patterns with some randomness
        for _ in range(max(left_pattern_count, right_pattern_count)):
            if available_patterns:
                if len(left_patterns) < left_pattern_count:
                    # Prefer more structured patterns for left hemisphere
                    structured_patterns = [p for p in available_patterns 
                                          if SACRED_GEOMETRY[p]['complexity'] >= 7]
                    if structured_patterns:
                        pattern = random.choice(structured_patterns)
                    else:
                        pattern = random.choice(available_patterns)
                    
                    left_patterns.append(pattern)
                    # Don't remove, allow same pattern in both hemispheres
                
                if len(right_patterns) < right_pattern_count:
                    # Prefer more fluid patterns for right hemisphere
                    fluid_patterns = [p for p in available_patterns 
                                     if p in ['flower_of_life', 'seed_of_life', 'vesica_piscis']]
                    if fluid_patterns:
                        pattern = random.choice(fluid_patterns)
                    else:
                        pattern = random.choice(available_patterns)
                    
                    right_patterns.append(pattern)
        
        # Store the selected patterns
        self.sacred_geometry = {
            'left_hemisphere': left_patterns,
            'right_hemisphere': right_patterns,
            'scaling_factor': 0.5 + (0.5 * np.random.random()),
            'rotation_angles': [np.random.random() * 2 * np.pi for _ in range(3)]  # Random 3D rotation
        }
        
        logger.info(f"Initialized sacred geometry: {len(left_patterns)} patterns for left hemisphere, "
                   f"{len(right_patterns)} patterns for right hemisphere")
    
    def _initialize_platonic_structures(self):
        """Initialize platonic solid structures for brain regions."""
        # Map brain regions to platonic solids
        region_solid_mapping = {
            'frontal': random.choice(['tetrahedron', 'dodecahedron']),
            'parietal': random.choice(['octahedron', 'icosahedron']),
            'temporal': random.choice(['hexahedron', 'tetrahedron']),
            'occipital': random.choice(['icosahedron', 'octahedron']),
            'limbic': random.choice(['dodecahedron', 'hexahedron']),
            'cerebellum': random.choice(['icosahedron', 'dodecahedron']),
            'brainstem': random.choice(['hexahedron', 'tetrahedron'])
        }
        
        # Create platonic structure configuration for each region
        self.platonic_structures = {}
        for region, solid in region_solid_mapping.items():
            # Create structure with variation
            self.platonic_structures[region] = {
                'solid': solid,
                'scale': 0.4 + (0.6 * np.random.random()),
                'rotation': [np.random.random() * 2 * np.pi for _ in range(3)],
                'complexity': PLATONIC_SOLIDS[solid].get('vertices', 4) / 2,
                'element': PLATONIC_SOLIDS[solid].get('element', 'aether'),
                'frequency': self._generate_brain_frequency(random.choice(['alpha', 'beta', 'theta'])),
                'connection_points': PLATONIC_SOLIDS[solid].get('vertices', 4)
            }
    
    def generate_natural_energy(self, duration=1.0, intensity=0.8):
        """
        Generate natural energy for brain formation without relying on mycelial network.
        
        Parameters:
            duration (float): Duration of energy generation in time units
            intensity (float): Intensity of energy generation (0-1)
            
        Returns:
            float: Amount of energy generated
        """
        # Calculate base energy from generators
        base_energy = self.energy_generation_rate * duration
        
        # Apply intensity modifier
        applied_energy = base_energy * intensity
        
        # Apply efficiency factor
        efficiency = sum(g['efficiency'] for g in self.energy_generators) / len(self.energy_generators)
        effective_energy = applied_energy * efficiency
        
        # Apply stability variations
        stability = sum(g['stability'] for g in self.energy_generators) / len(self.energy_generators)
        stability_factor = 0.7 + (0.3 * stability)  # Even with low stability, still get 70% energy
        
        # Calculate final energy generation
        final_energy = effective_energy * stability_factor
        
        # Update the brain seed energy level
        current_energy = self.base_energy_level
        new_energy = min(current_energy + final_energy, self.energy_capacity)
        energy_added = new_energy - current_energy
        self.base_energy_level = new_energy
        
        logger.info(f"Generated {energy_added:.2f} units of natural energy (capacity: {self.energy_capacity:.2f})")
        
        return energy_added
    
    def develop_initial_structure(self):
        """
        Develop the initial brain structure with hemispheres.
        
        Returns:
            dict: Structure development metrics
        """
        logger.info("Developing initial brain structure")
        
        # Check if we have enough energy
        required_energy = 50 + (20 * self.complexity)
        if self.base_energy_level < required_energy:
            logger.warning(f"Insufficient energy for initial structure development. "
                          f"Have: {self.base_energy_level:.2f}, Need: {required_energy:.2f}")
            return {
                'success': False,
                'energy_deficit': required_energy - self.base_energy_level,
                'progress': self.formation_progress
            }
        
        # Consume energy for structure development
        self.base_energy_level -= required_energy
        
        # Develop hemisphere structures
        left_complexity = random.uniform(0.8, 1.2) * self.complexity
        right_complexity = random.uniform(0.8, 1.2) * self.complexity
        
        # Configure left hemisphere - more logical/analytical
        self.hemisphere_structure['left'] = {
            'developed': True,
            'complexity': left_complexity,
            'energy': required_energy * 0.5,
            'primary_function': 'analytical',
            'sacred_geometry': self.sacred_geometry['left_hemisphere'],
            'frequency_range': [
                self._generate_brain_frequency('beta'),
                self._generate_brain_frequency('gamma')
            ],
            'energy_channels': int(2 + (left_complexity / 2)),
            'growth_rate': 0.3 + (0.1 * left_complexity)
        }
        
        # Configure right hemisphere - more creative/intuitive
        self.hemisphere_structure['right'] = {
            'developed': True,
            'complexity': right_complexity,
            'energy': required_energy * 0.5,
            'primary_function': 'creative',
            'sacred_geometry': self.sacred_geometry['right_hemisphere'],
            'frequency_range': [
                self._generate_brain_frequency('alpha'),
                self._generate_brain_frequency('theta')
            ],
            'energy_channels': int(2 + (right_complexity / 2)),
            'growth_rate': 0.3 + (0.1 * right_complexity)
        }
        
        # Update brain formation progress
        self.formation_progress = 0.3  # 30% - basic hemisphere structure formed
        
        # Update stability based on balanced development
        hemisphere_balance = 1.0 - abs(left_complexity - right_complexity) / (left_complexity + right_complexity)
        self.stability = 0.5 + (0.3 * hemisphere_balance)
        
        # Calculate structural integrity
        self.structural_integrity = 0.4 + (0.1 * self.complexity) + (0.2 * hemisphere_balance)
        
        logger.info(f"Initial brain structure developed with {self.formation_progress:.1%} completion")
        
        return {
            'success': True,
            'hemispheres_developed': True,
            'left_complexity': left_complexity,
            'right_complexity': right_complexity,
            'balance_factor': hemisphere_balance,
            'progress': self.formation_progress,
            'stability': self.stability
        }
    
    def develop_brain_regions(self):
        """
        Develop specialized brain regions within hemispheres.
        
        Returns:
            dict: Region development metrics
        """
        logger.info("Developing specialized brain regions")
        
        # Check if hemispheres are developed
        if not (self.hemisphere_structure['left']['developed'] and 
                self.hemisphere_structure['right']['developed']):
            logger.warning("Cannot develop regions before hemispheres are formed")
            return {
                'success': False,
                'reason': 'hemispheres_not_developed',
                'progress': self.formation_progress
            }
        
        # Check energy requirements
        required_energy = 80 + (30 * self.complexity)
        if self.base_energy_level < required_energy:
            logger.warning(f"Insufficient energy for region development. "
                          f"Have: {self.base_energy_level:.2f}, Need: {required_energy:.2f}")
            return {
                'success': False,
                'energy_deficit': required_energy - self.base_energy_level,
                'progress': self.formation_progress
            }
        
        # Consume energy for region development
        self.base_energy_level -= required_energy
        
        # Define brain regions to develop
        regions = {
            'frontal': {
                'primary_functions': ['executive_function', 'planning', 'personality'],
                'hemisphere_distribution': {'left': 0.5, 'right': 0.5},
                'energy_allocation': 0.2,
                'complexity_factor': 1.2
            },
            'parietal': {
                'primary_functions': ['sensory_processing', 'spatial_awareness'],
                'hemisphere_distribution': {'left': 0.6, 'right': 0.4},
                'energy_allocation': 0.15,
                'complexity_factor': 1.0
            },
            'temporal': {
                'primary_functions': ['auditory_processing', 'memory', 'language'],
                'hemisphere_distribution': {'left': 0.7, 'right': 0.3},
                'energy_allocation': 0.15,
                'complexity_factor': 1.1
            },
            'occipital': {
                'primary_functions': ['visual_processing'],
                'hemisphere_distribution': {'left': 0.5, 'right': 0.5},
                'energy_allocation': 0.1,
                'complexity_factor': 0.9
            },
            'limbic': {
                'primary_functions': ['emotion', 'memory_formation', 'motivation'],
                'hemisphere_distribution': {'left': 0.3, 'right': 0.7},
                'energy_allocation': 0.2,
                'complexity_factor': 1.3
            },
            'cerebellum': {
                'primary_functions': ['motor_control', 'coordination'],
                'hemisphere_distribution': {'left': 0.5, 'right': 0.5},
                'energy_allocation': 0.1,
                'complexity_factor': 0.8
            },
            'brainstem': {
                'primary_functions': ['autonomic_functions', 'consciousness'],
                'hemisphere_distribution': {'left': 0.5, 'right': 0.5},
                'energy_allocation': 0.1,
                'complexity_factor': 0.7
            }
        }
        
        # Develop each region
        self.region_structure = {}
        for region_name, region_info in regions.items():
            # Calculate region complexity
            base_complexity = self.complexity * region_info['complexity_factor']
            
            # Add randomness for uniqueness
            complexity_variation = 0.8 + (0.4 * np.random.random())
            final_complexity = base_complexity * complexity_variation
            
            # Calculate energy allocation
            energy_allocation = required_energy * region_info['energy_allocation']
            
            # Select appropriate platonic solid structure
            platonic_structure = self.platonic_structures.get(region_name, {
                'solid': random.choice(list(PLATONIC_SOLIDS.keys())),
                'scale': 0.5 + (0.5 * np.random.random())
            })
            
            # Create pockets within the region
            pockets = self._create_region_pockets(region_name, final_complexity)
            
            # Create region entry
            self.region_structure[region_name] = {
                'developed': True,
                'complexity': final_complexity,
                'energy': energy_allocation,
                'primary_functions': region_info['primary_functions'],
                'hemisphere_distribution': region_info['hemisphere_distribution'],
                'platonic_structure': platonic_structure,
                'pockets': pockets,
                'frequency': self._generate_brain_frequency(
                    random.choice(['alpha', 'beta', 'theta', 'gamma'])
                ),
                'connection_strength': 0.5 + (0.3 * np.random.random()),
                'development_level': 0.7 + (0.3 * np.random.random()),
                'white_noise_level': 0.1 + (0.2 * np.random.random())
            }
        
        # Create connections between regions
        self._create_region_connections()
        
        # Update brain formation progress
        self.formation_progress = 0.6  # 60% - regions formed
        
        # Update stability based on even development
        complexity_values = [r['complexity'] for r in self.region_structure.values()]
        complexity_deviation = np.std(complexity_values) / np.mean(complexity_values)
        region_balance = 1.0 - min(0.5, complexity_deviation)  # Lower deviation = better balance
        
        # Update metrics
        self.stability = 0.6 + (0.2 * region_balance)
        self.structural_integrity = 0.5 + (0.1 * self.complexity) + (0.2 * region_balance)
        self.resonance_coherence = 0.5 + (0.1 * self.complexity) + (0.1 * region_balance)
        
        logger.info(f"Brain regions developed with {self.formation_progress:.1%} completion")
        
        return {
            'success': True,
            'regions_developed': list(self.region_structure.keys()),
            'region_count': len(self.region_structure),
            'balance_factor': region_balance,
            'progress': self.formation_progress,
            'stability': self.stability
        }
    
    def _create_region_pockets(self, region_name, complexity):
        """Create pockets within a brain region with varying frequencies and colors."""
        # Determine number of pockets based on complexity
        pocket_count = max(3, int(complexity * 1.5))
        
        # Create pockets
        pockets = []
        for i in range(pocket_count):
            # Generate a color based on frequency (higher frequency = warmer colors)
            frequency = random.choice([
                self._generate_brain_frequency('delta'),
                self._generate_brain_frequency('theta'),
                self._generate_brain_frequency('alpha'),
                self._generate_brain_frequency('beta'),
                self._generate_brain_frequency('gamma')
            ])
            
            # Map frequency to color
            freq_value = frequency['frequency']
            if freq_value < 4:  # delta
                color = {'r': 50, 'g': 0, 'b': 100}  # deep purple
            elif freq_value < 8:  # theta
                color = {'r': 0, 'g': 0, 'b': 220}  # blue
            elif freq_value < 13:  # alpha
                color = {'r': 0, 'g': 150, 'b': 150}  # teal
            elif freq_value < 30:  # beta
                color = {'r': 0, 'g': 200, 'b': 0}  # green
            else:  # gamma
                color = {'r': 200, 'g': 100, 'b': 0}  # orange
            
            # Add some randomness to color
            for component in color:
                color[component] = min(255, max(0, color[component] + random.randint(-30, 30)))
            
            # Create pocket
            pocket = {
                'id': f'{region_name}_pocket_{i}',
                'size': 0.1 + (0.3 * np.random.random()),
                'position': np.array([
                    (np.random.random() - 0.5) * 2,  # x: -1 to 1
                    (np.random.random() - 0.5) * 2,  # y: -1 to 1
                    (np.random.random() - 0.5) * 2   # z: -1 to 1
                ]),
                'frequency': frequency,
                'color': color,
                'energy_level': 0.2 + (0.6 * np.random.random()),
                'connectivity': 0.3 + (0.5 * np.random.random()),
                'function': random.choice([
                    'processing', 'storage', 'routing', 'amplification', 
                    'filtering', 'synchronization', 'integration'
                ]),
                'platonic_element': random.choice([
                    solid_info['element'] for solid_info in PLATONIC_SOLIDS.values()
                ])
            }
            
            pockets.append(pocket)
        
        return pockets
    
    def _create_region_connections(self):
        """Create neural connections between brain regions."""
        # Create connection map
        connection_map = {}
        
        # Define basic region connections (simplified brain connectivity)
        base_connections = {
            'frontal': ['parietal', 'temporal', 'limbic'],
            'parietal': ['frontal', 'occipital', 'temporal'],
            'temporal': ['frontal', 'parietal', 'limbic', 'occipital'],
            'occipital': ['parietal', 'temporal'],
            'limbic': ['frontal', 'temporal', 'brainstem'],
            'cerebellum': ['brainstem', 'occipital', 'parietal'],
            'brainstem': ['cerebellum', 'limbic']
        }
        
        # Create connections with strength and complexity
        for region, connections in base_connections.items():
            if region not in self.region_structure:
                continue
                
            connection_map[region] = []
            
            for target in connections:
                if target not in self.region_structure:
                    continue
                    
                # Calculate connection strength based on region complexities
                source_complexity = self.region_structure[region]['complexity']
                target_complexity = self.region_structure[target]['complexity']
                
                # Higher complexity regions form stronger connections
                base_strength = 0.5 * (source_complexity + target_complexity) / (2 * self.complexity)
                
                # Add randomness for uniqueness
                strength_variation = 0.7 + (0.6 * np.random.random())
                connection_strength = min(1.0, base_strength * strength_variation)
                
                # Create connection
                connection = {
                    'target': target,
                    'strength': connection_strength,
                    'pathways': 1 + int(3 * connection_strength),
                    'frequency_sync': 0.3 + (0.7 * np.random.random())
                }
                
                connection_map[region].append(connection)
        
        # Store connection map in each region
        for region, connections in connection_map.items():
            self.region_structure[region]['connections'] = connections
    
    def apply_white_noise(self):
        """
        Apply white noise to unstructured areas of the brain.
        This creates background activity in empty pockets.
        
        Returns:
            dict: White noise application metrics
        """
        logger.info("Applying white noise to unstructured brain areas")
        
        # Check if regions are developed
        if not self.region_structure:
            logger.warning("Cannot apply white noise before regions are developed")
            return {
                'success': False,
                'reason': 'regions_not_developed',
                'progress': self.formation_progress
            }
        
        # Calculate energy required for white noise
        required_energy = 20 + (5 * self.complexity)
        
        # Check energy availability
        if self.base_energy_level < required_energy:
            logger.warning(f"Insufficient energy for white noise application. "
                          f"Have: {self.base_energy_level:.2f}, Need: {required_energy:.2f}")
            return {
                'success': False,
                'energy_deficit': required_energy - self.base_energy_level,
                'progress': self.formation_progress
            }
        
        # Consume energy
        self.base_energy_level -= required_energy
        
        # Apply white noise to each region
        noise_metrics = {}
        for region_name, region in self.region_structure.items():
            # Calculate noise level based on region's development
            base_noise = 1.0 - region['development_level']
            
            # Add randomness
            noise_variation = 0.7 + (0.6 * np.random.random())
            noise_level = base_noise * noise_variation
            
            # Store white noise level
            self.region_structure[region_name]['white_noise_level'] = noise_level
            
            # Create white noise frequencies (low amplitude across spectrum)
            white_noise_frequencies = []
            for band in BRAIN_FREQUENCIES:
                low, high = BRAIN_FREQUENCIES[band]
                
                # Multiple points across the band for proper noise distribution
                for _ in range(3):
                    freq = low + np.random.random() * (high - low)
                    amplitude = 0.1 + (noise_level * 0.2)  # Low amplitude
                    
                    white_noise_frequencies.append({
                        'frequency': freq,
                        'amplitude': amplitude,
                        'band': band,
                        'phase': 2 * np.pi * np.random.random()
                    })
            
            # Store noise frequencies
            self.region_structure[region_name]['white_noise_frequencies'] = white_noise_frequencies
            
            # Store metrics
            noise_metrics[region_name] = {
                'noise_level': noise_level,
                'frequency_count': len(white_noise_frequencies)
            }
        
        # Update brain formation progress
        self.formation_progress = 0.7  # 70% - white noise applied
        
        logger.info(f"White noise applied to {len(self.region_structure)} regions")
        
        return {
            'success': True,
            'noise_metrics': noise_metrics,
            'energy_used': required_energy,
            'progress': self.formation_progress
        }
    
    def prepare_for_soul_attachment(self):
        """
        Prepare the brain for soul attachment through the life cord.
        
        Returns:
            dict: Preparation metrics
        """
        logger.info("Preparing brain for soul attachment")
        
        # Check development progress
        if self.formation_progress < 0.6:
            logger.warning("Brain not sufficiently developed for soul attachment preparation")
            return {
                'success': False,
                'reason': 'insufficient_development',
                'required_progress': 0.6,
                'current_progress': self.formation_progress
            }
        
        # Calculate energy requirement
        required_energy = 100 + (20 * self.complexity)
        
        # Check energy availability
        if self.base_energy_level < required_energy:
            logger.warning(f"Insufficient energy for soul attachment preparation. "
                          f"Have: {self.base_energy_level:.2f}, Need: {required_energy:.2f}")
            return {
                'success': False,
                'energy_deficit': required_energy - self.base_energy_level,
                'progress': self.formation_progress
            }
        
        # Consume energy
        self.base_energy_level -= required_energy
        
        # Create attachment points for life cord
        attachment_points = []
        
        # Primary attachment point at center/core
        primary_point = {
            'id': 'core_attachment',
            'position': np.array([0, 0, 0]),
            'strength': 0.9,
            'frequency': 7.83,  # Schumann resonance
            'purpose': 'primary_connection',
            'capacity': 100.0,
            'stability': 0.8 + (0.2 * np.random.random())
        }
        attachment_points.append(primary_point)
        
        # Create region-specific attachment points
        for region_name, region in self.region_structure.items():
            # More attachment points for higher complexity regions
            point_count = 1 + int(region['complexity'] / 3)
            
            for i in range(point_count):
                # Calculate position based on region
                region_center = np.mean([p['position'] for p in region['pockets']], axis=0)
                
                # Add some variation
                variation = np.array([
                    (np.random.random() - 0.5) * 0.5,
                    (np.random.random() - 0.5) * 0.5,
                    (np.random.random() - 0.5) * 0.5
                ])
                
                position = region_center + variation
                
                # Create attachment point
                point = {
                    'id': f'{region_name}_attachment_{i}',
                    'position': position,
                    'strength': 0.5 + (0.4 * np.random.random()),
                    'frequency': region['frequency']['frequency'],
                    'purpose': 'region_connection',
                    'region': region_name,
                    'capacity': 20.0 + (30.0 * np.random.random()),
                    'stability': 0.6 + (0.3 * np.random.random())
                }
                
                attachment_points.append(point)
        
        # Create hemisphere attachment points
        for hemi in ['left', 'right']:
            hemisphere = self.hemisphere_structure[hemi]
            
            # Create attachment point
            point = {
                'id': f'{hemi}_hemisphere_attachment',
                'position': np.array([-0.5 if hemi == 'left' else 0.5, 0, 0]),
                'strength': 0.7 + (0.2 * np.random.random()),
                'frequency': hemisphere['frequency_range'][0]['frequency'],
                'purpose': 'hemisphere_connection',
                'hemisphere': hemi,
                'capacity': 50.0 + (30.0 * np.random.random()),
                'stability': 0.7 + (0.2 * np.random.random())
            }
            
            attachment_points.append(point)
        
        # Store attachment points
        self.attachment_points = attachment_points
        
        # Create resonance channels for soul communication
        self.resonance_channels = self._create_resonance_channels()
        
        # Update brain formation progress
        self.formation_progress = 0.9  # 90% - ready for soul attachment
        
        # Increase stability for attachment
        self.stability = 0.8
        self.structural_integrity = 0.8
        
        logger.info(f"Brain prepared for soul attachment with {len(attachment_points)} attachment points")
        
        return {
            'success': True,
            'attachment_points': len(attachment_points),
            'resonance_channels': len(self.resonance_channels),
            'stability': self.stability,
            'progress': self.formation_progress
        }
    
    def _create_resonance_channels(self):
        """Create resonance channels for soul-brain communication."""
        # Create channels for different types of communication
        channels = []
        
        # Channel types
        channel_types = [
            {'name': 'consciousness', 'base_frequency': 40.0, 'importance': 0.9},
            {'name': 'emotion', 'base_frequency': 10.0, 'importance': 0.8},
            {'name': 'intuition', 'base_frequency': 30.0, 'importance': 0.7},
            {'name': 'memory', 'base_frequency': 8.0, 'importance': 0.8},
            {'name': 'cognition', 'base_frequency': 20.0, 'importance': 0.85},
            {'name': 'creativity', 'base_frequency': 15.0, 'importance': 0.75},
            {'name': 'spiritual', 'base_frequency': 111.0, 'importance': 0.9}
        ]
        
        # Create each channel
        for channel_type in channel_types:
            # Base properties
            base_frequency = channel_type['base_frequency']
            
            # Add variations for uniqueness
            frequency_variation = 0.9 + (0.2 * np.random.random())
            final_frequency = base_frequency * frequency_variation
            
            # Create channel
            channel = {
                'name': channel_type['name'],
                'frequency': final_frequency,
                'bandwidth': 5.0 + (10.0 * np.random.random()),
                'capacity': 50.0 + (50.0 * channel_type['importance']),
                'priority': channel_type['importance'],
                'stability': 0.6 + (0.4 * channel_type['importance']),
                'modulation': random.choice(['amplitude', 'frequency', 'phase']),
                'harmonic_factor': GOLDEN_RATIO if random.random() > 0.5 else random.choice([2.0, 3.0, 4.0])
            }
            
            channels.append(channel)
        
        return channels
    
    def get_metrics(self):
        """Return current metrics of the brain seed."""
        metrics = {
            'energy_level': self.base_energy_level,
            'energy_capacity': self.energy_capacity,
            'energy_generation_rate': self.energy_generation_rate,
            'complexity': self.complexity,
            'formation_progress': self.formation_progress,
            'structural_integrity': self.structural_integrity,
            'resonance_coherence': self.resonance_coherence,
            'stability': self.stability,
            'uniqueness_factor': self.uniqueness_factor
        }
        
        # Add hemisphere metrics if developed
        if self.hemisphere_structure['left']['developed'] and self.hemisphere_structure['right']['developed']:
            metrics['hemispheres'] = {
                'left_complexity': self.hemisphere_structure['left']['complexity'],
                'right_complexity': self.hemisphere_structure['right']['complexity'],
                'balance': 1.0 - abs(
                    self.hemisphere_structure['left']['complexity'] - 
                    self.hemisphere_structure['right']['complexity']
                ) / (
                    self.hemisphere_structure['left']['complexity'] + 
                    self.hemisphere_structure['right']['complexity']
                )
            }
        
        # Add region metrics if developed
        if self.region_structure:
            metrics['regions'] = {
                'count': len(self.region_structure),
                'avg_complexity': np.mean([r['complexity'] for r in self.region_structure.values()]),
                'connection_count': sum(len(r.get('connections', [])) for r in self.region_structure.values())
            }
        
        # Add attachment metrics if prepared
        if hasattr(self, 'attachment_points'):
            metrics['soul_attachment'] = {
                'attachment_points': len(self.attachment_points),
                'resonance_channels': len(self.resonance_channels),
                'readiness': self.formation_progress
            }
        
        return metrics


# Factory function to create a BrainSeed
def create_brain_seed(resonant_soul, complexity=9):
    """
    Create a new brain seed instance.
    
    Parameters:
        resonant_soul: The resonant soul to connect
        complexity (int): Complexity level of the brain seed (1-10)
        
    Returns:
        BrainSeed: A new brain seed instance
    """
    logger.info(f"Creating new brain seed with complexity {complexity}")
    brain_seed = BrainSeed(resonant_soul, complexity)
    
    # Generate initial energy
    brain_seed.generate_natural_energy(duration=2.0, intensity=1.0)
    
    return brain_seed