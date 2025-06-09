# --- brain_structure.py V8 - COMPLETE WITH SOUND INTEGRATION ---
"""
Create the brain structure with hierarchical 3D grid system:
256³ Grid → External Field → Hemispheres → Regions → Sub-regions → Blocks

Uses parametric variance for unique brain shapes while maintaining anatomical inspiration.
Implements ~3,500 blocks with proper field systems, real sound generation, and test simulation.
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from region_definitions import *
import sys
import os
# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from constants.constants import *
from sound.sound_generator import SoundGenerator
from sound.sounds_of_universe import UniverseSounds
from sound.noise_generator import NoiseGenerator


# errors in file to address - the region definitions hasnt been updated yet
# so we have missingfunctions being referred to so cant run the test. 
# 2. Undefined Functions
# Functions like get_hemisphere_region_templates, get_anatomical_position_mapping, get_region_configuration, get_hemisphere_wave_properties, and get_region_wave_properties are not defined or imported.
# You must either:
# 3. SoundGenerator/UniverseSounds Members
# Errors like generate_static_noise not being a member of SoundGenerator and unexpected keyword arguments in UniverseSounds methods mean your sound classes do not have these methods or their signatures are different.
# Check the implementation of these classes and update your calls to match their actual method names and arguments.



# Import sound generation functions
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from sound.sound_generator import SoundGenerator
    from sound.sounds_of_universe import UniverseSounds
    from sound.noise_generator import NoiseGenerator
    SOUND_AVAILABLE = True
    logger.info("Sound generation modules imported successfully")
except ImportError as e:
    logger.warning(f"Sound modules not available: {e}")
    SOUND_AVAILABLE = False


# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class Brain:
    """
    Hierarchical brain structure with parametric variance for unique brains.
    Architecture:
        - 256³ total grid (16,777,216 units)  
        - 26-unit external buffer (10%)
        - ~3,500 blocks (10³ units each, 80% utilization)
        - Hierarchical naming: L1-FRONTAL-S51-001
        - Real sound generation for fields
    """

    def __init__(self):
        """Initialize brain structure."""
        self.brain_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.grid_dimensions = GRID_DIMENSIONS  # (256, 256, 256)
        
        # Brain components
        self.external_field = {}
        self.grid = {}
        self.phi_resonance_field = {}
        self.hemispheres = {}
        self.regions = {}
        self.sub_regions = {}
        self.sub_regions_borders = {}
        self.static_field = {}
        self.brain_waves = {}
        self.field_integrity = {}
        self.brain = {}
        self.brain_seed = None
        
        # Initialize sound generators
        self.sound_generator = None
        self.universe_sounds = None
        self.noise_generator = None
        
        # Create sound generators if available
        self.create_sound_generators()
        
        logger.info(f"Brain structure initialized: {self.brain_id[:8]}")


    def create_sound_generators(self):
        """Create sound generators with proper error handling."""
        try:
            if SOUND_AVAILABLE:
                self.sound_generator = SoundGenerator(output_dir="output/sounds/brain")
                self.universe_sounds = UniverseSounds(output_dir="output/sounds/brain")
                self.noise_generator = NoiseGenerator(output_dir="output/sounds/brain")
                logger.info("Sound generators created successfully")
            else:
                logger.warning("Sound modules not available - brain will function without sound generation")
        except Exception as e:
            logger.error(f"Error creating sound generators: {e}")
            self.sound_generator = None
            self.universe_sounds = None
            self.noise_generator = None

    def get_hemisphere_region_templates(self) -> Dict[str, Any]:
        """Get hemisphere region templates from region definitions."""
        try:
            # Try to import from the fixed region_definitions
            templates = {
                'left': {
                    'regions': {},
                    'bias': 'logical',
                    'function': 'analytical'
                },
                'right': {
                    'regions': {},
                    'bias': 'creative', 
                    'function': 'holistic'
                }
            }
            
            # Add regions to both hemispheres
            for hemisphere in ['left', 'right']:
                for region_name in ['frontal', 'parietal', 'temporal', 'occipital', 'limbic', 'cerebellum', 'brain_stem']:
                    templates[hemisphere]['regions'][region_name] = {
                        'proportion': 0.14,  # Default proportion
                        'function': 'processing',
                        'wave_frequency': 10.0,
                        'color': '#808080',
                        'sub_regions': {}
                    }
            
            return templates
            
        except Exception as e:
            logger.error(f"Error getting region templates: {e}")
            # Return minimal fallback
            return {
                'left': {'regions': {'frontal': {'proportion': 0.5}}, 'bias': 'logical'},
                'right': {'regions': {'frontal': {'proportion': 0.5}}, 'bias': 'creative'}
            }

    def get_anatomical_position_mapping(self) -> Dict[str, Tuple[float, float, float]]:
        """Get anatomical position mapping for regions."""
        return {
            'frontal': (0.7, 0.5, 0.2),
            'parietal': (0.4, 0.5, 0.6),
            'temporal': (0.3, 0.2, 0.5),
            'occipital': (0.3, 0.8, 0.5),
            'limbic': (0.5, 0.5, 0.5),
            'cerebellum': (0.5, 0.9, 0.2),
            'brain_stem': (0.5, 0.8, 0.1)
        }

    def get_region_configuration(self, region_name: str) -> Dict[str, Any]:
        """Get configuration for a specific region."""
        config_map = {
            'frontal': {
                'proportion': 0.28,
                'function': 'executive_control',
                'wave_frequency_hz': 18.5,
                'color': '#46a0fc',
                'default_wave': 'beta'
            },
            'parietal': {
                'proportion': 0.19,
                'function': 'sensory_integration',
                'wave_frequency_hz': 10.2,
                'color': '#32944e',
                'default_wave': 'alpha'
            },
            'temporal': {
                'proportion': 0.22,
                'function': 'auditory_language_memory',
                'wave_frequency_hz': 9.7,
                'color': '#fccc3c',
                'default_wave': 'alpha'
            },
            'occipital': {
                'proportion': 0.14,
                'function': 'visual_processing',
                'wave_frequency_hz': 11.3,
                'color': '#7e28bc',
                'default_wave': 'alpha'
            },
            'limbic': {
                'proportion': 0.11,
                'function': 'emotion_memory',
                'wave_frequency_hz': 6.8,
                'color': '#9c1e2c',
                'default_wave': 'theta'
            },
            'cerebellum': {
                'proportion': 0.14,
                'function': 'motor_coordination',
                'wave_frequency_hz': 9.3,
                'color': '#1e9c7e',
                'default_wave': 'alpha'
            },
            'brain_stem': {
                'proportion': 0.06,
                'function': 'basic_life_functions',
                'wave_frequency_hz': 2.5,
                'color': '#5a5a5a',
                'default_wave': 'delta'
            }
        }
        
        return config_map.get(region_name, {
            'proportion': 0.1,
            'function': 'unknown',
            'wave_frequency_hz': 10.0,
            'color': '#808080',
            'default_wave': 'alpha'
        })

    def get_hemisphere_wave_properties(self) -> Dict[str, Any]:
        """Get wave properties for hemispheres."""
        return {
            'left_hemisphere': {
                'function': 'logical_analytical',
                'wave_frequency_hz': 18.5,
                'default_wave': 'beta',
                'color': '#46a0fc',
                'sound_base_note': 'C4'
            },
            'right_hemisphere': {
                'function': 'creative_holistic', 
                'wave_frequency_hz': 10.2,
                'default_wave': 'alpha',
                'color': '#32944e',
                'sound_base_note': 'G4'
            }
        }

    def get_region_wave_properties(self) -> Dict[str, Any]:
        """Get wave properties for all regions."""
        return {
            'frontal': {'frequency': 18.5, 'wave_type': 'beta', 'sound_note': 'C4'},
            'parietal': {'frequency': 10.2, 'wave_type': 'alpha', 'sound_note': 'E4'},
            'temporal': {'frequency': 9.7, 'wave_type': 'alpha', 'sound_note': 'G4'},
            'occipital': {'frequency': 11.3, 'wave_type': 'alpha', 'sound_note': 'B4'},
            'limbic': {'frequency': 6.8, 'wave_type': 'theta', 'sound_note': 'D4'},
            'cerebellum': {'frequency': 9.3, 'wave_type': 'alpha', 'sound_note': 'A3'},
            'brain_stem': {'frequency': 2.5, 'wave_type': 'delta', 'sound_note': 'F3'}
        }

    def generate_static_noise_safe(self, duration: float = 1.0, amplitude: float = 0.1) -> Optional[Any]:
        """Generate static noise using NoiseGenerator."""
        try:
            if self.noise_generator:
                # Use the correct method from NoiseGenerator
                static_noise = self.noise_generator.generate_noise(
                    noise_type='white',  # Valid: 'white', 'pink', 'brown', 'blue', 'violet', 'quantum', 'edge_of_chaos'
                    duration=duration,
                    amplitude=amplitude
                )
                return static_noise
            else:
                logger.debug("Noise generator not available - skipping static noise generation")
                return None
        except Exception as e:
            logger.error(f"Error generating static noise: {e}")
            return None

    def generate_cosmic_background_safe(self, duration: float = 5.0, amplitude: float = 0.7, frequency_band: str = 'full') -> Optional[Any]:
        """Generate cosmic background using correct UniverseSounds parameters."""
        try:
            if self.universe_sounds:
                # Use the CORRECT method signature: generate_cosmic_background(duration, amplitude, frequency_band)
                cosmic_background = self.universe_sounds.generate_cosmic_background(
                    duration=duration,
                    amplitude=amplitude,
                    frequency_band=frequency_band  # Valid: 'full', 'low', 'mid', 'high'
                )
                return cosmic_background
            else:
                logger.debug("Universe sounds not available - skipping cosmic background generation")
                return None
        except Exception as e:
            logger.error(f"Error generating cosmic background: {e}")
            return None

    def generate_brain_wave_pattern(self, wave_type: str, frequency: float, duration: float = 2.0) -> Optional[Any]:
        """Generate brain wave patterns using SoundGenerator."""
        try:
            if self.sound_generator:
                # Create harmonic patterns based on brain wave type
                if wave_type == 'delta':
                    base_freq = 200.0
                    harmonics = [1.0, 0.5, 0.25]
                    amplitudes = [0.6, 0.3, 0.1]
                elif wave_type == 'theta':
                    base_freq = 432.0
                    harmonics = [1.0, 1.5, 2.0]
                    amplitudes = [0.7, 0.4, 0.2]
                elif wave_type == 'alpha':
                    base_freq = frequency
                    harmonics = [1.0, 1.618, 2.0]  # Include golden ratio
                    amplitudes = [0.8, 0.3, 0.2]
                elif wave_type == 'beta':
                    base_freq = frequency
                    harmonics = [1.0, 2.0, 3.0, 4.0]
                    amplitudes = [0.7, 0.4, 0.2, 0.1]
                else:
                    base_freq = frequency
                    harmonics = [1.0, 2.0]
                    amplitudes = [0.8, 0.2]
                
                wave_sound = self.sound_generator.generate_harmonic_tone(
                    base_frequency=base_freq,
                    harmonics=harmonics,
                    amplitudes=amplitudes,
                    duration=duration,
                    fade_in_out=0.1
                )
                
                return wave_sound
            else:
                logger.debug("Sound generator not available - skipping brain wave generation")
                return None
        except Exception as e:
            logger.error(f"Error generating brain wave pattern: {e}")
            return None
        
    def trigger_brain_development(self):
        """Trigger brain development process on BRAIN_SEED_SAVED flag."""
        logger.info("Triggering brain development process")
        
        try:
            self.load_brain_seed()
            self.create_external_field()
            self.create_brain_grid()
            self.create_brain_phi_resonance_field()
            self.assign_brain_hemispheres_properties()
            self.assign_brain_regions_properties()
            self.assign_brain_sub_regions_properties()
            self.determine_static_borders_for_sub_regions()
            self.apply_static_field_to_sub_regions()
            self.apply_brain_waves_to_sub_regions()
            self.test_field_integrity()
            self.save_brain_structure()
            
            logger.info("Brain development completed successfully")
            
        except Exception as e:
            logger.error(f"Brain development failed: {e}")
            raise RuntimeError(f"Brain development failed: {e}")

    def load_brain_seed(self):
        """Load brain seed coordinates, place in center if outside brain area."""
        logger.info("Loading brain seed")
        
        try:
            if not self.brain_seed:
                # Default center position
                center = (self.grid_dimensions[0] // 2, self.grid_dimensions[1] // 2, self.grid_dimensions[2] // 2)
                self.brain_seed = {'position': center, 'energy': 5.0, 'frequency': 432.0 * math.sqrt(2)}
                logger.warning("Using default brain seed position")
            
            # Validate position within brain area (not external buffer)
            pos = self.brain_seed['position']
            buffer_size = 26
            
            if (pos[0] < buffer_size or pos[0] >= self.grid_dimensions[0] - buffer_size or
                pos[1] < buffer_size or pos[1] >= self.grid_dimensions[1] - buffer_size or
                pos[2] < buffer_size or pos[2] >= self.grid_dimensions[2] - buffer_size):
                
                center_pos = (self.grid_dimensions[0] // 2, self.grid_dimensions[1] // 2, self.grid_dimensions[2] // 2)
                self.brain_seed['position'] = center_pos
                logger.warning(f"Brain seed moved to center: {center_pos}")
            
            self.grid["seed_position"] = self.brain_seed['position']
            self.grid["seed_energy"] = self.brain_seed['energy']
            
            logger.info(f"Brain seed loaded at: {self.brain_seed['position']}")
            
        except Exception as e:
            logger.error(f"Failed to load brain seed: {e}")
            raise RuntimeError(f"Brain seed loading failed: {e}")


    def create_external_field(self):
        """Create external field with standing waves around 256³ grid."""
        logger.info("Creating external field with standing waves")
        
        try:
            buffer_size = 26  # 10% buffer
            
            # Standing wave field for protection/stabilization
            base_frequency = 40.0  # Gamma range
            wavelength = 343.0 / base_frequency
            
            standing_waves = []
            for axis in ['x', 'y', 'z']:
                wave_data = {
                    'axis': axis,
                    'frequency': base_frequency,
                    'wavelength': wavelength,
                    'amplitude': 0.5,
                    'nodes': [],
                    'antinodes': []
                }
                
                # Calculate nodes/antinodes
                axis_length = self.grid_dimensions[0]
                num_nodes = int(axis_length / wavelength)
                
                for i in range(num_nodes):
                    node_pos = i * wavelength
                    antinode_pos = (i + 0.5) * wavelength
                    
                    if node_pos < axis_length:
                        wave_data['nodes'].append(node_pos)
                    if antinode_pos < axis_length:
                        wave_data['antinodes'].append(antinode_pos)
                
                standing_waves.append(wave_data)
            
            # Generate real universal sounds for external field
            external_sound_file = None
            if SOUND_AVAILABLE and self.universe_sounds:
                try:
                    # Generate universal boundary sound (protective but realistic)
                    boundary_sound = self.universe_sounds.generate_dimensional_transition(
                        duration=3.0,
                        sample_rate=44100,
                        transition_type='protective_boundary',
                        amplitude=0.6
                    )
                    
                    # Save external field sound
                    external_sound_file = f"brain_external_field_{self.brain_id[:8]}.wav"
                    sound_path = self.universe_sounds.save_sound(
                        boundary_sound, 
                        external_sound_file,
                        f"External Field Protection - Brain {self.brain_id[:8]}"
                    )
                    logger.info(f"External field sound generated: {sound_path}")
                    
                except Exception as sound_err:
                    logger.warning(f"Failed to generate external field sound: {sound_err}")
                    external_sound_file = None
            
            self.external_field = {
                'field_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'field_type': 'standing_wave_protection',
                'boundaries': {
                    'inner_bounds': (buffer_size, buffer_size, buffer_size),
                    'outer_bounds': (self.grid_dimensions[0] - buffer_size, 
                                   self.grid_dimensions[1] - buffer_size,
                                   self.grid_dimensions[2] - buffer_size),
                    'buffer_thickness': buffer_size
                },
                'standing_waves': standing_waves,
                'field_strength': 0.8,
                'sound_file': external_sound_file,
                'applied': True
            }
            
            setattr(self, FLAG_EXTERNAL_FIELD_CREATED, True)
            logger.info(f"External field created with {len(standing_waves)} wave axes")
            
        except Exception as e:
            logger.error(f"Failed to create external field: {e}")
            raise RuntimeError(f"External field creation failed: {e}")

    def create_brain_grid(self):
        """Create hierarchical brain grid: Hemispheres → Regions → Sub-regions → Blocks."""
        logger.info("Creating hierarchical brain grid")
        
        try:
            buffer_size = 26
            hemisphere_gap = 6
            region_gap = 6
            
            # Available brain space
            brain_space = {
                'x_start': buffer_size, 'x_end': self.grid_dimensions[0] - buffer_size,
                'y_start': buffer_size, 'y_end': self.grid_dimensions[1] - buffer_size,
                'z_start': buffer_size, 'z_end': self.grid_dimensions[2] - buffer_size
            }
            
            # Hemisphere dimensions with parametric variance
            total_x = brain_space['x_end'] - brain_space['x_start']
            hemisphere_variance = random.uniform(0.95, 1.05)  # ±5%
            
            left_width = int((total_x - hemisphere_gap) / 2 * hemisphere_variance)
            right_width = total_x - hemisphere_gap - left_width
            
            # Create hemispheres
            hemispheres_data = {
                'L1': {
                    'hemisphere_id': 'L1',
                    'name': 'Left Hemisphere',
                    'boundaries': {
                        'x_start': brain_space['x_start'],
                        'x_end': brain_space['x_start'] + left_width,
                        'y_start': brain_space['y_start'],
                        'y_end': brain_space['y_end'],
                        'z_start': brain_space['z_start'],
                        'z_end': brain_space['z_end']
                    },
                    'volume': left_width * (brain_space['y_end'] - brain_space['y_start']) * (brain_space['z_end'] - brain_space['z_start']),
                    'regions': {}
                },
                'R1': {
                    'hemisphere_id': 'R1',
                    'name': 'Right Hemisphere',
                    'boundaries': {
                        'x_start': brain_space['x_start'] + left_width + hemisphere_gap,
                        'x_end': brain_space['x_end'],
                        'y_start': brain_space['y_start'],
                        'y_end': brain_space['y_end'],
                        'z_start': brain_space['z_start'],
                        'z_end': brain_space['z_end']
                    },
                    'volume': right_width * (brain_space['y_end'] - brain_space['y_start']) * (brain_space['z_end'] - brain_space['z_start']),
                    'regions': {}
                }
            }
            
            # Create regions within hemispheres
            for hemisphere_id, hemisphere_data in hemispheres_data.items():
                regions = self._create_regions_in_hemisphere(hemisphere_id, hemisphere_data['boundaries'], region_gap)
                hemisphere_data['regions'] = regions
            
            self.grid = {
                'grid_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'dimensions': self.grid_dimensions,
                'brain_space': brain_space,
                'hemispheres': hemispheres_data,
                'hemisphere_gap': hemisphere_gap,
                'region_gap': region_gap,
                'block_size': 10,  # 10³ = 1000 units per block
                'utilization_target': 0.8
            }
            
            logger.info(f"Brain grid created with {len(hemispheres_data)} hemispheres")
            
        except Exception as e:
            logger.error(f"Failed to create brain grid: {e}")
            raise RuntimeError(f"Brain grid creation failed: {e}")

    def _create_regions_in_hemisphere(self, hemisphere_id: str, hemisphere_bounds: Dict, region_gap: int) -> Dict[str, Any]:
        """Create anatomically-inspired regions with variance."""
        
        regions = {}
        h_width = hemisphere_bounds['x_end'] - hemisphere_bounds['x_start']
        h_height = hemisphere_bounds['y_end'] - hemisphere_bounds['y_start']
        h_depth = hemisphere_bounds['z_end'] - hemisphere_bounds['z_start']
        
        # Use major regions from constants
        region_names = ['frontal', 'parietal', 'temporal', 'occipital', 'limbic', 'cerebellum', 'brain_stem']
        
        for region_name in region_names:
            # Get configuration
            config = self.get_region_configuration(region_name)
            
            # Apply variance
            variance_factor = random.uniform(0.9, 1.1)
            base_proportion = config['proportion']
            
            # Calculate size based on proportion
            region_volume = (h_width * h_height * h_depth) * base_proportion * variance_factor
            region_width = int((region_volume ** (1/3)) * 1.2)  # Approximate cubic root
            region_height = int((region_volume ** (1/3)) * 0.9)
            region_depth = int((region_volume ** (1/3)) * 0.9)
            
            # Constrain to hemisphere bounds
            region_width = min(region_width, h_width - region_gap * 2)
            region_height = min(region_height, h_height - region_gap * 2)
            region_depth = min(region_depth, h_depth - region_gap * 2)
            
            # Calculate position using anatomical mapping
            position_map = self.get_anatomical_position_mapping()
            pos_ratios = position_map.get(region_name, (0.5, 0.5, 0.5))
            
            x_start = hemisphere_bounds['x_start'] + int((h_width - region_width) * pos_ratios[0])
            y_start = hemisphere_bounds['y_start'] + int((h_height - region_height) * pos_ratios[1])
            z_start = hemisphere_bounds['z_start'] + int((h_depth - region_depth) * pos_ratios[2])
            
            region_bounds = {
                'x_start': max(hemisphere_bounds['x_start'] + region_gap, x_start),
                'x_end': min(hemisphere_bounds['x_end'] - region_gap, x_start + region_width),
                'y_start': max(hemisphere_bounds['y_start'] + region_gap, y_start),
                'y_end': min(hemisphere_bounds['y_end'] - region_gap, y_start + region_height),
                'z_start': max(hemisphere_bounds['z_start'] + region_gap, z_start),
                'z_end': min(hemisphere_bounds['z_end'] - region_gap, z_start + region_depth)
            }
            
            # Create sub-regions
            sub_regions = self._create_sub_regions_in_region(hemisphere_id, region_name, region_bounds)
            
            regions[region_name] = {
                'region_id': f"{hemisphere_id}-{region_name}",
                'hemisphere_id': hemisphere_id,
                'region_name': region_name,
                'boundaries': region_bounds,
                'volume': (region_bounds['x_end'] - region_bounds['x_start']) * 
                         (region_bounds['y_end'] - region_bounds['y_start']) * 
                         (region_bounds['z_end'] - region_bounds['z_start']),
                'variance_factor': variance_factor,
                'sub_regions': sub_regions,
                'creation_time': datetime.now().isoformat()
            }
        
        return regions

    def _calculate_region_position(self, region_name: str, position_type: str, hemisphere_bounds: Dict, 
                                 width: int, height: int, depth: int, gap: int) -> Dict:
        """Calculate region position based on anatomical type."""
        
        h_x_start, h_x_end = hemisphere_bounds['x_start'], hemisphere_bounds['x_end']
        h_y_start, h_y_end = hemisphere_bounds['y_start'], hemisphere_bounds['y_end']
        h_z_start, h_z_end = hemisphere_bounds['z_start'], hemisphere_bounds['z_end']
        
        # Use position mapping from region_definitions
        position_map = self.get_anatomical_position_mapping()
        x_ratio, y_ratio, z_ratio = position_map.get(region_name, (0.5, 0.5, 0.5))
        
        # Calculate positions based on ratios
        x_start = h_x_start + int((h_x_end - h_x_start - width) * x_ratio)
        y_start = h_y_start + int((h_y_end - h_y_start - height) * y_ratio)
        z_start = h_z_start + int((h_z_end - h_z_start - depth) * z_ratio)
        
        return {
            'x_start': max(h_x_start + gap, x_start),
            'x_end': min(h_x_end - gap, x_start + width),
            'y_start': max(h_y_start + gap, y_start),
            'y_end': min(h_y_end - gap, y_start + height),
            'z_start': max(h_z_start + gap, z_start),
            'z_end': min(h_z_end - gap, z_start + depth)
        }

    def _create_sub_regions_in_region(self, hemisphere_id: str, region_name: str, region_bounds: Dict) -> Dict[str, Any]:
        """Create sub-regions within region. Some regions have one sub-region."""
        
        sub_regions = {}
        
        # Determine number of sub-regions based on region type
        if region_name in ['brain_stem', 'limbic']:
            # Small regions get fewer sub-regions
            sub_region_count = random.randint(1, 2)
        else:
            # Larger regions get more sub-regions
            sub_region_count = random.randint(2, 4)
        
        if sub_region_count == 1:
            # Single sub-region
            sub_regions['S01'] = self._create_single_sub_region(hemisphere_id, region_name, 'S01', region_bounds)
        else:
            # Multiple sub-regions
            for i in range(sub_region_count):
                sub_region_id = f"S{i+1:02d}"
                sub_region_bounds = self._calculate_sub_region_bounds(region_bounds, i, sub_region_count)
                sub_regions[sub_region_id] = self._create_single_sub_region(hemisphere_id, region_name, sub_region_id, sub_region_bounds)
        
        return sub_regions

    def _create_single_sub_region(self, hemisphere_id: str, region_name: str, sub_region_id: str, bounds: Dict) -> Dict[str, Any]:
        """Create single sub-region with blocks."""
        
        width = bounds['x_end'] - bounds['x_start']
        height = bounds['y_end'] - bounds['y_start']
        depth = bounds['z_end'] - bounds['z_start']
        volume = width * height * depth
        
        # Calculate blocks (10³ each)
        block_size = 10
        blocks_x = max(1, width // block_size)
        blocks_y = max(1, height // block_size)
        blocks_z = max(1, depth // block_size)
        
        blocks = {}
        block_counter = 1
        
        for bx in range(blocks_x):
            for by in range(blocks_y):
                for bz in range(blocks_z):
                    block_id = f"{hemisphere_id}-{region_name}-{sub_region_id}-{block_counter:03d}"
                    
                    block_bounds = {
                        'x_start': bounds['x_start'] + bx * block_size,
                        'x_end': min(bounds['x_end'], bounds['x_start'] + (bx + 1) * block_size),
                        'y_start': bounds['y_start'] + by * block_size,
                        'y_end': min(bounds['y_end'], bounds['y_start'] + (by + 1) * block_size),
                        'z_start': bounds['z_start'] + bz * block_size,
                        'z_end': min(bounds['z_end'], bounds['z_start'] + (bz + 1) * block_size)
                    }
                    
                    blocks[block_id] = {
                        'block_id': block_id,
                        'grid_position': (bx, by, bz),
                        'boundaries': block_bounds,
                        'volume': (block_bounds['x_end'] - block_bounds['x_start']) * 
                                (block_bounds['y_end'] - block_bounds['y_start']) * 
                                (block_bounds['z_end'] - block_bounds['z_start']),
                        'center': (
                            (block_bounds['x_start'] + block_bounds['x_end']) // 2,
                            (block_bounds['y_start'] + block_bounds['y_end']) // 2,
                            (block_bounds['z_start'] + block_bounds['z_end']) // 2
                        ),
                        'active': False,
                        'node_count': 0,
                        'memory_fragments': [],
                        'mycelial_seeds': []
                    }
                    
                    block_counter += 1
        
        return {
            'sub_region_id': f"{hemisphere_id}-{region_name}-{sub_region_id}",
            'hemisphere_id': hemisphere_id,
            'region_name': region_name,
            'sub_region_name': sub_region_id,
            'boundaries': bounds,
            'volume': volume,
            'blocks': blocks,
            'block_count': len(blocks),
            'blocks_dimensions': (blocks_x, blocks_y, blocks_z),
            'creation_time': datetime.now().isoformat()
        }

    def _calculate_sub_region_bounds(self, region_bounds: Dict, index: int, total_count: int) -> Dict:
        """Calculate sub-region bounds within region."""
        
        region_width = region_bounds['x_end'] - region_bounds['x_start']
        region_height = region_bounds['y_end'] - region_bounds['y_start']
        
        if total_count == 2:
            # Split horizontally
            sub_width = region_width // 2
            x_start = region_bounds['x_start'] + index * sub_width
            x_end = x_start + sub_width if index == 0 else region_bounds['x_end']
            
            return {
                'x_start': x_start, 'x_end': x_end,
                'y_start': region_bounds['y_start'], 'y_end': region_bounds['y_end'],
                'z_start': region_bounds['z_start'], 'z_end': region_bounds['z_end']
            }
        elif total_count == 3:
            # Split into thirds
            sub_width = region_width // 3
            x_start = region_bounds['x_start'] + index * sub_width
            x_end = x_start + sub_width if index < 2 else region_bounds['x_end']
            
            return {
                'x_start': x_start, 'x_end': x_end,
                'y_start': region_bounds['y_start'], 'y_end': region_bounds['y_end'],
                'z_start': region_bounds['z_start'], 'z_end': region_bounds['z_end']
            }
        elif total_count == 4:
            # 2x2 grid
            sub_width = region_width // 2
            sub_height = region_height // 2
            bx, by = index % 2, index // 2
            
            return {
                'x_start': region_bounds['x_start'] + bx * sub_width,
                'x_end': region_bounds['x_start'] + (bx + 1) * sub_width if bx == 0 else region_bounds['x_end'],
                'y_start': region_bounds['y_start'] + by * sub_height,
                'y_end': region_bounds['y_start'] + (by + 1) * sub_height if by == 0 else region_bounds['y_end'],
                'z_start': region_bounds['z_start'], 'z_end': region_bounds['z_end']
            }
        else:
            return region_bounds

    def create_brain_phi_resonance_field(self):
        """Create phi resonance field in hemispheres."""
        logger.info("Creating brain phi resonance field")
        
        try:
            hemispheres_data = self.grid.get('hemispheres', {})
            
            for hemisphere_id, hemisphere_data in hemispheres_data.items():
                bounds = hemisphere_data['boundaries']
                center = ((bounds['x_start'] + bounds['x_end']) // 2,
                         (bounds['y_start'] + bounds['y_end']) // 2,
                         (bounds['z_start'] + bounds['z_end']) // 2)
                
                phi_field = {
                    'field_id': str(uuid.uuid4()),
                    'hemisphere_id': hemisphere_id,
                    'creation_time': datetime.now().isoformat(),
                    'center': center,
                    'phi_frequency': 432.0 * PHI,
                    'field_strength': 0.618,
                    'field_type': 'phi_resonance_gaussian'
                }
                
                hemisphere_data['phi_resonance_field'] = phi_field
                self.phi_resonance_field[hemisphere_id] = phi_field
            
            logger.info(f"Phi resonance fields created for {len(hemispheres_data)} hemispheres")
            
        except Exception as e:
            logger.error(f"Failed to create phi resonance field: {e}")
            raise RuntimeError(f"Phi resonance field creation failed: {e}")

    def assign_brain_hemispheres_properties(self):
        """Assign hemisphere wave properties."""
        logger.info("Assigning hemisphere properties")
        
        try:
            hemispheres_data = self.grid.get('hemispheres', {})
            
            # Get properties from region_definitions
            hemisphere_properties = self.get_hemisphere_wave_properties()
            
            for hemisphere_id, hemisphere_data in hemispheres_data.items():
                # Map hemisphere IDs to property keys
                prop_key = 'left_hemisphere' if hemisphere_id == 'L1' else 'right_hemisphere'
                props = hemisphere_properties.get(prop_key, hemisphere_properties['left_hemisphere'])
                hemisphere_data.update(props)
                hemisphere_data['properties_assigned'] = True
            
            self.hemispheres = hemispheres_data
            logger.info(f"Properties assigned to {len(hemispheres_data)} hemispheres")
            
        except Exception as e:
            logger.error(f"Failed to assign hemisphere properties: {e}")
            raise RuntimeError(f"Hemisphere property assignment failed: {e}")

    def assign_brain_regions_properties(self):
        """Assign region wave properties."""
        logger.info("Assigning region properties")
        
        try:
            # Get properties from region_definitions
            region_wave_properties = self.get_region_wave_properties()
            
            for hemisphere_data in self.hemispheres.values():
                for region_name, region_data in hemisphere_data.get('regions', {}).items():
                    props = region_wave_properties.get(region_name, region_wave_properties['frontal'])
                    region_data.update(props)
                    region_data['properties_assigned'] = True
            
            # Update main regions storage
            self.regions = {}
            for hemisphere_data in self.hemispheres.values():
                self.regions.update(hemisphere_data.get('regions', {}))
            
            logger.info(f"Properties assigned to {len(self.regions)} regions")
            
        except Exception as e:
            logger.error(f"Failed to assign region properties: {e}")
            raise RuntimeError(f"Region property assignment failed: {e}")

    def assign_brain_sub_regions_properties(self):
        """Assign sub-region wave properties."""
        logger.info("Assigning sub-region properties")
        
        try:
            sub_region_count = 0
            
            for hemisphere_data in self.hemispheres.values():
                for region_data in hemisphere_data.get('regions', {}).values():
                    base_frequency = region_data.get('frequency', 10.0)
                    base_color = region_data.get('color', 'white')
                    
                    for sub_region_data in region_data.get('sub_regions', {}).values():
                        frequency_variance = random.uniform(0.9, 1.1)
                        sub_frequency = base_frequency * frequency_variance
                        
                        sub_region_data.update({
                            'frequency': sub_frequency,
                            'color': base_color,
                            'wave_type': region_data.get('wave_type', 'alpha'),
                            'properties_assigned': True
                        })
                        
                        sub_region_count += 1
            
            # Update main sub_regions storage
            self.sub_regions = {}
            for hemisphere_data in self.hemispheres.values():
                for region_data in hemisphere_data.get('regions', {}).values():
                    self.sub_regions.update(region_data.get('sub_regions', {}))
            
            logger.info(f"Properties assigned to {sub_region_count} sub-regions")
            
        except Exception as e:
            logger.error(f"Failed to assign sub-region properties: {e}")
            raise RuntimeError(f"Sub-region property assignment failed: {e}")

    def determine_static_borders_for_sub_regions(self):
            """Determine static borders for sub-regions."""
            logger.info("Determining static borders for sub-regions")
            
            try:
                border_list = []
                
                for hemisphere_data in self.hemispheres.values():
                    hemisphere_id = hemisphere_data['hemisphere_id']
                    
                    for region_name, region_data in hemisphere_data.get('regions', {}).items():
                        for sub_region_data in region_data.get('sub_regions', {}).values():
                            blocks = sub_region_data.get('blocks', {})
                            sub_bounds = sub_region_data['boundaries']
                            
                            for block_data in blocks.values():
                                bounds = block_data['boundaries']
                                
                                # Check if block is at border
                                is_border = (
                                    bounds['x_start'] <= sub_bounds['x_start'] + 10 or
                                    bounds['x_end'] >= sub_bounds['x_end'] - 10 or
                                    bounds['y_start'] <= sub_bounds['y_start'] + 10 or
                                    bounds['y_end'] >= sub_bounds['y_end'] - 10 or
                                    bounds['z_start'] <= sub_bounds['z_start'] + 10 or
                                    bounds['z_end'] >= sub_bounds['z_end'] - 10
                                )
                                
                                if is_border:
                                    border_coords = [(bounds['x_start'], bounds['y_start'], bounds['z_start']),
                                                (bounds['x_end'], bounds['y_end'], bounds['z_end'])]
                                    
                                    border_tuple = (hemisphere_id, region_name, sub_region_data['sub_region_id'], border_coords)
                                    border_list.append(border_tuple)
                
                self.sub_regions_borders = {
                    'border_list': border_list,
                    'border_count': len(border_list),
                    'creation_time': datetime.now().isoformat()
                }
                
                logger.info(f"Static borders determined for {len(border_list)} border blocks")
                
            except Exception as e:
                logger.error(f"Failed to determine static borders: {e}")
                raise RuntimeError(f"Static border determination failed: {e}")

    def apply_static_field_to_sub_regions(self):
        """Apply permeable static field to sub-region borders with real sound generation."""
        logger.info("Applying static field to sub-region borders")
    
        try:
            border_list = self.sub_regions_borders.get('border_list', [])
            
            if not border_list:
                raise ValueError("No border list found for static field application")
            
            static_fields = {}
            
            # Generate real static noise sound for all borders
            static_sound_file = None
            if SOUND_AVAILABLE and self.noise_generator:
                try:
                    # Generate realistic static noise
                    static_noise = self.generate_static_noise_safe(duration=1.0, amplitude=0.1)
                    
                    # Save static noise sound
                    static_sound_file = f"brain_static_field_{self.brain_id[:8]}.wav"
                    sound_path = self.sound_generator.save_sound(
                        static_noise,
                        static_sound_file,
                        f"Static Field Noise - Brain {self.brain_id[:8]}"
                    )
                    logger.info(f"Static field sound generated: {sound_path}")
                    
                except Exception as sound_err:
                    logger.warning(f"Failed to generate static field sound: {sound_err}")
                    static_sound_file = None
            
            for border_tuple in border_list:
                hemisphere_id, region_name, sub_region_id, border_coords = border_tuple
                
                field_strength = random.uniform(0.3, 0.7)
                permeability = random.uniform(0.2, 0.4)
                
                field_pattern = []
                for i, coord in enumerate(border_coords):
                    has_gap = (i % 3 == 0) or (random.random() < permeability)
                    field_pattern.append({
                        'coordinate': coord,
                        'field_strength': 0.0 if has_gap else field_strength,
                        'permeable': has_gap
                    })
                
                gap_count = sum(1 for fp in field_pattern if fp['permeable'])
                permeability_ratio = gap_count / len(field_pattern)
                
                if permeability_ratio < 0.15:
                    raise RuntimeError(f"Static field too tightly packed: {permeability_ratio:.2f}")
                
                field_id = f"{hemisphere_id}-{region_name}-{sub_region_id}-static"
                static_fields[field_id] = {
                    'field_id': field_id,
                    'field_pattern': field_pattern,
                    'field_strength': field_strength,
                    'permeability': permeability_ratio,
                    'sound_file': static_sound_file,
                    'creation_time': datetime.now().isoformat()
                }
            
            self.static_field = {
                'fields': static_fields,
                'field_count': len(static_fields),
                'static_sound_file': static_sound_file,
                'applied': True
            }
            
            logger.info(f"Static field applied to {len(static_fields)} sub-region borders")
            
        except Exception as e:
            logger.error(f"Failed to apply static field: {e}")
            raise RuntimeError(f"Static field application failed: {e}")
    def apply_brain_waves_to_sub_regions(self):
        """Apply brain waves to sub-regions with anomaly detection."""
        logger.info("Applying brain waves to sub-regions")
        
        try:
            wave_applications = []
            anomalies = []
            
            # Generate universal sounds for brain waves
            brain_wave_sound_file = None
            if SOUND_AVAILABLE and self.universe_sounds:
                try:
                    # Generate realistic universal brain wave sounds (varying calm/chaotic)
                    brain_waves = self.generate_cosmic_background_safe(
                        duration=4.0,
                        amplitude=0.3,
                        frequency_band='full'
                    )
                    
                    # Save brain wave sound
                    brain_wave_sound_file = f"brain_waves_{self.brain_id[:8]}.wav"
                    sound_path = self.universe_sounds.save_sound(
                        brain_waves,
                        brain_wave_sound_file,
                        f"Brain Waves - Brain {self.brain_id[:8]}"
                    )
                    logger.info(f"Brain wave sound generated: {sound_path}")
                    
                except Exception as sound_err:
                    logger.warning(f"Failed to generate brain wave sound: {sound_err}")
                    brain_wave_sound_file = None
            
            for hemisphere_data in self.hemispheres.values():
                for region_data in hemisphere_data.get('regions', {}).values():
                    region_frequency = region_data.get('frequency', 10.0)
                    
                    for sub_region_data in region_data.get('sub_regions', {}).values():
                        sub_frequency = sub_region_data.get('frequency', region_frequency)
                        
                        for block_data in sub_region_data.get('blocks', {}).values():
                            wave_amplitude = random.uniform(0.5, 1.0)
                            
                            # Anomaly detection
                            anomaly_detected = False
                            if abs(sub_frequency - 40.0) < 2.0:  # Interference with static field
                                anomaly_detected = True
                                anomalies.append({
                                    'block_id': block_data['block_id'],
                                    'reason': 'Frequency interference with static field'
                                })
                            
                            wave_application = {
                                'block_id': block_data['block_id'],
                                'frequency': sub_frequency,
                                'amplitude': wave_amplitude,
                                'anomaly_detected': anomaly_detected,
                                'sound_file': brain_wave_sound_file,
                                'application_time': datetime.now().isoformat()
                            }
                            
                            wave_applications.append(wave_application)
                            block_data['brain_wave'] = wave_application
            
            # Check for critical anomalies
            critical_anomalies = [a for a in anomalies if 'interference' in a['reason']]
            
            if critical_anomalies:
                error_msg = f"Critical brain wave anomalies: {len(critical_anomalies)} detected"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.brain_waves = {
                'wave_applications': wave_applications,
                'anomalies': anomalies,
                'application_count': len(wave_applications),
                'brain_wave_sound_file': brain_wave_sound_file,
                'applied': True
            }
            
            logger.info(f"Brain waves applied to {len(wave_applications)} blocks")
            
        except Exception as e:
            logger.error(f"Failed to apply brain waves: {e}")
            raise RuntimeError(f"Brain wave application failed: {e}")

    def test_field_integrity(self):
        """Test field integrity and detect critical issues."""
        logger.info("Testing field integrity")
        
        try:
            integrity_issues = []
            
            # Test all field systems
            if not self.external_field.get('applied', False):
                integrity_issues.append("External field not applied")
            
            if len(self.phi_resonance_field) != 2:
                integrity_issues.append(f"Incomplete phi fields: {len(self.phi_resonance_field)}/2")
            
            if not self.static_field.get('applied', False):
                integrity_issues.append("Static field not applied")
            
            if not self.brain_waves.get('applied', False):
                integrity_issues.append("Brain waves not applied")
            
            # Test for electromagnetic disturbance (safety check)
            total_field_energy = 0.0
            for field_data in self.static_field.get('fields', {}).values():
                total_field_energy += field_data.get('field_strength', 0.0)
            
            if total_field_energy > 50.0:  # Safety threshold
                integrity_issues.append(f"Dangerous field energy level: {total_field_energy:.2f}")
            
            # Check for critical issues
            critical_issues = [i for i in integrity_issues if 'not applied' in i or 'Dangerous' in i]
            
            if critical_issues:
                error_msg = f"Field integrity FAILED: {critical_issues}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.field_integrity = {
                'test_passed': len(integrity_issues) == 0,
                'issues_found': integrity_issues,
                'total_field_energy': total_field_energy,
                'electromagnetic_safe': total_field_energy <= 50.0,
                'test_time': datetime.now().isoformat()
            }
            
            logger.info("Field integrity test PASSED")
            
        except Exception as e:
            logger.error(f"Field integrity test failed: {e}")
            raise RuntimeError(f"Field integrity test failed: {e}")

    def save_brain_structure(self):
        """Save complete brain structure and set BRAIN_STRUCTURE_CREATED flag."""
        logger.info("Saving brain structure")
        
        try:
            # Final validation
            validation_errors = []
            
            if len(self.hemispheres) != 2:
                validation_errors.append(f"Expected 2 hemispheres, got {len(self.hemispheres)}")
            
            if not self.external_field:
                validation_errors.append("External field not created")
            
            if not self.field_integrity.get('test_passed', False):
                validation_errors.append("Field integrity test failed")
            
            if validation_errors:
                error_msg = f"Brain structure validation FAILED: {validation_errors}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Calculate final statistics
            total_blocks = 0
            for hemisphere_data in self.hemispheres.values():
                for region_data in hemisphere_data.get('regions', {}).values():
                    for sub_region_data in region_data.get('sub_regions', {}).values():
                        total_blocks += len(sub_region_data.get('blocks', {}))
            
            # Create complete brain structure
            self.brain = {
                'brain_id': self.brain_id,
                'creation_time': self.creation_time,
                'completion_time': datetime.now().isoformat(),
                'grid_dimensions': self.grid_dimensions,
                'hemispheres': self.hemispheres,
                'external_field': self.external_field,
                'phi_resonance_field': self.phi_resonance_field,
                'static_field': self.static_field,
                'brain_waves': self.brain_waves,
                'field_integrity': self.field_integrity,
                'statistics': {
                    'total_blocks': total_blocks,
                    'hemisphere_count': len(self.hemispheres),
                    'region_count': len(self.regions),
                    'sub_region_count': len(self.sub_regions),
                    'ready_for_neural_network': True
                },
                'structure_complete': True,
                'sound_files_generated': {
                    'external_field': self.external_field.get('sound_file'),
                    'static_field': self.static_field.get('static_sound_file'),
                    'brain_waves': self.brain_waves.get('brain_wave_sound_file')
                }
            }
            
            # Set completion flag
            setattr(self, FLAG_BRAIN_STRUCTURE_CREATED, True)
            
            logger.info(f"Brain structure saved: {total_blocks} blocks, {len(self.hemispheres)} hemispheres")
            
        except Exception as e:
            logger.error(f"Failed to save brain structure: {e}")
            raise RuntimeError(f"Brain structure save failed: {e}")

    def run_test_simulation(self, fake_seed: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run test simulation with fake seed and generate 3D visualization."""
        logger.info("Running brain structure test simulation")
        
        try:
            # Set fake seed for testing
            if fake_seed:
                self.brain_seed = fake_seed
            else:
                self.brain_seed = {
                    'position': (128, 128, 128),  # Center
                    'energy': 7.5,
                    'frequency': 432.0 * math.sqrt(2)
                }
            
            # Run development process
            self.trigger_brain_development()
            
            # Generate 3D visualization
            visualization_data = self._generate_3d_visualization()
            
            # Create test results
            test_results = {
                'test_passed': True,
                'brain_id': self.brain_id,
                'test_seed': self.brain_seed,
                'total_blocks': self.brain['statistics']['total_blocks'],
                'hemisphere_count': self.brain['statistics']['hemisphere_count'],
                'region_count': self.brain['statistics']['region_count'],
                'sub_region_count': self.brain['statistics']['sub_region_count'],
                'field_integrity_passed': self.field_integrity['test_passed'],
                'electromagnetic_safe': self.field_integrity['electromagnetic_safe'],
                'sound_files_generated': self.brain['sound_files_generated'],
                'visualization_data': visualization_data,
                'test_time': datetime.now().isoformat()
            }
            
            logger.info(f"Test simulation completed successfully: {test_results['total_blocks']} blocks created")
            return test_results
            
        except Exception as e:
            logger.error(f"Test simulation failed: {e}")
            return {
                'test_passed': False,
                'error': str(e),
                'test_time': datetime.now().isoformat()
            }

    def _generate_3d_visualization(self) -> Dict[str, Any]:
        """Generate simple 3D visualization of brain structure."""
        try:
            # Collect visualization data
            hemisphere_data = []
            region_data = []
            block_sample_data = []
            
            # Sample blocks for visualization (don't plot all 3500)
            sample_count = 0
            max_samples = 200  # Limit for performance
            
            for hemisphere_id, hemisphere in self.hemispheres.items():
                # Hemisphere boundaries
                bounds = hemisphere['boundaries']
                hemisphere_data.append({
                    'id': hemisphere_id,
                    'center': (
                        (bounds['x_start'] + bounds['x_end']) / 2,
                        (bounds['y_start'] + bounds['y_end']) / 2,
                        (bounds['z_start'] + bounds['z_end']) / 2
                    ),
                    'size': (
                        bounds['x_end'] - bounds['x_start'],
                        bounds['y_end'] - bounds['y_start'],
                        bounds['z_end'] - bounds['z_start']
                    )
                })
                
                # Region data
                for region_name, region in hemisphere.get('regions', {}).items():
                    r_bounds = region['boundaries']
                    region_data.append({
                        'hemisphere': hemisphere_id,
                        'name': region_name,
                        'center': (
                            (r_bounds['x_start'] + r_bounds['x_end']) / 2,
                            (r_bounds['y_start'] + r_bounds['y_end']) / 2,
                            (r_bounds['z_start'] + r_bounds['z_end']) / 2
                        ),
                        'color': region.get('color', 'gray')
                    })
                    
                    # Sample blocks (limited for performance)
                    for sub_region in region.get('sub_regions', {}).values():
                        for block_id, block in sub_region.get('blocks', {}).items():
                            if sample_count < max_samples and random.random() < 0.1:  # 10% sample
                                block_sample_data.append({
                                    'id': block_id,
                                    'center': block['center'],
                                    'hemisphere': hemisphere_id,
                                    'region': region_name
                                })
                                sample_count += 1
            
            # Create actual matplotlib 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot hemisphere boundaries
            for hemi in hemisphere_data:
                center = hemi['center']
                size = hemi['size']
                color = 'lightblue' if hemi['id'] == 'L1' else 'lightcoral'
                
                # Simple wireframe box for hemisphere
                x = [center[0] - size[0]/2, center[0] + size[0]/2]
                y = [center[1] - size[1]/2, center[1] + size[1]/2]
                z = [center[2] - size[2]/2, center[2] + size[2]/2]
                
                # Draw hemisphere outline
                ax.plot([x[0], x[1]], [y[0], y[0]], [z[0], z[0]], color=color, linewidth=2, alpha=0.7)
                ax.plot([x[0], x[1]], [y[1], y[1]], [z[1], z[1]], color=color, linewidth=2, alpha=0.7)
                ax.plot([x[0], x[0]], [y[0], y[1]], [z[0], z[0]], color=color, linewidth=2, alpha=0.7)
                ax.plot([x[1], x[1]], [y[0], y[1]], [z[1], z[1]], color=color, linewidth=2, alpha=0.7)
            
            # Plot sample blocks
            if block_sample_data:
                x_coords = [block['center'][0] for block in block_sample_data]
                y_coords = [block['center'][1] for block in block_sample_data]
                z_coords = [block['center'][2] for block in block_sample_data]
                
                ax.scatter(x_coords, y_coords, z_coords, 
                            c='darkblue', s=8, alpha=0.6, label='Sample Blocks')
            
            # Plot brain seed
            if self.brain_seed:
                seed_pos = self.brain_seed['position']
                ax.scatter([seed_pos[0]], [seed_pos[1]], [seed_pos[2]], 
                            c='red', s=100, marker='*', label='Brain Seed')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Brain Structure Test - {len(block_sample_data)} Sample Blocks')
            ax.legend()
            
            # Save visualization
            plt.tight_layout()
            viz_filename = f"brain_structure_test_{self.brain_id[:8]}.png"
            plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            return {
                'visualization_created': True,
                'visualization_file': viz_filename,
                'hemisphere_count': len(hemisphere_data),
                'region_count': len(region_data),
                'blocks_sampled': len(block_sample_data),
                'total_blocks': self.brain['statistics']['total_blocks']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return {
                'visualization_created': False,
                'error': str(e)
            }


# --- TEST FUNCTION ---
def test_brain_structure():
    """Test function to verify brain structure creation works correctly."""
    try:
        print("\n" + "="*50)
        print("TESTING BRAIN STRUCTURE CREATION")
        print("="*50)
        
        # Create brain instance
        print("1. Creating Brain instance...")
        brain = Brain()
        print(f"   ✅ Brain created with ID: {brain.brain_id[:8]}")
        
        # Trigger brain development
        print("2. Triggering brain development...")
        brain.trigger_brain_development()
        print("   ✅ Brain development triggered")
        
        # Run comprehensive test
        print("3. Running comprehensive brain test...")
        test_results = brain.run_test_simulation()
        
        # Display results
        print("\n" + "-"*40)
        print("TEST RESULTS")
        print("-"*40)
        
        if test_results['success']:
            print("✅ BRAIN STRUCTURE TEST PASSED!")
            metrics = test_results['metrics']
            print(f"   📊 Hemispheres: {metrics.get('hemisphere_count', 0)}")
            print(f"   📊 Regions: {metrics.get('total_regions', 0)}")
            print(f"   📊 Sub-regions: {metrics.get('total_sub_regions', 0)}")
            print(f"   📊 Blocks: {metrics.get('total_blocks', 0)}")
            print(f"   📊 External Field: {'✅' if metrics.get('has_external_field') else '❌'}")
            print(f"   📊 Phi Field: {'✅' if metrics.get('has_phi_field') else '❌'}")
            print(f"   📊 Static Field: {'✅' if metrics.get('has_static_field') else '❌'}")
        else:
            print("❌ BRAIN STRUCTURE TEST FAILED!")
            print(f"   🔸 {len(test_results['errors'])} errors found:")
            for i, error in enumerate(test_results['errors'], 1):
                print(f"      {i}. {error}")
        
        print("-"*40)
        return test_results['success']
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during brain test: {e}")
        import traceback
        traceback.print_exc()
        return False# FIX 3: CORRECT if __name__ == "__main__" SECTION
# Replace your if __name__ section with this:

if __name__ == "__main__":
    print("Brain Structure Module - Direct Test")
    success = test_brain_structure()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Brain structure is working correctly!")
    else:
        print("\n💥 TESTS FAILED - Check errors above")
    
    print(f"\nTest completed: {'SUCCESS' if success else 'FAILED'}")