# --- womb_environment.py V8 COMPLETE ---
"""
Creates complete womb environment with all fields: standing waves, phi ratio fields, 
and merkaba sacred geometry. Provides protective enclosure for brain formation.
Triggered by brain seed creation and integrates with stress monitoring system.
"""

import logging
import uuid
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import math
from shared.constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("Conception")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class Womb:
    """
    Womb environment - protective enclosure for brain formation with sacred geometry fields.
    """   

    def __init__(self):
        """Initialize womb environment."""
        self.womb = {}
        self.dimensions = None
        self.temperature = None
        self.humidity = None
        self.ph_level = None
        self.nutrients = None
        self.protection_field = None
        self.comfort_field = None
        self.field_parameters = None
        self.stress_level = None
        self.womb_created = False
        self.love_resonance = None
        self.standing_waves = None
        self.phi_ratio = None
        self.merkaba = None
        self.field_strength = 1.0
        self.field_stability = 1.0
        self.creation_metrics = {
            'creation_time': None,
            'field_applications': 0,
            'stability_checks': 0,
            'errors': []
        }

    def create_3d_womb(self) -> Dict[str, Any]:
        """
        Create a 3D womb environment with initial protective field and all basic parameters.
        If womb already exists, return existing womb. Hard fail on creation failure.
        """
        try:
            if self.womb_created and self.womb:
                logger.info("Womb already exists, returning existing womb")
                return self.womb
            
            logger.info("Creating 3D womb environment...")
            
            # Define womb dimensions (slightly larger than brain to provide buffer)
            brain_size = GRID_DIMENSIONS  # (256, 256, 256)
            buffer = 32  # Additional space around brain
            
            self.dimensions = {
                'x_min': -buffer,
                'x_max': brain_size[0] + buffer,
                'y_min': -buffer, 
                'y_max': brain_size[1] + buffer,
                'z_min': -buffer,
                'z_max': brain_size[2] + buffer,
                'volume': (brain_size[0] + 2*buffer) * (brain_size[1] + 2*buffer) * (brain_size[2] + 2*buffer)
            }
            
            # Set optimal biological parameters
            self.temperature = 37.0  # Celsius - body temperature
            self.humidity = 0.98     # 98% humidity - amniotic fluid environment
            self.ph_level = 7.4      # Slightly alkaline - optimal for development
            self.nutrients = 1.0     # Full nutrient availability
            
            # Initialize protection and comfort fields
            self.protection_field = {
                'strength': 1.0,
                'type': 'electromagnetic_shield',
                'frequency_range': (0.1, 1000.0),  # Hz
                'attenuation_factor': 0.95,
                'active': True
            }
            
            self.comfort_field = {
                'strength': 0.8,
                'type': 'harmonic_resonance',
                'base_frequency': 528.0,  # Love frequency
                'harmonics': [528.0, 396.0, 417.0, 639.0, 741.0, 852.0, 963.0],  # Solfeggio
                'active': True
            }
            
            # Set love resonance frequency
            self.love_resonance = 528.0  # Hz - Solfeggio love frequency
            
            # Initialize field strength parameters
            self.field_strength = 1.0
            self.field_stability = 1.0
            
            # Create basic womb structure
            womb_id = str(uuid.uuid4())
            creation_time = datetime.now().isoformat()
            
            self.womb = {
                'womb_id': womb_id,
                'creation_time': creation_time,
                'type': 'protective_3d_environment',
                'status': 'active',
                'dimensions': self.dimensions,
                'biological_parameters': {
                    'temperature_celsius': self.temperature,
                    'humidity_percentage': self.humidity * 100,
                    'ph_level': self.ph_level,
                    'nutrient_availability': self.nutrients
                },
                'protection_field': self.protection_field,
                'comfort_field': self.comfort_field,
                'love_resonance_frequency': self.love_resonance,
                'field_strength': self.field_strength,
                'field_stability': self.field_stability,
                'fields_applied': {
                    'standing_waves': False,
                    'phi_ratio': False,
                    'merkaba': False
                }
            }
            
            self.womb_created = True
            self.creation_metrics['creation_time'] = creation_time
            
            logger.info(f"3D womb environment created: {womb_id}")
            logger.info(f"Dimensions: {self.dimensions['volume']:.0f} cubic units")
            logger.info(f"Temperature: {self.temperature}°C, pH: {self.ph_level}, Humidity: {self.humidity*100:.1f}%")
            
            return self.womb
            
        except Exception as e:
            error_msg = f"Womb creation failed: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            raise RuntimeError(f"Womb creation failed - hard fail: {e}")

    def _apply_standing_waves_stabilization(self, cycle: int) -> float:
        """
        Apply standing wave stabilization using womb dimensions.
        Create standing waves in womb and save data to womb dictionary.
        Return new field parameters created by standing waves.
        """
        try:
            logger.info(f"Applying standing wave stabilization (cycle {cycle})...")
            
            if not self.womb_created or not hasattr(self, 'dimensions') or self.dimensions is None:
                raise RuntimeError("Womb not created or dimensions not set - cannot apply standing waves")
            
            # Calculate standing wave parameters based on womb dimensions
            width = self.dimensions['x_max'] - self.dimensions['x_min']
            height = self.dimensions['y_max'] - self.dimensions['y_min'] 
            depth = self.dimensions['z_max'] - self.dimensions['z_min']
            
            # Use Schumann resonance as base frequency
            base_frequency = SCHUMANN_FREQUENCY  # 7.83 Hz
            
            # Calculate wavelengths for each dimension
            wave_speed = SPEED_OF_SOUND  # 343 m/s
            wavelength_base = wave_speed / base_frequency
            
            # Create standing wave nodes and antinodes for each dimension
            nodes_x = max(2, int(width / (wavelength_base / 4)))  # Quarter wavelength spacing
            nodes_y = max(2, int(height / (wavelength_base / 4)))
            nodes_z = max(2, int(depth / (wavelength_base / 4)))
            
            # Generate standing wave pattern
            standing_wave_pattern = {
                'pattern_id': str(uuid.uuid4()),
                'cycle': cycle,
                'base_frequency': base_frequency,
                'wavelength': wavelength_base,
                'dimensions': {
                    'x_nodes': nodes_x,
                    'y_nodes': nodes_y,
                    'z_nodes': nodes_z
                },
                'wave_equations': {},
                'field_strength_modifier': 1.0,
                'stability_enhancement': 0.0
            }
            
            # Calculate wave equations for each dimension
            for dim, nodes in [('x', nodes_x), ('y', nodes_y), ('z', nodes_z)]:
                dim_size = width if dim == 'x' else (height if dim == 'y' else depth)
                node_spacing = dim_size / (nodes - 1) if nodes > 1 else dim_size
                
                wave_equation = {
                    'dimension': dim,
                    'nodes': nodes,
                    'node_spacing': node_spacing,
                    'amplitude': BIRTH_STANDING_WAVE_AMPLITUDE,
                    'phase_shift': 0.0,
                    'harmonic_series': []
                }
                
                # Create harmonic series (up to 5 harmonics)
                for harmonic in range(1, 6):
                    harmonic_freq = base_frequency * harmonic
                    harmonic_amplitude = BIRTH_STANDING_WAVE_AMPLITUDE / harmonic
                    
                    wave_equation['harmonic_series'].append({
                        'harmonic_number': harmonic,
                        'frequency': harmonic_freq,
                        'amplitude': harmonic_amplitude,
                        'wavelength': wave_speed / harmonic_freq
                    })
                
                standing_wave_pattern['wave_equations'][dim] = wave_equation
            
            # Calculate field enhancement from standing waves
            # Standing waves create interference patterns that stabilize the field
            node_density = (nodes_x * nodes_y * nodes_z) / self.dimensions['volume']
            field_enhancement = min(0.3, node_density * 1000)  # Cap at 30% enhancement
            
            # Apply golden ratio modulation for optimal resonance
            phi_modulation = math.sin(cycle * PHI) * 0.1  # ±10% modulation
            total_enhancement = field_enhancement + phi_modulation
            
            standing_wave_pattern['field_strength_modifier'] = 1.0 + total_enhancement
            standing_wave_pattern['stability_enhancement'] = total_enhancement
            
            # Store standing wave data in womb
            self.standing_waves = standing_wave_pattern
            self.womb['standing_waves'] = standing_wave_pattern
            self.womb['fields_applied']['standing_waves'] = True
            
            # Update field parameters
            new_field_strength = self.field_strength * standing_wave_pattern['field_strength_modifier']
            self.field_strength = min(2.0, new_field_strength)  # Cap at 2x strength
            
            # Update womb field strength
            self.womb['field_strength'] = self.field_strength
            
            self.creation_metrics['field_applications'] += 1
            
            logger.info(f"Standing waves applied: {nodes_x}×{nodes_y}×{nodes_z} nodes, "
                       f"field enhancement: {total_enhancement:.3f}")
            
            return self.field_strength
            
        except Exception as e:
            error_msg = f"Standing wave application failed: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            raise RuntimeError(f"Standing wave stabilization failed: {e}")
    
    def _apply_phi_stabilization(self, cycle: int) -> float:
        """
        Apply phi ratio stabilization using womb dimensions.
        Create phi ratio field in womb and save data to womb dictionary.
        Return new field parameters created by phi ratio.
        """
        try:
            logger.info(f"Applying phi ratio stabilization (cycle {cycle})...")
            
            if not self.womb_created or not hasattr(self, 'dimensions') or not self.dimensions:
                raise RuntimeError("Womb not created or dimensions not set - cannot apply phi ratio field")
            
            if not hasattr(self, 'love_resonance') or self.love_resonance is None:
                self.love_resonance = 1.0  # Default value if not set
            
            # Calculate phi-based field parameters
            phi = PHI  # Golden ratio ≈ 1.618
            
            # Create phi spiral field based on womb dimensions
            center_x = (self.dimensions['x_max'] + self.dimensions['x_min']) / 2
            center_y = (self.dimensions['y_max'] + self.dimensions['y_min']) / 2
            center_z = (self.dimensions['z_max'] + self.dimensions['z_min']) / 2
            
            max_radius = min(
                (self.dimensions['x_max'] - self.dimensions['x_min']) / 2,
                (self.dimensions['y_max'] - self.dimensions['y_min']) / 2,
                (self.dimensions['z_max'] - self.dimensions['z_min']) / 2
            )
            
            # Generate phi spiral coordinates
            phi_spiral_points = []
            spiral_rotations = 3  # Number of complete rotations
            points_per_rotation = 21  # Fibonacci number
            total_points = spiral_rotations * points_per_rotation
            
            for i in range(total_points):
                # Phi-based spiral angle
                angle = i * 2 * math.pi * phi / points_per_rotation
                
                # Radius increases with phi progression
                radius_factor = (i / total_points) ** (1/phi)  # Phi-based growth
                radius = radius_factor * max_radius
                
                # 3D phi spiral coordinates
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                z = center_z + (radius_factor - 0.5) * max_radius * 0.5  # Vertical component
                
                phi_spiral_points.append({
                    'point_id': i,
                    'coordinates': (x, y, z),
                    'radius': radius,
                    'angle': angle,
                    'phi_factor': radius_factor
                })
            
            # Create phi field structure
            phi_field = {
                'field_id': str(uuid.uuid4()),
                'cycle': cycle,
                'phi_constant': phi,
                'center_coordinates': (center_x, center_y, center_z),
                'max_radius': max_radius,
                'spiral_points': phi_spiral_points,
                'field_equations': {},
                'resonance_zones': [],
                'field_strength_modifier': 1.0,
                'harmony_enhancement': 0.0
            }
            
            # Calculate phi resonance zones (areas of constructive interference)
            resonance_zones = []
            for i in range(0, len(phi_spiral_points), int(points_per_rotation/phi)):
                point = phi_spiral_points[i]
                zone = {
                    'zone_id': len(resonance_zones),
                    'center': point['coordinates'],
                    'radius': max_radius / (phi ** (len(resonance_zones) + 1)),
                    'resonance_strength': 1.0 / (len(resonance_zones) + 1),
                    'phi_harmonic': (len(resonance_zones) + 1) * phi
                }
                resonance_zones.append(zone)
            
            phi_field['resonance_zones'] = resonance_zones
            
            # Create field equations based on phi relationships
            phi_field['field_equations'] = {
                'radial_equation': f'r = r0 * φ^t',
                'angular_equation': f'θ = t * 2π * φ',
                'vertical_equation': f'z = z0 * (r/r_max - 0.5)',
                'field_strength': f'F = F0 * (1 + φ^(-n))',
                'resonance_frequency': self.love_resonance * phi
            }
            
            # Calculate field enhancement from phi ratios
            # Phi creates natural harmony and stability
            phi_zones = len(resonance_zones)
            harmony_enhancement = min(0.25, phi_zones * 0.05)  # Up to 25% enhancement
            
            # Apply cycle-based phi modulation
            cycle_phi_factor = math.cos(cycle / phi) * 0.1  # ±10% modulation
            total_enhancement = harmony_enhancement + cycle_phi_factor
            
            phi_field['field_strength_modifier'] = 1.0 + total_enhancement
            phi_field['harmony_enhancement'] = total_enhancement
            
            # Store phi field data in womb
            self.phi_ratio = phi_field
            self.womb['phi_ratio_field'] = phi_field
            self.womb['fields_applied']['phi_ratio'] = True
            
            # Update field parameters
            new_field_strength = self.field_strength * phi_field['field_strength_modifier']
            self.field_strength = min(2.5, new_field_strength)  # Cap at 2.5x strength
            
            # Update womb field strength
            self.womb['field_strength'] = self.field_strength
            
            self.creation_metrics['field_applications'] += 1
            
            logger.info(f"Phi ratio field applied: {phi_zones} resonance zones, "
                       f"harmony enhancement: {total_enhancement:.3f}")
            
            return self.field_strength
            
        except Exception as e:
            error_msg = f"Phi ratio application failed: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            raise RuntimeError(f"Phi ratio stabilization failed: {e}")
    
    def _apply_merkaba_stabilization(self, cycle: int) -> float:
        """
        Apply merkaba sacred geometry stabilization using womb dimensions.
        Create merkaba field in womb and save data to womb dictionary.
        Return new field parameters created by merkaba.
        """
        try:
            logger.info(f"Applying merkaba sacred geometry stabilization (cycle {cycle})...")
            
            if not self.womb_created or not hasattr(self, 'dimensions') or not self.dimensions:
                raise RuntimeError("Womb not created or dimensions not set - cannot apply merkaba field")
            
            # Calculate merkaba field parameters
            center_x = (self.dimensions['x_max'] + self.dimensions['x_min']) / 2
            center_y = (self.dimensions['y_max'] + self.dimensions['y_min']) / 2
            center_z = (self.dimensions['z_max'] + self.dimensions['z_min']) / 2
            
            # Merkaba is two interlocked tetrahedra (Star of David in 3D)
            # Calculate tetrahedron edge length based on womb size
            max_dimension = min(
                self.dimensions['x_max'] - self.dimensions['x_min'],
                self.dimensions['y_max'] - self.dimensions['y_min'],
                self.dimensions['z_max'] - self.dimensions['z_min']
            )
            
            # Edge length should fit within womb with some margin
            edge_length = max_dimension * 0.6  # 60% of available space
            
            # Calculate tetrahedron vertices for both upper and lower tetrahedra
            # Upper tetrahedron (pointing up)
            height = edge_length * math.sqrt(2/3)
            radius = edge_length * math.sqrt(3/8)
            
            upper_tetrahedron = {
                'tetrahedron_id': 'upper',
                'orientation': 'ascending',
                'vertices': [
                    (center_x, center_y + radius, center_z - height/3),  # Front
                    (center_x - radius * math.cos(math.pi/6), center_y - radius * math.sin(math.pi/6), center_z - height/3),  # Back left
                    (center_x + radius * math.cos(math.pi/6), center_y - radius * math.sin(math.pi/6), center_z - height/3),  # Back right
                    (center_x, center_y, center_z + 2*height/3)  # Apex
                ],
                'edge_length': edge_length,
                'volume': (edge_length ** 3) / (6 * math.sqrt(2))
            }
            
            # Lower tetrahedron (pointing down, interlocked)
            lower_tetrahedron = {
                'tetrahedron_id': 'lower',
                'orientation': 'descending',
                'vertices': [
                    (center_x, center_y + radius, center_z + height/3),  # Front
                    (center_x - radius * math.cos(math.pi/6), center_y - radius * math.sin(math.pi/6), center_z + height/3),  # Back left
                    (center_x + radius * math.cos(math.pi/6), center_y - radius * math.sin(math.pi/6), center_z + height/3),  # Back right
                    (center_x, center_y, center_z - 2*height/3)  # Apex
                ],
                'edge_length': edge_length,
                'volume': (edge_length ** 3) / (6 * math.sqrt(2))
            }
            
            # Calculate merkaba rotation (counter-rotating tetrahedra)
            rotation_speed = MERKABA_ROTATION_SPEED  # 7.23 Hz from constants
            upper_rotation = (cycle * rotation_speed) % 360  # Clockwise
            lower_rotation = (-cycle * rotation_speed) % 360  # Counter-clockwise
            
            # Create merkaba field structure
            merkaba_field = {
                'field_id': str(uuid.uuid4()),
                'cycle': cycle,
                'center_coordinates': (center_x, center_y, center_z),
                'edge_length': edge_length,
                'upper_tetrahedron': upper_tetrahedron,
                'lower_tetrahedron': lower_tetrahedron,
                'rotation_speed': rotation_speed,
                'current_rotations': {
                    'upper': upper_rotation,
                    'lower': lower_rotation
                },
                'energy_vortex': {},
                'protection_field': {},
                'field_strength_modifier': 1.0,
                'spiritual_enhancement': 0.0
            }
            
            # Calculate energy vortex created by counter-rotation
            vortex_strength = abs(math.sin(math.radians(upper_rotation - lower_rotation))) * 0.5
            energy_vortex = {
                'vortex_center': (center_x, center_y, center_z),
                'vortex_strength': vortex_strength,
                'rotation_differential': abs(upper_rotation - lower_rotation),
                'energy_flow_direction': 'ascending' if upper_rotation > lower_rotation else 'descending',
                'vortex_radius': radius,
                'energy_concentration': vortex_strength * 2.0
            }
            
            merkaba_field['energy_vortex'] = energy_vortex
            
            # Calculate protection field generated by merkaba
            protection_field = {
                'field_type': 'merkaba_shield',
                'protection_strength': 0.8 + vortex_strength * 0.4,  # 80-120% strength
                'field_harmonics': [
                    rotation_speed,
                    rotation_speed * PHI,
                    rotation_speed * 2,
                    rotation_speed * 3
                ],
                'interference_patterns': [],
                'spiritual_frequency': 741.0  # Solfeggio frequency for spiritual cleansing
            }
            
            # Create interference patterns from rotating tetrahedra
            for i in range(4):  # 4 major interference zones
                angle = i * 90  # 90-degree spacing
                interference_x = center_x + radius * math.cos(math.radians(angle))
                interference_y = center_y + radius * math.sin(math.radians(angle))
                
                pattern = {
                    'pattern_id': i,
                    'coordinates': (interference_x, interference_y, center_z),
                    'interference_type': 'constructive' if i % 2 == 0 else 'stabilizing',
                    'strength': 0.6 + vortex_strength * 0.3,
                    'frequency': protection_field['spiritual_frequency'] * (i + 1)
                }
                protection_field['interference_patterns'].append(pattern)
            
            merkaba_field['protection_field'] = protection_field
            
            # Calculate field enhancement from merkaba sacred geometry
            # Merkaba provides powerful spiritual protection and energy focusing
            geometric_enhancement = min(0.4, vortex_strength + 0.2)  # Up to 40% enhancement
            
            # Apply sacred geometry resonance based on cycle
            sacred_resonance = math.sin(cycle / 7.23) * 0.15  # Based on merkaba frequency
            total_enhancement = geometric_enhancement + sacred_resonance
            
            merkaba_field['field_strength_modifier'] = 1.0 + total_enhancement
            merkaba_field['spiritual_enhancement'] = total_enhancement
            
            # Store merkaba field data in womb
            self.merkaba = merkaba_field
            self.womb['merkaba_field'] = merkaba_field
            self.womb['fields_applied']['merkaba'] = True
            
            # Update field parameters
            new_field_strength = self.field_strength * merkaba_field['field_strength_modifier']
            self.field_strength = min(3.0, new_field_strength)  # Cap at 3x strength
            
            # Update womb field strength
            self.womb['field_strength'] = self.field_strength
            
            self.creation_metrics['field_applications'] += 1
            
            logger.info(f"Merkaba field applied: vortex strength {vortex_strength:.3f}, "
                       f"spiritual enhancement: {total_enhancement:.3f}")
            
            return self.field_strength
            
        except Exception as e:
            error_msg = f"Merkaba application failed: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            raise RuntimeError(f"Merkaba stabilization failed: {e}")

    def apply_all_field_stabilizations(self, development_cycles: int = 3) -> Dict[str, Any]:
        """
        Apply all field stabilizations in sequence: standing waves, phi ratio, merkaba.
        Perform for specified number of development cycles.
        """
        try:
            logger.info(f"Applying all field stabilizations for {development_cycles} cycles...")
            
            if not self.womb_created:
                raise RuntimeError("Womb not created - cannot apply field stabilizations")
            
            field_results = {
                'cycles_completed': 0,
                'standing_waves_results': [],
                'phi_ratio_results': [],
                'merkaba_results': [],
                'final_field_strength': 0.0,
                'total_enhancement': 0.0
            }
            
            initial_field_strength = self.field_strength
            
            for cycle in range(1, development_cycles + 1):
                logger.info(f"Field stabilization cycle {cycle}/{development_cycles}")
                
                # Apply standing waves
                standing_wave_strength = self._apply_standing_waves_stabilization(cycle)
                field_results['standing_waves_results'].append({
                    'cycle': cycle,
                    'field_strength': standing_wave_strength,
                    'enhancement': standing_wave_strength / initial_field_strength - 1
                })
                
                # Apply phi ratio field
                phi_field_strength = self._apply_phi_stabilization(cycle)
                field_results['phi_ratio_results'].append({
                    'cycle': cycle,
                    'field_strength': phi_field_strength,
                    'enhancement': phi_field_strength / initial_field_strength - 1
                })
                
                # Apply merkaba sacred geometry
                merkaba_field_strength = self._apply_merkaba_stabilization(cycle)
                field_results['merkaba_results'].append({
                    'cycle': cycle,
                    'field_strength': merkaba_field_strength,
                    'enhancement': merkaba_field_strength / initial_field_strength - 1
                })
                
                field_results['cycles_completed'] = cycle
                
                # Check field stability after each cycle
                self._check_field_stability()
            
            # Calculate final results
            field_results['final_field_strength'] = self.field_strength
            field_results['total_enhancement'] = (self.field_strength / initial_field_strength) - 1
            
            # Update womb with final field state
            self.womb['field_stabilization_complete'] = True
            self.womb['development_cycles'] = development_cycles
            self.womb['field_enhancement_factor'] = field_results['total_enhancement']
            
            logger.info(f"All field stabilizations complete: {field_results['total_enhancement']:.1%} enhancement")
            
            return field_results
            
        except Exception as e:
            error_msg = f"Field stabilization sequence failed: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            raise RuntimeError(f"Field stabilization failed: {e}")

    def _check_field_stability(self) -> bool:
        """Check overall field stability and integrity."""
        try:
            # Calculate stability based on field interactions
            stability_factors = []
            
            # Standing wave contribution
            if self.standing_waves:
                wave_stability = self.standing_waves.get('stability_enhancement', 0)
                stability_factors.append(max(0.5, 1.0 + wave_stability))
            
            # Phi ratio contribution  
            if self.phi_ratio:
                phi_stability = self.phi_ratio.get('harmony_enhancement', 0)
                stability_factors.append(max(0.5, 1.0 + phi_stability))
            
            # Merkaba contribution
            if self.merkaba:
                merkaba_stability = self.merkaba.get('spiritual_enhancement', 0)
                stability_factors.append(max(0.5, 1.0 + merkaba_stability))
            
            # Calculate overall stability
            if stability_factors:
                self.field_stability = np.mean(stability_factors)
            else:
                self.field_stability = 1.0
            
            # Update womb stability
            self.womb['field_stability'] = self.field_stability
            
            self.creation_metrics['stability_checks'] += 1
            
            # Check if stability is acceptable
            stable = bool(self.field_stability >= FIELD_INTEGRITY_THRESHOLD)
            
            if not stable:
                logger.warning(f"Field stability below threshold: {self.field_stability:.3f} < {FIELD_INTEGRITY_THRESHOLD}")
            
            return stable
            
        except Exception as e:
            logger.error(f"Field stability check failed: {e}")
            return False

    def save_womb(self) -> Dict[str, Any]:
        """
        Save womb environment with all parameters and field data to womb dictionary. 
        Set flag to WOMB_CREATED.
        """
        try:
            logger.info("Saving complete womb environment...")
            
            if not self.womb_created:
                raise RuntimeError("Womb not created - cannot save")
            
            # Ensure all field data is included
            self.womb['womb_parameters'] = {
                'dimensions': self.dimensions,
                'temperature': self.temperature,
                'humidity': self.humidity,
                'ph_level': self.ph_level,
                'nutrients': self.nutrients,
                'protection_field': self.protection_field,
                'comfort_field': self.comfort_field,
                'love_resonance': self.love_resonance
            }
            
            self.womb['field_data'] = {
                'standing_waves': self.standing_waves,
                'phi_ratio': self.phi_ratio,
                'merkaba': self.merkaba,
                'field_strength': self.field_strength,
                'field_stability': self.field_stability
            }
            
            # Add creation metrics
            self.womb['creation_metrics'] = self.creation_metrics
            
            # Set completion flag
            setattr(self, FLAG_WOMB_CREATED, True)
            self.womb['flag_womb_created'] = True
            
            # Calculate final womb score
            womb_score = self._calculate_womb_quality_score()
            self.womb['womb_quality_score'] = womb_score
            
            logger.info(f"Womb saved successfully with quality score: {womb_score:.2f}")
            logger.info(f"Fields applied: Standing waves: {bool(self.standing_waves)}, "
                       f"Phi ratio: {bool(self.phi_ratio)}, Merkaba: {bool(self.merkaba)}")
            
            return {'success': True, 'womb_id': self.womb['womb_id'], 'quality_score': womb_score}
            
        except Exception as e:
            error_msg = f"Failed to save womb: {e}"
            logger.error(error_msg)
            self.creation_metrics['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}

    def _calculate_womb_quality_score(self) -> float:
        """Calculate overall womb quality score based on all parameters."""
        try:
            score_components = []
            
            # Basic parameters score (25%)
            basic_score = 0.0
            if self.temperature == 37.0:  # Perfect body temperature
                basic_score += 0.25
            if self.humidity >= 0.95:  # High humidity like amniotic fluid
                basic_score += 0.25
            if 7.2 <= self.ph_level <= 7.6:  # Optimal pH range
                basic_score += 0.25
            if self.nutrients == 1.0:  # Full nutrients
                basic_score += 0.25
            score_components.append(basic_score)
            
            # Field strength score (25%)
            field_score = min(1.0, self.field_strength / 2.0)  # Normalize to max expected strength
            score_components.append(field_score)
            
            # Field stability score (25%)
            stability_score = self.field_stability
            score_components.append(stability_score)
            
            # Field completeness score (25%)
            fields_applied = sum([
                1 if self.standing_waves else 0,
                1 if self.phi_ratio else 0,
                1 if self.merkaba else 0
            ])
            completeness_score = fields_applied / 3.0
            score_components.append(completeness_score)
            
            # Calculate weighted average
            total_score = np.mean(score_components)
            
            return round(total_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating womb quality score: {e}")
            return 0.5  # Default moderate score

    def get_womb_for_integration(self) -> Dict[str, Any]:
        """
        Get complete womb data for integration with brain formation and birth processes.
        """
        try:
            if not self.womb_created or not self.womb:
                raise RuntimeError("Womb not created or saved")
            
            integration_data = {
                'womb_environment': self.womb,
                'field_parameters': {
                    'standing_waves_active': bool(self.standing_waves),
                    'phi_ratio_active': bool(self.phi_ratio),
                    'merkaba_active': bool(self.merkaba),
                    'field_strength': self.field_strength,
                    'field_stability': self.field_stability
                },
                'biological_parameters': {
                    'temperature': self.temperature,
                    'humidity': self.humidity,
                    'ph_level': self.ph_level,
                    'nutrients': self.nutrients
                },
                'protection_systems': {
                    'electromagnetic_shield': self.protection_field,
                    'harmonic_resonance': self.comfort_field,
                    'love_frequency': self.love_resonance
                },
                'ready_for_brain_formation': True,
                'ready_for_stress_monitoring': True
            }
            
            return integration_data
            
        except Exception as e:
            logger.error(f"Error getting womb for integration: {e}")
            raise RuntimeError(f"Womb integration data failed: {e}")

    def enhance_womb_for_brain_formation(self, brain_seed_energy: float) -> Dict[str, Any]:
        """
        Enhance womb environment specifically for brain formation process.
        Adjust field parameters based on brain seed energy.
        """
        try:
            logger.info(f"Enhancing womb for brain formation (seed energy: {brain_seed_energy:.2f})")
            
            if not self.womb_created:
                raise RuntimeError("Womb not created - cannot enhance")
            
            # Calculate enhancement factors based on brain seed energy
            energy_factor = min(2.0, brain_seed_energy / 100.0)  # Normalize to reasonable range
            
            enhancement_results = {
                'energy_enhancement': 0.0,
                'frequency_stabilization': 0.0,
                'growth_support': 0.0,
                'protection_boost': 0.0
            }
            
            # Energy enhancement
            energy_boost = energy_factor * WOMB_ENERGY_ENHANCEMENT_FACTOR
            self.field_strength *= (1.0 + energy_boost)
            enhancement_results['energy_enhancement'] = energy_boost
            
            # Frequency stabilization
            freq_stabilization = energy_factor * WOMB_FREQUENCY_STABILIZATION_FACTOR
            if self.comfort_field:
                self.comfort_field['strength'] *= (1.0 + freq_stabilization)
            enhancement_results['frequency_stabilization'] = freq_stabilization
            
            # Growth support enhancement
            growth_support = energy_factor * WOMB_GROWTH_ENHANCEMENT_FACTOR
            self.nutrients = min(1.0, self.nutrients * (1.0 + growth_support))
            enhancement_results['growth_support'] = growth_support
            
            # Protection boost
            protection_boost = energy_factor * 0.1  # 10% boost per energy factor
            if self.protection_field:
                self.protection_field['strength'] = min(1.0, 
                    self.protection_field['strength'] * (1.0 + protection_boost))
            enhancement_results['protection_boost'] = protection_boost
            
            # Update womb with enhancement data
            self.womb['brain_formation_enhancements'] = enhancement_results
            self.womb['enhanced_for_brain_formation'] = True
            self.womb['enhancement_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Womb enhanced for brain formation: "
                       f"energy +{energy_boost:.1%}, stability +{freq_stabilization:.1%}")
            
            return enhancement_results
            
        except Exception as e:
            logger.error(f"Womb enhancement failed: {e}")
            raise RuntimeError(f"Womb enhancement for brain formation failed: {e}")

    def get_womb_metrics(self) -> Dict[str, Any]:
        """Get comprehensive womb metrics for monitoring and debugging."""
        try:
            metrics = {
                'womb_id': self.womb.get('womb_id', 'unknown'),
                'creation_timestamp': self.creation_metrics.get('creation_time'),
                'womb_created': self.womb_created,
                'dimensions': self.dimensions,
                'biological_parameters': {
                    'temperature_celsius': self.temperature,
                    'humidity_percentage': self.humidity * 100 if self.humidity else 0,
                    'ph_level': self.ph_level,
                    'nutrient_availability': self.nutrients
                },
                'field_status': {
                    'field_strength': self.field_strength,
                    'field_stability': self.field_stability,
                    'standing_waves_applied': bool(self.standing_waves),
                    'phi_ratio_applied': bool(self.phi_ratio),
                    'merkaba_applied': bool(self.merkaba)
                },
                'creation_metrics': self.creation_metrics,
                'quality_score': self.womb.get('womb_quality_score', 0.0),
                'ready_for_use': self.womb_created and self.field_strength > 0.5
            }
            
            # Add field-specific details if available
            if self.standing_waves:
                metrics['standing_wave_details'] = {
                    'base_frequency': self.standing_waves.get('base_frequency'),
                    'enhancement': self.standing_waves.get('stability_enhancement'),
                    'node_count': f"{self.standing_waves.get('dimensions', {}).get('x_nodes', 0)}×"
                                 f"{self.standing_waves.get('dimensions', {}).get('y_nodes', 0)}×"
                                 f"{self.standing_waves.get('dimensions', {}).get('z_nodes', 0)}"
                }
            
            if self.phi_ratio:
                metrics['phi_ratio_details'] = {
                    'phi_constant': self.phi_ratio.get('phi_constant'),
                    'resonance_zones': len(self.phi_ratio.get('resonance_zones', [])),
                    'harmony_enhancement': self.phi_ratio.get('harmony_enhancement')
                }
            
            if self.merkaba:
                metrics['merkaba_details'] = {
                    'rotation_speed': self.merkaba.get('rotation_speed'),
                    'vortex_strength': self.merkaba.get('energy_vortex', {}).get('vortex_strength'),
                    'protection_strength': self.merkaba.get('protection_field', {}).get('protection_strength')
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting womb metrics: {e}")
            return {'error': str(e), 'womb_created': False}

# ===== UTILITY FUNCTIONS =====

def create_complete_womb_environment(development_cycles: int = 3) -> Womb:
    """
    Create complete womb environment with all fields applied.
    Returns fully configured Womb instance.
    """
    try:
        # Create womb instance
        womb = Womb()
        
        # Create 3D womb environment
        womb_data = womb.create_3d_womb()
        logger.info(f"Base womb created: {womb_data.get('womb_id')}")
        
        # Apply all field stabilizations
        field_results = womb.apply_all_field_stabilizations(development_cycles)
        logger.info(f"Field stabilizations complete: {field_results['total_enhancement']:.1%} enhancement")
        
        # Save complete womb
        save_result = womb.save_womb()
        if not save_result['success']:
            raise RuntimeError(f"Womb save failed: {save_result.get('error')}")
        
        logger.info(f"Complete womb environment created with quality score: {save_result['quality_score']:.2f}")
        
        return womb
        
    except Exception as e:
        logger.error(f"Failed to create complete womb environment: {e}")
        raise RuntimeError(f"Complete womb creation failed: {e}")

def test_womb_environment_functionality():
    """Test function to verify womb environment works correctly."""
    try:
        # Create complete womb
        womb = create_complete_womb_environment(development_cycles=2)
        
        # Test enhancement
        enhancement = womb.enhance_womb_for_brain_formation(150.0)
        print(f"Womb enhancement: {enhancement}")
        
        # Test metrics
        metrics = womb.get_womb_metrics()
        print(f"Womb quality score: {metrics['quality_score']:.2f}")
        print(f"Fields applied: SW={metrics['field_status']['standing_waves_applied']}, "
              f"Phi={metrics['field_status']['phi_ratio_applied']}, "
              f"Merkaba={metrics['field_status']['merkaba_applied']}")
        
        # Test integration data
        integration_data = womb.get_womb_for_integration()
        print(f"Ready for brain formation: {integration_data['ready_for_brain_formation']}")
        
        return True
        
    except Exception as e:
        print(f"Womb environment test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if script is executed directly
    test_womb_environment_functionality()

    

