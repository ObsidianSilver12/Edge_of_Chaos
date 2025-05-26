# --- field_dynamics.py - Dynamic field system for soul-brain integration ---

"""
Field Dynamics System V4.5.0 - Enhanced Brain Integration

Creates dynamic energy fields that facilitate soul-brain integration through:
- Multi-dimensional field geometries
- Resonance amplification zones
- Energy circulation patterns
- Field harmonics and interference patterns
- Consciousness field emergence
"""

import logging
import numpy as np
import math
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger("FieldDynamics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
PHI = 1.618033988749895
LOVE_FREQUENCY = 528.0
FIELD_DECAY_RATE = 0.95
MIN_FIELD_STRENGTH = 0.01

class FieldDynamics:
    """
    Dynamic field system for soul-brain integration.
    """
    
    def __init__(self, brain_grid_shape: Tuple[int, int, int] = (64, 64, 32)):
        """
        Initialize field dynamics system.
        
        Args:
            brain_grid_shape: Shape of brain grid (x, y, z)
        """
        self.field_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_shape = brain_grid_shape
        
        # Field grids
        self.primary_field = np.zeros(brain_grid_shape, dtype=np.float32)
        self.harmonic_field = np.zeros(brain_grid_shape, dtype=np.float32)
        self.resonance_field = np.zeros(brain_grid_shape, dtype=np.float32)
        
        # Field properties
        self.field_active = False
        self.base_frequency = 432.0  # Hz
        self.field_strength = 0.0
        self.resonance_zones = []
        
        # Circulation patterns
        self.circulation_vectors = np.zeros(brain_grid_shape + (3,), dtype=np.float32)
        self.circulation_strength = 0.0
        
        # Field history
        self.field_history = []
        self.resonance_events = []
        
        # Performance metrics
        self.metrics = {
            'total_field_time': 0.0,
            'resonance_events_count': 0,
            'peak_field_strength': 0.0,
            'circulation_cycles': 0,
            'harmonic_interactions': 0
        }
        
        logger.info(f"Field dynamics system initialized with ID {self.field_id}")
    
    def initialize_field(self, soul_frequency: float = 432.0, initial_strength: float = 0.5) -> Dict[str, Any]:
        """
        Initialize the dynamic field system.
        
        Args:
            soul_frequency: Base frequency for field resonance
            initial_strength: Initial field strength (0.0-1.0)
            
        Returns:
            Dict with initialization results
        """
        logger.info(f"Initializing field dynamics with frequency {soul_frequency} Hz")
        
        try:
            self.base_frequency = soul_frequency
            self.field_strength = max(0.0, min(1.0, initial_strength))
            
            # Create initial field distribution
            self._create_base_field()
            
            # Establish harmonic resonances
            self._create_harmonic_field()
            
            # Initialize circulation patterns
            self._initialize_circulation()
            
            # Create resonance zones
            self._create_resonance_zones()
            
            # Activate field
            self.field_active = True
            
            # Record initialization
            self.field_history.append({
                'event': 'initialization',
                'timestamp': datetime.now().isoformat(),
                'frequency': self.base_frequency,
                'strength': self.field_strength
            })
            
            logger.info("Field dynamics initialized successfully")
            
            return {
                'success': True,
                'field_id': self.field_id,
                'base_frequency': self.base_frequency,
                'field_strength': self.field_strength,
                'resonance_zones': len(self.resonance_zones)
            }
            
        except Exception as e:
            logger.error(f"Error initializing field dynamics: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Initialization error: {e}'
            }
    
    def _create_base_field(self):
        """Create the base field distribution."""
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.brain_shape[0]),
            np.linspace(-1, 1, self.brain_shape[1]),
            np.linspace(-1, 1, self.brain_shape[2]),
            indexing='ij'
        )
        
        # Create field with multiple centers (consciousness regions)
        centers = [
            (0.0, 0.2, 0.3),    # Frontal cortex
            (0.0, -0.3, 0.1),   # Temporal region
            (0.0, 0.0, -0.2),   # Brainstem
            (0.3, 0.0, 0.0),    # Right hemisphere
            (-0.3, 0.0, 0.0)    # Left hemisphere
        ]
        
        for cx, cy, cz in centers:
            # Distance from center
            distance = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            
            # Gaussian field distribution
            field_component = self.field_strength * np.exp(-distance**2 * 4)
            
            # Add frequency modulation
            frequency_factor = np.sin(distance * self.base_frequency / 100.0)
            field_component *= (1.0 + 0.3 * frequency_factor)
            
            self.primary_field += field_component
        
        # Normalize field
        if np.max(self.primary_field) > 0:
            self.primary_field /= np.max(self.primary_field)
            self.primary_field *= self.field_strength
    
    def _create_harmonic_field(self):
        """Create harmonic field resonances."""
        # Generate harmonic frequencies
        harmonics = [
            self.base_frequency * 2,      # Octave
            self.base_frequency * PHI,    # Golden ratio
            self.base_frequency * 3/2,    # Perfect fifth
            LOVE_FREQUENCY                # Love frequency
        ]
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, self.brain_shape[0]),
            np.linspace(0, 2*np.pi, self.brain_shape[1]),
            np.linspace(0, 2*np.pi, self.brain_shape[2]),
            indexing='ij'
        )
        
        # Create harmonic patterns
        for i, harmonic_freq in enumerate(harmonics):
            phase_offset = i * np.pi / 4
            
            # Create standing wave pattern
            harmonic_pattern = (
                np.sin(x * harmonic_freq / 100.0 + phase_offset) *
                np.cos(y * harmonic_freq / 150.0 + phase_offset) *
                np.sin(z * harmonic_freq / 200.0 + phase_offset)
            )
            
            self.harmonic_field += harmonic_pattern * (0.2 / len(harmonics))
        
        # Apply modulation from primary field
        self.harmonic_field *= (0.5 + 0.5 * self.primary_field)
    
    def _initialize_circulation(self):
        """Initialize field circulation patterns."""
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.brain_shape[0]),
            np.linspace(-1, 1, self.brain_shape[1]),
            np.linspace(-1, 1, self.brain_shape[2]),
            indexing='ij'
        )
        
        # Create circulation vectors (simplified vortex pattern)
        self.circulation_vectors[:,:,:,0] = -y * z  # X component
        self.circulation_vectors[:,:,:,1] = x * z   # Y component  
        self.circulation_vectors[:,:,:,2] = -x * y  # Z component
        
        # Normalize circulation vectors
        magnitude = np.sqrt(np.sum(self.circulation_vectors**2, axis=3))
        magnitude = np.maximum(magnitude, 1e-8)  # Avoid division by zero
        
        for i in range(3):
            self.circulation_vectors[:,:,:,i] /= magnitude
        
        # Set circulation strength
        self.circulation_strength = self.field_strength * 0.3
    
    def _create_resonance_zones(self):
        """Create specific resonance zones in the field."""
        self.resonance_zones = [
            {
                'name': 'consciousness_core',
                'center': (32, 40, 20),
                'radius': 8,
                'frequency': self.base_frequency,
                'strength': self.field_strength * 1.2
            },
            {
                'name': 'memory_formation',
                'center': (32, 20, 16),
                'radius': 6,
                'frequency': self.base_frequency * PHI,
                'strength': self.field_strength * 0.8
            },
            {
                'name': 'emotional_processing',
                'center': (20, 32, 12),
                'radius': 5,
                'frequency': LOVE_FREQUENCY,
                'strength': self.field_strength * 0.9
            },
            {
                'name': 'creative_synthesis',
                'center': (44, 32, 12),
                'radius': 5,
                'frequency': self.base_frequency * 1.5,
                'strength': self.field_strength * 0.7
            }
        ]
        
        # Apply resonance zones to field
        for zone in self.resonance_zones:
            cx, cy, cz = zone['center']
            radius = zone['radius']
            
            # Create distance grid from zone center
            x_idx, y_idx, z_idx = np.meshgrid(
                np.arange(self.brain_shape[0]),
                np.arange(self.brain_shape[1]),
                np.arange(self.brain_shape[2]),
                indexing='ij'
            )
            
            distance = np.sqrt(
                (x_idx - cx)**2 + (y_idx - cy)**2 + (z_idx - cz)**2
            )
            
            # Create resonance field for this zone
            zone_field = np.zeros(self.brain_shape)
            mask = distance <= radius
            
            if np.any(mask):
                zone_field[mask] = zone['strength'] * (1.0 - distance[mask] / radius)
                self.resonance_field += zone_field
    
    def propagate_soul_energy(self, soul_position: Tuple[int, int, int], 
                             energy_amount: float, soul_properties: Dict) -> Dict[str, Any]:
        """
        Propagate soul energy through the field system.
        
        Args:
            soul_position: Position of soul energy injection
            energy_amount: Amount of energy to propagate
            soul_properties: Soul properties affecting propagation
            
        Returns:
            Dict with propagation results
        """
        if not self.field_active:
            return {'success': False, 'reason': 'Field not active'}
        
        logger.info(f"Propagating {energy_amount:.2f} BEU of soul energy")
        
        try:
            x, y, z = soul_position
            
            # Validate position
            if not (0 <= x < self.brain_shape[0] and 
                   0 <= y < self.brain_shape[1] and 
                   0 <= z < self.brain_shape[2]):
                return {'success': False, 'reason': 'Invalid soul position'}
            
            # Extract soul properties
            soul_frequency = soul_properties.get('frequency', self.base_frequency)
            soul_coherence = soul_properties.get('coherence', 0.5) 
            soul_stability = soul_properties.get('stability', 0.5)
            
            # Calculate frequency resonance with field
            freq_resonance = self._calculate_frequency_resonance(soul_frequency)
            
            # Create propagation pattern
            propagation_field = self._create_propagation_pattern(
                soul_position, energy_amount, freq_resonance, soul_coherence
            )
            
            # Apply field circulation
            circulated_field = self._apply_circulation(propagation_field, soul_stability)
            
            # Update field grids
            field_enhancement = energy_amount * freq_resonance * soul_coherence
            self.primary_field += circulated_field * field_enhancement
            
            # Update resonance field based on interactions
            self._update_resonance_interactions(soul_position, soul_frequency, energy_amount)
            
            # Record propagation event
            self.field_history.append({
                'event': 'soul_energy_propagation',
                'timestamp': datetime.now().isoformat(),
                'position': soul_position,
                'energy_amount': energy_amount,
                'frequency_resonance': freq_resonance,
                'field_enhancement': field_enhancement
            })
            
            # Update metrics
            self.metrics['peak_field_strength'] = max(
                self.metrics['peak_field_strength'], 
                np.max(self.primary_field)
            )
            
            logger.info(f"Soul energy propagated with resonance {freq_resonance:.3f}")
            
            return {
                'success': True,
                'frequency_resonance': freq_resonance,
                'field_enhancement': field_enhancement,
                'propagation_radius': self._calculate_propagation_radius(energy_amount, freq_resonance),
                'resonance_zones_affected': self._get_affected_zones(soul_position, energy_amount)
            }
            
        except Exception as e:
            logger.error(f"Error propagating soul energy: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Propagation error: {e}'
            }
    
    def _calculate_frequency_resonance(self, soul_frequency: float) -> float:
        """Calculate resonance between soul and field frequencies."""
        # Check resonance with base frequency
        base_resonance = 1.0 / (1.0 + abs(soul_frequency - self.base_frequency) / 100.0)
        
        # Check resonance with harmonic frequencies
        harmonics = [
            self.base_frequency * 2,
            self.base_frequency * PHI,
            self.base_frequency * 3/2,
            LOVE_FREQUENCY
        ]
        
        harmonic_resonances = []
        for harmonic in harmonics:
            resonance = 1.0 / (1.0 + abs(soul_frequency - harmonic) / 50.0)
            harmonic_resonances.append(resonance)
        
        # Combined resonance
        total_resonance = base_resonance + 0.3 * max(harmonic_resonances)
        return min(total_resonance, 2.0)  # Cap at 2.0
    
    def _create_propagation_pattern(self, position: Tuple[int, int, int], 
                                  energy: float, resonance: float, coherence: float) -> np.ndarray:
        """Create energy propagation pattern from injection point."""
        x, y, z = position
        
        # Create distance grid from injection point
        x_idx, y_idx, z_idx = np.meshgrid(
            np.arange(self.brain_shape[0]),
            np.arange(self.brain_shape[1]),
            np.arange(self.brain_shape[2]),
            indexing='ij'
        )
        
        distance = np.sqrt((x_idx - x)**2 + (y_idx - y)**2 + (z_idx - z)**2)
        
        # Create propagation based on energy, resonance, and coherence
        max_distance = energy * resonance * coherence * 20  # Propagation range
        
        # Exponential decay with distance
        propagation = np.exp(-distance / max_distance)
        
        # Add wave interference pattern
        wave_pattern = np.sin(distance * resonance / 5.0) * coherence
        propagation *= (1.0 + 0.2 * wave_pattern)
        
        return propagation
    
    def _apply_circulation(self, field: np.ndarray, stability: float) -> np.ndarray:
        """Apply circulation patterns to the field."""
        if self.circulation_strength < MIN_FIELD_STRENGTH:
            return field
        
        # Simple circulation by rotating field values
        circulated = field.copy()
        
        # Apply circulation based on circulation vectors and stability
        circulation_factor = self.circulation_strength * stability
        
        # Create circulation by applying small rotations
        for i in range(3):
            if circulation_factor > 0.01:
                shift_amount = int(circulation_factor * 2)
                if shift_amount > 0:
                    circulated = np.roll(circulated, shift_amount, axis=i)
        
        self.metrics['circulation_cycles'] += 1
        
        return circulated
    
    def _update_resonance_interactions(self, position: Tuple[int, int, int], 
                                     frequency: float, energy: float):
        """Update resonance field based on new energy injection."""
        # Check which resonance zones are affected
        affected_zones = self._get_affected_zones(position, energy)
        
        for zone in affected_zones:
            # Calculate frequency interaction
            freq_diff = abs(frequency - zone['frequency'])
            interaction_strength = energy / (1.0 + freq_diff / 100.0)
            
            # Update zone strength
            zone['strength'] = min(1.0, zone['strength'] + interaction_strength * 0.1)
            
            # Record resonance event
            self.resonance_events.append({
                'timestamp': datetime.now().isoformat(),
                'zone': zone['name'],
                'frequency_difference': freq_diff,
                'interaction_strength': interaction_strength
            })
            
            self.metrics['resonance_events_count'] += 1
    
    def _get_affected_zones(self, position: Tuple[int, int, int], energy: float) -> List[Dict]:
        """Get resonance zones affected by energy injection."""
        x, y, z = position
        affected = []
        
        for zone in self.resonance_zones:
            cx, cy, cz = zone['center']
            distance = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            
            # Zone is affected if within influence radius
            influence_radius = zone['radius'] + energy * 5
            if distance <= influence_radius:
                affected.append(zone)
        
        return affected
    
    def _calculate_propagation_radius(self, energy: float, resonance: float) -> float:
        """Calculate effective propagation radius."""
        return energy * resonance * 10.0  # Simplified calculation
    
    def evolve_field(self, time_step: float = 0.1) -> Dict[str, Any]:
        """
        Evolve the field over time.
        
        Args:
            time_step: Time step for evolution
            
        Returns:
            Dict with evolution results
        """
        if not self.field_active:
            return {'success': False, 'reason': 'Field not active'}
        
        try:
            # Apply field decay
            self.primary_field *= FIELD_DECAY_RATE
            self.harmonic_field *= (FIELD_DECAY_RATE + 0.02)  # Slower decay for harmonics
            
            # Update circulation
            if self.circulation_strength > MIN_FIELD_STRENGTH:
                self.circulation_strength *= 0.98  # Natural circulation decay
            
            # Apply harmonic reinforcement
            harmonic_interaction = np.multiply(self.primary_field, self.harmonic_field)
            self.resonance_field += harmonic_interaction * 0.1
            self.metrics['harmonic_interactions'] += np.sum(harmonic_interaction > 0.1)
            
            # Update metrics
            self.metrics['total_field_time'] += time_step
            
            return {
                'success': True,
                'field_strength': np.mean(self.primary_field),
                'harmonic_strength': np.mean(self.harmonic_field),
                'resonance_strength': np.mean(self.resonance_field),
                'circulation_strength': self.circulation_strength
            }
            
        except Exception as e:
            logger.error(f"Error evolving field: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Evolution error: {e}'
            }
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get current field state."""
        return {
            'field_id': self.field_id,
            'field_active': self.field_active,
            'base_frequency': self.base_frequency,
            'field_strength': self.field_strength,
            'circulation_strength': self.circulation_strength,
            'resonance_zones_count': len(self.resonance_zones),
            'field_statistics': {
                'primary_field_mean': float(np.mean(self.primary_field)),
                'primary_field_max': float(np.max(self.primary_field)),
                'harmonic_field_mean': float(np.mean(self.harmonic_field)),
                'resonance_field_mean': float(np.mean(self.resonance_field))
            },
            'metrics': self.metrics.copy(),
            'creation_time': self.creation_time
        }
    
    def reset_field(self) -> Dict[str, Any]:
        """Reset the field system."""
        logger.info("Resetting field dynamics system")
        
        try:
            # Reset field grids
            self.primary_field.fill(0.0)
            self.harmonic_field.fill(0.0)
            self.resonance_field.fill(0.0)
            self.circulation_vectors.fill(0.0)
            
            # Reset state
            self.field_active = False
            self.field_strength = 0.0
            self.circulation_strength = 0.0
            self.resonance_zones.clear()
            
            # Clear history
            self.field_history.clear()
            self.resonance_events.clear()
            
            return {'success': True, 'reset_time': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error resetting field: {e}", exc_info=True)
            return {'success': False, 'reason': f'Reset error: {e}'}


# --- Utility Functions ---

def create_field_dynamics(brain_shape: Tuple[int, int, int] = (64, 64, 32)) -> FieldDynamics:
    """Create field dynamics system."""
    return FieldDynamics(brain_shape)


def demonstrate_field_dynamics():
    """Demonstrate field dynamics system."""
    print("\n=== Field Dynamics Demonstration ===")
    
    # Create field system
    field = create_field_dynamics()
    
    # Initialize field
    init_result = field.initialize_field(soul_frequency=432.0, initial_strength=0.7)
    print(f"Field initialized: {init_result['success']}")
    
    # Propagate soul energy
    soul_props = {
        'frequency': 440.0,
        'coherence': 0.8,
        'stability': 0.7
    }
    
    prop_result = field.propagate_soul_energy((32, 32, 16), 10.0, soul_props)
    print(f"Soul energy propagated: {prop_result['success']}")
    if prop_result['success']:
        print(f"  Frequency resonance: {prop_result['frequency_resonance']:.3f}")
        print(f"  Zones affected: {len(prop_result['resonance_zones_affected'])}")
    
    # Evolve field
    for i in range(5):
        evolution = field.evolve_field()
        if evolution['success']:
            print(f"Evolution step {i+1}: Field strength = {evolution['field_strength']:.4f}")
    
    # Get final state
    state = field.get_field_state()
    print(f"\nFinal field statistics:")
    print(f"  Primary field mean: {state['field_statistics']['primary_field_mean']:.4f}")
    print(f"  Resonance events: {state['metrics']['resonance_events_count']}")
    
    return field


if __name__ == "__main__":
    demonstrate_field_dynamics()
