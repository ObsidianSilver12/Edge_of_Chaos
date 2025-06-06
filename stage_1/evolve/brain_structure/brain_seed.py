# --- brain_seed.py (Refactored V6.0.0) ---

"""
Brain Seed - Simple Energy Spark (Sperm+Egg Concept)

Simple energy packet that triggers brain and mycelial network growth.
No complex energy management - just a spark to start development.
Interfaces with mycelial network for energy distribution.
"""

import logging
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import constants
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("BrainSeed")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SimpleBrainSeed:
    """
    Simple brain seed - energy spark that triggers brain development.
    Like sperm+egg - provides initial energy burst to start growth.
    Mycelial network takes over energy management after initial spark.
    """
    
    def __init__(self, initial_energy: float = 100.0, base_frequency: float = DEFAULT_BRAIN_SEED_FREQUENCY):
        """
        Initialize simple brain seed with energy spark.
        
        Args:
            initial_energy: Initial energy spark (BEU)
            base_frequency: Base frequency for development (Hz)
            
        Raises:
            ValueError: If energy is insufficient for development
        """
        # Hard validation - no fallbacks
        if initial_energy < MIN_BRAIN_SEED_ENERGY:
            raise ValueError(f"Insufficient energy for brain development. Required: {MIN_BRAIN_SEED_ENERGY}, Got: {initial_energy}")
        
        if base_frequency <= 0:
            raise ValueError(f"Invalid frequency for brain development. Must be > 0, Got: {base_frequency}")
        
        # Core identifiers
        self.seed_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Simple energy spark
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.base_frequency = base_frequency
        
        # Position and connections (set later)
        self.position = None
        self.brain_region = None
        self.mycelial_connection = None
        self.brain_structure = None
        
        # Development state
        self.spark_triggered = False
        self.energy_transferred = False
        self.development_started = False
        
        # Simple field properties
        self.field_radius = SEED_FIELD_RADIUS
        self.field_strength = min(1.0, initial_energy / 100.0)
        
        logger.info(f"Brain seed created: {self.seed_id[:8]} with {initial_energy} BEU at {base_frequency} Hz")
    
    def set_position(self, position: Tuple[int, int, int], brain_region: str) -> bool:
        """
        Set brain seed position.
        
        Args:
            position: (x, y, z) coordinates
            brain_region: Brain region name
            
        Returns:
            True if position set successfully
            
        Raises:
            ValueError: If position or region invalid
        """
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            raise ValueError(f"Position must be (x, y, z) tuple, got {position}")
        
        if not isinstance(brain_region, str) or not brain_region:
            raise ValueError(f"Brain region must be non-empty string, got {brain_region}")
        
        self.position = tuple(position)
        self.brain_region = brain_region
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Brain seed positioned at {position} in {brain_region}")
        return True
    
    def connect_to_mycelial_network(self, mycelial_network) -> bool:
        """
        Connect to mycelial network for energy distribution.
        
        Args:
            mycelial_network: Mycelial network instance
            
        Returns:
            True if connected successfully
            
        Raises:
            ValueError: If mycelial network invalid
        """
        if mycelial_network is None:
            raise ValueError("Mycelial network cannot be None")
        
        # Hard validation - ensure mycelial network can accept energy
        if not hasattr(mycelial_network, 'receive_energy_from_seed'):
            raise ValueError("Mycelial network must have receive_energy_from_seed method")
        
        self.mycelial_connection = mycelial_network
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Brain seed connected to mycelial network")
        return True
    
    def trigger_development_spark(self) -> Dict[str, Any]:
        """
        Trigger the development spark - transfer energy to mycelial network.
        This is the key moment - like fertilization.
        
        Returns:
            Dict with spark results
            
        Raises:
            RuntimeError: If spark cannot be triggered
        """
        # Hard validation - all prerequisites must be met
        if self.position is None:
            raise RuntimeError("Cannot trigger spark: Position not set")
        
        if self.mycelial_connection is None:
            raise RuntimeError("Cannot trigger spark: Mycelial network not connected")
        
        if self.spark_triggered:
            raise RuntimeError("Cannot trigger spark: Already triggered")
        
        if self.current_energy < MIN_BRAIN_SEED_ENERGY:
            raise RuntimeError(f"Cannot trigger spark: Insufficient energy {self.current_energy}")
        
        logger.info("Triggering development spark")
        
        try:
            # Transfer energy to mycelial network
            energy_transfer_result = self.mycelial_connection.receive_energy_from_seed(
                energy_amount=self.current_energy,
                frequency=self.base_frequency,
                position=self.position,
                region=self.brain_region
            )
            
            if not energy_transfer_result.get('success', False):
                raise RuntimeError(f"Energy transfer failed: {energy_transfer_result}")
            
            # Mark spark as triggered
            self.spark_triggered = True
            self.energy_transferred = True
            self.development_started = True
            self.current_energy = 0.0  # All energy transferred
            
            # Create field pattern at position
            field_pattern = self._create_initial_field_pattern()
            
            self.last_updated = datetime.now().isoformat()
            
            spark_result = {
                'success': True,
                'energy_transferred': self.initial_energy,
                'frequency_set': self.base_frequency,
                'position': self.position,
                'region': self.brain_region,
                'field_pattern': field_pattern,
                'mycelial_response': energy_transfer_result
            }
            
            logger.info("Development spark triggered successfully")
            return spark_result
            
        except Exception as e:
            logger.error(f"Spark trigger failed: {e}")
            raise RuntimeError(f"Failed to trigger development spark: {e}")
    
    def _create_initial_field_pattern(self) -> Dict[str, Any]:
        """
        Create initial field pattern around seed position.
        Simple radial pattern for mycelial network to build on.
        
        Returns:
            Field pattern data
        """
        if self.position is None:
            return {}
        
        x, y, z = self.position
        
        # Simple radial field pattern
        field_points = []
        for dx in range(-int(self.field_radius), int(self.field_radius) + 1):
            for dy in range(-int(self.field_radius), int(self.field_radius) + 1):
                for dz in range(-int(self.field_radius), int(self.field_radius) + 1):
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if distance <= self.field_radius:
                        # Field strength decreases with distance
                        strength = self.field_strength * (1.0 - distance / self.field_radius)
                        
                        field_points.append({
                            'position': (x + dx, y + dy, z + dz),
                            'strength': strength,
                            'frequency': self.base_frequency
                        })
        
        return {
            'pattern_type': 'radial',
            'center': self.position,
            'radius': self.field_radius,
            'base_strength': self.field_strength,
            'base_frequency': self.base_frequency,
            'field_points': field_points
        }
    
    def get_seed_state(self) -> Dict[str, Any]:
        """
        Get current seed state.
        
        Returns:
            Complete seed state information
        """
        return {
            'seed_id': self.seed_id,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'energy': {
                'initial_energy': self.initial_energy,
                'current_energy': self.current_energy,
                'base_frequency': self.base_frequency
            },
            'position': {
                'coordinates': self.position,
                'brain_region': self.brain_region,
                'field_radius': self.field_radius,
                'field_strength': self.field_strength
            },
            'connections': {
                'mycelial_connected': self.mycelial_connection is not None,
                'brain_structure_connected': self.brain_structure is not None
            },
            'development': {
                'spark_triggered': self.spark_triggered,
                'energy_transferred': self.energy_transferred,
                'development_started': self.development_started
            }
        }
    
    def validate_development_readiness(self) -> Dict[str, Any]:
        """
        Validate that seed is ready for development.
        
        Returns:
            Validation results
        """
        validation = {
            'ready': True,
            'checks': {},
            'errors': []
        }
        
        # Check energy
        energy_check = self.current_energy >= MIN_BRAIN_SEED_ENERGY
        validation['checks']['sufficient_energy'] = energy_check
        if not energy_check:
            validation['ready'] = False
            validation['errors'].append(f"Insufficient energy: {self.current_energy} < {MIN_BRAIN_SEED_ENERGY}")
        
        # Check position
        position_check = self.position is not None
        validation['checks']['position_set'] = position_check
        if not position_check:
            validation['ready'] = False
            validation['errors'].append("Position not set")
        
        # Check mycelial connection
        mycelial_check = self.mycelial_connection is not None
        validation['checks']['mycelial_connected'] = mycelial_check
        if not mycelial_check:
            validation['ready'] = False
            validation['errors'].append("Mycelial network not connected")
        
        # Check not already triggered
        not_triggered_check = not self.spark_triggered
        validation['checks']['not_already_triggered'] = not_triggered_check
        if not not_triggered_check:
            validation['ready'] = False
            validation['errors'].append("Development already triggered")
        
        return validation


# --- Utility Functions ---

def create_simple_brain_seed(initial_energy: float = 100.0, 
                            base_frequency: float = DEFAULT_BRAIN_SEED_FREQUENCY) -> SimpleBrainSeed:
    """
    Create a simple brain seed with validation.
    
    Args:
        initial_energy: Initial energy spark (BEU)
        base_frequency: Base development frequency (Hz)
        
    Returns:
        SimpleBrainSeed instance
        
    Raises:
        ValueError: If parameters invalid
    """
    logger.info(f"Creating simple brain seed with {initial_energy} BEU at {base_frequency} Hz")
    
    try:
        seed = SimpleBrainSeed(initial_energy=initial_energy, base_frequency=base_frequency)
        return seed
        
    except Exception as e:
        logger.error(f"Failed to create brain seed: {e}")
        raise ValueError(f"Brain seed creation failed: {e}")


def demonstrate_simple_brain_seed():
    """Demonstrate the simplified brain seed."""
    print("\n=== Simple Brain Seed Demonstration ===")
    
    try:
        # Create seed
        seed = create_simple_brain_seed(initial_energy=150.0, base_frequency=7.83)
        print(f"Created seed: {seed.seed_id[:8]}")
        
        # Set position
        seed.set_position((128, 128, 64), "limbic")
        print(f"Position set: {seed.position} in {seed.brain_region}")
        
        # Check state
        state = seed.get_seed_state()
        print(f"Seed state: {state['development']}")
        
        # Validate readiness
        validation = seed.validate_development_readiness()
        print(f"Ready for development: {validation['ready']}")
        if validation['errors']:
            print(f"Missing: {validation['errors']}")
        
        return seed
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demonstrate simple brain seed
    demo_seed = demonstrate_simple_brain_seed()
    
    if demo_seed:
        print("\nSimple Brain Seed demonstration completed successfully!")
    else:
        print("\nERROR: Simple Brain Seed demonstration failed")

# --- End of brain_seed.py ---