# --- brain_seed_integration.py (Simplified V6.0.0) ---

"""
Simplified Brain Seed Integration

Simple integration of brain seed with brain structure.
No complex field dynamics - just connect, position, and trigger.
Mycelial network handles all the complexity after spark.
"""

import logging
import uuid
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import constants
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("BrainSeedIntegration")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SimpleBrainSeedIntegration:
    """
    Simplified brain seed integration.
    Architecture principle: Simple spark trigger, let mycelial network handle complexity.
    """
    
    def __init__(self):
        """Initialize simplified integration handler."""
        self.integration_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Simple state tracking
        self.integration_completed = False
        self.seed_id = None
        self.brain_id = None
        self.integration_position = None
        
        logger.info("SimpleBrainSeedIntegration initialized")
    
    def integrate_seed_with_structure(self, brain_seed, brain_structure) -> Dict[str, Any]:
        """
        Simple integration: position seed, connect to mycelial, trigger spark.
        
        Args:
            brain_seed: SimpleBrainSeed instance
            brain_structure: BrainGrid instance
            
        Returns:
            Dict containing integration results
            
        Raises:
            ValueError: If integration cannot proceed
            RuntimeError: If integration fails
        """
        # Validate inputs
        if not hasattr(brain_seed, 'seed_id'):
            raise ValueError("Invalid brain_seed: missing seed_id")
            
        if not hasattr(brain_structure, 'brain_id'):
            raise ValueError("Invalid brain_structure: missing brain_id")
        
        self.seed_id = brain_seed.seed_id
        self.brain_id = brain_structure.brain_id
        
        logger.info(f"Integrating seed {brain_seed.seed_id[:8]} with brain {brain_structure.brain_id[:8]}")
        
        integration_start = datetime.now().isoformat()
        
        try:
            # Phase 1: Find optimal position (simple - just use limbic center)
            optimal_position = self._find_simple_position(brain_structure)
            self.integration_position = optimal_position
            
            # Phase 2: Set seed position
            brain_seed.set_position(optimal_position, REGION_LIMBIC)
            
            # Phase 3: Connect to mycelial network (if brain has one)
            mycelial_connection = self._connect_to_mycelial_network(brain_seed, brain_structure)
            
            # Phase 4: Trigger development spark
            spark_result = brain_seed.trigger_development_spark()
            
            # Phase 5: Notify brain structure of integration
            brain_notification = self._notify_brain_structure(brain_structure, spark_result)
            
            # Mark integration complete
            self.integration_completed = True
            self.last_updated = datetime.now().isoformat()
            
            # Return simple integration metrics
            integration_result = {
                'success': True,
                'integration_id': self.integration_id,
                'seed_id': brain_seed.seed_id,
                'brain_id': brain_structure.brain_id,
                'integration_start': integration_start,
                'integration_end': datetime.now().isoformat(),
                'position': optimal_position,
                'region': REGION_LIMBIC,
                'energy_transferred': spark_result['energy_transferred'],
                'frequency_set': spark_result['frequency_set'],
                'mycelial_connected': mycelial_connection['connected'],
                'brain_notified': brain_notification['success']
            }
            
            logger.info(f"Integration completed successfully at {optimal_position}")
            return integration_result
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            integration_result = {
                'success': False,
                'integration_id': self.integration_id,
                'error': str(e),
                'integration_start': integration_start,
                'integration_end': datetime.now().isoformat()
            }
            raise RuntimeError(f"Brain seed integration failed: {e}")
    
    def _find_simple_position(self, brain_structure) -> Tuple[int, int, int]:
        """
        Find simple position for seed - just use limbic center.
        Architecture principle: Start simple, let mycelial network optimize.
        
        Args:
            brain_structure: Brain structure instance
            
        Returns:
            Position coordinates
        """
        # Simple approach: use limbic region center from constants
        limbic_location = REGION_LOCATIONS.get(REGION_LIMBIC, (0.5, 0.5, 0.4))
        
        # Convert to grid coordinates
        x = int(limbic_location[0] * brain_structure.dimensions[0])
        y = int(limbic_location[1] * brain_structure.dimensions[1])
        z = int(limbic_location[2] * brain_structure.dimensions[2])
        
        # Ensure within bounds
        x = max(0, min(x, brain_structure.dimensions[0] - 1))
        y = max(0, min(y, brain_structure.dimensions[1] - 1))
        z = max(0, min(z, brain_structure.dimensions[2] - 1))
        
        position = (x, y, z)
        logger.info(f"Simple position selected: {position} (limbic center)")
        return position
    
    def _connect_to_mycelial_network(self, brain_seed, brain_structure) -> Dict[str, Any]:
        """
        Connect brain seed to mycelial network.
        
        Args:
            brain_seed: Brain seed instance
            brain_structure: Brain structure instance
            
        Returns:
            Connection result
        """
        # Check if brain structure has mycelial network interface
        if hasattr(brain_structure, 'mycelial_network'):
            try:
                brain_seed.connect_to_mycelial_network(brain_structure.mycelial_network)
                logger.info("Brain seed connected to existing mycelial network")
                return {'connected': True, 'type': 'existing_network'}
            except Exception as e:
                logger.warning(f"Failed to connect to existing mycelial network: {e}")
        
        # Check if brain structure can create mycelial interface
        if hasattr(brain_structure, 'create_mycelial_interface'):
            try:
                mycelial_interface = brain_structure.create_mycelial_interface()
                brain_seed.connect_to_mycelial_network(mycelial_interface)
                logger.info("Brain seed connected to new mycelial interface")
                return {'connected': True, 'type': 'new_interface'}
            except Exception as e:
                logger.warning(f"Failed to create mycelial interface: {e}")
        
        # Fallback: create minimal interface
        logger.info("Creating minimal mycelial interface for brain seed")
        minimal_interface = self._create_minimal_mycelial_interface(brain_structure)
        brain_seed.connect_to_mycelial_network(minimal_interface)
        
        return {'connected': True, 'type': 'minimal_interface'}
    
    def _create_minimal_mycelial_interface(self, brain_structure):
        """
        Create minimal mycelial interface for brain seed integration.
        
        Args:
            brain_structure: Brain structure instance
            
        Returns:
            Minimal mycelial interface
        """
        class MinimalMycelialInterface:
            def __init__(self, brain_structure):
                self.brain_structure = brain_structure
                self.energy_received = 0.0
                self.frequency_set = 0.0
                
            def receive_energy_from_seed(self, energy_amount, frequency, position, region):
                """Receive energy from brain seed."""
                try:
                    # Store energy in brain structure
                    if hasattr(self.brain_structure, 'set_field_value'):
                        # Use existing field interface
                        self.brain_structure.set_field_value(position, 'energy', energy_amount)
                        self.brain_structure.set_field_value(position, 'frequency', frequency)
                    else:
                        # Create simple storage
                        if not hasattr(self.brain_structure, 'seed_energy'):
                            self.brain_structure.seed_energy = {}
                        self.brain_structure.seed_energy[position] = {
                            'energy': energy_amount,
                            'frequency': frequency,
                            'region': region
                        }
                    
                    self.energy_received = energy_amount
                    self.frequency_set = frequency
                    
                    logger.info(f"Minimal interface received {energy_amount} BEU at {frequency} Hz")
                    return {'success': True, 'energy_stored': energy_amount}
                    
                except Exception as e:
                    logger.error(f"Minimal interface failed to receive energy: {e}")
                    return {'success': False, 'error': str(e)}
        
        return MinimalMycelialInterface(brain_structure)
    
    def _notify_brain_structure(self, brain_structure, spark_result) -> Dict[str, Any]:
        """
        Notify brain structure that seed integration is complete.
        
        Args:
            brain_structure: Brain structure instance
            spark_result: Result from spark trigger
            
        Returns:
            Notification result
        """
        try:
            # Try to use brain structure's integration notification method
            if hasattr(brain_structure, 'notify_seed_integration'):
                result = brain_structure.notify_seed_integration(
                    seed_id=self.seed_id,
                    position=self.integration_position,
                    energy_transferred=spark_result['energy_transferred'],
                    frequency=spark_result['frequency_set']
                )
                logger.info("Brain structure notified via notify_seed_integration")
                return {'success': True, 'method': 'notify_seed_integration', 'result': result}
            
            # Fallback: set flag attribute
            brain_structure.seed_integrated = True
            brain_structure.seed_integration_data = {
                'seed_id': self.seed_id,
                'position': self.integration_position,
                'energy': spark_result['energy_transferred'],
                'frequency': spark_result['frequency_set'],
                'integration_time': datetime.now().isoformat()
            }
            logger.info("Brain structure notified via attribute flag")
            return {'success': True, 'method': 'attribute_flag'}
            
        except Exception as e:
            logger.warning(f"Failed to notify brain structure: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get current integration status.
        
        Returns:
            Integration status information
        """
        return {
            'integration_id': self.integration_id,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'integration_completed': self.integration_completed,
            'seed_id': self.seed_id,
            'brain_id': self.brain_id,
            'integration_position': self.integration_position
        }


# --- Utility Functions ---

def integrate_brain_seed_simple(brain_seed, brain_structure) -> Dict[str, Any]:
    """
    Simple utility function for brain seed integration.
    
    Args:
        brain_seed: SimpleBrainSeed instance
        brain_structure: BrainGrid instance
        
    Returns:
        Integration result
    """
    logger.info("Starting simple brain seed integration")
    
    try:
        integrator = SimpleBrainSeedIntegration()
        result = integrator.integrate_seed_with_structure(brain_seed, brain_structure)
        
        logger.info("Simple brain seed integration completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Simple brain seed integration failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def demonstrate_simple_integration():
    """Demonstrate simplified brain seed integration."""
    print("\n=== Simple Brain Seed Integration Demonstration ===")
    
    try:
        # Import brain seed
        from stage_1.brain_formation.brain.brain_seed import create_simple_brain_seed
        
        # Create mock brain structure
        class MockBrainStructure:
            def __init__(self):
                self.brain_id = str(uuid.uuid4())
                self.dimensions = (256, 256, 256)
                self.seed_energy = {}
                
            def set_field_value(self, position, field_type, value):
                if not hasattr(self, 'fields'):
                    self.fields = {}
                self.fields[(position, field_type)] = value
                return True
        
        # Create brain seed and structure
        brain_seed = create_simple_brain_seed(initial_energy=150.0)
        brain_structure = MockBrainStructure()
        
        print(f"Created seed: {brain_seed.seed_id[:8]}")
        print(f"Created brain: {brain_structure.brain_id[:8]}")
        
        # Integrate
        result = integrate_brain_seed_simple(brain_seed, brain_structure)
        
        print(f"Integration success: {result['success']}")
        if result['success']:
            print(f"Position: {result['position']}")
            print(f"Energy transferred: {result['energy_transferred']} BEU")
            print(f"Frequency set: {result['frequency_set']} Hz")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demonstrate simple integration
    demo_result = demonstrate_simple_integration()
    
    if demo_result and demo_result.get('success'):
        print("\nSimple Brain Seed Integration demonstration completed successfully!")
    else:
        print("\nERROR: Simple Brain Seed Integration demonstration failed")

# --- End of brain_seed_integration.py ---