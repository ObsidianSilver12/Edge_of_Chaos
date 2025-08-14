# --- START OF FILE stage_1/evolve/mycelial/mycelial_quantum_seeds.py ---

"""
Mycelial Quantum Seeds Module (V4.5.0 - Brain Integration)

Implements quantum entanglement system for brain regions.
Creates 1-2 quantum seeds per subregion for survival/emotional/creative workloads.
Uses 432Hz quantum frequency distinct from brain frequencies.
Provides efficient cross-region communication and coordination.
"""

import logging
import os
import sys
import numpy as np
import uuid
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
import math
import random

# Import constants from the main constants module
from shared.constants.constants import *

# --- Logging ---
logger = logging.getLogger('MycelialQuantumSeeds')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Metrics Tracking ---
try:
    import metrics_tracking as metrics
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics tracking module not found. Metrics will not be recorded.")
    METRICS_AVAILABLE = False
    class MetricsPlaceholder:
        def record_metrics(self, *args, **kwargs): pass
    metrics = MetricsPlaceholder()


class QuantumSeed:
    """
    Individual quantum seed for mycelial network entanglement.
    """
    
    def __init__(self, position: Tuple[int, int, int], seed_type: str = "general", 
                 workload_focus: str = "survival"):
        """
        Initialize quantum seed.
        
        Args:
            position: (x, y, z) position in brain grid
            seed_type: Type of seed ("primary", "secondary", "general")
            workload_focus: Focus area ("survival", "emotional", "creative")
        """
        self.seed_id = str(uuid.uuid4())
        self.position = position
        self.seed_type = seed_type
        self.workload_focus = workload_focus
        self.creation_time = datetime.now().isoformat()
        
        # Quantum properties
        self.quantum_frequency = QUANTUM_ENTANGLEMENT_FREQUENCY  # 432Hz distinct from brain
        self.quantum_phase = random.random() * 2 * math.pi  # Random initial phase
        self.entanglement_strength = 0.0
        self.coherence = 0.0
        self.energy_level = 0.0
        
        # Entanglement connections
        self.entangled_seeds: Set[str] = set()  # IDs of entangled seeds
        self.entanglement_data: Dict[str, Dict[str, Any]] = {}  # Detailed entanglement info
        
        # Workload capabilities
        self.workload_capacity = self._calculate_workload_capacity()
        self.current_workload = 0.0
        self.processing_efficiency = QUANTUM_EFFICIENCY
        
        # Network properties
        self.network_role = "node"  # "node", "hub", "relay"
        self.connection_radius = 50.0  # Maximum connection distance
        self.signal_strength = 1.0
        
        # Performance metrics
        self.total_processing_time = 0.0
        self.successful_entanglements = 0
        self.failed_entanglements = 0
        self.data_transmitted = 0.0
        
        logger.debug(f"Quantum seed {self.seed_id[:8]} created at {position} for {workload_focus}")
    
    def _calculate_workload_capacity(self) -> float:
        """
        Calculate processing capacity based on seed type and focus.
        
        Returns:
            Processing capacity (0.0-1.0)
        """
        # Base capacity by type
        base_capacity = {
            "primary": 1.0,
            "secondary": 0.7,
            "general": 0.5
        }.get(self.seed_type, 0.5)
        
        # Workload focus modifier
        focus_modifier = {
            "survival": 1.0,      # Maximum efficiency for survival
            "emotional": 0.8,     # Good efficiency for emotional processing
            "creative": 0.6       # Moderate efficiency for creative tasks
        }.get(self.workload_focus, 0.5)
        
        return base_capacity * focus_modifier
    
    def entangle_with_seed(self, other_seed: 'QuantumSeed', distance: float) -> bool:
        """
        Create quantum entanglement with another seed.
        
        Args:
            other_seed: Other quantum seed to entangle with
            distance: Distance between seeds
            
        Returns:
            True if entanglement successful, False otherwise
        """
        # Check if already entangled
        if other_seed.seed_id in self.entangled_seeds:
            logger.debug(f"Seeds {self.seed_id[:8]} and {other_seed.seed_id[:8]} already entangled")
            return True
        
        # Check distance constraint
        if distance > self.connection_radius:
            logger.debug(f"Distance {distance:.1f} exceeds connection radius {self.connection_radius}")
            return False
        
        try:
            # Calculate entanglement strength based on distance and compatibility
            # Closer seeds have stronger entanglement
            distance_factor = max(0.1, 1.0 - (distance / self.connection_radius))
            
            # Compatibility based on workload focus
            compatibility = self._calculate_compatibility(other_seed)
            
            # Base entanglement strength
            base_strength = distance_factor * compatibility * QUANTUM_EFFICIENCY
            
            # Add quantum coherence factor (phase alignment)
            phase_diff = abs(self.quantum_phase - other_seed.quantum_phase)
            phase_alignment = 1.0 - (phase_diff / (2 * math.pi))
            
            # Final entanglement strength
            entanglement_strength = base_strength * (0.7 + 0.3 * phase_alignment)
            
            # Create entanglement if strength is sufficient
            if entanglement_strength > 0.3:  # Minimum threshold
                # Store entanglement data
                self.entangled_seeds.add(other_seed.seed_id)
                other_seed.entangled_seeds.add(self.seed_id)
                
                self.entanglement_data[other_seed.seed_id] = {
                    'strength': entanglement_strength,
                    'distance': distance,
                    'compatibility': compatibility,
                    'phase_alignment': phase_alignment,
                    'established_at': datetime.now().isoformat(),
                    'data_exchanged': 0.0,
                    'last_communication': None
                }
                
                other_seed.entanglement_data[self.seed_id] = {
                    'strength': entanglement_strength,
                    'distance': distance,
                    'compatibility': compatibility,
                    'phase_alignment': phase_alignment,
                    'established_at': datetime.now().isoformat(),
                    'data_exchanged': 0.0,
                    'last_communication': None
                }
                
                # Update individual seed properties
                self.entanglement_strength = max(self.entanglement_strength, entanglement_strength)
                other_seed.entanglement_strength = max(other_seed.entanglement_strength, entanglement_strength)
                
                self.successful_entanglements += 1
                other_seed.successful_entanglements += 1
                
                logger.debug(f"Entanglement established between {self.seed_id[:8]} and {other_seed.seed_id[:8]} "
                           f"with strength {entanglement_strength:.3f}")
                
                return True
            else:
                # Entanglement too weak
                self.failed_entanglements += 1
                other_seed.failed_entanglements += 1
                logger.debug(f"Entanglement failed: insufficient strength {entanglement_strength:.3f}")
                return False
                
        except Exception as e:
            logger.error(f"Error during entanglement: {e}", exc_info=True)
            self.failed_entanglements += 1
            other_seed.failed_entanglements += 1
            return False
    
    def _calculate_compatibility(self, other_seed: 'QuantumSeed') -> float:
        """
        Calculate compatibility with another seed.
        
        Args:
            other_seed: Other quantum seed
            
        Returns:
            Compatibility factor (0.0-1.0)
        """
        # Same workload focus = high compatibility
        if self.workload_focus == other_seed.workload_focus:
            return 1.0
        
        # Compatible focus combinations
        compatibility_matrix = {
            ("survival", "emotional"): 0.8,
            ("emotional", "creative"): 0.7,
            ("survival", "creative"): 0.6
        }
        
        # Check both directions
        key1 = (self.workload_focus, other_seed.workload_focus)
        key2 = (other_seed.workload_focus, self.workload_focus)
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.4))
    
    def process_workload(self, workload_type: str, workload_data: Dict[str, Any], 
                        processing_time: float) -> Dict[str, Any]:
        """
        Process a workload using quantum processing capabilities.
        
        Args:
            workload_type: Type of workload ("survival", "emotional", "creative")
            workload_data: Workload data
            processing_time: Time to spend processing
            
        Returns:
            Processing results
        """
        # Check if workload type matches focus
        efficiency_modifier = 1.0 if workload_type == self.workload_focus else 0.7
        
        # Check current capacity
        if self.current_workload >= self.workload_capacity:
            return {
                'success': False,
                'reason': 'Seed at maximum capacity',
                'current_workload': self.current_workload,
                'capacity': self.workload_capacity
            }
        
        try:
            # Calculate processing power available
            available_capacity = self.workload_capacity - self.current_workload
            processing_power = min(available_capacity, processing_time * efficiency_modifier)
            
            # Simulate processing
            self.current_workload += processing_power
            self.total_processing_time += processing_time
            
            # Create processing results based on workload type
            if workload_type == "survival":
                results = {
                    'threat_assessment': random.uniform(0.0, 1.0),
                    'resource_availability': random.uniform(0.0, 1.0),
                    'action_urgency': random.uniform(0.0, 1.0)
                }
            elif workload_type == "emotional":
                results = {
                    'emotional_valence': random.uniform(-1.0, 1.0),
                    'arousal_level': random.uniform(0.0, 1.0),
                    'social_relevance': random.uniform(0.0, 1.0)
                }
            elif workload_type == "creative":
                results = {
                    'novelty_score': random.uniform(0.0, 1.0),
                    'pattern_recognition': random.uniform(0.0, 1.0),
                    'synthesis_quality': random.uniform(0.0, 1.0)
                }
            else:
                results = {
                    'general_processing': random.uniform(0.0, 1.0)
                }
            
            # Add quantum processing signature
            results['quantum_processed'] = True
            results['processing_efficiency'] = efficiency_modifier * self.processing_efficiency
            results['entanglement_boost'] = min(1.2, 1.0 + self.entanglement_strength * 0.2)
            
            # Gradually reduce workload (processing completion)
            workload_decay = processing_time * 0.1
            self.current_workload = max(0.0, self.current_workload - workload_decay)
            
            logger.debug(f"Workload processed by {self.seed_id[:8]}: {workload_type} "
                        f"(efficiency={efficiency_modifier:.2f})")
            
            return {
                'success': True,
                'results': results,
                'processing_power_used': processing_power,
                'efficiency': efficiency_modifier,
                'quantum_boost': results['entanglement_boost']
            }
            
        except Exception as e:
            logger.error(f"Error processing workload: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Processing error: {e}'
            }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current quantum seed state.
        
        Returns:
            Dict with current state information
        """
        return {
            'seed_id': self.seed_id,
            'position': self.position,
            'seed_type': self.seed_type,
            'workload_focus': self.workload_focus,
            'quantum_frequency': self.quantum_frequency,
            'entanglement_strength': self.entanglement_strength,
            'coherence': self.coherence,
            'energy_level': self.energy_level,
            'current_workload': self.current_workload,
            'entangled_count': len(self.entangled_seeds),
            'processing_efficiency': self.processing_efficiency
        }


class MycelialQuantumNetwork:
    """
    Network of quantum seeds for mycelial brain communication.
    """
    
    def __init__(self, brain_structure=None):
        """
        Initialize quantum network.
        
        Args:
            brain_structure: Brain structure to create network for
        """
        self.network_id = str(uuid.uuid4())
        self.brain_structure = brain_structure
        self.creation_time = datetime.now().isoformat()
        
        # Quantum seeds
        self.seeds: Dict[str, QuantumSeed] = {}
        self.seeds_by_position: Dict[Tuple[int, int, int], List[str]] = {}
        self.seeds_by_workload: Dict[str, List[str]] = {
            'survival': [],
            'emotional': [],
            'creative': []
        }
        
        # Network properties
        self.total_entanglements = 0
        self.network_coherence = 0.0
        self.processing_load = 0.0
        
        logger.info(f"Mycelial quantum network {self.network_id[:8]} initialized")
    
    def create_quantum_seeds(self) -> Dict[str, Any]:
        """
        Create quantum seeds throughout the brain structure.
        Uses 1-2 seeds per subregion as specified.
        
        Returns:
            Dict with creation results
        """
        if not self.brain_structure:
            return {
                'success': False,
                'reason': 'No brain structure available'
            }
        
        logger.info("Creating quantum seeds throughout brain structure")
        
        seeds_created = 0
        dimensions = self.brain_structure.dimensions
        subregion_size = 32  # 32x32x32 blocks per subregion
        
        try:
            # Process each subregion
            for x in range(0, dimensions[0], subregion_size):
                for y in range(0, dimensions[1], subregion_size):
                    for z in range(0, dimensions[2], subregion_size):
                        # Define subregion bounds
                        x_end = min(x + subregion_size, dimensions[0])
                        y_end = min(y + subregion_size, dimensions[1])
                        z_end = min(z + subregion_size, dimensions[2])
                        
                        # Check if subregion has sufficient activity
                        subregion_slice = (
                            slice(x, x_end),
                            slice(y, y_end),
                            slice(z, z_end)
                        )
                        
                        avg_energy = np.mean(self.brain_structure.energy_grid[subregion_slice])
                        avg_mycelial = np.mean(self.brain_structure.mycelial_density_grid[subregion_slice])
                        
                        # Only create seeds in active subregions
                        if avg_energy > 0.1 or avg_mycelial > 0.1:
                            # Determine number of seeds (1-2 per subregion)
                            num_seeds = 2 if (avg_energy > 0.3 and avg_mycelial > 0.2) else 1
                            
                            # Create seeds for this subregion
                            for seed_idx in range(num_seeds):
                                # Find position within subregion
                                position = self._find_optimal_position(x, y, z, x_end, y_end, z_end)
                                
                                if position:
                                    # Determine seed properties
                                    seed_type = "primary" if seed_idx == 0 else "secondary"
                                    workload_focus = self._determine_workload_focus(position)
                                    
                                    # Create quantum seed
                                    seed = QuantumSeed(
                                        position=position,
                                        seed_type=seed_type,
                                        workload_focus=workload_focus
                                    )
                                    
                                    # Add to network
                                    self._add_seed_to_network(seed)
                                    seeds_created += 1
            
            logger.info(f"Created {seeds_created} quantum seeds in network")
            
            return {
                'success': True,
                'seeds_created': seeds_created,
                'total_seeds': len(self.seeds)
            }
            
        except Exception as e:
            logger.error(f"Error creating quantum seeds: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Creation error: {e}',
                'seeds_created': seeds_created
            }
    
    def _find_optimal_position(self, x_start, y_start, z_start, x_end, y_end, z_end) -> Optional[Tuple[int, int, int]]:
        """
        Find optimal position for seed within subregion.
        
        Returns:
            Optimal position or None if not found
        """
        best_position = None
        best_score = -1.0
        
        # Sample positions within subregion
        for _ in range(10):  # Try up to 10 positions
            x = random.randint(x_start, x_end - 1)
            y = random.randint(y_start, y_end - 1)
            z = random.randint(z_start, z_end - 1)
            
            # Calculate score based on brain structure properties
            energy = self.brain_structure.energy_grid[x, y, z]
            mycelial = self.brain_structure.mycelial_density_grid[x, y, z]
            resonance = self.brain_structure.resonance_grid[x, y, z]
            
            score = energy * 0.4 + mycelial * 0.4 + resonance * 0.2
            
            if score > best_score:
                best_score = score
                best_position = (x, y, z)
        
        return best_position
    
    def _determine_workload_focus(self, position: Tuple[int, int, int]) -> str:
        """
        Determine workload focus based on position in brain.
        
        Args:
            position: Position in brain grid
            
        Returns:
            Workload focus ("survival", "emotional", "creative")
        """
        x, y, z = position
        
        # Get brain region at this position
        region = self.brain_structure.region_grid[x, y, z]
        
        # Map regions to workload focus
        region_focus_map = {
            REGION_BRAIN_STEM: "survival",
            REGION_LIMBIC: "emotional",
            REGION_FRONTAL: "creative",
            REGION_TEMPORAL: "emotional",
            REGION_PARIETAL: "creative",
            REGION_OCCIPITAL: "survival",
            REGION_CEREBELLUM: "survival"
        }
        
        return region_focus_map.get(region, "survival")
    
    def _add_seed_to_network(self, seed: QuantumSeed):
        """
        Add seed to network tracking structures.
        
        Args:
            seed: Quantum seed to add
        """
        # Add to main dictionary
        self.seeds[seed.seed_id] = seed
        
        # Add to position tracking
        if seed.position not in self.seeds_by_position:
            self.seeds_by_position[seed.position] = []
        self.seeds_by_position[seed.position].append(seed.seed_id)
        
        # Add to workload tracking
        self.seeds_by_workload[seed.workload_focus].append(seed.seed_id)
    
    def establish_entanglements(self) -> Dict[str, Any]:
        """
        Establish quantum entanglements between seeds.
        
        Returns:
            Dict with entanglement results
        """
        logger.info("Establishing quantum entanglements between seeds")
        
        entanglements_created = 0
        entanglements_failed = 0
        
        try:
            seed_list = list(self.seeds.values())
            
            # Try to entangle each seed with nearby seeds
            for i, seed1 in enumerate(seed_list):
                for j, seed2 in enumerate(seed_list[i+1:], i+1):
                    # Calculate distance
                    pos1 = np.array(seed1.position)
                    pos2 = np.array(seed2.position)
                    distance = np.linalg.norm(pos2 - pos1)
                    
                    # Try to establish entanglement
                    if seed1.entangle_with_seed(seed2, distance):
                        entanglements_created += 1
                    else:
                        entanglements_failed += 1
            
            # Update network totals
            self.total_entanglements = entanglements_created
            
            logger.info(f"Established {entanglements_created} entanglements, {entanglements_failed} failed")
            
            return {
                'success': True,
                'entanglements_created': entanglements_created,
                'entanglements_failed': entanglements_failed,
                'total_entanglements': self.total_entanglements
            }
            
        except Exception as e:
            logger.error(f"Error establishing entanglements: {e}", exc_info=True)
            return {
                'success': False,
                'reason': f'Entanglement error: {e}',
                'entanglements_created': entanglements_created
            }
    
    def get_network_state(self) -> Dict[str, Any]:
        """
        Get current network state.
        
        Returns:
            Dict with network state information
        """
        # Calculate network coherence
        if self.seeds:
            coherence_values = [seed.coherence for seed in self.seeds.values()]
            self.network_coherence = np.mean(coherence_values)
        
        # Calculate processing load
        if self.seeds:
            workload_values = [seed.current_workload for seed in self.seeds.values()]
            capacity_values = [seed.workload_capacity for seed in self.seeds.values()]
            self.processing_load = np.sum(workload_values) / np.sum(capacity_values) if capacity_values else 0.0
        
        return {
            'network_id': self.network_id,
            'total_seeds': len(self.seeds),
            'total_entanglements': self.total_entanglements,
            'network_coherence': self.network_coherence,
            'processing_load': self.processing_load,
            'seeds_by_workload': {k: len(v) for k, v in self.seeds_by_workload.items()},
            'creation_time': self.creation_time
        }


# --- Utility Functions ---

def create_mycelial_quantum_network(brain_structure) -> MycelialQuantumNetwork:
    """
    Create and initialize a mycelial quantum network.
    
    Args:
        brain_structure: Brain structure to create network for
        
    Returns:
        MycelialQuantumNetwork instance
    """
    logger.info("Creating mycelial quantum network")
    
    try:
        # Create network
        network = MycelialQuantumNetwork(brain_structure=brain_structure)
        
        # Create quantum seeds
        seed_result = network.create_quantum_seeds()
        if not seed_result['success']:
            raise RuntimeError(f"Failed to create quantum seeds: {seed_result['reason']}")
        
        # Establish entanglements
        entanglement_result = network.establish_entanglements()
        if not entanglement_result['success']:
            logger.warning(f"Entanglement creation had issues: {entanglement_result['reason']}")
        
        logger.info(f"Quantum network created with {len(network.seeds)} seeds "
                   f"and {network.total_entanglements} entanglements")
        
        return network
        
    except Exception as e:
        logger.error(f"Error creating quantum network: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create quantum network: {e}")


# --- Module Test Function ---
def test_mycelial_quantum_network():
    """
    Test mycelial quantum network functionality.
    """
    logger.info("=== Testing Mycelial Quantum Network ===")
    
    # Create mock brain structure
    class MockBrainStructure:
        def __init__(self):
            self.dimensions = (128, 128, 128)
            self.energy_grid = np.random.random((128, 128, 128)) * 0.8
            self.mycelial_density_grid = np.random.random((128, 128, 128)) * 0.6
            self.resonance_grid = np.random.random((128, 128, 128)) * 0.5
            self.region_grid = np.full((128, 128, 128), REGION_LIMBIC, dtype=object)
    
    # Create test brain structure
    brain_structure = MockBrainStructure()
    
    # Create quantum network
    network = create_mycelial_quantum_network(brain_structure)
    
    # Test network state
    state = network.get_network_state()
    logger.info(f"Network state: {state}")
    
    # Test individual seed processing
    if network.seeds:
        seed = list(network.seeds.values())[0]
        
        # Test workload processing
        result = seed.process_workload(
            workload_type="survival",
            workload_data={"test": "data"},
            processing_time=1.0
        )
        logger.info(f"Workload processing result: {result['success']}")
        
        # Test seed state
        seed_state = seed.get_state()
        logger.info(f"Seed state: {seed_state}")
    
    logger.info("=== Quantum Network Tests Completed ===")
    return network


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("=== Mycelial Quantum Seeds Module Standalone Execution ===")
    
    # Test quantum network
    try:
        network = test_mycelial_quantum_network()
        print("Mycelial quantum network tests passed successfully!")
        print(f"Created network: {network.network_id}")
    except Exception as e:
        logger.error(f"Quantum network tests failed: {e}", exc_info=True)
        print(f"ERROR: Quantum network tests failed: {e}")
        sys.exit(1)
    
    logger.info("=== Mycelial Quantum Seeds Module Execution Complete ===")

# --- END OF FILE stage_1/evolve/mycelial/mycelial_quantum_seeds.py ---