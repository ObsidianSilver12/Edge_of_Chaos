# field_dynamics.py (Updated V6.0.0 - Brain Structure Integration)

"""
Brain Field Dynamics (Version 6.0.0 - Brain Structure Integration)

Integrates with brain_structure.py's static field patterns and standing waves.
Provides state-dependent field modulation while brain structure handles complex calculations.
"""

import numpy as np
import logging
import random
import uuid
from typing import Tuple, Dict, List, Optional, Any, Set

from constants.constants import *

logger = logging.getLogger("BrainFieldDynamics")
if not logger.handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


class BrainFieldDynamics:
    """
    Manages dynamic field modulation working with brain structure's static patterns.
    Brain structure handles standing waves, phi ratios, and static foundations.
    This class handles state-dependent modulation only.
    """
    
    def __init__(self, brain_structure_dimensions: Tuple[int, int, int]):
        if brain_structure_dimensions != GRID_DIMENSIONS:
            raise ValueError(f"BrainFieldDynamics must be initialized with GRID_DIMENSIONS {GRID_DIMENSIONS}, got {brain_structure_dimensions}")
        
        self.dimensions: Tuple[int, int, int] = brain_structure_dimensions
        self.field_system_id: str = str(uuid.uuid4())
        
        # Reference to brain structure (set by brain structure)
        self.brain_structure_ref = None
        
        # Simple state tracking
        self.current_brain_state: str = BRAIN_STATE_DORMANT
        self.active_regions_in_current_state: Set[str] = set()
        self.last_specific_thought_signature: Optional[str] = None
        self.last_processing_intensity: float = 0.0
        
        # Basic field properties
        self.static_patterns_available: bool = False
        self.field_active: bool = False
        self.chaos_level: float = FIELD_CHAOS_LEVEL_DEFAULT
        
        # Dynamic field modulations (small arrays for state changes)
        self.state_modulation_cache: Dict[str, np.ndarray] = {}
        self.current_modulation = np.zeros(self.dimensions, dtype=np.float32)
        
        logger.info(f"BrainFieldDynamics system {self.field_system_id} initialized for {self.dimensions} grid.")
    
    def set_brain_structure_reference(self, brain_structure):
        """Set reference to brain structure."""
        self.brain_structure_ref = brain_structure
        
        # Check if brain structure has static patterns
        if (hasattr(brain_structure, 'static_fields_created') and 
            brain_structure.static_fields_created):
            self.static_patterns_available = True
            logger.info("Brain structure static patterns detected and available")
        else:
            logger.warning("Brain structure static patterns not yet available")
    
    def validate_field_integrity(self) -> bool:
        """Validates that brain structure's static patterns are available and valid."""
        logger.debug("Validating field integrity via brain structure...")
        
        if not self.brain_structure_ref:
            logger.error("Integrity Check Failed: No brain structure reference.")
            return False
        
        if not self.static_patterns_available:
            logger.error("Integrity Check Failed: Brain structure static patterns not available.")
            return False
        
        # Check if brain structure has required static systems
        required_attributes = [
            'static_fields_created',
            'standing_waves_calculated', 
            'regions_defined'
        ]
        
        for attr in required_attributes:
            if not hasattr(self.brain_structure_ref, attr):
                logger.error(f"Integrity Check Failed: Brain structure missing {attr}")
                return False
            
            if not getattr(self.brain_structure_ref, attr):
                logger.error(f"Integrity Check Failed: Brain structure {attr} is False")
                return False
        
        logger.debug("Field integrity validated successfully via brain structure.")
        return True
    
    def update_fields_for_new_state(self, 
                                   new_brain_state: str, 
                                   active_regions_now: Optional[Set[str]] = None,
                                   processing_intensity: float = 0.5,
                                   specific_thought_signature: Optional[str] = None):
        """
        Update dynamic field modulations for new brain state.
        Brain structure handles static patterns, this handles state-dependent changes.
        """
        if not self.static_patterns_available:
            logger.warning("Cannot update fields: Brain structure static patterns not available")
            return
        
        active_regions_now = active_regions_now if active_regions_now is not None else set()
        
        # Check if state actually changed
        state_changed = (new_brain_state != self.current_brain_state or
                        active_regions_now != self.active_regions_in_current_state or
                        specific_thought_signature != self.last_specific_thought_signature or
                        abs(processing_intensity - self.last_processing_intensity) > FLOAT_EPSILON)
        
        if not state_changed:
            logger.debug(f"Brain state unchanged ('{new_brain_state}'). Adding minor chaos only.")
            self._inject_biological_chaos(self.current_modulation, intensity_scale=0.05)
            return
        
        logger.info(f"Updating field dynamics: '{self.current_brain_state}' -> '{new_brain_state}'. "
                   f"Active: {active_regions_now}. Intensity: {processing_intensity:.2f}")
        
        # Update state tracking
        self.current_brain_state = new_brain_state
        self.active_regions_in_current_state = active_regions_now
        self.last_specific_thought_signature = specific_thought_signature
        self.last_processing_intensity = processing_intensity
        
        # Create cache key for this state
        cache_key = f"{new_brain_state}_{hash(frozenset(active_regions_now))}_{processing_intensity:.2f}"
        
        # Check cache first
        if FIELD_STATE_CACHE_ENABLED and cache_key in self.state_modulation_cache:
            logger.debug(f"Using cached modulation for state: {cache_key}")
            self.current_modulation = self.state_modulation_cache[cache_key].copy()
        else:
            # Calculate new modulation
            self.current_modulation = self._calculate_state_modulation(
                new_brain_state, active_regions_now, processing_intensity, specific_thought_signature
            )
            
            # Cache the modulation
            if FIELD_STATE_CACHE_ENABLED:
                self.state_modulation_cache[cache_key] = self.current_modulation.copy()
        
        # Apply biological chaos
        self._inject_biological_chaos(self.current_modulation, intensity_scale=processing_intensity)
        
        # Mark field as active
        self.field_active = True
        
        logger.info(f"Field dynamics updated for state '{new_brain_state}'")
    
    def _calculate_state_modulation(self, brain_state: str, active_regions: Set[str], 
                                   intensity: float, thought_signature: Optional[str]) -> np.ndarray:
        """Calculate state-dependent field modulation."""
        modulation = np.zeros(self.dimensions, dtype=np.float32)
        
        try:
            # State-specific modulations
            if brain_state == BRAIN_STATE_FORMATION:
                # Formation state: enhance development around brain seed
                if hasattr(self.brain_structure_ref, 'seed_integration_data'):
                    seed_data = self.brain_structure_ref.seed_integration_data
                    if seed_data and 'position' in seed_data:
                        position = seed_data['position']
                        strength = intensity * 0.2
                        self._add_radial_modulation(modulation, position, strength, radius=20)
            
            elif brain_state == BRAIN_STATE_AWARE_PROCESSING:
                # Aware processing: enhance active regions
                for region_name in active_regions:
                    if region_name in self.brain_structure_ref.regions:
                        region = self.brain_structure_ref.regions[region_name]
                        center = region['center']
                        radius = region['radius']
                        strength = intensity * 0.15
                        self._add_radial_modulation(modulation, center, strength, radius)
            
            elif brain_state == BRAIN_STATE_SOUL_ATTACHED_SETTLING:
                # Soul settling: enhance around soul attachment point
                if hasattr(self.brain_structure_ref, 'seed_integration_data'):
                    seed_data = self.brain_structure_ref.seed_integration_data
                    if seed_data and 'position' in seed_data:
                        position = seed_data['position']
                        strength = intensity * 0.3
                        self._add_radial_modulation(modulation, position, strength, radius=15)
            
            elif brain_state == 'calming':
                # Calming state: gentle, uniform enhancement
                base_strength = intensity * 0.1
                modulation.fill(base_strength)
            
            # Add thought signature effects if present
            if thought_signature == 'mother_comfort':
                # Mother comfort: gentle limbic enhancement
                if 'limbic' in self.brain_structure_ref.regions:
                    limbic_region = self.brain_structure_ref.regions['limbic']
                    center = limbic_region['center']
                    radius = limbic_region['radius'] * 1.2
                    strength = intensity * 0.25
                    self._add_radial_modulation(modulation, center, strength, radius)
            
        except Exception as e:
            logger.error(f"Error calculating state modulation: {e}")
        
        return modulation
    
    def _add_radial_modulation(self, modulation: np.ndarray, center: Tuple[int, int, int], 
                              strength: float, radius: float):
        """Add radial modulation around a center point."""
        cx, cy, cz = center
        
        # Create coordinate grids
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        z = np.arange(self.dimensions[2])
        
        # Calculate distances from center
        x_dist = (x - cx) ** 2
        y_dist = (y[:, np.newaxis] - cy) ** 2
        z_dist = (z[:, np.newaxis, np.newaxis] - cz) ** 2
        
        distance = np.sqrt(x_dist[np.newaxis, np.newaxis, :] + 
                          y_dist[np.newaxis, :, np.newaxis] + 
                          z_dist[:, np.newaxis, np.newaxis])
        
        # Apply modulation with falloff
        mask = distance <= radius
        if np.any(mask):
            falloff = 1.0 - (distance / radius)
            modulation[mask] += strength * falloff[mask]
    
    def _inject_biological_chaos(self, field_array: np.ndarray, intensity_scale: float = 1.0):
        """Adds small random noise for biological realism."""
        if self.chaos_level > FLOAT_EPSILON:
            noise_std_dev = self.chaos_level * intensity_scale * 0.05
            if noise_std_dev < FLOAT_EPSILON:
                noise_std_dev = FLOAT_EPSILON * 10
            
            noise = np.random.normal(0, noise_std_dev, field_array.shape).astype(np.float32)
            field_array += noise
        
        return field_array
    
    def get_combined_field_value_at_point(self, position: Tuple[int, int, int]) -> float:
        """
        Get combined field value at point.
        Combines brain structure's static patterns with dynamic modulation.
        """
        if not self.static_patterns_available or not self.brain_structure_ref:
            logger.warning("Cannot get field value: static patterns not available")
            return 0.0
        
        # Validate position
        x, y, z = position
        if not (0 <= x < self.dimensions[0] and 
                0 <= y < self.dimensions[1] and 
                0 <= z < self.dimensions[2]):
            logger.error(f"Position {position} out of bounds {self.dimensions}")
            return 0.0
        
        try:
            # Get static field value from brain structure
            static_value = 0.0
            if hasattr(self.brain_structure_ref, 'get_field_value'):
                static_value = self.brain_structure_ref.get_field_value(position, 'energy')
            elif (hasattr(self.brain_structure_ref, 'static_field_foundation') and 
                  self.brain_structure_ref.static_field_foundation is not None):
                static_value = float(self.brain_structure_ref.static_field_foundation[x, y, z])
            
            # Add dynamic modulation
            dynamic_value = float(self.current_modulation[x, y, z])
            
            return static_value + dynamic_value
            
        except Exception as e:
            logger.error(f"Error getting field value at {position}: {e}")
            return 0.0
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get current field state."""
        return {
            'field_system_id': self.field_system_id,
            'field_active': self.field_active,
            'current_brain_state': self.current_brain_state,
            'static_patterns_available': self.static_patterns_available,
            'brain_structure_connected': self.brain_structure_ref is not None,
            'active_regions': list(self.active_regions_in_current_state),
            'processing_intensity': self.last_processing_intensity,
            'modulation_cache_size': len(self.state_modulation_cache),
            'chaos_level': self.chaos_level
        }


# === Quantum Seeds Coordinator ===

class QuantumSeedsCoordinator:
    """
    Coordinates quantum seeds network with enhanced mycelial network.
    Handles cross-region communication and workload distribution.
    """
    
    def __init__(self, enhanced_network=None):
        self.coordinator_id = str(uuid.uuid4())
        self.enhanced_network = enhanced_network
        self.quantum_network = None
        self.coordination_active = False
        
        logger.info(f"Quantum seeds coordinator initialized: {self.coordinator_id[:8]}")
    
    def set_quantum_network(self, quantum_network):
        """Set the quantum seeds network."""
        self.quantum_network = quantum_network
        self.coordination_active = True
        logger.info("Quantum seeds network connected to coordinator")
    
    def coordinate_cross_region_processing(self, workload_type: str, 
                                         source_region: str, 
                                         target_regions: List[str]) -> Dict[str, Any]:
        """Use quantum entanglement for efficient cross-region processing."""
        if not self.quantum_network:
            return {'success': False, 'reason': 'No quantum network available'}
        
        try:
            # Find quantum seeds in source region
            source_seeds = self._find_seeds_in_region(source_region)
            
            # Find quantum seeds in target regions
            target_seeds = []
            for region in target_regions:
                region_seeds = self._find_seeds_in_region(region)
                target_seeds.extend(region_seeds)
            
            # Process workload using entangled seeds
            processing_results = []
            successful_coordinations = 0
            
            for target_seed in target_seeds:
                # Check if any source seed is entangled with target
                if self._has_entanglement(source_seeds, target_seed):
                    result = target_seed.process_workload(
                        workload_type=workload_type,
                        workload_data={'source_region': source_region},
                        processing_time=1.0
                    )
                    
                    if result.get('success', False):
                        processing_results.append(result)
                        successful_coordinations += 1
            
            # Report to enhanced network if available
            if self.enhanced_network and hasattr(self.enhanced_network, 'metrics'):
                self.enhanced_network.metrics['quantum_coordinations'] = (
                    self.enhanced_network.metrics.get('quantum_coordinations', 0) + 
                    successful_coordinations
                )
            
            logger.info(f"Quantum coordination completed: {successful_coordinations} successful")
            
            return {
                'success': True,
                'workload_type': workload_type,
                'source_region': source_region,
                'target_regions': target_regions,
                'source_seeds_found': len(source_seeds),
                'target_seeds_found': len(target_seeds),
                'successful_coordinations': successful_coordinations,
                'processing_results': processing_results
            }
            
        except Exception as e:
            logger.error(f"Quantum coordination error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _find_seeds_in_region(self, region_name: str) -> List:
        """Find quantum seeds that focus on specific brain region."""
        if not self.quantum_network or not hasattr(self.quantum_network, 'seeds'):
            return []
        
        region_seeds = []
        
        # Map region names to workload focus types
        region_focus_map = {
            'brain_stem': 'survival',
            'limbic': 'emotional', 
            'frontal': 'creative',
            'temporal': 'emotional',
            'parietal': 'creative',
            'occipital': 'survival',
            'cerebellum': 'survival'
        }
        
        target_focus = region_focus_map.get(region_name, 'survival')
        
        for seed_id, seed in self.quantum_network.seeds.items():
            if hasattr(seed, 'workload_focus') and seed.workload_focus == target_focus:
                region_seeds.append(seed)
        
        return region_seeds
    
    def _has_entanglement(self, source_seeds: List, target_seed) -> bool:
        """Check if any source seed is entangled with target seed."""
        if not hasattr(target_seed, 'entangled_seeds'):
            return False
        
        for source_seed in source_seeds:
            if hasattr(source_seed, 'seed_id'):
                if source_seed.seed_id in target_seed.entangled_seeds:
                    return True
        
        return False
    
    def optimize_entanglement_network(self) -> Dict[str, Any]:
        """Optimize quantum entanglement network for better coordination."""
        if not self.quantum_network:
            return {'success': False, 'reason': 'No quantum network available'}
        
        try:
            # Get current entanglement statistics
            total_seeds = len(self.quantum_network.seeds)
            total_entanglements = self.quantum_network.total_entanglements
            
            # Calculate optimization metrics
            avg_entanglements_per_seed = total_entanglements / total_seeds if total_seeds > 0 else 0
            
            optimization_result = {
                'success': True,
                'total_seeds': total_seeds,
                'total_entanglements': total_entanglements,
                'avg_entanglements_per_seed': avg_entanglements_per_seed,
                'network_efficiency': min(1.0, avg_entanglements_per_seed / 5.0)  # Target 5 entanglements per seed
            }
            
            # Report optimization to enhanced network
            if self.enhanced_network and hasattr(self.enhanced_network, 'metrics'):
                self.enhanced_network.metrics['quantum_network_efficiency'] = optimization_result['network_efficiency']
            
            logger.info(f"Quantum network optimization: {optimization_result['network_efficiency']:.2f} efficiency")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Quantum network optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            'coordinator_id': self.coordinator_id,
            'coordination_active': self.coordination_active,
            'quantum_network_connected': self.quantum_network is not None,
            'enhanced_network_connected': self.enhanced_network is not None,
            'quantum_network_state': (
                self.quantum_network.get_network_state() 
                if self.quantum_network else None
            )
        }


# === Enhanced Mycelial Network Integration ===

def integrate_quantum_coordinator_with_enhanced_network(enhanced_network, quantum_network):
    """
    Integrate quantum coordinator with enhanced mycelial network.
    
    Args:
        enhanced_network: Enhanced mycelial network instance
        quantum_network: Quantum seeds network instance
        
    Returns:
        Configured quantum coordinator
    """
    logger.info("Integrating quantum coordinator with enhanced mycelial network")
    
    # Create quantum coordinator
    quantum_coordinator = QuantumSeedsCoordinator(enhanced_network)
    quantum_coordinator.set_quantum_network(quantum_network)
    
    # Add quantum coordinator to enhanced network
    enhanced_network.quantum_seeds_coordinator = quantum_coordinator
    
    # Add quantum coordination method to enhanced network
    def coordinate_with_quantum_seeds(self, workload_type: str, regions: List[str]):
        """Use quantum entanglement for efficient cross-region processing."""
        if hasattr(self, 'quantum_seeds_coordinator'):
            return self.quantum_seeds_coordinator.coordinate_cross_region_processing(
                workload_type, regions[0] if regions else 'limbic', regions[1:] if len(regions) > 1 else []
            )
        else:
            return {'success': False, 'reason': 'Quantum coordinator not available'}
    
    # Bind method to enhanced network
    import types
    enhanced_network.coordinate_with_quantum_seeds = types.MethodType(
        coordinate_with_quantum_seeds, enhanced_network
    )
    
    logger.info("Quantum coordinator integrated with enhanced mycelial network")
    
    return quantum_coordinator