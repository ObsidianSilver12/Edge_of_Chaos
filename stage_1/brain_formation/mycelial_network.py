# --- mycelial_network.py V8 COMPLETE - Part 1 ---
"""
Create the mycelial network and distribute mycelial seeds within each sub region of the brain.
Provides basic functionality for controlling sleep/wake cycle, monitoring state and responding to stimuli
or changes in neural network activity. This is the "circulatory system" of the brain - separate from
neural network but part of the brain. Handles autonomic functions, energy management, state monitoring,
learning, sleep/wake cycles, etc. Uses quantum entanglement between seeds only (not neural nodes).
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

class MycelialNetwork:
    def __init__(self):
        """Initialize the mycelial network - the brain's circulatory system."""
        self.mycelial_network = {}
        self.brain_structure = None
        self.neural_network = None
        self.energy_storage = None
        self.seeds = {}  # seed_id -> seed data
        self.routes = {}  # route_id -> route data
        self.entanglements = {}  # quantum entanglement pairs
        self.active_seeds = []
        self.inactive_seeds = []
        self.brain_state = {
            'current_state': 'awake',
            'natural_frequency': 0.0,
            'current_state_frequency': 0.0,
            'wave_pattern': 'alpha'
        }
        self.sleep_wake_cycle_active = False
        self.liminal_state_active = False
        self.soul_attached = False
        self.energy_thresholds = {
            'critical_low': 0.1,
            'low': 0.3,
            'optimal': 0.7,
            'high': 0.9
        }
        self.metrics = {
            'total_seeds': 0,
            'active_seeds': 0,
            'inactive_seeds': 0,
            'entangled_pairs': 0,
            'quantum_communications': 0,
            'sleep_cycles': 0,
            'wake_cycles': 0,
            'energy_conservations': 0,
            'errors': []
        }

    def create_mycelial_network(self):
        """
        Initialize the mycelial network. Create optimal placement between sub-regions placing
        more mycelial seeds for larger areas. Check coordinates are within bounds and not occupied.
        Ensure network is not too dense to avoid overcrowding and energy inefficiencies.
        Triggered by flag NEURAL_NETWORK_CREATED.
        """
        try:
            logger.info("Creating mycelial network - the brain's circulatory system...")
            
            # Load required data
            self._load_brain_and_neural_data()
            
            # Verify brain structure is loaded
            if not self.brain_structure:
                raise RuntimeError("Brain structure not initialized")
            
            # Calculate energy requirements first
            total_energy_needed = self._calculate_total_energy_requirements()
            
            # Get available energy from storage
            available_energy = self._get_available_energy()
            
            if available_energy < total_energy_needed:
                logger.warning(f"Insufficient energy for full mycelial network. "
                             f"Need: {total_energy_needed:.2f}, Available: {available_energy:.2f}")
                # Reduce network density to fit available energy
                density_factor = available_energy / total_energy_needed
            else:
                density_factor = 1.0
            
            # Create seeds for each sub-region
            seed_counter = 0
            for hemisphere_name, hemisphere in self.brain_structure['hemispheres'].items():
                for region_name, region in hemisphere['regions'].items():
                    for sub_region_name, sub_region in region['sub_regions'].items():
                        
                        # Calculate sub region volume
                        bounds = sub_region['bounds']
                        volume = ((bounds['x_end'] - bounds['x_start']) * 
                                (bounds['y_end'] - bounds['y_start']) * 
                                (bounds['z_end'] - bounds['z_start']))
                        
                        # Determine number of seeds based on volume and density factor
                        base_seed_count = max(1, int(volume / 15000))  # 1 seed per 15k volume units
                        adjusted_seed_count = max(1, int(base_seed_count * density_factor))
                        
                        # Create seeds with optimal placement
                        region_seeds = self._create_optimal_seed_placement(
                            bounds, adjusted_seed_count, hemisphere_name, region_name, sub_region_name)
                        
                        # Store seeds
                        for seed in region_seeds:
                            seed_id = seed['seed_id']
                            self.seeds[seed_id] = seed
                            self.inactive_seeds.append(seed_id)
                            seed_counter += 1
                        
                        logger.debug(f"Created {len(region_seeds)} mycelial seeds in "
                                   f"{hemisphere_name}.{region_name}.{sub_region_name}")
            
            self.metrics['total_seeds'] = len(self.seeds)
            self.metrics['inactive_seeds'] = len(self.inactive_seeds)
            
            logger.info(f"Mycelial network created: {len(self.seeds)} seeds across brain regions")
            
            # Create routes between seeds
            self.create_mycelial_routes()
            
        except Exception as e:
            logger.error(f"Failed to create mycelial network: {e}")
            self.metrics['errors'].append(f"Network creation failed: {e}")
            raise RuntimeError(f"Mycelial network creation failed: {e}")

    def _load_brain_and_neural_data(self):
        """Load brain structure and neural network data for mycelial network creation."""
        try:
            # Try to get brain structure (should be available from brain_structure.py)
            if not hasattr(self, 'brain_structure') or not self.brain_structure:
                # Create minimal structure for testing if not available
                logger.warning("Brain structure not loaded - creating minimal test structure")
                self.brain_structure = {
                    'hemispheres': {
                        'left': {
                            'regions': {
                                'frontal': {
                                    'sub_regions': {
                                        'prefrontal': {
                                            'bounds': {'x_start': 0, 'x_end': 50, 'y_start': 0, 'y_end': 50, 'z_start': 150, 'z_end': 200}
                                        }
                                    }
                                },
                                'limbic': {
                                    'sub_regions': {
                                        'hippocampus': {
                                            'bounds': {'x_start': 20, 'x_end': 40, 'y_start': 20, 'y_end': 40, 'z_start': 120, 'z_end': 140}
                                        }
                                    }
                                }
                            }
                        },
                        'right': {
                            'regions': {
                                'frontal': {
                                    'sub_regions': {
                                        'prefrontal': {
                                            'bounds': {'x_start': 50, 'x_end': 100, 'y_start': 0, 'y_end': 50, 'z_start': 150, 'z_end': 200}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            
            # Try to get neural network data (should be available from neural_network.py)
            if not hasattr(self, 'neural_network') or not self.neural_network:
                logger.warning("Neural network data not loaded - mycelial network will function independently")
                self.neural_network = {'nodes': {}, 'synapses': {}}
            
        except Exception as e:
            logger.error(f"Failed to load brain and neural data: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def _calculate_total_energy_requirements(self) -> float:
        """Calculate total energy needed for the mycelial network."""
        try:
            # Base energy per seed
            base_energy_per_seed = 0.5  # SEU per seed
            
            # Check if brain structure exists
            if not hasattr(self, 'brain_structure') or not self.brain_structure:
                logger.warning("No brain structure available - using default energy requirement")
                return 10.0
            
            # Calculate rough seed count based on brain structure
            total_volume = 0
            for hemisphere in self.brain_structure['hemispheres'].values():
                for region in hemisphere['regions'].values():
                    for sub_region in region['sub_regions'].values():
                        bounds = sub_region['bounds']
                        volume = ((bounds['x_end'] - bounds['x_start']) * 
                                (bounds['y_end'] - bounds['y_start']) * 
                                (bounds['z_end'] - bounds['z_start']))
                        total_volume += volume
            
            estimated_seeds = max(10, int(total_volume / 15000))  # Conservative estimate
            total_energy = estimated_seeds * base_energy_per_seed
            
            logger.debug(f"Estimated energy requirements: {estimated_seeds} seeds × {base_energy_per_seed} = {total_energy:.2f} SEU")
            return total_energy
            
        except Exception as e:
            logger.error(f"Error calculating energy requirements: {e}")
            return 10.0  # Conservative fallback

    def _get_available_energy(self) -> float:
        """Get available energy from energy storage system."""
        try:
            if hasattr(self, 'energy_storage') and self.energy_storage:
                return self.energy_storage.get('current_energy', 50.0)  # Default if not set
            else:
                logger.warning("Energy storage not available - using default energy value")
                return 50.0  # Default energy value for testing
                
        except Exception as e:
            logger.error(f"Error getting available energy: {e}")
            return 25.0  # Conservative fallback

    def _create_optimal_seed_placement(self, bounds: Dict, count: int, hemisphere: str, 
                                     region: str, sub_region: str) -> List[Dict]:
        """Create optimal placement of mycelial seeds within sub-region bounds."""
        seeds = []
        
        # Calculate sub-region center and dimensions
        center_x = (bounds['x_start'] + bounds['x_end']) / 2
        center_y = (bounds['y_start'] + bounds['y_end']) / 2
        center_z = (bounds['z_start'] + bounds['z_end']) / 2
        
        width = bounds['x_end'] - bounds['x_start']
        height = bounds['y_end'] - bounds['y_start']
        depth = bounds['z_end'] - bounds['z_start']
        
        # Use golden ratio spiral for optimal distribution
        for i in range(count):
            # Golden angle spiral in 3D
            angle1 = i * 2.4  # Golden angle
            angle2 = i * 0.618 * math.pi  # Phi-based vertical distribution
            
            # Radius increases with sqrt for even distribution
            radius_factor = math.sqrt(i / max(1, count - 1))
            max_radius = min(width, height, depth) * 0.4  # Stay within bounds
            
            # Calculate position
            x = center_x + (radius_factor * max_radius * math.cos(angle1) * 
                           math.cos(angle2) * width / max(width, height, depth))
            y = center_y + (radius_factor * max_radius * math.sin(angle1) * 
                           math.cos(angle2) * height / max(width, height, depth))
            z = center_z + (radius_factor * max_radius * math.sin(angle2) * 
                           depth / max(width, height, depth))
            
            # Ensure within bounds with small buffer
            buffer = 2.0
            x = max(bounds['x_start'] + buffer, min(bounds['x_end'] - buffer, x))
            y = max(bounds['y_start'] + buffer, min(bounds['y_end'] - buffer, y))
            z = max(bounds['z_start'] + buffer, min(bounds['z_end'] - buffer, z))
            
            # Check if position conflicts with neural nodes
            if self._position_conflicts_with_neural_nodes(x, y, z):
                # Adjust position slightly
                x += random.uniform(-3, 3)
                y += random.uniform(-3, 3)
                z += random.uniform(-3, 3)
                
                # Re-clamp to bounds
                x = max(bounds['x_start'] + buffer, min(bounds['x_end'] - buffer, x))
                y = max(bounds['y_start'] + buffer, min(bounds['y_end'] - buffer, y))
                z = max(bounds['z_start'] + buffer, min(bounds['z_end'] - buffer, z))
            
            # Create seed
            seed = {
                'seed_id': str(uuid.uuid4()),
                'coordinates': {'x': x, 'y': y, 'z': z},
                'hemisphere': hemisphere,
                'region': region,
                'sub_region': sub_region,
                'status': 'inactive',
                'base_frequency': random.uniform(0.5, 4.0),  # Low frequency for efficiency
                'energy_consumption': random.uniform(0.3, 0.7),  # SEU per activation
                'quantum_state': 'unentangled',
                'entangled_with': [],
                'creation_time': datetime.now().isoformat(),
                'activation_count': 0,
                'routes': []
            }
            seeds.append(seed)
        
        return seeds

    def _position_conflicts_with_neural_nodes(self, x: float, y: float, z: float) -> bool:
        """Check if position conflicts with existing neural network nodes."""
        try:
            if not self.neural_network or 'nodes' not in self.neural_network:
                return False
            
            min_distance = 8.0  # Minimum distance from neural nodes
            
            for node in self.neural_network['nodes'].values():
                if 'coordinates' in node:
                    node_coords = node['coordinates']
                    distance = math.sqrt(
                        (x - node_coords['x'])**2 +
                        (y - node_coords['y'])**2 +
                        (z - node_coords['z'])**2
                    )
                    if distance < min_distance:
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking neural node conflicts: {e}")
            return False  # Assume no conflict if error
        
    def create_mycelial_routes(self):
        """
        Create the most optimal routes between mycelial seeds. Functions similar to synapses
        but routes are only deactivated due to seed decay, not communication failures.
        Routes are used for energy return to storage, not quantum communication between seeds.
        """
        try:
            logger.info("Creating mycelial routes for energy transport...")
            
            route_counter = 0
            
            # Create routes within each sub-region first
            seeds_by_subregion = {}
            for seed_id, seed in self.seeds.items():
                region_key = f"{seed['hemisphere']}.{seed['region']}.{seed['sub_region']}"
                if region_key not in seeds_by_subregion:
                    seeds_by_subregion[region_key] = []
                seeds_by_subregion[region_key].append(seed_id)
            
            # Create intra-region routes (within same sub-region)
            for region_key, seed_ids in seeds_by_subregion.items():
                if len(seed_ids) > 1:
                    for i, seed_id in enumerate(seed_ids):
                        for j, other_seed_id in enumerate(seed_ids):
                            if i != j:
                                distance = self._calculate_seed_distance(seed_id, other_seed_id)
                                
                                # Connect nearby seeds within sub-region
                                if distance < 25:  # Close proximity threshold
                                    route_id = f"intra_route_{route_counter}"
                                    route = {
                                        'route_id': route_id,
                                        'type': 'intra_region',
                                        'from_seed': seed_id,
                                        'to_seed': other_seed_id,
                                        'distance': distance,
                                        'status': 'active',
                                        'efficiency': max(0.5, 1.0 - distance/50),
                                        'energy_cost': distance * 0.01,  # Energy cost for transport
                                        'creation_time': datetime.now().isoformat()
                                    }
                                    
                                    self.routes[route_id] = route
                                    self.seeds[seed_id]['routes'].append(route_id)
                                    route_counter += 1
            
            # Create inter-region routes (between different sub-regions)
            all_seed_ids = list(self.seeds.keys())
            for seed_id in all_seed_ids:
                seed = self.seeds[seed_id]
                nearest_external = []
                
                # Find nearest seeds in other sub-regions
                for other_seed_id in all_seed_ids:
                    if seed_id != other_seed_id:
                        other_seed = self.seeds[other_seed_id]
                        
                        # Check if in different sub-region
                        if (seed['hemisphere'] != other_seed['hemisphere'] or
                            seed['region'] != other_seed['region'] or
                            seed['sub_region'] != other_seed['sub_region']):
                            
                            distance = self._calculate_seed_distance(seed_id, other_seed_id)
                            nearest_external.append((other_seed_id, distance))
                
                # Sort by distance and connect to nearest 2-3 external seeds
                nearest_external.sort(key=lambda x: x[1])
                for other_seed_id, distance in nearest_external[:3]:  # Connect to 3 nearest
                    if distance < 80:  # Maximum inter-region distance
                        route_id = f"inter_route_{route_counter}"
                        route = {
                            'route_id': route_id,
                            'type': 'inter_region',
                            'from_seed': seed_id,
                            'to_seed': other_seed_id,
                            'distance': distance,
                            'status': 'active',
                            'efficiency': max(0.2, 0.8 - distance/100),
                            'energy_cost': distance * 0.02,  # Higher cost for long routes
                            'creation_time': datetime.now().isoformat()
                        }
                        
                        self.routes[route_id] = route
                        self.seeds[seed_id]['routes'].append(route_id)
                        route_counter += 1
            
            logger.info(f"Mycelial routes created: {len(self.routes)} total routes")
            
        except Exception as e:
            logger.error(f"Failed to create mycelial routes: {e}")
            raise RuntimeError(f"Mycelial route creation failed: {e}")

    def _calculate_seed_distance(self, seed1_id: str, seed2_id: str) -> float:
        """Calculate distance between two mycelial seeds."""
        seed1 = self.seeds[seed1_id]
        seed2 = self.seeds[seed2_id]
        coord1 = seed1['coordinates']
        coord2 = seed2['coordinates']
        
        return math.sqrt(
            (coord1['x'] - coord2['x'])**2 +
            (coord1['y'] - coord2['y'])**2 +
            (coord1['z'] - coord2['z'])**2
        )

    def activate_mycelial_seeds(self):
        """
        Activate the mycelial seeds to allow quantum entanglement communication.
        Update energy storage with energy used for activation.
        """
        try:
            logger.info("Activating mycelial seeds...")
            
            # Calculate energy needed for activation
            total_activation_energy = 0
            activated_count = 0
            
            for seed_id in self.inactive_seeds.copy():  # Copy list to avoid modification during iteration
                seed = self.seeds[seed_id]
                activation_energy = seed['energy_consumption']
                
                # Check if we have enough energy
                available_energy = self._get_available_energy()
                if available_energy >= activation_energy:
                    # Activate seed
                    seed['status'] = 'active'
                    seed['activation_count'] += 1
                    total_activation_energy += activation_energy
                    
                    # Move to active list
                    self.inactive_seeds.remove(seed_id)
                    self.active_seeds.append(seed_id)
                    activated_count += 1
                    
                    # Subtract energy from storage
                    self._consume_energy(activation_energy)
                else:
                    logger.warning(f"Insufficient energy to activate seed {seed_id}")
                    break  # Stop if we run out of energy
            
            # Update metrics
            self.metrics['active_seeds'] = len(self.active_seeds)
            self.metrics['inactive_seeds'] = len(self.inactive_seeds)
            
            logger.info(f"Activated {activated_count} mycelial seeds, energy consumed: {total_activation_energy:.2f} SEU")
            
        except Exception as e:
            logger.error(f"Failed to activate mycelial seeds: {e}")
            self.metrics['errors'].append(f"Seed activation failed: {e}")
            raise RuntimeError(f"Mycelial seed activation failed: {e}")

    def entangle_mycelial_seeds(self):
        """
        Entangle all mycelial seeds using quantum entanglement for communication.
        Uses distance-based strategy for efficient entanglement network.
        Check network stability and re-entangle if needed. Save as flag SEEDS_ENTANGLED.
        """
        try:
            logger.info("Creating quantum entanglement network between mycelial seeds...")
            
            if len(self.active_seeds) < 2:
                logger.warning("Need at least 2 active seeds for entanglement")
                return
            
            # Strategy: Create entanglement based on optimal connectivity
            # Each seed should be entangled with 2-4 others for redundancy
            entanglement_pairs = []
            
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                
                # Find closest active seeds for entanglement
                distances = []
                for other_seed_id in self.active_seeds:
                    if seed_id != other_seed_id and other_seed_id not in seed['entangled_with']:
                        distance = self._calculate_seed_distance(seed_id, other_seed_id)
                        distances.append((other_seed_id, distance))
                
                # Sort by distance and entangle with closest 2-3 seeds
                distances.sort(key=lambda x: x[1])
                target_entanglements = min(3, len(distances))
                
                for other_seed_id, distance in distances[:target_entanglements]:
                    # Create quantum entanglement pair
                    entanglement_id = str(uuid.uuid4())
                    entanglement = {
                        'entanglement_id': entanglement_id,
                        'seed1': seed_id,
                        'seed2': other_seed_id,
                        'strength': max(0.3, 1.0 - distance/100),  # Distance affects strength
                        'frequency_sync': True,  # All seeds have same base frequency
                        'status': 'stable',
                        'creation_time': datetime.now().isoformat(),
                        'communication_count': 0
                    }
                    
                    self.entanglements[entanglement_id] = entanglement
                    entanglement_pairs.append((seed_id, other_seed_id))
                    
                    # Update seeds with entanglement info
                    self.seeds[seed_id]['entangled_with'].append(other_seed_id)
                    self.seeds[other_seed_id]['entangled_with'].append(seed_id)
                    self.seeds[seed_id]['quantum_state'] = 'entangled'
                    self.seeds[other_seed_id]['quantum_state'] = 'entangled'
            
            # Check network stability
            stable = self._check_entanglement_network_stability()
            
            if not stable:
                logger.warning("Entanglement network unstable, re-entangling isolated seeds...")
                self._fix_entanglement_network()
            
            # Set completion flag
            setattr(self, FLAG_SEEDS_ENTANGLED, True)
            self.metrics['entangled_pairs'] = len(self.entanglements)
            
            logger.info(f"Quantum entanglement network created: {len(self.entanglements)} entanglement pairs")
            
        except Exception as e:
            logger.error(f"Failed to entangle mycelial seeds: {e}")
            raise RuntimeError(f"Mycelial seed entanglement failed: {e}")

    def _check_entanglement_network_stability(self) -> bool:
        """Check if the entanglement network is stable (no isolated nodes, no loops)."""
        try:
            # Check for isolated seeds (not entangled with anyone)
            isolated_seeds = []
            for seed_id in self.active_seeds:
                if len(self.seeds[seed_id]['entangled_with']) == 0:
                    isolated_seeds.append(seed_id)
            
            if isolated_seeds:
                logger.debug(f"Found {len(isolated_seeds)} isolated seeds")
                return False
            
            # Check for network connectivity (simplified)
            # All seeds should be reachable from any other seed
            if len(self.active_seeds) > 1:
                start_seed = self.active_seeds[0]
                reachable = self._find_reachable_seeds(start_seed)
                
                if len(reachable) != len(self.active_seeds):
                    logger.debug(f"Network not fully connected: {len(reachable)}/{len(self.active_seeds)} reachable")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking entanglement stability: {e}")
            return False

    def _find_reachable_seeds(self, start_seed: str) -> set:
        """Find all seeds reachable from start seed through entanglement network."""
        reachable = set()
        queue = [start_seed]
        
        while queue:
            current_seed = queue.pop(0)
            if current_seed in reachable:
                continue
                
            reachable.add(current_seed)
            
            # Add entangled seeds to queue
            for entangled_seed in self.seeds[current_seed]['entangled_with']:
                if entangled_seed not in reachable:
                    queue.append(entangled_seed)
        
        return reachable

    def _fix_entanglement_network(self):
        """Fix entanglement network by connecting isolated seeds."""
        try:
            # Find isolated seeds
            isolated_seeds = []
            for seed_id in self.active_seeds:
                if len(self.seeds[seed_id]['entangled_with']) == 0:
                    isolated_seeds.append(seed_id)
            
            # Connect isolated seeds to nearest connected seeds
            for isolated_seed in isolated_seeds:
                # Find nearest connected seed
                nearest_connected = None
                min_distance = float('inf')
                
                for seed_id in self.active_seeds:
                    if (seed_id != isolated_seed and 
                        len(self.seeds[seed_id]['entangled_with']) > 0):
                        distance = self._calculate_seed_distance(isolated_seed, seed_id)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_connected = seed_id
                
                # Create entanglement with nearest connected seed
                if nearest_connected:
                    entanglement_id = str(uuid.uuid4())
                    entanglement = {
                        'entanglement_id': entanglement_id,
                        'seed1': isolated_seed,
                        'seed2': nearest_connected,
                        'strength': max(0.3, 1.0 - min_distance/100),
                        'frequency_sync': True,
                        'status': 'stable',
                        'creation_time': datetime.now().isoformat(),
                        'communication_count': 0
                    }
                    
                    self.entanglements[entanglement_id] = entanglement
                    self.seeds[isolated_seed]['entangled_with'].append(nearest_connected)
                    self.seeds[nearest_connected]['entangled_with'].append(isolated_seed)
                    self.seeds[isolated_seed]['quantum_state'] = 'entangled'
                    
                    logger.debug(f"Connected isolated seed {isolated_seed} to {nearest_connected}")
            
        except Exception as e:
            logger.error(f"Error fixing entanglement network: {e}")

    def _consume_energy(self, amount: float):
        """Consume energy from storage (placeholder - should integrate with energy_storage.py)."""
        try:
            if hasattr(self, 'energy_storage') and self.energy_storage:
                current = self.energy_storage.get('current_energy', 0)
                self.energy_storage.update_current_energy(max(0, current - amount))
            else:
                logger.debug(f"Energy consumption simulated: {amount:.2f} SEU")
        except Exception as e:
            logger.debug(f"Error consuming energy: {e}")

    def activate_liminal_state_for_attachment(self):
        """
        Activate liminal state by attuning one active mycelial seed to life cord spiritual anchor.
        Create resonance/vibration from natural seed frequency up to spiritual channel frequency.
        Set flag status to LIMINAL_STATE_ACTIVE when resonance achieved.
        """
        try:
            logger.info("Activating liminal state for soul attachment...")
            
            if not self.active_seeds:
                raise RuntimeError("No active mycelial seeds available for liminal state activation")
            
            # Find best seed for liminal state (closest to brain stem area)
            brain_stem_coords = {'x': 128, 'y': 128, 'z': 100}  # Approximate brain stem location
            best_seed = None
            min_distance = float('inf')
            
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                coords = seed['coordinates']
                distance = math.sqrt(
                    (coords['x'] - brain_stem_coords['x'])**2 +
                    (coords['y'] - brain_stem_coords['y'])**2 +
                    (coords['z'] - brain_stem_coords['z'])**2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_seed = seed_id
            
            if not best_seed:
                raise RuntimeError("Could not find suitable seed for liminal state")
            
            # Get life cord spiritual anchor frequency (should be available from life_cord.py)
            spiritual_anchor_frequency = self._get_spiritual_anchor_frequency()
            
            # Attune seed to spiritual frequency through resonance
            seed = self.seeds[best_seed]
            base_frequency = seed['base_frequency']
            
            # Calculate resonance steps to reach spiritual frequency
            frequency_ratio = spiritual_anchor_frequency / base_frequency
            resonance_steps = []
            
            # Create harmonic series to bridge frequencies
            current_freq = base_frequency
            step = 1
            while current_freq < spiritual_anchor_frequency * 0.95:
                # Use harmonic progression with phi scaling
                next_freq = current_freq * (1 + PHI/10)  # Gradual phi-based increase
                resonance_steps.append({
                    'step': step,
                    'frequency': next_freq,
                    'amplitude': 1.0 / step  # Decreasing amplitude
                })
                current_freq = next_freq
                step += 1
                
                if step > 20:  # Prevent infinite loop
                    break
            
            # Apply resonance progression
            liminal_seed_data = {
                'seed_id': best_seed,
                'original_frequency': base_frequency,
                'target_frequency': spiritual_anchor_frequency,
                'current_frequency': base_frequency,
                'resonance_steps': resonance_steps,
                'attunement_progress': 0.0,
                'liminal_state': 'attuning'
            }
            
            # Perform frequency attunement
            for i, step_data in enumerate(resonance_steps):
                progress = (i + 1) / len(resonance_steps)
                liminal_seed_data['current_frequency'] = step_data['frequency']
                liminal_seed_data['attunement_progress'] = progress
                
                # Check resonance achievement (simplified)
                if abs(step_data['frequency'] - spiritual_anchor_frequency) < 0.1:
                    liminal_seed_data['liminal_state'] = 'resonant'
                    break
            
            # Update seed with liminal state info
            seed['liminal_state_data'] = liminal_seed_data
            seed['status'] = 'liminal_active'
            
            # Set flags
            setattr(self, FLAG_LIMINAL_STATE_ACTIVE, True)
            self.liminal_state_active = True
            
            logger.info(f"Liminal state activated using seed {best_seed}, "
                       f"attunement progress: {liminal_seed_data['attunement_progress']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to activate liminal state: {e}")
            raise RuntimeError(f"Liminal state activation failed: {e}")

    def _get_spiritual_anchor_frequency(self) -> float:
        """Get spiritual anchor frequency from life cord (placeholder)."""
        try:
            # This should integrate with life_cord.py to get actual spiritual anchor frequency
            # For now, use a reasonable default based on spiritual frequencies
            return 528.0  # Hz - Love frequency from Solfeggio scale
        except Exception as e:
            logger.warning(f"Could not get spiritual anchor frequency: {e}")
            return 528.0  # Default fallback

    def attach_soul(self):
        """
        Guide soul to limbic region through vibration between soul and limbic sub region.
        Set flag to SOUL_ATTACHED when resonance achieved. Hard fail if resonance not achieved.
        """
        try:
            logger.info("Guiding soul to limbic region through mycelial network...")
            
            if not self.liminal_state_active:
                raise RuntimeError("Liminal state not active - cannot attach soul")
            
            # Find limbic region coordinates
            limbic_coords = self._find_limbic_region_center()
            if not limbic_coords:
                raise RuntimeError("Could not locate limbic region for soul attachment")
            
            # Get soul frequency from life cord (should be available)
            soul_frequency = self._get_soul_frequency()
            
            # Find mycelial seeds in limbic region
            limbic_seeds = []
            for seed_id, seed in self.seeds.items():
                if (seed['region'] == 'limbic' and seed['status'] in ['active', 'liminal_active']):
                    limbic_seeds.append(seed_id)
            
            if not limbic_seeds:
                raise RuntimeError("No active mycelial seeds in limbic region for soul attachment")
            
            # Create vibration bridge between soul and limbic region
            target_seed = limbic_seeds[0]  # Use first available limbic seed
            target_seed_data = self.seeds[target_seed]
            target_frequency = target_seed_data['base_frequency']
            
            # Calculate frequency adjustment needed for resonance
            frequency_difference = abs(soul_frequency - target_frequency)
            resonance_threshold = 0.5  # Hz tolerance
            
            if frequency_difference > resonance_threshold:
                # Adjust soul frequency to match limbic frequency (like aura vibration)
                logger.info(f"Adjusting soul frequency from {soul_frequency:.2f}Hz to {target_frequency:.2f}Hz")
                
                # Create gradual frequency transition
                transition_steps = 10
                freq_step = (target_frequency - soul_frequency) / transition_steps
                
                adjusted_soul_frequency = soul_frequency
                for step in range(transition_steps):
                    adjusted_soul_frequency += freq_step
                    
                    # Check if resonance achieved at each step
                    current_difference = abs(adjusted_soul_frequency - target_frequency)
                    if current_difference <= resonance_threshold:
                        logger.info(f"Soul frequency resonance achieved at {adjusted_soul_frequency:.2f}Hz")
                        break
                
                # Final check
                if abs(adjusted_soul_frequency - target_frequency) > resonance_threshold:
                    raise RuntimeError(f"Failed to achieve soul-limbic resonance. "
                                     f"Difference: {abs(adjusted_soul_frequency - target_frequency):.2f}Hz "
                                     f"(threshold: {resonance_threshold}Hz)")
            
            # Create soul attachment record
            soul_attachment_data = {
                'attachment_location': limbic_coords,
                'attachment_seed': target_seed,
                'soul_frequency': soul_frequency,
                'adjusted_frequency': target_frequency,
                'resonance_achieved': True,
                'attachment_time': datetime.now().isoformat(),
                'attachment_strength': 1.0
            }
            
            # Update target seed with soul attachment
            target_seed_data['soul_attachment'] = soul_attachment_data
            target_seed_data['status'] = 'soul_attached'
            
            # Set flags
            setattr(self, FLAG_SOUL_ATTACHED, True)
            self.soul_attached = True
            
            logger.info(f"Soul successfully attached to limbic region via seed {target_seed}")
            
        except Exception as e:
            logger.error(f"Soul attachment failed: {e}")
            # Hard fail as requested
            raise RuntimeError(f"Soul attachment failed - wrong process followed: {e}")

    def _find_limbic_region_center(self) -> Optional[Dict]:
        """Find center coordinates of limbic region."""
        try:
            if not self.brain_structure or 'hemispheres' not in self.brain_structure:
                logger.error("Brain structure not initialized")
                return None
                
            for hemisphere in self.brain_structure['hemispheres'].values():
                if not hemisphere or 'regions' not in hemisphere:
                    continue
                    
                for region_name, region in hemisphere['regions'].items():
                    if region_name == 'limbic' and region and 'sub_regions' in region:
                        for sub_region in region['sub_regions'].values():
                            if not sub_region or 'bounds' not in sub_region:
                                continue
                                
                            bounds = sub_region['bounds']
                            return {
                                'x': (bounds['x_start'] + bounds['x_end']) / 2,
                                'y': (bounds['y_start'] + bounds['y_end']) / 2,
                                'z': (bounds['z_start'] + bounds['z_end']) / 2
                            }
            
            # If no specific limbic region, use approximate location
            return {'x': 128, 'y': 128, 'z': 120}
            
        except Exception as e:
            logger.error(f"Error finding limbic region: {e}")
            return None

    def _get_soul_frequency(self) -> float:
        """Get soul frequency from life cord or soul data (placeholder)."""
        try:
            # This should integrate with life_cord.py or soul data to get actual frequency
            # For now, use a reasonable default
            return 40.0  # Hz - typical soul frequency
        except Exception as e:
            logger.warning(f"Could not get soul frequency: {e}")
            return 40.0  # Default fallback

    def test_quantum_communication(self):
        """
        Test quantum communication between random mycelial seeds.
        Send pulse and measure signal-to-noise ratio to determine communication success.
        """
        try:
            logger.info("Testing quantum communication between mycelial seeds...")
            
            if len(self.active_seeds) < 2:
                logger.warning("Need at least 2 active seeds for communication test")
                return None
            
            # Select random seeds for testing
            test_pairs = []
            num_tests = min(5, len(self.active_seeds) // 2)  # Test up to 5 pairs
            
            for _ in range(num_tests):
                # Pick two random active seeds
                sender = random.choice(self.active_seeds)
                receiver = random.choice([s for s in self.active_seeds if s != sender])
                
                # Check if they're entangled
                if receiver in self.seeds[sender]['entangled_with']:
                    test_pairs.append((sender, receiver))
            
            if not test_pairs:
                logger.warning("No entangled seed pairs available for testing")
                return None
            
            # Perform communication tests
            test_results = []
            
            for sender_id, receiver_id in test_pairs:
                sender = self.seeds[sender_id]
                receiver = self.seeds[receiver_id]
                
                # Communication parameters
                pulse_frequency = sender['base_frequency']
                pulse_power = 1.0  # Normalized power
                pulse_duration = 0.1  # seconds
                
                # Calculate distance effect
                distance = self._calculate_seed_distance(sender_id, receiver_id)
                distance_attenuation = max(0.1, 1.0 - distance/100)
                
                # Simulate quantum communication
                # In quantum entangled system, distance shouldn't matter much
                quantum_efficiency = 0.9  # High efficiency for entangled seeds
                noise_level = random.uniform(0.05, 0.15)  # Background noise
                
                received_power = pulse_power * quantum_efficiency * distance_attenuation
                signal_to_noise = received_power / noise_level if noise_level > 0 else 100
                
                # Determine communication success
                success_threshold = 5.0  # Minimum S/N ratio
                communication_successful = signal_to_noise >= success_threshold
                
                test_result = {
                    'sender': sender_id,
                    'receiver': receiver_id,
                    'distance': distance,
                    'pulse_frequency': pulse_frequency,
                    'sent_power': pulse_power,
                    'received_power': received_power,
                    'noise_level': noise_level,
                    'signal_to_noise_ratio': signal_to_noise,
                    'success': communication_successful,
                    'test_time': datetime.now().isoformat()
                }
                
                test_results.append(test_result)
                
                # Update entanglement communication count
                for entanglement in self.entanglements.values():
                    if ((entanglement['seed1'] == sender_id and entanglement['seed2'] == receiver_id) or
                        (entanglement['seed1'] == receiver_id and entanglement['seed2'] == sender_id)):
                        entanglement['communication_count'] += 1
                        break
                
                self.metrics['quantum_communications'] += 1
            
            # Log results
            successful_tests = sum(1 for result in test_results if result['success'])
            success_rate = successful_tests / len(test_results) if test_results else 0
            
            logger.info(f"Quantum communication test completed: {successful_tests}/{len(test_results)} "
                       f"successful ({success_rate:.1%} success rate)")
            
            for result in test_results:
                logger.debug(f"Test {result['sender'][:8]} → {result['receiver'][:8]}: "
                           f"S/N={result['signal_to_noise_ratio']:.2f}, "
                           f"Success={result['success']}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing quantum communication: {e}")
            self.metrics['errors'].append(f"Quantum communication test failed: {e}")
            return None

    def deactivate_mycelial_seeds(self):
        """
        Monitor energy storage levels and deactivate seeds in non-active sub-regions to conserve energy.
        If insufficient, activate energy creation process and don't reactivate until reserves adequate.
        """
        try:
            logger.info("Monitoring energy levels for mycelial seed management...")
            
            current_energy = self._get_available_energy()
            total_energy_capacity = 100.0  # Should get from energy storage
            energy_percentage = current_energy / total_energy_capacity
            
            if energy_percentage < self.energy_thresholds['critical_low']:
                logger.warning(f"Critical energy level: {energy_percentage:.1%} - emergency seed deactivation")
                self._emergency_seed_deactivation()
                
            elif energy_percentage < self.energy_thresholds['low']:
                logger.info(f"Low energy level: {energy_percentage:.1%} - deactivating non-essential seeds")
                self._selective_seed_deactivation()
                
            elif energy_percentage > self.energy_thresholds['high']:
                logger.info(f"High energy level: {energy_percentage:.1%} - reactivating seeds")
                self._reactivate_seeds()
            
            self.metrics['energy_conservations'] += 1
            
        except Exception as e:
            logger.error(f"Error in energy-based seed management: {e}")
            self.metrics['errors'].append(f"Energy management failed: {e}")

    def _emergency_seed_deactivation(self):
        """Emergency deactivation of all non-essential seeds."""
        try:
            essential_regions = ['limbic', 'brain_stem']  # Critical regions to keep active
            deactivated_count = 0
            
            for seed_id in self.active_seeds.copy():
                seed = self.seeds[seed_id]
                
                # Keep essential regions and soul-attached seeds active
                if (seed['region'] not in essential_regions and 
                    seed['status'] != 'soul_attached'):
                    
                    seed['status'] = 'inactive'
                    self.active_seeds.remove(seed_id)
                    self.inactive_seeds.append(seed_id)
                    deactivated_count += 1
            
            logger.warning(f"Emergency deactivation: {deactivated_count} seeds deactivated")
            
        except Exception as e:
            logger.error(f"Error in emergency deactivation: {e}")

    def _selective_seed_deactivation(self):
        """Selective deactivation based on neural activity levels."""
        try:
            # Get neural activity data if available
            neural_activity = self._get_neural_activity_by_region()
            deactivated_count = 0
            
            for seed_id in self.active_seeds.copy():
                seed = self.seeds[seed_id]
                region_key = f"{seed['hemisphere']}.{seed['region']}.{seed['sub_region']}"
                
                # Check neural activity in this region
                activity_level = neural_activity.get(region_key, 0.5)  # Default medium activity
                
                # Deactivate seeds in low-activity regions
                if (activity_level < 0.3 and 
                    seed['status'] not in ['soul_attached', 'liminal_active']):
                    
                    seed['status'] = 'inactive'
                    self.active_seeds.remove(seed_id)
                    self.inactive_seeds.append(seed_id)
                    deactivated_count += 1
            
            logger.info(f"Selective deactivation: {deactivated_count} seeds in low-activity regions")
            
        except Exception as e:
            logger.error(f"Error in selective deactivation: {e}")

    def _reactivate_seeds(self):
        """Reactivate seeds when energy levels are sufficient."""
        try:
            reactivated_count = 0
            
            # Reactivate some inactive seeds if we have energy
            for seed_id in self.inactive_seeds.copy()[:5]:  # Limit reactivation
                seed = self.seeds[seed_id]
                activation_energy = seed['energy_consumption']
                
                if self._get_available_energy() >= activation_energy:
                    seed['status'] = 'active'
                    self.inactive_seeds.remove(seed_id)
                    self.active_seeds.append(seed_id)
                    self._consume_energy(activation_energy)
                    reactivated_count += 1
            
            if reactivated_count > 0:
                logger.info(f"Reactivated {reactivated_count} mycelial seeds")
            
        except Exception as e:
            logger.error(f"Error reactivating seeds: {e}")

    def _get_neural_activity_by_region(self) -> Dict[str, float]:
        """Get neural activity levels by region (placeholder)."""
        try:
            # This should integrate with neural_network.py to get actual activity data
            # For now, simulate activity levels
            activity = {}
            
            if self.neural_network and 'nodes' in self.neural_network:
                # Calculate activity based on active neural nodes
                for node_id, node in self.neural_network['nodes'].items():
                    if 'coordinates' in node:
                        # Determine region from coordinates (simplified)
                        coords = node['coordinates']
                        region_key = f"region_{int(coords['x']//50)}_{int(coords['y']//50)}"
                        
                        if region_key not in activity:
                            activity[region_key] = 0
                        
                        if node.get('status') == 'active':
                            activity[region_key] += 0.1
            
            # Normalize activity levels
            for region in activity:
                activity[region] = min(1.0, activity[region])
            
            return activity
            
        except Exception as e:
            logger.debug(f"Error getting neural activity: {e}")
            return {}  # Return empty dict if error
        
    def sleep_wake_cycle(self):
        """
        Control sleep/wake cycles naturally through drowsiness/alertness and brain wave changes.
        Update brain state and manage subconscious regions separately.
        """
        try:
            current_state = self.brain_state['current_state']
            current_energy = self._get_available_energy()
            energy_percentage = current_energy / 100.0  # Assuming 100 is max capacity
            
            if current_state == 'awake':
                # Check if sleep is needed (low energy or stress relief)
                if (energy_percentage < self.energy_thresholds['low'] or
                    getattr(self, 'stress_relief_needed', False)):
                    
                    logger.info("Initiating sleep cycle...")
                    self._activate_sleep_cycle()
                    
            elif current_state == 'sleeping':
                # Check if wake is needed (high energy or external stimuli)
                if energy_percentage > self.energy_thresholds['optimal']:
                    logger.info("Initiating wake cycle...")
                    self._activate_wake_cycle()
            
            self.sleep_wake_cycle_active = True
            
        except Exception as e:
            logger.error(f"Error in sleep/wake cycle management: {e}")
            self.metrics['errors'].append(f"Sleep/wake cycle failed: {e}")

    def _activate_sleep_cycle(self):
        """Activate sleep cycle by inducing drowsiness and switching to delta waves."""
        try:
            # Create drowsiness by adjusting mycelial seed frequencies
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                if seed['region'] not in ['brain_stem', 'medulla']:  # Keep vital functions awake
                    # Lower frequency for drowsiness
                    seed['sleep_frequency'] = seed['base_frequency'] * 0.3
            
            # Switch brain waves to delta (0.5-4 Hz for deep sleep)
            delta_frequency = random.uniform(0.5, 4.0)
            
            # Update brain state
            self.brain_state['current_state'] = 'sleeping'
            self.brain_state['current_state_frequency'] = delta_frequency
            self.brain_state['wave_pattern'] = 'delta'
            
            # Keep subconscious regions (autonomic functions) awake
            subconscious_regions = ['brain_stem', 'medulla', 'pons']
            awake_subconscious_seeds = []
            
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                if seed['region'] in subconscious_regions:
                    awake_subconscious_seeds.append(seed_id)
                    # Keep these at normal frequency
                    seed['sleep_frequency'] = seed['base_frequency']
            
            # Set sleep flag
            setattr(self, FLAG_SLEEPING, True)
            self.metrics['sleep_cycles'] += 1
            
            # Energy healing during sleep
            healing_energy = 2.0  # Small energy boost from sleep
            self._add_energy(healing_energy)
            
            logger.info(f"Sleep cycle activated: Delta waves at {delta_frequency:.2f}Hz, "
                       f"{len(awake_subconscious_seeds)} subconscious seeds remain active")
            
        except Exception as e:
            logger.error(f"Error activating sleep cycle: {e}")

    def _activate_wake_cycle(self):
        """Activate wake cycle through alertness and alpha waves."""
        try:
            # Create alertness by switching to alpha waves temporarily
            alpha_frequency = random.uniform(8.0, 12.0)  # Alpha range
            
            # Update brain state to alpha temporarily
            self.brain_state['current_state'] = 'waking'
            self.brain_state['current_state_frequency'] = alpha_frequency
            self.brain_state['wave_pattern'] = 'alpha'
            
            # Put subconscious regions to brief sleep during alpha state
            subconscious_regions = ['brain_stem', 'medulla', 'pons']
            sleeping_subconscious = []
            
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                if seed['region'] in subconscious_regions:
                    sleeping_subconscious.append(seed_id)
                    # Brief sleep for subconscious during alpha
                    seed['alpha_sleep_frequency'] = seed['base_frequency'] * 0.5
            
            # Heighten alpha state with energy boost
            alpha_energy_boost = 1.5
            self._add_energy(alpha_energy_boost)
            
            # After 5 minutes (simulated), complete wake cycle
            self._complete_wake_cycle()
            
            logger.info(f"Wake cycle activated: Alpha waves at {alpha_frequency:.2f}Hz, "
                       f"{len(sleeping_subconscious)} subconscious seeds briefly sleeping")
            
        except Exception as e:
            logger.error(f"Error activating wake cycle: {e}")

    def _complete_wake_cycle(self):
        """Complete wake cycle by returning to natural state."""
        try:
            # Return all seeds to natural frequencies
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                # Remove sleep/alpha frequencies, return to base
                if 'sleep_frequency' in seed:
                    del seed['sleep_frequency']
                if 'alpha_sleep_frequency' in seed:
                    del seed['alpha_sleep_frequency']
            
            # Update brain state to awake with natural frequency
            natural_frequency = random.uniform(8.0, 30.0)  # Normal awake range
            self.brain_state['current_state'] = 'awake'
            self.brain_state['current_state_frequency'] = natural_frequency
            self.brain_state['wave_pattern'] = 'beta'  # Normal awake state
            
            # Clear sleep flags
            setattr(self, FLAG_AWAKE, True)
            if hasattr(self, FLAG_SLEEPING):
                delattr(self, FLAG_SLEEPING)
            
            self.metrics['wake_cycles'] += 1
            
            logger.info(f"Wake cycle completed: Beta waves at {natural_frequency:.2f}Hz")
            
        except Exception as e:
            logger.error(f"Error completing wake cycle: {e}")

    def _add_energy(self, amount: float):
        """Add energy to storage (placeholder)."""
        try:
            if hasattr(self, 'energy_storage') and self.energy_storage is not None:
                current = getattr(self.energy_storage, 'current_energy', 0)
                if hasattr(self.energy_storage, 'update_current_energy'):
                    self.energy_storage.update_current_energy(current + amount)
                else:
                    setattr(self.energy_storage, 'current_energy', current + amount)
            else:
                logger.debug(f"Energy addition simulated: +{amount:.2f} SEU")
        except Exception as e:
            logger.debug(f"Error adding energy: {e}")

    def activate_liminal_state_miscarriage(self):
        """
        Activate liminal state for miscarriage - return soul fragment to liminal space.
        Extract all energy from networks and return with soul. Triggered by MISCARRY flag.
        """
        try:
            logger.warning("Activating liminal state for miscarriage - returning soul to liminal space...")
            
            # Extract all energy from mycelial network
            total_extracted_energy = 0
            for seed_id in self.active_seeds:
                seed = self.seeds[seed_id]
                energy_to_extract = seed['energy_consumption'] * seed['activation_count']
                total_extracted_energy += energy_to_extract
                seed['status'] = 'energy_extracted'
            
            # Extract energy from neural network (if available)
            neural_energy_extracted = 0
            if self.neural_network and 'nodes' in self.neural_network:
                for node in self.neural_network['nodes'].values():
                    if node.get('status') == 'active':
                        neural_energy_extracted += 0.5  # Estimated neural energy
            
            total_extracted_energy += neural_energy_extracted
            
            # Collect soul aspects and memories for return
            soul_package = {
                'extracted_energy': total_extracted_energy,
                'sephiroth_aspects': [],  # Should collect from memory fragments
                'identity_aspects': [],   # Should collect from memory fragments
                'emotional_memories': [], # Should collect from memory fragments
                'extraction_time': datetime.now().isoformat(),
                'reason': 'miscarriage'
            }
            
            # Prepare for soul return through liminal state
            if self.liminal_state_active:
                liminal_seed = None
                for seed_id, seed in self.seeds.items():
                    if seed.get('status') == 'liminal_active':
                        liminal_seed = seed_id
                        break
                
                if liminal_seed:
                    # Use liminal seed to return soul
                    logger.info(f"Returning soul through liminal seed {liminal_seed}")
                    soul_package['return_method'] = 'liminal_seed'
                    soul_package['liminal_seed'] = liminal_seed
            
            # Log the miscarriage event
            logger.warning(f"Miscarriage process complete. Soul returned with {total_extracted_energy:.2f} SEU energy")
            logger.warning("Simulation termination initiated...")
            
            # Set termination flag
            setattr(self, FLAG_SIMULATION_TERMINATED, True)
            
            return soul_package
            
        except Exception as e:
            logger.error(f"Error in miscarriage process: {e}")
            raise RuntimeError(f"Miscarriage process failed: {e}")

    def monitor_states(self):
        """
        Monitor all states with flags and trigger appropriate functions.
        Central monitoring system for the mycelial network.
        """
        try:
            # Check each flag and trigger corresponding functions
            
            # WOMB_CREATED - Trigger brain seed creation (handled elsewhere)
            if getattr(self, FLAG_WOMB_CREATED, False):
                logger.debug("Womb created flag detected")
            
            # BRAIN_STRUCTURE_CREATED - Trigger neural network creation
            if getattr(self, FLAG_BRAIN_STRUCTURE_CREATED, False):
                if not getattr(self, FLAG_NEURAL_NETWORK_CREATED, False):
                    logger.debug("Brain structure ready - neural network should be created")
            
            # NEURAL_NETWORK_CREATED - Trigger mycelial network creation
            if getattr(self, FLAG_NEURAL_NETWORK_CREATED, False):
                if not hasattr(self, 'mycelial_network') or not self.mycelial_network:
                    logger.debug("Neural network ready - creating mycelial network")
                    self.create_mycelial_network()
            
# SEEDS_ENTANGLED - Trigger energy store creation (handled by energy_storage.py)
            if getattr(self, FLAG_SEEDS_ENTANGLED, False):
                logger.debug("Seeds entangled - energy store should be created")
            
            # Monitor energy levels and trigger sleep/wake cycles
            current_energy = self._get_available_energy()
            energy_percentage = current_energy / 100.0
            
            if energy_percentage < self.energy_thresholds['low']:
                if self.brain_state['current_state'] != 'sleeping':
                    self.sleep_wake_cycle()
            elif energy_percentage > self.energy_thresholds['high']:
                if self.brain_state['current_state'] == 'sleeping':
                    self.sleep_wake_cycle()
            
            # STRESS_RELIEVED - Trigger sleep for healing
            if getattr(self, FLAG_STRESS_RELIEVED, False):
                self.stress_relief_needed = True
                self.sleep_wake_cycle()
                delattr(self, FLAG_STRESS_RELIEVED)  # Clear flag after handling
            
            # MISCARRY - Trigger miscarriage process
            if getattr(self, FLAG_MISCARRY, False):
                self.activate_liminal_state_miscarriage()
                return  # Exit monitoring after miscarriage
            
            # Monitor energy levels for seed deactivation/activation
            if energy_percentage < self.energy_thresholds['critical_low']:
                self.deactivate_mycelial_seeds()
            elif energy_percentage > self.energy_thresholds['optimal']:
                self._reactivate_seeds()
            
            # Monitor neural activity for energy management
            if self.neural_network and 'nodes' in self.neural_network:
                self._monitor_neural_activity_changes()
            
            # FIELD_DISTURBANCE - Trigger field diagnosis (handled elsewhere)
            if getattr(self, FLAG_FIELD_DISTURBANCE, False):
                logger.debug("Field disturbance detected - diagnosis should be triggered")
            
        except Exception as e:
            logger.error(f"Error in state monitoring: {e}")
            self.metrics['errors'].append(f"State monitoring failed: {e}")

    def _monitor_neural_activity_changes(self):
        """Monitor neural network for activity changes and manage energy accordingly."""
        try:
            if not self.neural_network or 'nodes' not in self.neural_network:
                return
            
            active_nodes = []
            inactive_nodes = []
            
            for node_id, node in self.neural_network['nodes'].items():
                if node.get('status') == 'active':
                    active_nodes.append(node_id)
                else:
                    inactive_nodes.append(node_id)
            
            # Trigger energy management based on node changes
            # This should integrate with energy_storage.py functions:
            # - add_energy_to_node() for newly active nodes
            # - remove_energy_from_node() for newly inactive nodes
            
            logger.debug(f"Neural activity: {len(active_nodes)} active, {len(inactive_nodes)} inactive nodes")
            
        except Exception as e:
            logger.debug(f"Error monitoring neural activity: {e}")

    def save_mycelial_network(self) -> Dict[str, Any]:
        """
        Save complete mycelial network state for integration with other brain components.
        Returns dictionary with all mycelial network data.
        """
        try:
            mycelial_network_data = {
                'mycelial_network_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'seeds': self.seeds,
                'routes': self.routes,
                'entanglements': self.entanglements,
                'active_seeds': self.active_seeds,
                'inactive_seeds': self.inactive_seeds,
                'brain_state': self.brain_state,
                'energy_thresholds': self.energy_thresholds,
                'metrics': self.metrics,
                'flags': {
                    FLAG_SEEDS_ENTANGLED: getattr(self, FLAG_SEEDS_ENTANGLED, False),
                    FLAG_LIMINAL_STATE_ACTIVE: getattr(self, FLAG_LIMINAL_STATE_ACTIVE, False),
                    FLAG_SOUL_ATTACHED: getattr(self, FLAG_SOUL_ATTACHED, False),
                    FLAG_SLEEPING: getattr(self, FLAG_SLEEPING, False),
                    FLAG_AWAKE: getattr(self, FLAG_AWAKE, False)
                },
                'network_ready_for_energy_storage': getattr(self, FLAG_SEEDS_ENTANGLED, False),
                'soul_attachment_ready': self.liminal_state_active
            }
            
            # Store in instance for access by other components
            self.mycelial_network = mycelial_network_data
            
            logger.info(f"Mycelial network saved: {len(self.seeds)} seeds, {len(self.entanglements)} entanglements")
            return mycelial_network_data
            
        except Exception as e:
            logger.error(f"Failed to save mycelial network: {e}")
            raise RuntimeError(f"Mycelial network save failed: {e}")

    def get_mycelial_network_for_integration(self) -> Dict[str, Any]:
        """
        Get mycelial network data for integration with other brain components.
        Used by energy storage and memory distribution systems.
        """
        try:
            if not hasattr(self, 'mycelial_network') or not self.mycelial_network:
                # Save current state if not already saved
                return self.save_mycelial_network()
            
            return self.mycelial_network
            
        except Exception as e:
            logger.error(f"Error getting mycelial network for integration: {e}")
            raise RuntimeError(f"Mycelial network integration data failed: {e}")

    def get_limbic_seeds_for_soul_attachment(self) -> List[str]:
        """Get active mycelial seeds in limbic region for soul attachment."""
        try:
            limbic_seeds = []
            for seed_id, seed in self.seeds.items():
                if (seed['region'] == 'limbic' and 
                    seed['status'] in ['active', 'liminal_active']):
                    limbic_seeds.append(seed_id)
            
            return limbic_seeds
            
        except Exception as e:
            logger.error(f"Error getting limbic seeds: {e}")
            return []

    def mycelial_network_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report for mycelial network."""
        try:
            # Update current metrics
            self.metrics['active_seeds'] = len(self.active_seeds)
            self.metrics['inactive_seeds'] = len(self.inactive_seeds)
            
            # Calculate efficiency metrics
            total_seeds = len(self.seeds)
            activation_rate = len(self.active_seeds) / max(1, total_seeds)
            entanglement_rate = len(self.entanglements) / max(1, len(self.active_seeds))
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'mycelial_network_metrics': {
                    'seeds': {
                        'total': total_seeds,
                        'active': len(self.active_seeds),
                        'inactive': len(self.inactive_seeds),
                        'activation_rate': activation_rate * 100
                    },
                    'quantum_network': {
                        'entanglements': len(self.entanglements),
                        'entanglement_rate': entanglement_rate,
                        'quantum_communications': self.metrics['quantum_communications']
                    },
                    'brain_state': {
                        'current_state': self.brain_state['current_state'],
                        'frequency': self.brain_state['current_state_frequency'],
                        'wave_pattern': self.brain_state['wave_pattern']
                    },
                    'cycles': {
                        'sleep_cycles': self.metrics['sleep_cycles'],
                        'wake_cycles': self.metrics['wake_cycles'],
                        'energy_conservations': self.metrics['energy_conservations']
                    },
                    'system_status': {
                        'liminal_state_active': self.liminal_state_active,
                        'soul_attached': self.soul_attached,
                        'total_errors': len(self.metrics['errors'])
                    }
                }
            }
            
            # Log detailed report
            logger.info("=== MYCELIAL NETWORK METRICS REPORT ===")
            logger.info(f"Seeds: {total_seeds} total ({len(self.active_seeds)} active, "
                       f"{len(self.inactive_seeds)} inactive) - {activation_rate:.1%} activation rate")
            logger.info(f"Quantum Network: {len(self.entanglements)} entanglements, "
                       f"{self.metrics['quantum_communications']} communications")
            logger.info(f"Brain State: {self.brain_state['current_state']} at "
                       f"{self.brain_state['current_state_frequency']:.1f}Hz ({self.brain_state['wave_pattern']})")
            logger.info(f"Cycles: {self.metrics['sleep_cycles']} sleep, {self.metrics['wake_cycles']} wake, "
                       f"{self.metrics['energy_conservations']} energy conservations")
            
            if self.metrics['errors']:
                logger.warning(f"Errors: {len(self.metrics['errors'])}")
                for i, error in enumerate(self.metrics['errors'][-3:]):  # Show last 3 errors
                    logger.warning(f"  Error {i+1}: {error}")
            
            logger.info("=== END MYCELIAL NETWORK METRICS ===")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating mycelial network metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'mycelial_network_metrics': {}
            }

# ===== END OF MYCELIAL_NETWORK.PY CLASS =====

# Usage example and test functions
def test_mycelial_network_creation():
    """Test function to verify mycelial network creation works correctly."""
    try:
        mn = MycelialNetwork()
        
        # Test network creation
        mn.create_mycelial_network()
        
        # Test seed activation
        mn.activate_mycelial_seeds()
        
        # Test entanglement
        mn.entangle_mycelial_seeds()
        
        # Test quantum communication
        result = mn.test_quantum_communication()
        print(f"Quantum communication test: {len(result) if result else 0} tests performed")
        
        # Test liminal state
        mn.activate_liminal_state_for_attachment()
        
        # Test soul attachment
        mn.attach_soul()
        
        # Generate metrics report
        metrics = mn.mycelial_network_metrics_report()
        print(f"Mycelial network created successfully with {mn.metrics['total_seeds']} seeds")
        
        return True
        
    except Exception as e:
        print(f"Mycelial network test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if script is executed directly
    test_mycelial_network_creation()






# # --- mycelial_network.py ---
# """
# create the mycelial network and distribute mycelial seeds within each sub region of the brain.
# provide basic functionality for controlling sleep/wake cycle, monitoring state and responding to stimuli
# or changes in neural network activity. does not include any memory functionality or energy processing
# and control. so feedback loops in this are mainly stimuli and sleep/wake cycles. As baby brain doesn't 
# yet have any real memory based on experience or any learned processes we can only look at basic mycelial
# network functionality. sleep/wake cycles, monitoring state and responding to stimuli. we have included the
# triggers for some of the processes within neural network and energy storage files so we should only have 1 
# big function in here for monitoring all the states so the triggers can be triggered. consider if this is not 
# needed and is better to keep as part of the trigger.
# """

# import logging
# import uuid
# import random
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# from datetime import datetime
# import math

# from constants.constants import *
# from memory_definitions import *

# # --- Logging Setup ---
# logger = logging.getLogger("Conception")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# class MycelialNetwork:
#     def __init__(self):
#         self.mycelial_network = {}
        
#     def create_mycelial_network(self):
#         """
#         initialise the mycelial network. create an optimal placement between sub-regions placing
#         more mycelial seeds for larger areas to ensure optimal connectivity. check that the 
#         x,y,z coordinates are within the bounds of the sub region and are not already occupied 
#         by other mycelial seeds or neural network nodes. ensure that the mycelial network is not 
#         too dense in any one area to avoid overcrowding and energy inefficiencies. ensure that the 
#         energy utilised is sufficient to support the mycelial network and does not exceed the 
#         energy storage capacity. If energy is insufficient reduce the 
#         number of mycelial seeds until energy is sufficient. do not create the paths between mycelial 
#         seeds as that will be done as a separate process.triggered by flag BRAIN_STRUCTURE_CREATED.
#         """

#     def create_mycelial_routes(self):
#         """
#         create the most optimal routes between mycelial seeds.this functions in a similar way to synapses
#         between nodes. the difference is the routes for mycelial seeds are not activated as frequently as
#         the synapses between nodes. the routes are only deactivated due to mycelial seed decay. why is this
#         different to neural network synapses? this is because the mycelial network does not use routes
#         for communication between mycelial seeds. the routes are only used to return energy to storage (
#         although we dont physically simulate that right now - we might at a later stage need to add some
#         timing based on route return to energy storage to add processing delays if needed).
#         """

#     def activate_mycelial_seeds(self):
#         """
#         activate the mycelial seeds. this will activate the mycelial seeds and allow them to communicate 
#         with each other using quantum entanglement.update the mycelial network dictionary or numpy array
#         with the changes to state and update the energy storage dictionary with the energy used to activate 
#         the seeds.thats a basic subtraction process based on number of seeds created x energy utilised per seed
#         and that is subtracted from current energy storage.double check that you are using consistent names 
#         for dictionaries and variables and are not duplicating existing ones. 
#         """

#     def entangle_mycelial_seeds(self):
#         """
#         entangle all of the mycelial seeds together. this will create a network of mycelial threads
#         that can be used to transport energy and information between mycelial seeds in a quantum manner.
#         we would need to decide how best to do this - my thought is the base frequency is going to be the
#         same for each seed so we dont need to look for resonance between seeds. we can create a strategy
#         to entangle one seed at a time based on distance from each seed (might be computationally expensive)
#         or we can entangle all seeds at once using a matrix of all seeds and their distances from each other.
#         or some other method that we havent thought of yet. once all seeds entangled we need to check that
#         the network is stable and does not have any loops or dead ends. if it does we need to re-entangle
#         unentangled nodes until we get a stable network. Save as flag SEEDS_ENTANGLED.
#         """

#     def activate_liminal_state_for_attachment(self):
#         """
#         activate the liminal state by attuning one active mycelial seed to the life cord spiritual anchor. 
#         like the aura this is a resonance/vibration of the natural seed frequency up to the frequency of the spiritual
#         channel. when resonance is achieved this will activate the liminal state and allow the soul to be passed through the
#         life cord that attaches to the brain stem. from the brain stem the soul will be transferred/guided to its new home. 
#         Set flag status to LIMINAL_STATE_ACTIVE.
#         """

#     def attach_soul(self):
#         """
#         guide the soul will to its new home in the limbic region through creating vibration between soul and limbic sub region.
#         set Flag to SOUL_ATTACHED when resonance achieved. If resonance not achieved must hard fail because then the wrong process
#         was followed. remember this must work like the aura we vibrate soul up or down to match the frequency of the target area.
#         """

#     def test_quantum_communication(self):
#         """
#         test the quantum communication between mycelial seeds. this will involve sending a signal from one seed to another
#         and measuring the signal at the receiving seed. the signal will be a pulse of energy that will be sent at a certain
#         frequency. the frequency will be the same for all seeds. the pulse will be sent at a certain power level. the power
#         level will be the same for all seeds. the pulse will be sent for a certain amount of time. the time will be the same
#         for all seeds. the pulse will be received at the receiving seed and the power level will be measured. the power level
#         will be compared to the power level of the pulse that was sent. the difference in power level will be used to calculate
#         the signal to noise ratio. the signal to noise ratio will be used to determine if the communication was successful.
#         This was a prepopulated function that was added as an inline recommendation so please check if its viable.
#         If not create a different test. I would not test all seeds i would pick a few seeds at random and test the communication
#         between them.
#         """

#     def deactivate_mycelial_seeds(self):
#         """
#         monitor the energy storage values if it drops below a certain threshold, deactivate mycelial seeds in
#         non-active sub-regions to conserve energy. a non-active sub-region is a sub-region where there is a low
#         number of nodes and synapses active or none active. if this does not help bring energy to a certain threshold
#         then activate an energy creation process to increase energy levels and dont reactivate the deactivated nodes
#         until there is enough energy plus reserve energy to sustain the brain.
#         """


#     def sleep_wake_cycle(self):
#         """
#         can trigger from monitor_states due to stress but is also triggered in timed cycles to ensure the brain is getting
#         enough rest. to activate sleep cycle naturally create a bit of drowsiness and then switch each sub region to sleep mode by 
#         changing the brain waves to delta waves. Do not switch all sub regions to sleep mode as some which are more
#         responsible for sub conscious processes should remain awake. Update the flag status to SLEEPING and update the
#         brain state as well including adding a current state frequency to the brain state dictionary. this allows us to have the
#         natural frequency of the brain and the current state frequency so that if we need to calculate field effects for any
#         reasons we would be able to do so on the fly.to activate awake cycle naturally create a bit of alertness by switching all
#         brain waves to alpha temporarily and then switch each sub region to its natural state by updating current state brain wave frequency.
#         During the Alpha state put all the sub consciouse regios that were awake to sleep and wake them up after 5 minutes. Update the 
#         flag status to AWAKE and update the brain state as well including adding a current state frequency to the brain state dictionary. 
#         allow the main sub conscious regions to sleep periodically when the brain is quiet and not doing anything but is still awake. Only
#         allow very short cycles of sleep for those regions. Update energy levels by a small amount as sleep heals then use that energy to
#         heighten the Alpha state momentarily. no values are stored as this is an in and out process.
#         """

#     def activate_liminal_state_miscarriage(self):
#         """
#         activate liminal state and allow soul fragment to be returned to soul in liminal space. the soul will
#         return to the liminal space with all its energy. so we must extract all energy from mycelial network and neural network
#         and return this to the soul. all personal identity aspects, sephiroth aspects and emotional memories will be returned with the soul.
#         then terminate simulation. this is triggered from the def apply_heal_womb(self): when the flag status is set to MISCARRY
#         """

#     def monitor_states(self):
#         """
#         monitor all states that have flags set that trigger a function and ensure this is properly passed to the function. 
#         """	
#         # WOMB_CREATED - Trigger def create_brain_seed 
#         # BRAIN_SEED_PLACED - no not monitored just used for metrics/info in terminal output
#         # BRAIN_SEED_READY - no not monitored just used for metrics/info in terminal output
#         # BRAIN_SEED_SAVED - def trigger_brain_development(self):
#         # BRAIN_STRUCTURE_CREATED - trigger def create_neural_network(self):
#         # BRAIN_STRUCTURE_CREATED - trigger def load_aspects(self): delay by 1 second so that the other function can start
#         # BRAIN_STRUCTURE_CREATED - trigger def monitor_stress(self): 
#         # NEURAL_NETWORK_CREATED - trigger def create_mycelial_network(self):
#         # NEURAL_NETWORK_CREATED - trigger def monitor_stress(self): 
#         # SEEDS_ENTANGLED - trigger def create_energy_store(self):
#         # STORE_CREATED - trigger  
#         # Monitor - energy storage levels and trigger def sleep_wake_cycle(self):if low = sleep, if high = wake. This trigger also activates
#         # after a womb healing to allow model to sleep and heal. the flag status that triggers this is STRESS_RELIEVED.
#         # MISCARRY - def trigger_miscarriage(self): and use the def activate_liminal_state_miscarriage(self): within the trigger function
#         # Monitor - energy storage levels and trigger def deactivate_mycelial_seeds(self):if low = activate seeds, if high = deactivate seeds
#         # Monitor - check for active nodes if node is active trigger add_energy_to_node(self), also track nodes
#         # that go from inactive to active then trigger add_energy_to_node(self)
#         # FIELD_DISTURBANCE - def diagnose_repair_field(self)
#         # Monitor - check for deactivated nodes if node is deactivated trigger def remove_energy_from_node(self)
#         # Monitor - check for when a synapses route becomes active (So moves from DEACTIVATED_ROUTE to ACTIVE_ROUTE) if is active trigger add_energy_to_synaptic_routes(self)
#         # Monitor - check for when a synapses route becomes deactivated (So moves from ACTIVE_ROUTE to DEACTIVATED_ROUTE) if it is deactivated trigger remove_energy_from_synaptic_routes(self)
        
