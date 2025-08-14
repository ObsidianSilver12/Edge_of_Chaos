"""
Simplified Energy Storage System - Creation Functions Only
Receives energy from Creator, stores in deep limbic region (anterior cingulate),
provides basic placement function for initial node activation.
All tracking moved to energy_system.py in stage_3.
"""
from shared.constants.constants import *
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import uuid
import random
import math

# --- Logging Setup ---
logger = logging.getLogger("EnergyStorage")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EnergyStorage:
    """
    Simplified Energy Storage System - Creation Functions Only
    
    Responsibilities:
    - Receive energy from Creator (more than 2 weeks base requirement)
    - Create energy store in deep limbic region (anterior cingulate)
    - Initial energy placement for node activation during brain formation
    - NO tracking, NO transfers, NO returns (moved to energy_system.py)
    """
    
    def __init__(self, brain_structure: Dict[str, Any] = None):
        """Initialize energy storage with brain structure reference."""
        self.storage_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        
        # Energy storage data (creation only)
        self.energy_store = {}
        self.deep_limbic_location = None
        self.available_energy_seu = 0.0
        self.total_capacity_seu = 0.0
        
        # Basic creation history only
        self.creation_events = []
        
        logger.info(f"Energy storage system initialized: {self.storage_id[:8]}")
    
    def receive_energy_from_creator(self, base_energy_multiplier: float = 1.2) -> Dict[str, Any]:
        """
        Receive initial energy from Creator - more than 2 weeks base requirement.
        
        Args:
            base_energy_multiplier: Multiplier for base energy (default 1.2 = 20% more)
        
        Returns:
            Dict with energy received details
        """
        logger.info("🌟 Receiving energy from Creator...")
        
        try:
            # Calculate base 2-week energy requirement using synapse energy
            base_energy_joules = SYNAPSE_ENERGY_JOULES * SYNAPSES_COUNT_FOR_MIN_ENERGY * SECONDS_IN_14_DAYS
            
            # Convert to SEU (Soul Energy Units)
            base_energy_seu = base_energy_joules * SEU_PER_JOULE
            
            # Creator provides more than base requirement with random variance
            variance_factor = random.uniform(base_energy_multiplier, base_energy_multiplier + 0.3)
            received_energy_seu = base_energy_seu * variance_factor
            
            # Store received energy data
            energy_receipt = {
                'receipt_id': str(uuid.uuid4()),
                'received_time': datetime.now().isoformat(),
                'base_requirement_joules': base_energy_joules,
                'base_requirement_seu': base_energy_seu,
                'variance_factor': variance_factor,
                'received_energy_seu': received_energy_seu,
                'energy_source': 'creator',
                'multiplier_used': base_energy_multiplier,
                'excess_energy_seu': received_energy_seu - base_energy_seu
            }
            
            # Update available energy
            self.available_energy_seu = received_energy_seu
            self.total_capacity_seu = received_energy_seu * 1.1  # 10% overhead for storage
            
            # Log creation event
            self.creation_events.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_received_from_creator',
                'amount_seu': received_energy_seu,
                'source': 'creator'
            })
            
            logger.info(f"✅ Energy received from Creator: {received_energy_seu:.1f} SEU")
            logger.info(f"   Base requirement: {base_energy_seu:.1f} SEU")
            logger.info(f"   Excess energy: {energy_receipt['excess_energy_seu']:.1f} SEU")
            
            return energy_receipt
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to receive energy from Creator: {e}")
            raise RuntimeError(f"Energy reception from Creator failed: {e}") from e
    
    def create_energy_store_in_limbic(self) -> Dict[str, Any]:
        """
        Create energy store SPECIFICALLY in anterior cingulate - NO FALLBACKS.
        This is critical for system function.
        """
        logger.info("🧠 Creating energy store in anterior cingulate (STRICT)...")
        
        try:
            if not self.brain_structure:
                raise RuntimeError("Brain structure not provided - cannot locate limbic region")
            
            # Locate limbic system
            limbic_system = self.brain_structure['regions']['limbic_system']
            
            # MUST be anterior_cingulate specifically - no fallbacks
            if 'anterior_cingulate' not in limbic_system['sub_regions']:
                raise RuntimeError("anterior_cingulate sub-region not found in limbic system - HARD FAIL")
            
            anterior_cingulate = limbic_system['sub_regions']['anterior_cingulate']
            
            # Check that anterior_cingulate has nodes
            if len(anterior_cingulate.get('nodes', {})) == 0:
                raise RuntimeError("anterior_cingulate has no nodes - HARD FAIL")
            
            # Get anterior_cingulate bounds
            ac_bounds = anterior_cingulate['boundaries']
            
            # Calculate center of anterior_cingulate
            center_x = (ac_bounds['x_start'] + ac_bounds['x_end']) // 2
            center_y = (ac_bounds['y_start'] + ac_bounds['y_end']) // 2
            center_z = (ac_bounds['z_start'] + ac_bounds['z_end']) // 2
            
            self.deep_limbic_location = (center_x, center_y, center_z)
            
            # Find node closest to center in anterior_cingulate ONLY
            energy_node = None
            selected_node_id = None
            min_distance = float('inf')
            
            # Search through anterior_cingulate nodes only
            for node_id, node_data in anterior_cingulate['nodes'].items():
                node_coords = node_data['coordinates']
                distance = math.sqrt(
                    (node_coords[0] - center_x)**2 + 
                    (node_coords[1] - center_y)**2 + 
                    (node_coords[2] - center_z)**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    energy_node = node_data
                    selected_node_id = node_id
            
            if not energy_node or not selected_node_id:
                raise RuntimeError("No suitable node found in anterior_cingulate - HARD FAIL")
            
            # Create the energy store
            self.energy_store = {
                'store_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'location': {
                    'region': 'limbic_system',
                    'sub_region': 'anterior_cingulate',  # MUST be anterior_cingulate
                    'node_id': selected_node_id,
                    'coordinates': self.deep_limbic_location,
                    'node_coordinates': energy_node['coordinates'],
                    'distance_from_center': min_distance
                },
                'capacity_seu': self.total_capacity_seu,
                'stored_energy_seu': self.available_energy_seu,
                'storage_type': 'quantum_crystalline',
                'efficiency': 1.0,  # Perfect efficiency - no energy loss
                'status': 'active',
                'occupies_node': True
            }
            
            # Mark the node as energy storage
            energy_node['energy_storage'] = True
            energy_node['storage_id'] = self.energy_store['store_id']
            energy_node['energy_density'] = 'maximum'
            
            # Log creation event
            self.creation_events.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_store_created',
                'location': self.deep_limbic_location,
                'sub_region': 'anterior_cingulate',
                'node_id': selected_node_id,
                'capacity_seu': self.total_capacity_seu
            })
            
            logger.info("✅ Energy store created in anterior_cingulate (STRICT)")
            logger.info(f"   Location: {self.deep_limbic_location}")
            logger.info(f"   Node ID: {selected_node_id}")
            logger.info(f"   Capacity: {self.total_capacity_seu:.1f} SEU")
            
            return self.energy_store
            
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error(f"HARD FAIL - Energy store creation in anterior_cingulate failed: {e}")
            raise RuntimeError(f"Energy store creation failed: {e}") from e
    
    def activate_initial_nodes(self, nodes_to_activate: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Activate initial nodes during brain formation by placing energy.
        Updates node status in brain structure but does NOT track ongoing usage.
        
        Args:
            nodes_to_activate: List of node data with coordinates and energy requirements
            
        Returns:
            Activation results summary
        """
        logger.info(f"🔋 Activating {len(nodes_to_activate)} initial nodes...")
        
        try:
            # Calculate energy per node (synapse firing x 10)
            energy_per_node = SYNAPSE_ENERGY_JOULES * 10 * SEU_PER_JOULE
            
            activation_results = {
                'activation_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'total_nodes': len(nodes_to_activate),
                'successful_activations': 0,
                'failed_activations': 0,
                'total_energy_used': 0.0,
                'activated_nodes': []
            }
            
            for node_data in nodes_to_activate:
                try:
                    node_id = node_data['node_id']
                    coordinates = node_data['coordinates']
                    
                    # Check if enough energy available
                    if self.available_energy_seu < energy_per_node:
                        logger.warning(f"Insufficient energy for node {node_id}")
                        activation_results['failed_activations'] += 1
                        continue
                    
                    # Place energy and activate node
                    self.available_energy_seu -= energy_per_node
                    
                    # Update node status in brain structure (not tracked here)
                    node_data['status'] = 'active'
                    node_data['energy_level'] = energy_per_node
                    node_data['activation_time'] = datetime.now().isoformat()
                    
                    activation_results['successful_activations'] += 1
                    activation_results['total_energy_used'] += energy_per_node
                    activation_results['activated_nodes'].append({
                        'node_id': node_id,
                        'coordinates': coordinates,
                        'energy_placed': energy_per_node
                    })
                    
                    logger.debug(f"Node {node_id} activated with {energy_per_node:.2f} SEU")
                    
                except Exception as e:
                    logger.error(f"Failed to activate node {node_data.get('node_id', 'unknown')}: {e}")
                    activation_results['failed_activations'] += 1
            
            # Log activation event
            self.creation_events.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'initial_nodes_activated',
                'total_nodes': activation_results['total_nodes'],
                'successful': activation_results['successful_activations'],
                'energy_used': activation_results['total_energy_used']
            })
            
            logger.info(f"✅ Node activation complete: {activation_results['successful_activations']}/{activation_results['total_nodes']} successful")
            logger.info(f"   Energy used: {activation_results['total_energy_used']:.1f} SEU")
            logger.info(f"   Remaining energy: {self.available_energy_seu:.1f} SEU")
            
            return activation_results
            
        except Exception as e:
            logger.error(f"Failed to activate initial nodes: {e}")
            raise RuntimeError(f"Node activation failed: {e}") from e
    
    def get_available_energy(self) -> float:
        """Get current available energy amount."""
        return self.available_energy_seu
    
    def get_energy_storage_status(self) -> Dict[str, Any]:
        """Get basic status of energy storage system."""
        return {
            'storage_id': self.storage_id,
            'creation_time': self.creation_time,
            'energy_store': self.energy_store.copy() if self.energy_store else {},
            'deep_limbic_location': self.deep_limbic_location,
            'available_energy_seu': self.available_energy_seu,
            'total_capacity_seu': self.total_capacity_seu,
            'utilization_percentage': (self.available_energy_seu / max(1, self.total_capacity_seu)) * 100,
            'creation_events_count': len(self.creation_events),
            'status': 'active' if self.available_energy_seu > 0 else 'empty'
        }


# ===== UTILITY FUNCTIONS =====

def create_energy_storage_with_brain(brain_structure: Dict[str, Any]) -> EnergyStorage:
    """
    Create energy storage system with brain structure.
    
    Args:
        brain_structure: Complete brain structure from AnatomicalBrain
    
    Returns:
        Configured EnergyStorage instance
    """
    try:
        energy_system = EnergyStorage(brain_structure)
        
        # Initialize with Creator energy
        energy_system.receive_energy_from_creator()
        
        # Create storage in limbic region
        energy_system.create_energy_store_in_limbic()
        
        logger.info("✅ Energy storage system created and initialized with brain structure")
        return energy_system
        
    except (ValueError, KeyError, RuntimeError) as e:
        logger.error(f"Failed to create energy storage with brain: {e}")
        raise RuntimeError(f"Energy storage creation failed: {e}") from e

def test_energy_storage_system():
    """Test the simplified energy storage system."""
    print("\n" + "="*50)
    print("⚡ TESTING SIMPLIFIED ENERGY STORAGE SYSTEM")
    print("="*50)
    
    try:
        # Mock brain structure for testing
        mock_brain = {
            'regions': {
                'limbic_system': {
                    'sub_regions': {
                        'anterior_cingulate': {
                            'boundaries': {
                                'x_start': 120, 'x_end': 140,
                                'y_start': 120, 'y_end': 140,
                                'z_start': 120, 'z_end': 140
                            },
                            'blocks': {
                                'limbic_system-anterior_cingulate-001': {
                                    'block_id': 'limbic_system-anterior_cingulate-001',
                                    'center': (130, 130, 130),
                                    'boundaries': {
                                        'x_start': 125, 'x_end': 135,
                                        'y_start': 125, 'y_end': 135,
                                        'z_start': 125, 'z_end': 135
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Test energy storage creation
        print("1. Creating energy storage system...")
        energy_system = create_energy_storage_with_brain(mock_brain)
        print(f"   ✅ System created: {energy_system.storage_id[:8]}")
        
        # Test node activation
        print("\n2. Testing initial node activation...")
        test_nodes = [
            {'node_id': 'test_node_1', 'coordinates': (100, 100, 100)},
            {'node_id': 'test_node_2', 'coordinates': (150, 150, 150)}
        ]
        
        results = energy_system.activate_initial_nodes(test_nodes)
        print(f"   ✅ Activated: {results['successful_activations']}/{results['total_nodes']} nodes")
        
        # Get final status
        status = energy_system.get_energy_storage_status()
        print("\n" + "="*40)
        print("📊 FINAL STATUS")
        print("="*40)
        print(f"Available energy: {status['available_energy_seu']:.1f} SEU")
        print(f"Utilization: {status['utilization_percentage']:.1f}%")
        print(f"Creation events: {status['creation_events_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_energy_storage_system()















# # --- energy_storage.py V8 COMPLETE ---
# """
# Creates energy storage system in the limbic system for the brain after mycelial network is formed.
# Triggered by SEEDS_ENTANGLED flag. Manages energy distribution to nodes and synapses with 
# field disturbance detection and repair. Calculates 2-week energy requirement with 5-10% variance.
# """

# import logging
# import uuid
# import random
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# from datetime import datetime
# import math
# from shared.constants.constants import *

# # --- Logging Setup ---
# logger = logging.getLogger("Conception")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# # Energy Storage System Class
# class EnergyStorage:
#     def __init__(self):
#         """Initialize the energy storage system."""
#         self.energy_storage = {}
#         self.energy_amount = 0  # Initial value set to 0
#         self.active_nodes_energy = {}  # Track energy in active nodes
#         self.active_routes_energy = {}  # Track energy in active routes
#         self.field_disturbances = {}  # Track field disturbances by region
#         self.energy_history = []  # Track energy changes over time
#         self.metrics = {
#             'total_energy_stored': 0.0,
#             'total_energy_distributed': 0.0,
#             'total_energy_returned': 0.0,
#             'field_repairs': 0,
#             'energy_depletion_events': 0,
#             'creation_time': None
#         }
        
#         # Energy distribution thresholds
#         self.energy_thresholds = {
#             'critical_low': ENERGY_THRESHOLD_CRITICAL_LOW,
#             'low': ENERGY_THRESHOLD_LOW,
#             'optimal': ENERGY_THRESHOLD_OPTIMAL,
#             'high': ENERGY_THRESHOLD_HIGH
#         }
        
#     def create_energy_store(self):
#         """
#         Create the energy store within the deep limbic sub region adjacent to the limbic system.
#         Triggered by flag SEEDS_ENTANGLED. Calculate 2-week energy with 5-10% random variance.
#         Save flag as STORE_CREATED.
#         """
#         try:
#             logger.info("Creating energy storage system in limbic region...")
            
#             # Calculate base 2-week energy requirement
#             # Based on typical brain energy consumption: ~20 watts continuous
#             base_energy_joules = 20.0 * 14 * 24 * 60 * 60  # 20W for 14 days in joules
            
#             # Convert to SEU (Soul Energy Units)
#             base_energy_seu = base_energy_joules * SEU_PER_JOULE
            
#             # Add random variance (5-10%)
#             variance_factor = random.uniform(1.05, 1.10)  # 5-10% increase
#             total_energy_seu = base_energy_seu * variance_factor
            
#             # Store energy in dictionary format
#             self.energy_storage = {
#                 'energy_store_id': str(uuid.uuid4()),
#                 'creation_time': datetime.now().isoformat(),
#                 'location': 'deep_limbic_sub_region',
#                 'energy_type': 'initial_creator_energy',
#                 'energy_amount_seu': float(total_energy_seu),
#                 'base_requirement_seu': float(base_energy_seu),
#                 'variance_factor': float(variance_factor),
#                 'energy_distributed_seu': 0.0,
#                 'energy_returned_seu': 0.0,
#                 'energy_available_seu': float(total_energy_seu),
#                 'capacity_seu': float(total_energy_seu * 1.2),  # 20% overhead
#                 'status': 'active',
#                 'storage_efficiency': 1.0  # Perfect efficiency - no energy loss
#             }
            
#             # Initialize tracking arrays
#             self.energy_amount = total_energy_seu
#             self.energy_history.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'event': 'store_created',
#                 'energy_level': total_energy_seu,
#                 'action': 'initial_creation'
#             })
            
#             # Update metrics
#             self.metrics['total_energy_stored'] = total_energy_seu
#             self.metrics['creation_time'] = self.energy_storage['creation_time']
            
#             # Set flag indicating store creation is complete
#             setattr(self, FLAG_STORE_CREATED, True)
            
#             logger.info(f"Energy store created: {total_energy_seu:.1f} SEU "
#                        f"(base: {base_energy_seu:.1f}, variance: {variance_factor:.3f})")
            
#             return self.energy_storage
            
#         except ValueError as e:
#             logger.error(f"Failed to create energy store: {e}")
#             raise RuntimeError(f"Energy store creation failed: {e}") from e

#     def add_energy_to_node(self, node_id: str, node_coordinates: Tuple[float, float, float], 
#                           energy_required: float = 1.0) -> bool:
#         """
#         Add energy to an active node and perform local field calculation.
#         Check for local disturbances around the node coordinates.
#         Set flag to FIELD_DISTURBANCE if disturbance found.
#         """
#         try:
#             # Check if enough energy is available
#             if self.energy_storage['energy_available_seu'] < energy_required:
#                 logger.warning(f"Insufficient energy for node {node_id}: "
#                              f"Required {energy_required}, Available {self.energy_storage['energy_available_seu']}")
#                 return False
            
#             # Calculate local field stability
#             field_stability = self._calculate_local_field_stability(node_coordinates)
            
#             # Add energy to node
#             if node_id not in self.active_nodes_energy:
#                 self.active_nodes_energy[node_id] = {
#                     'coordinates': node_coordinates,
#                     'energy_level': 0.0,
#                     'activation_time': datetime.now().isoformat(),
#                     'field_stability': field_stability
#                 }
            
#             # Transfer energy
#             self.active_nodes_energy[node_id]['energy_level'] += energy_required
#             self.energy_storage['energy_available_seu'] -= energy_required
#             self.energy_storage['energy_distributed_seu'] += energy_required
            
#             # Update field stability
#             self.active_nodes_energy[node_id]['field_stability'] = field_stability
            
#             # Check for field disturbances
#             if field_stability < 0.7:  # Threshold for field disturbance
#                 disturbance_detected = True
#                 setattr(self, FLAG_FIELD_DISTURBANCE, True)
                
#                 # Record disturbance
#                 region_key = self._get_region_key(node_coordinates)
#                 if region_key not in self.field_disturbances:
#                     self.field_disturbances[region_key] = []
                
#                 self.field_disturbances[region_key].append({
#                     'node_id': node_id,
#                     'coordinates': node_coordinates,
#                     'field_stability': field_stability,
#                     'disturbance_time': datetime.now().isoformat(),
#                     'disturbance_type': 'low_field_stability'
#                 })
                
#                 logger.warning(f"Field disturbance detected at node {node_id}: "
#                              f"stability {field_stability:.3f}")
                
#                 # Trigger field repair
#                 self.diagnose_repair_field(region_key)
#             else:
#                 disturbance_detected = False
            
#             # Log energy addition
#             self.energy_history.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'event': 'energy_added_to_node',
#                 'node_id': node_id,
#                 'energy_amount': energy_required,
#                 'remaining_energy': self.energy_storage['energy_available_seu'],
#                 'field_stability': field_stability,
#                 'disturbance_detected': disturbance_detected
#             })
            
#             # Update metrics
#             self.metrics['total_energy_distributed'] += energy_required
            
#             logger.debug(f"Energy added to node {node_id}: {energy_required:.2f} SEU, "
#                         f"field stability: {field_stability:.3f}")
            
#             return True
            
#         except (ValueError, KeyError, RuntimeError) as e:
#             logger.error(f"Failed to add energy to node {node_id}: {e}")
#             return False

#     def diagnose_repair_field(self, region_key: str):
#         """
#         Diagnose and repair field disturbances per sub region.
#         Apply appropriate repair based on disturbance type and severity.
#         """
#         try:
#             logger.info(f"Diagnosing field disturbances in region {region_key}...")
            
#             if region_key not in self.field_disturbances:
#                 logger.warning(f"No disturbances recorded for region {region_key}")
#                 return
            
#             disturbances = self.field_disturbances[region_key]
            
#             # Analyze disturbance patterns
#             avg_stability = np.mean([d['field_stability'] for d in disturbances])
#             # Count disturbances in the region
#             disturbance_count = len(disturbances)            
#             # Count active nodes and synapses in region
#             region_nodes = self._count_active_nodes_in_region(region_key)
#             region_synapses = self._count_active_synapses_in_region(region_key)
            
#             # Log diagnostic information - now using disturbance_count
#             logger.info(f"Region {region_key} analysis: {disturbance_count} disturbances, "
#                        f"avg stability: {avg_stability:.3f}, nodes: {region_nodes}, synapses: {region_synapses}")
            
#             repair_actions = []
            
#             # Determine repair strategy based on disturbance type
#             if avg_stability < 0.3:  # Severe instability
#                 logger.warning(f"Severe field instability in {region_key}: {avg_stability:.3f} "
#                               f"({disturbance_count} disturbances)")
                
#                 if region_nodes > 10 or region_synapses > 20:
#                     # Too many active components - deactivate some
#                     repair_actions.append('deactivate_excess_nodes')
#                     repair_actions.append('deactivate_excess_synapses')
#                 else:
#                     # Insufficient energy - create more energy
#                     repair_actions.append('create_additional_energy')
                    
#             elif avg_stability < 0.5:  # Moderate instability
#                 logger.info(f"Moderate field instability in {region_key}: {avg_stability:.3f} "
#                            f"({disturbance_count} disturbances)")
                
#                 if region_nodes > 5:
#                     # Moderate load - selective deactivation
#                     repair_actions.append('selective_node_deactivation')
#                 else:
#                     # Boost energy temporarily
#                     repair_actions.append('temporary_energy_boost')
                    
#             elif avg_stability < 0.7:  # Minor instability
#                 logger.debug(f"Minor field instability in {region_key}: {avg_stability:.3f} "
#                             f"({disturbance_count} disturbances)")
#                 repair_actions.append('minor_energy_adjustment')
            
#             # Apply repair actions
#             for action in repair_actions:
#                 self._apply_repair_action(region_key, action, disturbances)
            
#             # Clear resolved disturbances
#             resolved_count = len(disturbances)
#             del self.field_disturbances[region_key]
            
#             # Update metrics
#             self.metrics['field_repairs'] += resolved_count
            
#             logger.info(f"Field repair complete for {region_key}: {resolved_count} disturbances resolved, "
#                        f"{len(repair_actions)} repair actions applied")
            
#         except (ValueError, KeyError, RuntimeError) as e:
#             logger.error(f"Failed to repair field in region {region_key}: {e}")

#     def _apply_repair_action(self, region_key: str, action: str, disturbances: List[Dict]):
#         """Apply specific repair action to resolve field disturbances."""
#         try:
#             if action == 'deactivate_excess_nodes':
#                 # Deactivate nodes with lowest field stability
#                 nodes_to_deactivate = sorted(disturbances, key=lambda x: x['field_stability'])[:3]
#                 for disturbance in nodes_to_deactivate:
#                     self.remove_energy_from_node(disturbance['node_id'])
#                 logger.info(f"Deactivated {len(nodes_to_deactivate)} excess nodes in {region_key}")
                
#             elif action == 'selective_node_deactivation':
#                 # Deactivate most problematic node
#                 worst_node = min(disturbances, key=lambda x: x['field_stability'])
#                 self.remove_energy_from_node(worst_node['node_id'])
#                 logger.info(f"Selectively deactivated worst node in {region_key}")
                
#             elif action == 'create_additional_energy':
#                 # Generate emergency energy (limited)
#                 emergency_energy = min(100.0, self.energy_storage['capacity_seu'] * 0.05)
#                 self.energy_storage['energy_available_seu'] += emergency_energy
#                 logger.info(f"Created {emergency_energy:.1f} SEU emergency energy for {region_key}")
                
#             elif action == 'temporary_energy_boost':
#                 # Boost energy temporarily
#                 boost_energy = 50.0
#                 self.energy_storage['energy_available_seu'] += boost_energy
#                 logger.info(f"Applied {boost_energy:.1f} SEU temporary boost to {region_key}")
                
#             elif action == 'minor_energy_adjustment':
#                 # Minor adjustment
#                 adjust_energy = 10.0
#                 self.energy_storage['energy_available_seu'] += adjust_energy
#                 logger.debug(f"Applied {adjust_energy:.1f} SEU minor adjustment to {region_key}")
                
#         except (ValueError, KeyError, RuntimeError) as e:
#             logger.error(f"Failed to apply repair action {action} to {region_key}: {e}")

#     def remove_energy_from_node(self, node_id: str) -> float:
#         """
#         Remove energy from a deactivated node and return it to energy storage.
#         Update the energy storage dictionary/numpy array.
#         """
#         try:
#             if node_id not in self.active_nodes_energy:
#                 logger.warning(f"Node {node_id} not found in active nodes")
#                 return 0.0
            
#             # Get energy from node
#             node_energy = self.active_nodes_energy[node_id]['energy_level']
            
#             # Apply storage efficiency loss
#             recoverable_energy = node_energy * self.energy_storage['storage_efficiency']
#             energy_transformation = 0.0
            
#             # Return energy to storage
#             self.energy_storage['energy_available_seu'] += recoverable_energy
#             self.energy_storage['energy_returned_seu'] += recoverable_energy
            
#             # Remove node from active tracking
#             del self.active_nodes_energy[node_id]
            
#             # Log energy removal with perfect conservation
#             self.energy_history.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'event': 'energy_removed_from_node',
#                 'node_id': node_id,
#                 'energy_returned': recoverable_energy,
#                 'energy_transformed': energy_transformation,
#                 'total_available': self.energy_storage['energy_available_seu'],
#                 'conservation_perfect': True
#             })
            
#             # Update metrics
#             self.metrics['total_energy_returned'] += recoverable_energy
            
#             logger.debug(f"Energy removed from node {node_id}: {recoverable_energy:.2f} SEU returned perfectly, "
#                         f"energy conservation maintained")
            
#             return recoverable_energy
            
#         except (ValueError, KeyError, RuntimeError) as e:
#             logger.error(f"Failed to remove energy from node {node_id}: {e}")
#             return 0.0

#     def add_energy_to_synaptic_routes(self, route_id: str, from_node: str, to_node: str,
#                                      energy_required: float = 0.5) -> bool:
#         """
#         Add energy to a synaptic route when activated.
#         Update energy storage and set ACTIVE_ROUTE flag.
#         """
#         try:
#             # Check energy availability
#             if self.energy_storage['energy_available_seu'] < energy_required:
#                 logger.warning(f"Insufficient energy for route {route_id}: "
#                              f"Required {energy_required}, Available {self.energy_storage['energy_available_seu']}")
#                 return False
            
#             # Add route to active tracking
#             if route_id not in self.active_routes_energy:
#                 self.active_routes_energy[route_id] = {
#                     'from_node': from_node,
#                     'to_node': to_node,
#                     'energy_level': 0.0,
#                     'activation_time': datetime.now().isoformat()
#                 }
            
#             # Transfer energy
#             self.active_routes_energy[route_id]['energy_level'] += energy_required
#             self.energy_storage['energy_available_seu'] -= energy_required
#             self.energy_storage['energy_distributed_seu'] += energy_required
            
#             # Set active route flag
#             setattr(self, 'ACTIVE_ROUTE', True)
            
#             # Log energy addition
#             self.energy_history.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'event': 'energy_added_to_route',
#                 'route_id': route_id,
#                 'from_node': from_node,
#                 'to_node': to_node,
#                 'energy_amount': energy_required,
#                 'remaining_energy': self.energy_storage['energy_available_seu']
#             })
            
#             # Update metrics
#             self.metrics['total_energy_distributed'] += energy_required
            
#             logger.debug(f"Energy added to route {route_id} ({from_node}->{to_node}): {energy_required:.2f} SEU")
            
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to add energy to route {route_id}: {e}")
#             return False

#     def remove_energy_from_synaptic_routes(self, route_id: str) -> float:
#         """
#         Remove energy from a deactivated synaptic route and return to storage.
#         Set DEACTIVATED_ROUTE flag.
#         """
#         try:
#             if route_id not in self.active_routes_energy:
#                 logger.warning(f"Route {route_id} not found in active routes")
#                 return 0.0
            
#             # Get energy from route
#             route_energy = self.active_routes_energy[route_id]['energy_level']
            
#             # Apply storage efficiency loss
#             recoverable_energy = route_energy * self.energy_storage['storage_efficiency']
#             energy_transformation = 0.0
            
#             # Return energy to storage
#             self.energy_storage['energy_available_seu'] += recoverable_energy
#             self.energy_storage['energy_returned_seu'] += recoverable_energy
            
#             # Remove route from active tracking
#             route_info = self.active_routes_energy[route_id]
#             del self.active_routes_energy[route_id]
            
#             # Set deactivated route flag
#             setattr(self, 'DEACTIVATED_ROUTE', True)
            
#             # Log energy removal with perfect conservation
#             self.energy_history.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'event': 'energy_removed_from_route',
#                 'route_id': route_id,
#                 'from_node': route_info['from_node'],
#                 'to_node': route_info['to_node'],
#                 'energy_returned': recoverable_energy,
#                 'energy_transformed': energy_transformation,
#                 'total_available': self.energy_storage['energy_available_seu'],
#                 'conservation_perfect': True
#             })
            
#             # Update metrics
#             self.metrics['total_energy_returned'] += recoverable_energy
            
#             logger.debug(f"Energy removed from route {route_id}: {recoverable_energy:.2f} SEU returned")
            
#             return recoverable_energy
            
#         except Exception as e:
#             logger.error(f"Failed to remove energy from route {route_id}: {e}")
#             return 0.0

#     def monitor_energy_levels(self) -> Dict[str, Any]:
#         """
#         Monitor overall energy levels and trigger actions based on thresholds.
#         Return current energy status and recommendations.
#         """
#         try:
#             current_level = self.energy_storage['energy_available_seu']
#             capacity = self.energy_storage['capacity_seu']
#             percentage = (current_level / capacity) * 100 if capacity > 0 else 0
            
#             status = {
#                 'timestamp': datetime.now().isoformat(),
#                 'energy_level_seu': current_level,
#                 'capacity_seu': capacity,
#                 'percentage': percentage,
#                 'status': 'unknown',
#                 'recommendations': []
#             }
            
#             # Determine status and recommendations
#             if percentage < (self.energy_thresholds['critical_low'] * 100):
#                 status['status'] = 'critical_low'
#                 status['recommendations'].extend([
#                     'emergency_node_deactivation',
#                     'suspend_non_essential_processes',
#                     'activate_energy_conservation_mode'
#                 ])
#                 self.metrics['energy_depletion_events'] += 1
#                 logger.error(f"CRITICAL: Energy level at {percentage:.1f}% - emergency measures required")
                
#             elif percentage < (self.energy_thresholds['low'] * 100):
#                 status['status'] = 'low'
#                 status['recommendations'].extend([
#                     'selective_node_deactivation',
#                     'reduce_synaptic_activity',
#                     'prioritize_essential_functions'
#                 ])
#                 logger.warning(f"LOW: Energy level at {percentage:.1f}% - conservation needed")
                
#             elif percentage < (self.energy_thresholds['optimal'] * 100):
#                 status['status'] = 'moderate'
#                 status['recommendations'].append('monitor_usage_patterns')
#                 logger.info(f"MODERATE: Energy level at {percentage:.1f}% - normal operation")
                
#             else:
#                 status['status'] = 'optimal'
#                 status['recommendations'].append('normal_operation')
#                 logger.debug(f"OPTIMAL: Energy level at {percentage:.1f}% - excellent")
            
#             # Add current usage statistics
#             status['usage_stats'] = {
#                 'active_nodes': len(self.active_nodes_energy),
#                 'active_routes': len(self.active_routes_energy),
#                 'total_distributed': self.energy_storage['energy_distributed_seu'],
#                 'total_returned': self.energy_storage['energy_returned_seu'],
#                 'efficiency': (self.energy_storage['energy_returned_seu'] / 
#                              max(1, self.energy_storage['energy_distributed_seu'])) * 100
#             }
            
#             return status
            
#         except Exception as e:
#             logger.error(f"Failed to monitor energy levels: {e}")
#             return {'status': 'error', 'error': str(e)}

#     def _calculate_local_field_stability(self, coordinates: Tuple[float, float, float]) -> float:
#         """Calculate local field stability around given coordinates."""
#         try:
#             x, y, z = coordinates
            
#             # Calculate distance from brain center (simplified)
#             brain_center = (128, 128, 128)  # Center of 256x256x256 grid
#             distance_from_center = math.sqrt(
#                 (x - brain_center[0])**2 + 
#                 (y - brain_center[1])**2 + 
#                 (z - brain_center[2])**2
#             )
            
#             # Base stability decreases with distance from center
#             distance_factor = max(0.3, 1.0 - (distance_from_center / 200))
            
#             # Count nearby active nodes (field interference)
#             nearby_interference = 0
#             for node_data in self.active_nodes_energy.values():
#                 node_coords = node_data['coordinates']
#                 distance = math.sqrt(
#                     (x - node_coords[0])**2 + 
#                     (y - node_coords[1])**2 + 
#                     (z - node_coords[2])**2
#                 )
#                 if distance < 20:  # Within interference range
#                     nearby_interference += 1
            
#             # Interference penalty
#             interference_factor = max(0.2, 1.0 - (nearby_interference * 0.1))
            
#             # Random field fluctuation
#             random_factor = random.uniform(0.9, 1.1)
            
#             # Calculate final stability
#             stability = distance_factor * interference_factor * random_factor
            
#             return max(0.0, min(1.0, stability))
            
#         except (ValueError, TypeError) as e:
#             logger.error(f"Failed to calculate field stability: {e}")
#             return 0.5  # Default moderate stability

#     def _get_region_key(self, coordinates: Tuple[float, float, float]) -> str:
#         """Get region key for coordinates (simplified spatial hashing)."""
#         x, y, z = coordinates
#         return f"region_{int(x//50)}_{int(y//50)}_{int(z//50)}"

#     def _count_active_nodes_in_region(self, region_key: str) -> int:
#         """Count active nodes in a specific region."""
#         count = 0
#         for node_data in self.active_nodes_energy.values():
#             node_region = self._get_region_key(node_data['coordinates'])
#             if node_region == region_key:
#                 count += 1
#         return count

#     def _count_active_synapses_in_region(self, region_key: str) -> int:
#         """Count active synapses in a specific region (simplified)."""
#         # Simplified: assume 2 synapses per node on average
#         return self._count_active_nodes_in_region(region_key) * 2

#     def get_energy_storage_status(self) -> Dict[str, Any]:
#         """
#         Get complete energy storage status for integration with other brain components.
#         """
#         try:
#             status = {
#                 'energy_storage_id': self.energy_storage.get('energy_store_id'),
#                 'creation_time': self.energy_storage.get('creation_time'),
#                 'current_energy_seu': self.energy_storage.get('energy_available_seu', 0),
#                 'total_capacity_seu': self.energy_storage.get('capacity_seu', 0),
#                 'energy_distributed_seu': self.energy_storage.get('energy_distributed_seu', 0),
#                 'energy_returned_seu': self.energy_storage.get('energy_returned_seu', 0),
#                 'storage_efficiency': self.energy_storage.get('storage_efficiency', 0.95),
#                 'active_nodes_count': len(self.active_nodes_energy),
#                 'active_routes_count': len(self.active_routes_energy),
#                 'field_disturbances_count': sum(len(d) for d in self.field_disturbances.values()),
#                 'metrics': self.metrics.copy(),
#                 'flags': {
#                     FLAG_STORE_CREATED: getattr(self, FLAG_STORE_CREATED, False),
#                     FLAG_FIELD_DISTURBANCE: getattr(self, FLAG_FIELD_DISTURBANCE, False),
#                     'ACTIVE_ROUTE': getattr(self, 'ACTIVE_ROUTE', False),
#                     'DEACTIVATED_ROUTE': getattr(self, 'DEACTIVATED_ROUTE', False)
#                 },
#                 'energy_thresholds': self.energy_thresholds.copy(),
#                 'ready_for_integration': True
#             }
            
#             return status
            
#         except Exception as e:
#             logger.error(f"Failed to get energy storage status: {e}")
#             return {'error': str(e), 'ready_for_integration': False}

#     def generate_energy_report(self) -> Dict[str, Any]:
#         """
#         Generate comprehensive energy storage report for monitoring and debugging.
#         """
#         try:
#             # Current status
#             current_status = self.monitor_energy_levels()
            
#             # Energy history summary
#             history_summary = {
#                 'total_events': len(self.energy_history),
#                 'recent_events': self.energy_history[-10:] if self.energy_history else [],
#                 'event_types': {}
#             }
            
#             # Count event types
#             for event in self.energy_history:
#                 event_type = event.get('event', 'unknown')
#                 history_summary['event_types'][event_type] = history_summary['event_types'].get(event_type, 0) + 1
            
#             # Field disturbance summary
#             disturbance_summary = {
#                 'regions_affected': len(self.field_disturbances),
#                 'total_disturbances': sum(len(d) for d in self.field_disturbances.values()),
#                 'disturbance_types': {}
#             }
            
#             for region_disturbances in self.field_disturbances.values():
#                 for disturbance in region_disturbances:
#                     dtype = disturbance.get('disturbance_type', 'unknown')
#                     disturbance_summary['disturbance_types'][dtype] = disturbance_summary['disturbance_types'].get(dtype, 0) + 1
            
#             # Resource utilization
#             resource_utilization = {
#                 'energy_utilization_percentage': (
#                     self.energy_storage.get('energy_distributed_seu', 0) / 
#                     max(1, self.energy_storage.get('energy_amount_seu', 1))
#                 ) * 100,
#                 'node_density': len(self.active_nodes_energy),
#                 'route_density': len(self.active_routes_energy),
#                 'average_node_energy': np.mean([n['energy_level'] for n in self.active_nodes_energy.values()]) if self.active_nodes_energy else 0,
#                 'average_route_energy': np.mean([r['energy_level'] for r in self.active_routes_energy.values()]) if self.active_routes_energy else 0
#             }
            
#             # Compile comprehensive report
#             report = {
#                 'report_id': str(uuid.uuid4()),
#                 'timestamp': datetime.now().isoformat(),
#                 'energy_storage_system': {
#                     'store_id': self.energy_storage.get('energy_store_id'),
#                     'status': current_status['status'],
#                     'energy_level_seu': current_status['energy_level_seu'],
#                     'capacity_percentage': current_status['percentage'],
#                     'storage_location': self.energy_storage.get('location'),
#                     'creation_time': self.energy_storage.get('creation_time')
#                 },
#                 'current_status': current_status,
#                 'energy_history': history_summary,
#                 'field_disturbances': disturbance_summary,
#                 'resource_utilization': resource_utilization,
#                 'metrics': self.metrics.copy(),
#                 'recommendations': current_status.get('recommendations', []),
#                 'system_health': self._assess_system_health()
#             }
            
#             # Log report summary
#             logger.info("=== ENERGY STORAGE SYSTEM REPORT ===")
#             logger.info(f"Status: {current_status['status'].upper()} - {current_status['percentage']:.1f}% capacity")
#             logger.info(f"Active Nodes: {len(self.active_nodes_energy)}, Active Routes: {len(self.active_routes_energy)}")
#             logger.info(f"Energy Distributed: {self.energy_storage.get('energy_distributed_seu', 0):.1f} SEU")
#             logger.info(f"Energy Returned: {self.energy_storage.get('energy_returned_seu', 0):.1f} SEU")
#             logger.info(f"Field Repairs: {self.metrics['field_repairs']}, Depletion Events: {self.metrics['energy_depletion_events']}")
            
#             if current_status.get('recommendations'):
#                 logger.info(f"Recommendations: {', '.join(current_status['recommendations'])}")
            
#             logger.info("=== END ENERGY STORAGE REPORT ===")
            
#             return report
            
#         except Exception as e:
#             logger.error(f"Failed to generate energy report: {e}")
#             return {'error': str(e), 'timestamp': datetime.now().isoformat()}

#     def _assess_system_health(self) -> str:
#         """Assess overall system health based on metrics and current state."""
#         try:
#             health_score = 100.0
            
#             # Energy level factor
#             current_percentage = (self.energy_storage.get('energy_available_seu', 0) / 
#                                 max(1, self.energy_storage.get('capacity_seu', 1))) * 100
            
#             if current_percentage < 20:
#                 health_score -= 40
#             elif current_percentage < 50:
#                 health_score -= 20
#             elif current_percentage < 70:
#                 health_score -= 10
            
#             # Field disturbance factor
#             total_disturbances = sum(len(d) for d in self.field_disturbances.values())
#             health_score -= min(30, total_disturbances * 2)
            
#             # Energy depletion events factor
#             health_score -= min(20, self.metrics['energy_depletion_events'] * 5)
            
#             # Storage efficiency factor
#             efficiency = self.energy_storage.get('storage_efficiency', 1.0)  # Perfect efficiency
#             if efficiency < 0.99:  # Should always be perfect
#                 health_score -= 5  # Minor penalty if somehow not perfect
            
#             # Determine health category
#             health_score = max(0, health_score)
            
#             if health_score >= 90:
#                 return 'excellent'
#             elif health_score >= 75:
#                 return 'good'
#             elif health_score >= 60:
#                 return 'fair'
#             elif health_score >= 40:
#                 return 'poor'
#             else:
#                 return 'critical'
                
#         except Exception as e:
#             logger.error(f"Failed to assess system health: {e}")
#             return 'unknown'

#     def cleanup_inactive_resources(self):
#         """
#         Clean up tracking for inactive nodes and routes to prevent memory leaks.
#         """
#         try:
#             cleanup_time = datetime.now()
#             nodes_cleaned = 0
#             routes_cleaned = 0
            
#             # Clean up nodes with zero energy
#             nodes_to_remove = []
#             for node_id, node_data in self.active_nodes_energy.items():
#                 if node_data['energy_level'] <= 0:
#                     nodes_to_remove.append(node_id)
            
#             for node_id in nodes_to_remove:
#                 del self.active_nodes_energy[node_id]
#                 nodes_cleaned += 1
            
#             # Clean up routes with zero energy
#             routes_to_remove = []
#             for route_id, route_data in self.active_routes_energy.items():
#                 if route_data['energy_level'] <= 0:
#                     routes_to_remove.append(route_id)
            
#             for route_id in routes_to_remove:
#                 del self.active_routes_energy[route_id]
#                 routes_cleaned += 1
            
#             # Clean up old field disturbances (resolved ones)
#             old_disturbances = []
#             for region_key, disturbances in self.field_disturbances.items():
#                 if not disturbances:  # Empty list
#                     old_disturbances.append(region_key)
            
#             for region_key in old_disturbances:
#                 del self.field_disturbances[region_key]
            
#             # Trim energy history if too long
#             if len(self.energy_history) > 1000:
#                 self.energy_history = self.energy_history[-500:]  # Keep last 500 events
            
#             logger.debug(f"Cleanup complete: {nodes_cleaned} nodes, {routes_cleaned} routes, "
#                         f"{len(old_disturbances)} disturbance regions cleaned")
            
#         except Exception as e:
#             logger.error(f"Failed to cleanup inactive resources: {e}")

# # ===== UTILITY FUNCTIONS =====

# def create_energy_storage_system() -> EnergyStorage:
#     """
#     Create and initialize a new energy storage system.
#     Returns configured EnergyStorage instance.
#     """
#     try:
#         energy_system = EnergyStorage()
#         logger.info("Energy storage system initialized successfully")
#         return energy_system
        
#     except Exception as e:
#         logger.error(f"Failed to create energy storage system: {e}")
#         raise RuntimeError(f"Energy storage system creation failed: {e}")

# def test_energy_storage_functionality():
#     """
#     Test function to verify energy storage system works correctly.
#     """
#     try:
#         # Create energy storage
#         energy_system = EnergyStorage()
        
#         # Create energy store
#         store = energy_system.create_energy_store()
#         print(f"Energy store created: {store['energy_amount_seu']:.1f} SEU")
        
#         # Test node energy addition
#         success = energy_system.add_energy_to_node("test_node_1", (100, 100, 150), 10.0)
#         print(f"Node energy addition: {'Success' if success else 'Failed'}")
        
#         # Test route energy addition
#         success = energy_system.add_energy_to_synaptic_routes("test_route_1", "node_1", "node_2", 5.0)
#         print(f"Route energy addition: {'Success' if success else 'Failed'}")
        
#         # Monitor energy levels
#         status = energy_system.monitor_energy_levels()
#         print(f"Energy status: {status['status']} ({status['percentage']:.1f}%)")
        
#         # Generate report
#         report = energy_system.generate_energy_report()
#         print(f"System health: {report['system_health']}")
        
#         return True
        
#     except Exception as e:
#         print(f"Energy storage test failed: {e}")
#         return False

# if __name__ == "__main__":
#     # Run test if script is executed directly
#     test_energy_storage_functionality()
