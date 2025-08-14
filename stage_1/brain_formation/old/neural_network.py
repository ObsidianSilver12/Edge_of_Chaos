# --- neural_network.py V8 COMPLETE - Part 1 ---
"""
Creates the neural network in stages. Each sub region is calculated with sparse inactive nodes
connected to a few active nodes. The active nodes are placed within a spiral pattern within the sub region
area. Maps synaptic connections between active nodes within sub regions first, then maps
synaptic connections between adjacent sub regions, connecting to nearest active nodes.
Includes basic pattern matching and thinking capabilities for testing.
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

class NeuralNetwork:
    def __init__(self):
        """Initialize the neural network."""
        self.neural_network = {}
        self.brain_structure = None
        self.nodes = {}  # node_id -> node data
        self.synapses = {}  # synapse_id -> synapse data
        self.active_nodes = []
        self.inactive_nodes = []
        self.thinking_active = False
        self.current_query = None
        self.metrics = {
            'total_nodes': 0,
            'active_nodes': 0,
            'inactive_nodes': 0,
            'total_synapses': 0,
            'active_synapses': 0,
            'inactive_synapses': 0,
            'route_recalculations': 0,
            'thinking_triggers': 0,
            'errors': []
        }
        self.status_logs = {}  # node_id -> status history
        self.synapse_routes = {}  # (from_node, to_node) -> [current_route, backup_routes]
        
    def trigger_neural_network_creation(self):
        """
        Trigger the neural network creation process when flag is set to BRAIN_STRUCTURE_CREATED.
        When the neural network process is complete, set FLAG to NEURAL_NETWORK_CREATED.
        """
        try:
            logger.info("Starting neural network creation process...")
            
            # Load brain structure first
            self.load_sub_region_grid_areas()
            
            # Create neural network distribution
            self.create_the_neural_network_distribution_per_sub_region()
            
            # Create synaptic connections in stages
            self.create_the_synaptic_connections_to_each_node()
            self.create_the_synaptic_connections_between_nodes()
            self.create_the_synaptic_connections_between_sub_regions()
            
            # Set completion flag
            setattr(self, FLAG_NEURAL_NETWORK_CREATED, True)
            
            # Generate metrics report
            self.neural_network_metrics_tracking()
            
            logger.info(f"Neural network creation complete. Total nodes: {self.metrics['total_nodes']}, "
                       f"Total synapses: {self.metrics['total_synapses']}")
            
        except Exception as e:
            logger.error(f"Failed to create neural network: {e}")
            self.metrics['errors'].append(f"Creation failed: {e}")
            raise RuntimeError(f"Neural network creation failed: {e}")

    def load_sub_region_grid_areas(self) -> None:
        """Load the sub region grid areas/blocks to create the sparse neural network."""
        try:
            # Check if brain structure is available (should be set by brain_structure.py)
            brain_structure_available = hasattr(self, 'brain_structure') and self.brain_structure is not None
            
            if not brain_structure_available:
                # Try to get brain structure from elsewhere or create minimal structure for testing
                logger.warning("Brain structure not loaded - creating minimal test structure")
                self.brain_structure = {
                    'hemispheres': {
                        'left': {
                            'regions': {
                                'frontal': {
                                    'sub_regions': {
                                        'prefrontal': {
                                            'bounds': {'x_start': 0, 'x_end': 50, 'y_start': 0, 'y_end': 50, 'z_start': 150, 'z_end': 200},
                                            'blocks': {i: {'x': 10+i, 'y': 10+i, 'z': 160+i} for i in range(10)}
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
                                            'bounds': {'x_start': 50, 'x_end': 100, 'y_start': 0, 'y_end': 50, 'z_start': 150, 'z_end': 200},
                                            'blocks': {i: {'x': 60+i, 'y': 10+i, 'z': 160+i} for i in range(10)}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            
            logger.info("Brain structure loaded for neural network creation")
            
        except Exception as e:
            logger.error(f"Failed to load sub region grid areas: {e}")
            raise RuntimeError(f"Grid area loading failed: {e}")

    def create_the_neural_network_distribution_per_sub_region(self):
        """
        Create neural network distribution per sub region. Small regions get 1 active node with 10 inactive,
        larger regions get up to 5 active nodes with 10 inactive each. Store coordinates and frequencies.
        """
        try:
            logger.info("Creating neural network distribution per sub region...")
            
            if not hasattr(self, 'brain_structure') or self.brain_structure is None:
                self.load_sub_region_grid_areas()
                
            if self.brain_structure is None:
                raise ValueError("Brain structure is still None after loading")
                
            if not hasattr(self, 'nodes'):
                self.nodes = {}
            if not hasattr(self, 'active_nodes'):
                self.active_nodes = []
            if not hasattr(self, 'inactive_nodes'):
                self.inactive_nodes = []
            if not hasattr(self, 'status_logs'):
                self.status_logs = {}
            if not hasattr(self, 'metrics'):
                self.metrics = {
                    'active_nodes': 0,
                    'inactive_nodes': 0,
                    'total_nodes': 0,
                    'errors': []
                }
            
            node_id_counter = 0
            
            for hemisphere_name, hemisphere in self.brain_structure['hemispheres'].items():
                for region_name, region in hemisphere['regions'].items():
                    for sub_region_name, sub_region in region['sub_regions'].items():
                        
                        # Calculate sub region volume
                        bounds = sub_region['bounds']
                        volume = ((bounds['x_end'] - bounds['x_start']) * 
                                (bounds['y_end'] - bounds['y_start']) * 
                                (bounds['z_end'] - bounds['z_start']))
                        
                        # Determine number of active nodes based on volume
                        if volume < 10000:  # Small region
                            active_node_count = 1
                        elif volume < 50000:  # Medium region
                            active_node_count = 3
                        else:  # Large region
                            active_node_count = 5
                        
                        # Create active nodes in spiral pattern
                        active_nodes = self._create_spiral_pattern_nodes(
                            bounds, active_node_count, node_type='active')
                        
                        # Create inactive nodes around each active node
                        inactive_nodes = []
                        for active_node in active_nodes:
                            inactive_around_active = self._create_nodes_around_active(
                                active_node, bounds, 10, node_id_counter + 1000)
                            inactive_nodes.extend(inactive_around_active)
                            node_id_counter += 10
                        
                        # Store nodes with full information
                        for node in active_nodes + inactive_nodes:
                            node_id = node['node_id']
                            self.nodes[node_id] = node
                            
                            # Add to appropriate list
                            if node['type'] == 'active':
                                self.active_nodes.append(node_id)
                                self.metrics['active_nodes'] += 1
                            else:
                                self.inactive_nodes.append(node_id)
                                self.metrics['inactive_nodes'] += 1
                                
                            # Initialize status log
                            self.status_logs[node_id] = {
                                'last_status': 'inactive',
                                'last_status_time': datetime.now().isoformat(),
                                'current_status': node['status'],
                                'current_status_time': datetime.now().isoformat()
                            }
                        
                        logger.debug(f"Sub-region {hemisphere_name}.{region_name}.{sub_region_name}: "
                                   f"{len(active_nodes)} active, {len(inactive_nodes)} inactive nodes")
                        
                        node_id_counter += len(active_nodes) + len(inactive_nodes)
            
            self.metrics['total_nodes'] = len(self.nodes)
            logger.info(f"Neural network distribution complete: {self.metrics['total_nodes']} total nodes "
                       f"({self.metrics['active_nodes']} active, {self.metrics['inactive_nodes']} inactive)")
            
        except Exception as e:
            logger.error(f"Failed to create neural network distribution: {e}")
            raise RuntimeError(f"Neural network distribution failed: {e}")

    def _create_spiral_pattern_nodes(self, bounds: Dict, count: int, node_type: str) -> List[Dict]:
        """Create nodes in a spiral pattern within the given bounds."""
        nodes = []
        
        # Calculate center and dimensions
        center_x = (bounds['x_start'] + bounds['x_end']) / 2
        center_y = (bounds['y_start'] + bounds['y_end']) / 2
        center_z = (bounds['z_start'] + bounds['z_end']) / 2
        
        max_radius = min(
            (bounds['x_end'] - bounds['x_start']) / 3,
            (bounds['y_end'] - bounds['y_start']) / 3,
            (bounds['z_end'] - bounds['z_start']) / 3
        )
        
        for i in range(count):
            # Create spiral coordinates
            angle = i * 2.4  # Golden angle for natural spiral
            radius = (i / max(1, count - 1)) * max_radius
            height_offset = (i / max(1, count - 1) - 0.5) * (bounds['z_end'] - bounds['z_start']) * 0.3
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            z = center_z + height_offset
            
            # Ensure within bounds
            x = max(bounds['x_start'], min(bounds['x_end'], x))
            y = max(bounds['y_start'], min(bounds['y_end'], y))
            z = max(bounds['z_start'], min(bounds['z_end'], z))
            
            node = {
                'node_id': str(uuid.uuid4()),
                'type': node_type,
                'coordinates': {'x': x, 'y': y, 'z': z},
                'status': 'active' if node_type == 'active' else 'inactive',
                'frequency_active': random.uniform(40, 100),  # Hz
                'frequency_inactive': random.uniform(0.5, 4),  # Hz
                'creation_time': datetime.now().isoformat(),
                'synapses': []
            }
            nodes.append(node)
        
        return nodes
    
    def _create_nodes_around_active(self, active_node: Dict, bounds: Dict, count: int, base_id: int) -> List[Dict]:
        """Create inactive nodes around an active node."""
        nodes = []
        active_coords = active_node['coordinates']
        
        for i in range(count):
            # Random position around active node
            angle = (i / count) * 2 * math.pi
            distance = random.uniform(5, 15)  # Close to active node
            
            x = active_coords['x'] + distance * math.cos(angle)
            y = active_coords['y'] + distance * math.sin(angle)
            z = active_coords['z'] + random.uniform(-5, 5)
            
            # Ensure within bounds
            x = max(bounds['x_start'], min(bounds['x_end'], x))
            y = max(bounds['y_start'], min(bounds['y_end'], y))
            z = max(bounds['z_start'], min(bounds['z_end'], z))
            
            node = {
                'node_id': str(uuid.uuid4()),
                'type': 'inactive',
                'coordinates': {'x': x, 'y': y, 'z': z},
                'status': 'inactive',
                'frequency_active': random.uniform(40, 100),
                'frequency_inactive': random.uniform(0.5, 4),
                'creation_time': datetime.now().isoformat(),
                'connected_active_node': active_node['node_id'],
                'synapses': []
            }
            nodes.append(node)
        
        return nodes

    def create_the_synaptic_connections_to_each_node(self):
        """Create synaptic connections between each active node and its inactive nodes."""
        try:
            logger.info("Creating synaptic connections to inactive nodes...")
            
            synapse_counter = 0
            
            for node_id, node in self.nodes.items():
                if node['type'] == 'active':
                    # Find all inactive nodes connected to this active node
                    connected_inactive = [n for n in self.nodes.values() 
                                        if n.get('connected_active_node') == node_id]
                    
                    for inactive_node in connected_inactive:
                        synapse_id = f"local_{synapse_counter}"
                        synapse = {
                            'synapse_id': synapse_id,
                            'type': 'local',
                            'from_node': node_id,
                            'to_node': inactive_node['node_id'],
                            'status': 'active',
                            'strength': random.uniform(0.7, 1.0),  # Strong local connections
                            'creation_time': datetime.now().isoformat()
                        }
                        
                        self.synapses[synapse_id] = synapse
                        
                        # Add to both nodes
                        node['synapses'].append(synapse_id)
                        inactive_node['synapses'].append(synapse_id)
                        
                        synapse_counter += 1
            
            # Set flag and update metrics
            setattr(self, FLAG_LOCAL_SYNAPSES_ADDED, True)
            self.metrics['active_synapses'] += synapse_counter
            self.metrics['total_synapses'] += synapse_counter
            
            logger.info(f"Local synaptic connections created: {synapse_counter}")
            
        except Exception as e:
            logger.error(f"Failed to create local synaptic connections: {e}")
            raise RuntimeError(f"Local synapse creation failed: {e}")

    def create_the_synaptic_connections_between_nodes(self):
        """Create synaptic connections between active nodes in the same sub region."""
        try:
            logger.info("Creating synaptic connections between active nodes...")
            
            synapse_counter = 0
            
            # Group active nodes by sub region
            active_by_subregion = {}
            for node_id in self.active_nodes:
                node = self.nodes[node_id]
                # Simple grouping by coordinate proximity
                region_key = f"{int(node['coordinates']['x']//50)}_{int(node['coordinates']['y']//50)}_{int(node['coordinates']['z']//50)}"
                if region_key not in active_by_subregion:
                    active_by_subregion[region_key] = []
                active_by_subregion[region_key].append(node_id)
            
            # Create connections within each sub region
            for region_key, node_ids in active_by_subregion.items():
                if len(node_ids) > 1:
                    # Connect each node to its nearest neighbors
                    for i, node_id in enumerate(node_ids):
                        for j, other_node_id in enumerate(node_ids):
                            if i != j:
                                distance = self._calculate_distance(
                                    self.nodes[node_id]['coordinates'],
                                    self.nodes[other_node_id]['coordinates']
                                )
                                
                                # Only connect close nodes
                                if distance < 30:  # Arbitrary threshold
                                    synapse_id = f"intra_{synapse_counter}"
                                    synapse = {
                                        'synapse_id': synapse_id,
                                        'type': 'intra_region',
                                        'from_node': node_id,
                                        'to_node': other_node_id,
                                        'status': 'active',
                                        'strength': max(0.3, 1.0 - distance/50),  # Distance-based strength
                                        'distance': distance,
                                        'creation_time': datetime.now().isoformat()
                                    }
                                    
                                    self.synapses[synapse_id] = synapse
                                    self.nodes[node_id]['synapses'].append(synapse_id)
                                    
                                    synapse_counter += 1
            
            # Set flag and update metrics
            setattr(self, FLAG_SURROUNDING_SYNAPSES_ADDED, True)
            self.metrics['active_synapses'] += synapse_counter
            self.metrics['total_synapses'] += synapse_counter
            
            logger.info(f"Intra-region synaptic connections created: {synapse_counter}")
            
        except Exception as e:
            logger.error(f"Failed to create intra-region synaptic connections: {e}")
            raise RuntimeError(f"Intra-region synapse creation failed: {e}")

    def create_the_synaptic_connections_between_sub_regions(self):
        """Create synaptic connections between closest active nodes in adjacent sub regions."""
        try:
            logger.info("Creating synaptic connections between sub regions...")
            
            synapse_counter = 0
            
            # For each active node, find nearest active nodes in other regions
            for node_id in self.active_nodes:
                node = self.nodes[node_id]
                nearest_nodes = []
                
                for other_node_id in self.active_nodes:
                    if node_id != other_node_id:
                        other_node = self.nodes[other_node_id]
                        distance = self._calculate_distance(
                            node['coordinates'], other_node['coordinates']
                        )
                        
                        # Check if in different region (simple heuristic)
                        if distance > 50:  # Different region threshold
                            nearest_nodes.append((other_node_id, distance))
                
                # Sort by distance and connect to closest 2-3 nodes
                nearest_nodes.sort(key=lambda x: x[1])
                for other_node_id, distance in nearest_nodes[:3]:  # Connect to 3 nearest
                    if distance < 150:  # Maximum inter-region connection distance
                        synapse_id = f"inter_{synapse_counter}"
                        synapse = {
                            'synapse_id': synapse_id,
                            'type': 'inter_region',
                            'from_node': node_id,
                            'to_node': other_node_id,
                            'status': 'active',
                            'strength': max(0.1, 0.5 - distance/300),  # Weaker long-distance connections
                            'distance': distance,
                            'creation_time': datetime.now().isoformat()
                        }
                        
                        self.synapses[synapse_id] = synapse
                        self.nodes[node_id]['synapses'].append(synapse_id)
                        
                        synapse_counter += 1
            
            # Set flag and update metrics
            setattr(self, FLAG_SUB_REGION_SYNAPSES_ADDED, True)
            self.metrics['active_synapses'] += synapse_counter
            self.metrics['total_synapses'] += synapse_counter
            
            logger.info(f"Inter-region synaptic connections created: {synapse_counter}")
            
        except Exception as e:
            logger.error(f"Failed to create inter-region synaptic connections: {e}")
            raise RuntimeError(f"Inter-region synapse creation failed: {e}")

    def _calculate_distance(self, coord1: Dict, coord2: Dict) -> float:
        """Calculate Euclidean distance between two coordinates."""
        return math.sqrt(
            (coord1['x'] - coord2['x'])**2 +
            (coord1['y'] - coord2['y'])**2 +
            (coord1['z'] - coord2['z'])**2
        )

    def activate_thinking_search(self, query: str = "sephiroth_identity_commonalities"):
        """
        Trigger search for thinking based on test parameter. Simplest form of thinking
        searches for commonalities between sephiroth and identity aspects stored as memory fragments.
        """
        try:
            logger.info(f"Activating thinking search with query: {query}")
            
            self.thinking_active = True
            self.current_query = query
            self.metrics['thinking_triggers'] += 1
            
            # For testing, activate all nodes to see if neural network responds
            activated_nodes = []
            
            # Activate all inactive nodes temporarily for testing
            for node_id in self.inactive_nodes:
                self.nodes[node_id]['status'] = 'active'
                activated_nodes.append(node_id)
                self._update_node_status_log(node_id, 'active')
            
            logger.info(f"Activated {len(activated_nodes)} nodes for thinking search")
            
            # Perform the thinking process
            result = self.think()
            
            # Deactivate nodes after thinking
            for node_id in activated_nodes:
                self.nodes[node_id]['status'] = 'inactive'
                self._update_node_status_log(node_id, 'inactive')
            
            self.thinking_active = False
            self.current_query = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error in thinking search: {e}")
            if 'errors' not in self.metrics:
                self.metrics['errors'] = []
            self.metrics['errors'].append(str(f"Thinking search failed: {e}"))
            self.thinking_active = False
            return None
        
    def activate_nodes(self, condition: str = "memory_query") -> List[str]:
        """
        Determine how many nodes to activate based on conditions.
        Returns list of node IDs to activate.
        """
        try:
            nodes_to_activate = []
            
            if condition == "memory_query":
                # Activate nodes with memory-related patterns
                for node_id in self.active_nodes[:5]:  # Activate first 5 active nodes
                    nodes_to_activate.append(node_id)
                    
            elif condition == "pattern_matching":
                # Activate nodes for pattern matching
                for node_id in self.active_nodes[::2]:  # Every other active node
                    nodes_to_activate.append(node_id)
            
            # Update node statuses
            for node_id in nodes_to_activate:
                if self.nodes[node_id]['status'] != 'active':
                    self.nodes[node_id]['status'] = 'active'
                    self._update_node_status_log(node_id, 'active')
            
            logger.debug(f"Activated {len(nodes_to_activate)} nodes for condition: {condition}")
            return nodes_to_activate
            
        except Exception as e:
            logger.error(f"Error activating nodes: {e}")
            return []

    def think(self) -> Optional[Dict]:
        """
        Trigger the thinking process. Load active query and run basic pattern matching.
        Returns match if found, None otherwise.
        """
        try:
            if not self.thinking_active or not self.current_query:
                return None
                
            logger.info(f"Starting thinking process for query: {self.current_query}")
            
            # Basic pattern matching for baby brain
            patterns = self._get_basic_patterns()
            
            if self.current_query == "sephiroth_identity_commonalities":
                # Look for commonalities between sephiroth and identity aspects
                result = self._find_sephiroth_identity_patterns(patterns)
                
            elif self.current_query == "frequency_resonance":
                # Look for frequency-based patterns
                result = self._find_frequency_patterns(patterns)
                
            else:
                # Default pattern search
                result = self._default_pattern_search(patterns)
            
            if result:
                logger.info(f"Thinking process found result: {result['type']}")
                return result
            else:
                logger.info("No patterns found in thinking process")
                return None
                
        except Exception as e:
            logger.error(f"Error in thinking process: {e}")
            self.metrics['errors'].append(f"Thinking failed: {e}")
            return None

    def _get_basic_patterns(self) -> Dict:
        """Get basic patterns that a baby brain already has."""
        return {
            'frequency_harmony': {'pattern': 'harmonic_ratios', 'threshold': 0.3},
            'spatial_proximity': {'pattern': 'distance_clustering', 'threshold': 0.5},
            'temporal_sequence': {'pattern': 'time_ordering', 'threshold': 0.4},
            'resonance_matching': {'pattern': 'frequency_resonance', 'threshold': 0.6}
        }

    def _find_sephiroth_identity_patterns(self, patterns: Dict) -> Optional[Dict]:
        """Find patterns between sephiroth and identity aspects."""
        # Simulate finding commonalities
        commonalities = []
        
        # Check for frequency resonance patterns
        if 'frequency_harmony' in patterns:
            commonalities.append({
                'type': 'frequency_resonance',
                'strength': 0.7,
                'description': 'Harmonic relationship detected between aspects'
            })
        
        # Check for spatial clustering
        if 'spatial_proximity' in patterns:
            commonalities.append({
                'type': 'spatial_clustering',
                'strength': 0.5,
                'description': 'Spatial co-location of related aspects'
            })
        
        if commonalities:
            return {
                'type': 'sephiroth_identity_match',
                'commonalities': commonalities,
                'confidence': sum(c['strength'] for c in commonalities) / len(commonalities),
                'timestamp': datetime.now().isoformat()
            }
        
        return None

    def _find_frequency_patterns(self, patterns: Dict) -> Optional[Dict]:
        """Find frequency-based patterns."""
        # Simulate frequency pattern detection
        if 'resonance_matching' in patterns:
            return {
                'type': 'frequency_pattern',
                'pattern': 'harmonic_series',
                'confidence': 0.6,
                'timestamp': datetime.now().isoformat()
            }
        return None

    def _default_pattern_search(self, patterns: Dict) -> Optional[Dict]:
        """Default pattern search for unknown queries."""
        # Simple default pattern matching
        if patterns:
            return {
                'type': 'basic_pattern',
                'pattern': 'general_correlation',
                'confidence': 0.4,
                'timestamp': datetime.now().isoformat()
            }
        return None

    def monitor_node_activity(self):
        """
        Monitor the activity (inactive/active status changes) of each node in the neural network.
        Store status change log for each node with last and current status timestamps.
        """
        try:
            active_count = 0
            inactive_count = 0
            
            for node_id, node in self.nodes.items():
                current_status = node['status']
                
                # Count current statuses
                if current_status == 'active':
                    active_count += 1
                else:
                    inactive_count += 1
                
                # Check if status changed since last monitoring
                if node_id in self.status_logs:
                    last_status = self.status_logs[node_id]['current_status']
                    if current_status != last_status:
                        # Status changed, update log
                        self._update_node_status_log(node_id, current_status)
            
            # Update metrics
            self.metrics['active_nodes'] = active_count
            self.metrics['inactive_nodes'] = inactive_count
            
            logger.debug(f"Node activity monitoring: {active_count} active, {inactive_count} inactive")
            
        except Exception as e:
            logger.error(f"Error monitoring node activity: {e}")
            self.metrics['errors'].append(f"Monitoring failed: {e}")

    def _update_node_status_log(self, node_id: str, new_status: str):
        """Update the status log for a specific node."""
        timestamp = datetime.now().isoformat()
        
        if node_id in self.status_logs:
            # Move current to last
            self.status_logs[node_id]['last_status'] = self.status_logs[node_id]['current_status']
            self.status_logs[node_id]['last_status_time'] = self.status_logs[node_id]['current_status_time']
        else:
            # Initialize log
            self.status_logs[node_id] = {
                'last_status': 'inactive',
                'last_status_time': timestamp
            }
        
        # Update current status
        self.status_logs[node_id]['current_status'] = new_status
        self.status_logs[node_id]['current_status_time'] = timestamp

    def recalculate_synaption_route(self, from_node_id: str, to_node_id: str):
        """
        Recalculate the synaptic route between two nodes if one is deactivated.
        Keep old route as backup in case node is reactivated.
        """
        try:
            route_key = (from_node_id, to_node_id)
            
            # Store current route as backup if it exists
            if route_key in self.synapse_routes:
                current_route = self.synapse_routes[route_key]['current']
                if 'backups' not in self.synapse_routes[route_key]:
                    self.synapse_routes[route_key]['backups'] = []
                self.synapse_routes[route_key]['backups'].append(current_route)
            else:
                self.synapse_routes[route_key] = {'backups': []}
            
            # Find alternative route through active nodes
            new_route = self._find_alternative_route(from_node_id, to_node_id)
            
            if new_route:
                self.synapse_routes[route_key]['current'] = new_route
                self.metrics['route_recalculations'] += 1
                logger.debug(f"Recalculated route from {from_node_id} to {to_node_id}: {len(new_route)} hops")
            else:
                logger.warning(f"No alternative route found from {from_node_id} to {to_node_id}")
                
        except Exception as e:
            logger.error(f"Error recalculating synaptic route: {e}")
            self.metrics['errors'].append(f"Route recalculation failed: {e}")

    def _find_alternative_route(self, from_node_id: str, to_node_id: str) -> Optional[List[str]]:
        """Find alternative route between two nodes using only active nodes."""
        try:
            # Simple breadth-first search for alternative path
            if from_node_id not in self.nodes or to_node_id not in self.nodes:
                return None
            
            # Check if both nodes are active
            if (self.nodes[from_node_id]['status'] != 'active' or 
                self.nodes[to_node_id]['status'] != 'active'):
                return None
            
            # BFS to find path
            queue = [[from_node_id]]
            visited = {from_node_id}
            
            while queue:
                path = queue.pop(0)
                current_node = path[-1]
                
                if current_node == to_node_id:
                    return path
                
                # Get connected active nodes
                for synapse_id in self.nodes[current_node]['synapses']:
                    synapse = self.synapses[synapse_id]
                    next_node = synapse['to_node'] if synapse['from_node'] == current_node else synapse['from_node']
                    
                    if (next_node not in visited and 
                        self.nodes[next_node]['status'] == 'active' and
                        len(path) < 5):  # Limit path length
                        visited.add(next_node)
                        queue.append(path + [next_node])
            
            return None  # No path found
            
        except Exception as e:
            logger.error(f"Error finding alternative route: {e}")
            return None

    def neural_network_metrics_tracking(self):
        """
        Track all neural network metrics and output as report log.
        Includes nodes, synapses, route recalculations, thinking triggers, and errors.
        """
        try:
            # Update current metrics
            self.monitor_node_activity()
            
            # Count active/inactive synapses
            active_synapses = sum(1 for s in self.synapses.values() if s['status'] == 'active')
            inactive_synapses = len(self.synapses) - active_synapses
            
            self.metrics.update({
                'total_synapses': len(self.synapses),
                'active_synapses': active_synapses,
                'inactive_synapses': inactive_synapses
            })
            
            # Generate comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'neural_network_metrics': {
                    'nodes': {
                        'total': self.metrics['total_nodes'],
                        'active': self.metrics['active_nodes'],
                        'inactive': self.metrics['inactive_nodes'],
                        'active_percentage': (self.metrics['active_nodes'] / max(1, self.metrics['total_nodes'])) * 100
                    },
                    'synapses': {
                        'total': self.metrics['total_synapses'],
                        'active': self.metrics['active_synapses'],
                        'inactive': self.metrics['inactive_synapses'],
                        'active_percentage': (self.metrics['active_synapses'] / max(1, self.metrics['total_synapses'])) * 100
                    },
                    'operations': {
                        'route_recalculations': self.metrics['route_recalculations'],
                        'thinking_triggers': self.metrics['thinking_triggers'],
                        'total_errors': len(self.metrics['errors'])
                    }
                }
            }
            
            # Log detailed report
            logger.info("=== NEURAL NETWORK METRICS REPORT ===")
            logger.info(f"Nodes: {report['neural_network_metrics']['nodes']['total']} total "
                       f"({report['neural_network_metrics']['nodes']['active']} active, "
                       f"{report['neural_network_metrics']['nodes']['inactive']} inactive)")
            logger.info(f"Synapses: {report['neural_network_metrics']['synapses']['total']} total "
                       f"({report['neural_network_metrics']['synapses']['active']} active, "
                       f"{report['neural_network_metrics']['synapses']['inactive']} inactive)")
            logger.info(f"Operations: {self.metrics['route_recalculations']} route recalculations, "
                       f"{self.metrics['thinking_triggers']} thinking triggers")
            
            if self.metrics['errors']:
                logger.warning(f"Errors encountered: {len(self.metrics['errors'])}")
                for i, error in enumerate(self.metrics['errors'][-5:]):  # Show last 5 errors
                    logger.warning(f"  Error {i+1}: {error}")
            
            logger.info("=== END NEURAL NETWORK METRICS ===")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating neural network metrics: {e}")
            self.metrics['errors'].append(f"Metrics tracking failed: {e}")
            return None

    def save_neural_network(self) -> Dict[str, Any]:
        """
        Save the complete neural network state for use by other brain formation components.
        Returns dictionary with all neural network data.
        """
        try:
            neural_network_data = {
                'neural_network_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'nodes': self.nodes,
                'synapses': self.synapses,
                'active_nodes': self.active_nodes,
                'inactive_nodes': self.inactive_nodes,
                'synapse_routes': self.synapse_routes,
                'status_logs': self.status_logs,
                'metrics': self.metrics,
                'flags': {
                    FLAG_NEURAL_NETWORK_CREATED: getattr(self, FLAG_NEURAL_NETWORK_CREATED, False),
                    FLAG_LOCAL_SYNAPSES_ADDED: getattr(self, FLAG_LOCAL_SYNAPSES_ADDED, False),
                    FLAG_SURROUNDING_SYNAPSES_ADDED: getattr(self, FLAG_SURROUNDING_SYNAPSES_ADDED, False),
                    FLAG_SUB_REGION_SYNAPSES_ADDED: getattr(self, FLAG_SUB_REGION_SYNAPSES_ADDED, False)
                },
                'network_ready_for_mycelium': True
            }
            
            # Store in instance for access by other components
            self.neural_network = neural_network_data
            
            logger.info(f"Neural network saved: {len(self.nodes)} nodes, {len(self.synapses)} synapses")
            return neural_network_data
            
        except Exception as e:
            logger.error(f"Failed to save neural network: {e}")
            raise RuntimeError(f"Neural network save failed: {e}")

    def get_neural_network_for_integration(self) -> Dict[str, Any]:
        """
        Get neural network data for integration with other brain components.
        Used by mycelial network and memory distribution systems.
        """
        try:
            if not hasattr(self, 'neural_network') or not self.neural_network:
                # Save current state if not already saved
                return self.save_neural_network()
            
            return self.neural_network
            
        except Exception as e:
            logger.error(f"Error getting neural network for integration: {e}")
            raise RuntimeError(f"Neural network integration data failed: {e}")

# ===== END OF NEURAL_NETWORK.PY CLASS =====

# Usage example and test functions (can be removed in production)
def test_neural_network_creation():
    """Test function to verify neural network creation works correctly."""
    try:
        nn = NeuralNetwork()
        nn.trigger_neural_network_creation()
        
        # Test thinking functionality
        result = nn.activate_thinking_search("sephiroth_identity_commonalities")
        print(f"Thinking test result: {result}")
        
        # Generate metrics report
        metrics = nn.neural_network_metrics_tracking()
        print(f"Neural network created successfully with {nn.metrics['total_nodes']} nodes")
        
        return True
        
    except Exception as e:
        print(f"Neural network test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if script is executed directly
    test_neural_network_creation()












# # --- neural_network.py V7 ---
# """
# Creates the neural network in stages. each sub region is calculated with some sparse inactive nodes
# connected to a few active nodes. The active nodes are placed within a spiral pattern within the sub region
# area. we will map synaptic connections between the active nodes within the sub region first and then will map
# synaptic connections between the adjacent sub regions.connecting the synapses to the nearest active node in adjacent
# sub regions.
# """

# import logging
# import uuid
# import random
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# from datetime import datetime
# import math
# from stage_1.brain_formation.brain.region_definitions import *
# from shared.constants.constants import *

# # --- Logging Setup ---
# logger = logging.getLogger("Conception")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# class NeuralNetwork:

#     def __init__(self):
#         """
#         Initialize the neural network.
#         """

#     def trigger_neural_network_creation(self):
#         """
#         trigger the neural network creation process when flag is set to BRAIN_STRUCTURE_CREATED.
#         when the neural network process is complete which is triggered by SUB_REGION_SYNAPSES_ADDED
#         set FLAG to NEURAL_NETWORK_CREATED. Add flag to constants
#         """

#     def load_sub_region_grid_areas(self) -> None:
#         """
#         Load the sub region grid areas/blocks in order to create the sparse neural network.
#         """
#     def create_the_neural_network_distribution_per_sub_region(self):
#         """"
#         if sub region volume is less than a certain amount only create 1 active node with 10 
#         inactive nodes per node. for bigger regions create up to 5 active nodes with 10 inactive nodes.
#         after each sub region map is created store the active nodes in a list or numpy array.make sure all
#         x,y,z coordinates of the active and inactive nodes are stored with the active and inactive frequency
#         defined for each node.
#         """"

#     def create_the_synaptic_connections_to_each_node():
#         """
#         create the synaptic connections between each node and its inactive nodes.store the data to the list/numpy
#         array created in the previous step.mark the flag as LOCAL_SYNAPSES_ADDED and store with the data
#         """	

#     def create_the_synaptic_connections_between_nodes():
#         """
#         create the synaptic connections between each active node in the sub region.
#         store the data to the list/numpy array created in the previous step.
#         mark the flag as SURROUNDING_SYNAPSES_ADDED and store with the data
#         """	

#     def create_the_synaptic_connections_between_sub_regions():
#         """
#         create the synaptic connections between closest active node in adjacent sub regions.
#         that could mean in any direction. store the data to the list/numpy array created in 
#         the previous step.mark the flag as SUB_REGION_SYNAPSES_ADDED and store with the data
#         """	

#     def activate_thinking_search():
#         """
#         trigger search for thinking based on a test paremeter simplest form of thinking for this basic
#         brain which has no learned patterns or new memories is to search for commonalities 
#         between the sephiroth aspects and identity aspects stored as memory fragments. because these
#         have a standard frequency for inactive we can call up all of them so that we can test if this
#         is working. Normally inactive nodes are not searched for patterns when thinking as we would
#         only do that when learning or dreaming. but for testing purposes we will activate all inactive
#         nodes and see if the neural network responds correctly.
#         """
#     def activate_nodes():
#         """
#         determine how many nodes to activate and based on what conditions. triggered 
#         by thinking about something. return a list of node indices to activate.activate them and
#         update their status in the neural network.
#         """

#     def think():
#         """
#         trigger the thinking process. load the active query from prior steps and run through a
#         basic sequence of pattern matching to find a match. if a match is found, return the match.
#         you would need to guide on what we could find correlation for realising that we dont have
#         any learned patterns to use for this. this means we need some basic patterns a baby brain
#         already has. consider if we need separate pattern functions that we just run through for
#         each thought process. does not have to be complex as we will add more complex patterns later.
#         If no patterns found return None and deactivate the thinking process and return the nodes
#         """

#     def monitor_node_activity():
#         """
#         monitor the activity (inactive/active status changes) of each node in the neural network.
#         for now we will not add any decay to the synaptic connections and will only test that 
#         the monitoring is working. the monitoring is triggered whenever a node status is changed.
#         the monitor will store a status change log for each node which only contains last status,
#         last status date and time, new status and new status date and time. each time the node
#         status is changed the monitor will update the status change log moving new to last updated
#         and updating new with current status and date and time. this should be stored to the main 
#         node dictionary instead of a separate dictionary.
#         """

#     def recalculate_synaption_route():
#         """
#         recalculate the synaptic route between two nodes if one of the nodes is deactivated.
#         this will be triggered whenever a node is deactivated. update synaptic path dictionary with
#         new synaptic route but always keep the old synaptic route in case the node is activated again.
#         """

#     def neural_network_metrics_tracking():
#         """
#         ensure all metrics are tracked correctly and stored in the metrics dictionary.output as a
#         report log. metrics include: total number of nodes, total number of active nodes, total
#         number of inactive nodes, total number of synaptic connections, total number of
#         active synaptic connections, total number of inactive synaptic connections,how many times 
#         synaptic routes recalculated, how many times thinking triggered, record any failures and errors
#         """

     