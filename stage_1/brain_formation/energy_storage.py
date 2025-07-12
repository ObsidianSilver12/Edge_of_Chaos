# --- energy_storage.py V8 COMPLETE ---
"""
Creates energy storage system in the limbic system for the brain after mycelial network is formed.
Triggered by SEEDS_ENTANGLED flag. Manages energy distribution to nodes and synapses with 
field disturbance detection and repair. Calculates 2-week energy requirement with 5-10% variance.
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

class EnergyStorage:
    def __init__(self):
        """Initialize the energy storage system."""
        self.energy_storage = {}
        self.energy_amount = 0  # Initial value set to 0
        self.active_nodes_energy = {}  # Track energy in active nodes
        self.active_routes_energy = {}  # Track energy in active routes
        self.field_disturbances = {}  # Track field disturbances by region
        self.energy_history = []  # Track energy changes over time
        self.metrics = {
            'total_energy_stored': 0.0,
            'total_energy_distributed': 0.0,
            'total_energy_returned': 0.0,
            'field_repairs': 0,
            'energy_depletion_events': 0,
            'creation_time': None
        }
        
        # Energy distribution thresholds
        self.energy_thresholds = {
            'critical_low': ENERGY_THRESHOLD_CRITICAL_LOW,
            'low': ENERGY_THRESHOLD_LOW,
            'optimal': ENERGY_THRESHOLD_OPTIMAL,
            'high': ENERGY_THRESHOLD_HIGH
        }
        
    def create_energy_store(self):
        """
        Create the energy store within the deep limbic sub region adjacent to the limbic system.
        Triggered by flag SEEDS_ENTANGLED. Calculate 2-week energy with 5-10% random variance.
        Save flag as STORE_CREATED.
        """
        try:
            logger.info("Creating energy storage system in limbic region...")
            
            # Calculate base 2-week energy requirement
            # Based on typical brain energy consumption: ~20 watts continuous
            base_energy_joules = 20.0 * 14 * 24 * 60 * 60  # 20W for 14 days in joules
            
            # Convert to SEU (Soul Energy Units)
            base_energy_seu = base_energy_joules * SEU_PER_JOULE
            
            # Add random variance (5-10%)
            variance_factor = random.uniform(1.05, 1.10)  # 5-10% increase
            total_energy_seu = base_energy_seu * variance_factor
            
            # Store energy in dictionary format
            self.energy_storage = {
                'energy_store_id': str(uuid.uuid4()),
                'creation_time': datetime.now().isoformat(),
                'location': 'deep_limbic_sub_region',
                'energy_type': 'initial_creator_energy',
                'energy_amount_seu': float(total_energy_seu),
                'base_requirement_seu': float(base_energy_seu),
                'variance_factor': float(variance_factor),
                'energy_distributed_seu': 0.0,
                'energy_returned_seu': 0.0,
                'energy_available_seu': float(total_energy_seu),
                'capacity_seu': float(total_energy_seu * 1.2),  # 20% overhead
                'status': 'active',
                'storage_efficiency': 1.0  # Perfect efficiency - no energy loss
            }
            
            # Initialize tracking arrays
            self.energy_amount = total_energy_seu
            self.energy_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'store_created',
                'energy_level': total_energy_seu,
                'action': 'initial_creation'
            })
            
            # Update metrics
            self.metrics['total_energy_stored'] = total_energy_seu
            self.metrics['creation_time'] = self.energy_storage['creation_time']
            
            # Set flag indicating store creation is complete
            setattr(self, FLAG_STORE_CREATED, True)
            
            logger.info(f"Energy store created: {total_energy_seu:.1f} SEU "
                       f"(base: {base_energy_seu:.1f}, variance: {variance_factor:.3f})")
            
            return self.energy_storage
            
        except Exception as e:
            logger.error(f"Failed to create energy store: {e}")
            raise RuntimeError(f"Energy store creation failed: {e}")

    def add_energy_to_node(self, node_id: str, node_coordinates: Tuple[float, float, float], 
                          energy_required: float = 1.0) -> bool:
        """
        Add energy to an active node and perform local field calculation.
        Check for local disturbances around the node coordinates.
        Set flag to FIELD_DISTURBANCE if disturbance found.
        """
        try:
            # Check if enough energy is available
            if self.energy_storage['energy_available_seu'] < energy_required:
                logger.warning(f"Insufficient energy for node {node_id}: "
                             f"Required {energy_required}, Available {self.energy_storage['energy_available_seu']}")
                return False
            
            # Calculate local field stability
            field_stability = self._calculate_local_field_stability(node_coordinates)
            
            # Add energy to node
            if node_id not in self.active_nodes_energy:
                self.active_nodes_energy[node_id] = {
                    'coordinates': node_coordinates,
                    'energy_level': 0.0,
                    'activation_time': datetime.now().isoformat(),
                    'field_stability': field_stability
                }
            
            # Transfer energy
            self.active_nodes_energy[node_id]['energy_level'] += energy_required
            self.energy_storage['energy_available_seu'] -= energy_required
            self.energy_storage['energy_distributed_seu'] += energy_required
            
            # Update field stability
            self.active_nodes_energy[node_id]['field_stability'] = field_stability
            
            # Check for field disturbances
            if field_stability < 0.7:  # Threshold for field disturbance
                disturbance_detected = True
                setattr(self, FLAG_FIELD_DISTURBANCE, True)
                
                # Record disturbance
                region_key = self._get_region_key(node_coordinates)
                if region_key not in self.field_disturbances:
                    self.field_disturbances[region_key] = []
                
                self.field_disturbances[region_key].append({
                    'node_id': node_id,
                    'coordinates': node_coordinates,
                    'field_stability': field_stability,
                    'disturbance_time': datetime.now().isoformat(),
                    'disturbance_type': 'low_field_stability'
                })
                
                logger.warning(f"Field disturbance detected at node {node_id}: "
                             f"stability {field_stability:.3f}")
                
                # Trigger field repair
                self.diagnose_repair_field(region_key)
            else:
                disturbance_detected = False
            
            # Log energy addition
            self.energy_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_added_to_node',
                'node_id': node_id,
                'energy_amount': energy_required,
                'remaining_energy': self.energy_storage['energy_available_seu'],
                'field_stability': field_stability,
                'disturbance_detected': disturbance_detected
            })
            
            # Update metrics
            self.metrics['total_energy_distributed'] += energy_required
            
            logger.debug(f"Energy added to node {node_id}: {energy_required:.2f} SEU, "
                        f"field stability: {field_stability:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add energy to node {node_id}: {e}")
            return False

    def diagnose_repair_field(self, region_key: str):
        """
        Diagnose and repair field disturbances per sub region.
        Apply appropriate repair based on disturbance type and severity.
        """
        try:
            logger.info(f"Diagnosing field disturbances in region {region_key}...")
            
            if region_key not in self.field_disturbances:
                logger.warning(f"No disturbances recorded for region {region_key}")
                return
            
            disturbances = self.field_disturbances[region_key]
            
            # Analyze disturbance patterns
            avg_stability = np.mean([d['field_stability'] for d in disturbances])
            disturbance_count = len(disturbances)
            
            # Count active nodes and synapses in region
            region_nodes = self._count_active_nodes_in_region(region_key)
            region_synapses = self._count_active_synapses_in_region(region_key)
            
            repair_actions = []
            
            # Determine repair strategy based on disturbance type
            if avg_stability < 0.3:  # Severe instability
                logger.warning(f"Severe field instability in {region_key}: {avg_stability:.3f}")
                
                if region_nodes > 10 or region_synapses > 20:
                    # Too many active components - deactivate some
                    repair_actions.append('deactivate_excess_nodes')
                    repair_actions.append('deactivate_excess_synapses')
                else:
                    # Insufficient energy - create more energy
                    repair_actions.append('create_additional_energy')
                    
            elif avg_stability < 0.5:  # Moderate instability
                logger.info(f"Moderate field instability in {region_key}: {avg_stability:.3f}")
                
                if region_nodes > 5:
                    # Moderate load - selective deactivation
                    repair_actions.append('selective_node_deactivation')
                else:
                    # Boost energy temporarily
                    repair_actions.append('temporary_energy_boost')
                    
            elif avg_stability < 0.7:  # Minor instability
                logger.debug(f"Minor field instability in {region_key}: {avg_stability:.3f}")
                repair_actions.append('minor_energy_adjustment')
            
            # Apply repair actions
            for action in repair_actions:
                self._apply_repair_action(region_key, action, disturbances)
            
            # Clear resolved disturbances
            resolved_count = len(disturbances)
            del self.field_disturbances[region_key]
            
            # Update metrics
            self.metrics['field_repairs'] += resolved_count
            
            logger.info(f"Field repair complete for {region_key}: {resolved_count} disturbances resolved")
            
        except Exception as e:
            logger.error(f"Failed to repair field in region {region_key}: {e}")

    def _apply_repair_action(self, region_key: str, action: str, disturbances: List[Dict]):
        """Apply specific repair action to resolve field disturbances."""
        try:
            if action == 'deactivate_excess_nodes':
                # Deactivate nodes with lowest field stability
                nodes_to_deactivate = sorted(disturbances, key=lambda x: x['field_stability'])[:3]
                for disturbance in nodes_to_deactivate:
                    self.remove_energy_from_node(disturbance['node_id'])
                logger.info(f"Deactivated {len(nodes_to_deactivate)} excess nodes in {region_key}")
                
            elif action == 'selective_node_deactivation':
                # Deactivate most problematic node
                worst_node = min(disturbances, key=lambda x: x['field_stability'])
                self.remove_energy_from_node(worst_node['node_id'])
                logger.info(f"Selectively deactivated worst node in {region_key}")
                
            elif action == 'create_additional_energy':
                # Generate emergency energy (limited)
                emergency_energy = min(100.0, self.energy_storage['capacity_seu'] * 0.05)
                self.energy_storage['energy_available_seu'] += emergency_energy
                logger.info(f"Created {emergency_energy:.1f} SEU emergency energy for {region_key}")
                
            elif action == 'temporary_energy_boost':
                # Boost energy temporarily
                boost_energy = 50.0
                self.energy_storage['energy_available_seu'] += boost_energy
                logger.info(f"Applied {boost_energy:.1f} SEU temporary boost to {region_key}")
                
            elif action == 'minor_energy_adjustment':
                # Minor adjustment
                adjust_energy = 10.0
                self.energy_storage['energy_available_seu'] += adjust_energy
                logger.debug(f"Applied {adjust_energy:.1f} SEU minor adjustment to {region_key}")
                
        except Exception as e:
            logger.error(f"Failed to apply repair action {action} to {region_key}: {e}")

    def remove_energy_from_node(self, node_id: str) -> float:
        """
        Remove energy from a deactivated node and return it to energy storage.
        Update the energy storage dictionary/numpy array.
        """
        try:
            if node_id not in self.active_nodes_energy:
                logger.warning(f"Node {node_id} not found in active nodes")
                return 0.0
            
            # Get energy from node
            node_energy = self.active_nodes_energy[node_id]['energy_level']
            
            # Apply storage efficiency loss
            recoverable_energy = node_energy * self.energy_storage['storage_efficiency']
            energy_transformation = 0.0
            
            # Return energy to storage
            self.energy_storage['energy_available_seu'] += recoverable_energy
            self.energy_storage['energy_returned_seu'] += recoverable_energy
            
            # Remove node from active tracking
            del self.active_nodes_energy[node_id]
            
            # Log energy removal with perfect conservation
            self.energy_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_removed_from_node',
                'node_id': node_id,
                'energy_returned': recoverable_energy,
                'energy_transformed': energy_transformation,
                'total_available': self.energy_storage['energy_available_seu'],
                'conservation_perfect': True
            })
            
            # Update metrics
            self.metrics['total_energy_returned'] += recoverable_energy
            
            logger.debug(f"Energy removed from node {node_id}: {recoverable_energy:.2f} SEU returned perfectly, "
                        f"energy conservation maintained")
            
            return recoverable_energy
            
        except Exception as e:
            logger.error(f"Failed to remove energy from node {node_id}: {e}")
            return 0.0

    def add_energy_to_synaptic_routes(self, route_id: str, from_node: str, to_node: str,
                                     energy_required: float = 0.5) -> bool:
        """
        Add energy to a synaptic route when activated.
        Update energy storage and set ACTIVE_ROUTE flag.
        """
        try:
            # Check energy availability
            if self.energy_storage['energy_available_seu'] < energy_required:
                logger.warning(f"Insufficient energy for route {route_id}: "
                             f"Required {energy_required}, Available {self.energy_storage['energy_available_seu']}")
                return False
            
            # Add route to active tracking
            if route_id not in self.active_routes_energy:
                self.active_routes_energy[route_id] = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'energy_level': 0.0,
                    'activation_time': datetime.now().isoformat()
                }
            
            # Transfer energy
            self.active_routes_energy[route_id]['energy_level'] += energy_required
            self.energy_storage['energy_available_seu'] -= energy_required
            self.energy_storage['energy_distributed_seu'] += energy_required
            
            # Set active route flag
            setattr(self, 'ACTIVE_ROUTE', True)
            
            # Log energy addition
            self.energy_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_added_to_route',
                'route_id': route_id,
                'from_node': from_node,
                'to_node': to_node,
                'energy_amount': energy_required,
                'remaining_energy': self.energy_storage['energy_available_seu']
            })
            
            # Update metrics
            self.metrics['total_energy_distributed'] += energy_required
            
            logger.debug(f"Energy added to route {route_id} ({from_node}->{to_node}): {energy_required:.2f} SEU")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add energy to route {route_id}: {e}")
            return False

    def remove_energy_from_synaptic_routes(self, route_id: str) -> float:
        """
        Remove energy from a deactivated synaptic route and return to storage.
        Set DEACTIVATED_ROUTE flag.
        """
        try:
            if route_id not in self.active_routes_energy:
                logger.warning(f"Route {route_id} not found in active routes")
                return 0.0
            
            # Get energy from route
            route_energy = self.active_routes_energy[route_id]['energy_level']
            
            # Apply storage efficiency loss
            recoverable_energy = route_energy * self.energy_storage['storage_efficiency']
            energy_transformation = 0.0
            
            # Return energy to storage
            self.energy_storage['energy_available_seu'] += recoverable_energy
            self.energy_storage['energy_returned_seu'] += recoverable_energy
            
            # Remove route from active tracking
            route_info = self.active_routes_energy[route_id]
            del self.active_routes_energy[route_id]
            
            # Set deactivated route flag
            setattr(self, 'DEACTIVATED_ROUTE', True)
            
            # Log energy removal with perfect conservation
            self.energy_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'energy_removed_from_route',
                'route_id': route_id,
                'from_node': route_info['from_node'],
                'to_node': route_info['to_node'],
                'energy_returned': recoverable_energy,
                'energy_transformed': energy_transformation,
                'total_available': self.energy_storage['energy_available_seu'],
                'conservation_perfect': True
            })
            
            # Update metrics
            self.metrics['total_energy_returned'] += recoverable_energy
            
            logger.debug(f"Energy removed from route {route_id}: {recoverable_energy:.2f} SEU returned")
            
            return recoverable_energy
            
        except Exception as e:
            logger.error(f"Failed to remove energy from route {route_id}: {e}")
            return 0.0

    def monitor_energy_levels(self) -> Dict[str, Any]:
        """
        Monitor overall energy levels and trigger actions based on thresholds.
        Return current energy status and recommendations.
        """
        try:
            current_level = self.energy_storage['energy_available_seu']
            capacity = self.energy_storage['capacity_seu']
            percentage = (current_level / capacity) * 100 if capacity > 0 else 0
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'energy_level_seu': current_level,
                'capacity_seu': capacity,
                'percentage': percentage,
                'status': 'unknown',
                'recommendations': []
            }
            
            # Determine status and recommendations
            if percentage < (self.energy_thresholds['critical_low'] * 100):
                status['status'] = 'critical_low'
                status['recommendations'].extend([
                    'emergency_node_deactivation',
                    'suspend_non_essential_processes',
                    'activate_energy_conservation_mode'
                ])
                self.metrics['energy_depletion_events'] += 1
                logger.error(f"CRITICAL: Energy level at {percentage:.1f}% - emergency measures required")
                
            elif percentage < (self.energy_thresholds['low'] * 100):
                status['status'] = 'low'
                status['recommendations'].extend([
                    'selective_node_deactivation',
                    'reduce_synaptic_activity',
                    'prioritize_essential_functions'
                ])
                logger.warning(f"LOW: Energy level at {percentage:.1f}% - conservation needed")
                
            elif percentage < (self.energy_thresholds['optimal'] * 100):
                status['status'] = 'moderate'
                status['recommendations'].append('monitor_usage_patterns')
                logger.info(f"MODERATE: Energy level at {percentage:.1f}% - normal operation")
                
            else:
                status['status'] = 'optimal'
                status['recommendations'].append('normal_operation')
                logger.debug(f"OPTIMAL: Energy level at {percentage:.1f}% - excellent")
            
            # Add current usage statistics
            status['usage_stats'] = {
                'active_nodes': len(self.active_nodes_energy),
                'active_routes': len(self.active_routes_energy),
                'total_distributed': self.energy_storage['energy_distributed_seu'],
                'total_returned': self.energy_storage['energy_returned_seu'],
                'efficiency': (self.energy_storage['energy_returned_seu'] / 
                             max(1, self.energy_storage['energy_distributed_seu'])) * 100
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to monitor energy levels: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_local_field_stability(self, coordinates: Tuple[float, float, float]) -> float:
        """Calculate local field stability around given coordinates."""
        try:
            x, y, z = coordinates
            
            # Calculate distance from brain center (simplified)
            brain_center = (128, 128, 128)  # Center of 256x256x256 grid
            distance_from_center = math.sqrt(
                (x - brain_center[0])**2 + 
                (y - brain_center[1])**2 + 
                (z - brain_center[2])**2
            )
            
            # Base stability decreases with distance from center
            distance_factor = max(0.3, 1.0 - (distance_from_center / 200))
            
            # Count nearby active nodes (field interference)
            nearby_interference = 0
            for node_data in self.active_nodes_energy.values():
                node_coords = node_data['coordinates']
                distance = math.sqrt(
                    (x - node_coords[0])**2 + 
                    (y - node_coords[1])**2 + 
                    (z - node_coords[2])**2
                )
                if distance < 20:  # Within interference range
                    nearby_interference += 1
            
            # Interference penalty
            interference_factor = max(0.2, 1.0 - (nearby_interference * 0.1))
            
            # Random field fluctuation
            random_factor = random.uniform(0.9, 1.1)
            
            # Calculate final stability
            stability = distance_factor * interference_factor * random_factor
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Failed to calculate field stability: {e}")
            return 0.5  # Default moderate stability

    def _get_region_key(self, coordinates: Tuple[float, float, float]) -> str:
        """Get region key for coordinates (simplified spatial hashing)."""
        x, y, z = coordinates
        return f"region_{int(x//50)}_{int(y//50)}_{int(z//50)}"

    def _count_active_nodes_in_region(self, region_key: str) -> int:
        """Count active nodes in a specific region."""
        count = 0
        for node_data in self.active_nodes_energy.values():
            node_region = self._get_region_key(node_data['coordinates'])
            if node_region == region_key:
                count += 1
        return count

    def _count_active_synapses_in_region(self, region_key: str) -> int:
        """Count active synapses in a specific region (simplified)."""
        # Simplified: assume 2 synapses per node on average
        return self._count_active_nodes_in_region(region_key) * 2

    def get_energy_storage_status(self) -> Dict[str, Any]:
        """
        Get complete energy storage status for integration with other brain components.
        """
        try:
            status = {
                'energy_storage_id': self.energy_storage.get('energy_store_id'),
                'creation_time': self.energy_storage.get('creation_time'),
                'current_energy_seu': self.energy_storage.get('energy_available_seu', 0),
                'total_capacity_seu': self.energy_storage.get('capacity_seu', 0),
                'energy_distributed_seu': self.energy_storage.get('energy_distributed_seu', 0),
                'energy_returned_seu': self.energy_storage.get('energy_returned_seu', 0),
                'storage_efficiency': self.energy_storage.get('storage_efficiency', 0.95),
                'active_nodes_count': len(self.active_nodes_energy),
                'active_routes_count': len(self.active_routes_energy),
                'field_disturbances_count': sum(len(d) for d in self.field_disturbances.values()),
                'metrics': self.metrics.copy(),
                'flags': {
                    FLAG_STORE_CREATED: getattr(self, FLAG_STORE_CREATED, False),
                    FLAG_FIELD_DISTURBANCE: getattr(self, FLAG_FIELD_DISTURBANCE, False),
                    'ACTIVE_ROUTE': getattr(self, 'ACTIVE_ROUTE', False),
                    'DEACTIVATED_ROUTE': getattr(self, 'DEACTIVATED_ROUTE', False)
                },
                'energy_thresholds': self.energy_thresholds.copy(),
                'ready_for_integration': True
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get energy storage status: {e}")
            return {'error': str(e), 'ready_for_integration': False}

    def generate_energy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive energy storage report for monitoring and debugging.
        """
        try:
            # Current status
            current_status = self.monitor_energy_levels()
            
            # Energy history summary
            history_summary = {
                'total_events': len(self.energy_history),
                'recent_events': self.energy_history[-10:] if self.energy_history else [],
                'event_types': {}
            }
            
            # Count event types
            for event in self.energy_history:
                event_type = event.get('event', 'unknown')
                history_summary['event_types'][event_type] = history_summary['event_types'].get(event_type, 0) + 1
            
            # Field disturbance summary
            disturbance_summary = {
                'regions_affected': len(self.field_disturbances),
                'total_disturbances': sum(len(d) for d in self.field_disturbances.values()),
                'disturbance_types': {}
            }
            
            for region_disturbances in self.field_disturbances.values():
                for disturbance in region_disturbances:
                    dtype = disturbance.get('disturbance_type', 'unknown')
                    disturbance_summary['disturbance_types'][dtype] = disturbance_summary['disturbance_types'].get(dtype, 0) + 1
            
            # Resource utilization
            resource_utilization = {
                'energy_utilization_percentage': (
                    self.energy_storage.get('energy_distributed_seu', 0) / 
                    max(1, self.energy_storage.get('energy_amount_seu', 1))
                ) * 100,
                'node_density': len(self.active_nodes_energy),
                'route_density': len(self.active_routes_energy),
                'average_node_energy': np.mean([n['energy_level'] for n in self.active_nodes_energy.values()]) if self.active_nodes_energy else 0,
                'average_route_energy': np.mean([r['energy_level'] for r in self.active_routes_energy.values()]) if self.active_routes_energy else 0
            }
            
            # Compile comprehensive report
            report = {
                'report_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'energy_storage_system': {
                    'store_id': self.energy_storage.get('energy_store_id'),
                    'status': current_status['status'],
                    'energy_level_seu': current_status['energy_level_seu'],
                    'capacity_percentage': current_status['percentage'],
                    'storage_location': self.energy_storage.get('location'),
                    'creation_time': self.energy_storage.get('creation_time')
                },
                'current_status': current_status,
                'energy_history': history_summary,
                'field_disturbances': disturbance_summary,
                'resource_utilization': resource_utilization,
                'metrics': self.metrics.copy(),
                'recommendations': current_status.get('recommendations', []),
                'system_health': self._assess_system_health()
            }
            
            # Log report summary
            logger.info("=== ENERGY STORAGE SYSTEM REPORT ===")
            logger.info(f"Status: {current_status['status'].upper()} - {current_status['percentage']:.1f}% capacity")
            logger.info(f"Active Nodes: {len(self.active_nodes_energy)}, Active Routes: {len(self.active_routes_energy)}")
            logger.info(f"Energy Distributed: {self.energy_storage.get('energy_distributed_seu', 0):.1f} SEU")
            logger.info(f"Energy Returned: {self.energy_storage.get('energy_returned_seu', 0):.1f} SEU")
            logger.info(f"Field Repairs: {self.metrics['field_repairs']}, Depletion Events: {self.metrics['energy_depletion_events']}")
            
            if current_status.get('recommendations'):
                logger.info(f"Recommendations: {', '.join(current_status['recommendations'])}")
            
            logger.info("=== END ENERGY STORAGE REPORT ===")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate energy report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _assess_system_health(self) -> str:
        """Assess overall system health based on metrics and current state."""
        try:
            health_score = 100.0
            
            # Energy level factor
            current_percentage = (self.energy_storage.get('energy_available_seu', 0) / 
                                max(1, self.energy_storage.get('capacity_seu', 1))) * 100
            
            if current_percentage < 20:
                health_score -= 40
            elif current_percentage < 50:
                health_score -= 20
            elif current_percentage < 70:
                health_score -= 10
            
            # Field disturbance factor
            total_disturbances = sum(len(d) for d in self.field_disturbances.values())
            health_score -= min(30, total_disturbances * 2)
            
            # Energy depletion events factor
            health_score -= min(20, self.metrics['energy_depletion_events'] * 5)
            
            # Storage efficiency factor
            efficiency = self.energy_storage.get('storage_efficiency', 1.0)  # Perfect efficiency
            if efficiency < 0.99:  # Should always be perfect
                health_score -= 5  # Minor penalty if somehow not perfect
            
            # Determine health category
            health_score = max(0, health_score)
            
            if health_score >= 90:
                return 'excellent'
            elif health_score >= 75:
                return 'good'
            elif health_score >= 60:
                return 'fair'
            elif health_score >= 40:
                return 'poor'
            else:
                return 'critical'
                
        except Exception as e:
            logger.error(f"Failed to assess system health: {e}")
            return 'unknown'

    def cleanup_inactive_resources(self):
        """
        Clean up tracking for inactive nodes and routes to prevent memory leaks.
        """
        try:
            cleanup_time = datetime.now()
            nodes_cleaned = 0
            routes_cleaned = 0
            
            # Clean up nodes with zero energy
            nodes_to_remove = []
            for node_id, node_data in self.active_nodes_energy.items():
                if node_data['energy_level'] <= 0:
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                del self.active_nodes_energy[node_id]
                nodes_cleaned += 1
            
            # Clean up routes with zero energy
            routes_to_remove = []
            for route_id, route_data in self.active_routes_energy.items():
                if route_data['energy_level'] <= 0:
                    routes_to_remove.append(route_id)
            
            for route_id in routes_to_remove:
                del self.active_routes_energy[route_id]
                routes_cleaned += 1
            
            # Clean up old field disturbances (resolved ones)
            old_disturbances = []
            for region_key, disturbances in self.field_disturbances.items():
                if not disturbances:  # Empty list
                    old_disturbances.append(region_key)
            
            for region_key in old_disturbances:
                del self.field_disturbances[region_key]
            
            # Trim energy history if too long
            if len(self.energy_history) > 1000:
                self.energy_history = self.energy_history[-500:]  # Keep last 500 events
            
            logger.debug(f"Cleanup complete: {nodes_cleaned} nodes, {routes_cleaned} routes, "
                        f"{len(old_disturbances)} disturbance regions cleaned")
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive resources: {e}")

# ===== UTILITY FUNCTIONS =====

def create_energy_storage_system() -> EnergyStorage:
    """
    Create and initialize a new energy storage system.
    Returns configured EnergyStorage instance.
    """
    try:
        energy_system = EnergyStorage()
        logger.info("Energy storage system initialized successfully")
        return energy_system
        
    except Exception as e:
        logger.error(f"Failed to create energy storage system: {e}")
        raise RuntimeError(f"Energy storage system creation failed: {e}")

def test_energy_storage_functionality():
    """
    Test function to verify energy storage system works correctly.
    """
    try:
        # Create energy storage
        energy_system = EnergyStorage()
        
        # Create energy store
        store = energy_system.create_energy_store()
        print(f"Energy store created: {store['energy_amount_seu']:.1f} SEU")
        
        # Test node energy addition
        success = energy_system.add_energy_to_node("test_node_1", (100, 100, 150), 10.0)
        print(f"Node energy addition: {'Success' if success else 'Failed'}")
        
        # Test route energy addition
        success = energy_system.add_energy_to_synaptic_routes("test_route_1", "node_1", "node_2", 5.0)
        print(f"Route energy addition: {'Success' if success else 'Failed'}")
        
        # Monitor energy levels
        status = energy_system.monitor_energy_levels()
        print(f"Energy status: {status['status']} ({status['percentage']:.1f}%)")
        
        # Generate report
        report = energy_system.generate_energy_report()
        print(f"System health: {report['system_health']}")
        
        return True
        
    except Exception as e:
        print(f"Energy storage test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if script is executed directly
    test_energy_storage_functionality()
