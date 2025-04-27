"""
time_decay.py - Module for managing memory and neural connection decay over time.

This module implements the time decay mechanism that causes inactive neural 
network bonds to dissolve and updates memory map coordinates accordingly.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TimeDecay')

class TimeDecay:
    """
    Manages the decay of inactive neural network bonds and updates memory map coordinates.
    
    The time decay mechanism ensures that unused or infrequently accessed information
    gradually decays, allowing the system to prioritize more relevant or recent information.
    However, it also maintains a record of last known positions to enable recovery if needed.
    """
    
    def __init__(self, brain_structure):
        """
        Initialize the time decay mechanism.
        
        Parameters:
            brain_structure: The brain structure to manage decay for
        """
        self.brain_structure = brain_structure
        
        # Dictionary to track node activity timestamps
        # {node_id: {'last_access': timestamp, 'access_count': count}}
        self.node_activity = {}
        
        # Dictionary to track last known positions of decayed nodes
        # {node_id: {'position': 3D_coordinate, 'token_id': wbs_token, 'decay_time': timestamp}}
        self.last_known_positions = {}
        
        # Dictionary to track connection strengths between nodes
        # {(node1_id, node2_id): {'strength': value, 'last_access': timestamp}}
        self.connection_strengths = {}
        
        # Base decay rates by memory type
        self.memory_type_decay_rates = {
            'survival': 0.0001,   # Very slow decay for survival information
            'emotional': 0.001,   # Slow decay for emotional memories
            'procedural': 0.002,  # Slow decay for procedural (how-to) memories
            'semantic': 0.005,    # Medium decay for semantic (factual) knowledge
            'episodic': 0.007,    # Medium decay for episodic (event) memories
            'working': 0.05,      # Fast decay for working memory
            'dimensional': 0.002, # Slow decay for dimensional information
            'ephemeral': 0.1      # Very fast decay for ephemeral info
        }
        
        # Decay modifiers for different brain states
        self.state_decay_modifiers = {
            'sleep': 0.5,         # Slower decay during sleep
            'dream': 0.3,         # Much slower decay during dream states
            'meditation': 0.7,    # Moderately slower decay during meditation
            'focused': 1.2,       # Slightly faster decay when focused (to clear irrelevant info)
            'learning': 0.8,      # Slower decay when actively learning
            'default': 1.0        # Default/normal decay rate
        }
        
        # Current brain state for decay calculations
        self.current_state = 'default'
        
        # Minimum connection strength before decay
        self.min_connection_strength = 0.01
        
        # Define grid coordinate limits
        self.grid_limits = {
            'x': (-100, 100),  # X coordinate range
            'y': (-100, 100),  # Y coordinate range
            'z': (-100, 100)   # Z coordinate range
        }
        
        logger.info("Time decay mechanism initialized")
    
    def register_node(self, node_id, token_id, position, memory_type='semantic'):
        """
        Register a node for decay tracking.
        
        Parameters:
            node_id (str): Unique identifier for the node
            token_id (str): WBS token ID
            position (tuple/array): 3D position coordinates
            memory_type (str): Type of memory
            
        Returns:
            bool: True if registration successful
        """
        # Create node activity record
        self.node_activity[node_id] = {
            'last_access': datetime.now(),
            'access_count': 1,
            'memory_type': memory_type,
            'token_id': token_id,
            'position': position,
            'decay_level': 0.0  # No decay initially
        }
        
        logger.info(f"Registered node {node_id} of type {memory_type} at position {position}")
        return True
    
    def register_connection(self, node1_id, node2_id, initial_strength=1.0):
        """
        Register a connection between nodes for decay tracking.
        
        Parameters:
            node1_id (str): First node ID
            node2_id (str): Second node ID
            initial_strength (float): Initial connection strength
            
        Returns:
            bool: True if registration successful
        """
        # Ensure nodes exist
        if node1_id not in self.node_activity or node2_id not in self.node_activity:
            logger.warning(f"Cannot register connection: one or both nodes not found")
            return False
        
        # Create sorted connection key (to ensure the same connection isn't stored twice)
        conn_key = tuple(sorted([node1_id, node2_id]))
        
        # Create connection record
        self.connection_strengths[conn_key] = {
            'strength': initial_strength,
            'last_access': datetime.now(),
            'creation_time': datetime.now(),
            'access_count': 1
        }
        
        logger.info(f"Registered connection between {node1_id} and {node2_id} with strength {initial_strength}")
        return True
    
    def access_node(self, node_id):
        """
        Record access to a node, updating its activity timestamp.
        
        Parameters:
            node_id (str): The node being accessed
            
        Returns:
            bool: True if node was found and updated
        """
        if node_id not in self.node_activity:
            logger.warning(f"Cannot access node {node_id}: not found")
            return False
        
        # Update node activity
        self.node_activity[node_id]['last_access'] = datetime.now()
        self.node_activity[node_id]['access_count'] += 1
        
        # Reset decay level when accessed
        self.node_activity[node_id]['decay_level'] = 0.0
        
        # If node was previously decayed, restore it from last known position
        if node_id in self.last_known_positions:
            logger.info(f"Restoring previously decayed node {node_id}")
            # Could restore additional properties here if needed
            del self.last_known_positions[node_id]
        
        return True
    
    def access_connection(self, node1_id, node2_id, strengthen_amount=0.1):
        """
        Record access to a connection, updating its timestamp and strengthening it.
        
        Parameters:
            node1_id (str): First node ID
            node2_id (str): Second node ID
            strengthen_amount (float): Amount to strengthen the connection
            
        Returns:
            bool: True if connection was found and updated
        """
        # Create sorted connection key
        conn_key = tuple(sorted([node1_id, node2_id]))
        
        if conn_key not in self.connection_strengths:
            logger.warning(f"Cannot access connection {conn_key}: not found")
            return False
        
        # Update connection activity
        self.connection_strengths[conn_key]['last_access'] = datetime.now()
        self.connection_strengths[conn_key]['access_count'] += 1
        
        # Strengthen the connection (up to a maximum of 1.0)
        current_strength = self.connection_strengths[conn_key]['strength']
        self.connection_strengths[conn_key]['strength'] = min(1.0, current_strength + strengthen_amount)
        
        return True
    
    def update_decay(self, elapsed_time=None):
        """
        Update decay for all nodes and connections based on elapsed time.
        
        Parameters:
            elapsed_time (timedelta, optional): Time elapsed since last update
            
        Returns:
            dict: Decay update metrics
        """
        # If no elapsed time specified, use a default of 1 hour
        if elapsed_time is None:
            elapsed_time = timedelta(hours=1)
        
        # Get elapsed time in seconds
        elapsed_seconds = elapsed_time.total_seconds()
        
        # Get current state's decay modifier
        decay_modifier = self.state_decay_modifiers.get(self.current_state, 1.0)
        
        # Track metrics
        metrics = {
            'nodes_updated': 0,
            'nodes_decayed': 0,
            'connections_updated': 0,
            'connections_dissolved': 0
        }
        
        # Update node decay
        decayed_nodes = self._update_node_decay(elapsed_seconds, decay_modifier)
        metrics['nodes_updated'] = len(self.node_activity)
        metrics['nodes_decayed'] = len(decayed_nodes)
        
        # Update connection decay
        dissolved_connections = self._update_connection_decay(elapsed_seconds, decay_modifier)
        metrics['connections_updated'] = len(self.connection_strengths)
        metrics['connections_dissolved'] = len(dissolved_connections)
        
        logger.info(f"Updated decay: {metrics['nodes_decayed']} nodes decayed, "
                   f"{metrics['connections_dissolved']} connections dissolved")
        
        return metrics
    
    def _update_node_decay(self, elapsed_seconds, decay_modifier):
        """
        Update decay for all nodes.
        
        Parameters:
            elapsed_seconds (float): Elapsed time in seconds
            decay_modifier (float): Modification factor for decay rate
            
        Returns:
            list: IDs of nodes that were decayed
        """
        decayed_nodes = []
        now = datetime.now()
        
        # Process each node
        for node_id, activity in list(self.node_activity.items()):
            # Calculate time since last access
            last_access = activity['last_access']
            inactive_time = (now - last_access).total_seconds()
            
            # Get base decay rate for this memory type
            memory_type = activity.get('memory_type', 'semantic')
            base_decay_rate = self.memory_type_decay_rates.get(memory_type, 0.005)
            
            # Calculate actual decay rate, modified by state and access count
            # More frequently accessed nodes decay more slowly
            access_modifier = 1.0 / (1.0 + 0.1 * activity['access_count'])
            decay_rate = base_decay_rate * decay_modifier * access_modifier
            
            # Calculate decay amount for this update
            decay_amount = decay_rate * elapsed_seconds
            
            # Update decay level
            activity['decay_level'] += decay_amount
            
            # Check if node has decayed beyond threshold (1.0 = fully decayed)
            if activity['decay_level'] >= 1.0:
                # Store last known position before removing
                self.last_known_positions[node_id] = {
                    'position': activity['position'],
                    'token_id': activity['token_id'],
                    'decay_time': now,
                    'memory_type': memory_type
                }
                
                # Remove from active nodes
                del self.node_activity[node_id]
                decayed_nodes.append(node_id)
                
                logger.info(f"Node {node_id} decayed after {inactive_time:.2f} seconds of inactivity")
        
        return decayed_nodes
    
    def _update_connection_decay(self, elapsed_seconds, decay_modifier):
        """
        Update decay for all connections.
        
        Parameters:
            elapsed_seconds (float): Elapsed time in seconds
            decay_modifier (float): Modification factor for decay rate
            
        Returns:
            list: Keys of connections that were dissolved
        """
        dissolved_connections = []
        now = datetime.now()
        
        # Base decay rate for connections
        base_connection_decay = 0.003  # Default connection decay rate
        
        # Process each connection
        for conn_key, conn_data in list(self.connection_strengths.items()):
            # Calculate time since last access
            last_access = conn_data['last_access']
            inactive_time = (now - last_access).total_seconds()
            
            # Calculate decay rate, modified by state and access count
            # More frequently accessed connections decay more slowly
            access_modifier = 1.0 / (1.0 + 0.05 * conn_data['access_count'])
            decay_rate = base_connection_decay * decay_modifier * access_modifier
            
            # Calculate decay amount for this update
            decay_amount = decay_rate * elapsed_seconds
            
            # Weaken the connection
            current_strength = conn_data['strength']
            new_strength = max(0.0, current_strength - decay_amount)
            
            # Update connection strength
            conn_data['strength'] = new_strength
            
            # Check if connection has dissolved below minimum strength
            if new_strength < self.min_connection_strength:
                # Remove connection
                del self.connection_strengths[conn_key]
                dissolved_connections.append(conn_key)
                
                logger.info(f"Connection {conn_key} dissolved after {inactive_time:.2f} seconds of inactivity")
        
        return dissolved_connections
    
    def set_brain_state(self, state):
        """
        Set the current brain state to modify decay rates.
        
        Parameters:
            state (str): Brain state (sleep, dream, meditation, focused, learning, default)
            
        Returns:
            float: New decay modifier
        """
        if state in self.state_decay_modifiers:
            self.current_state = state
            logger.info(f"Brain state set to {state} with decay modifier {self.state_decay_modifiers[state]}")
            return self.state_decay_modifiers[state]
        else:
            logger.warning(f"Unknown brain state {state}, using default")
            self.current_state = 'default'
            return self.state_decay_modifiers['default']
    
    def restore_node(self, node_id):
        """
        Restore a previously decayed node using its last known position.
        
        Parameters:
            node_id (str): ID of node to restore
            
        Returns:
            bool: True if restoration successful
        """
        if node_id not in self.last_known_positions:
            logger.warning(f"Cannot restore node {node_id}: no last known position")
            return False
        
        # Get last known position data
        position_data = self.last_known_positions[node_id]
        
# Register node again
        self.register_node(
            node_id=node_id,
            token_id=position_data['token_id'],
            position=position_data['position'],
            memory_type=position_data['memory_type']
        )
        
        # Remove from last known positions
        del self.last_known_positions[node_id]
        
        logger.info(f"Successfully restored node {node_id}")
        return True
    
    def update_node_position(self, node_id, new_position):
        """
        Update a node's position in the 3D memory grid.
        
        Parameters:
            node_id (str): ID of node to update
            new_position (tuple/array): New 3D position coordinates
            
        Returns:
            bool: True if update successful
        """
        if node_id not in self.node_activity:
            logger.warning(f"Cannot update position of node {node_id}: not found")
            return False
        
        # Store previous position
        prev_position = self.node_activity[node_id]['position']
        
        # Update position
        self.node_activity[node_id]['position'] = new_position
        
        # Mark as accessed
        self.node_activity[node_id]['last_access'] = datetime.now()
        
        logger.info(f"Updated position of node {node_id} from {prev_position} to {new_position}")
        return True
    
    def calculate_decay_trajectory(self, node_id, projection_time):
        """
        Calculate the projected decay trajectory for a node.
        
        Parameters:
            node_id (str): ID of node to project
            projection_time (timedelta): Time to project into the future
            
        Returns:
            dict: Decay trajectory data
        """
        if node_id not in self.node_activity:
            logger.warning(f"Cannot calculate decay trajectory for node {node_id}: not found")
            return None
        
        # Get node data
        node_data = self.node_activity[node_id]
        
        # Get memory type and base decay rate
        memory_type = node_data.get('memory_type', 'semantic')
        base_decay_rate = self.memory_type_decay_rates.get(memory_type, 0.005)
        
        # Apply modifiers
        decay_modifier = self.state_decay_modifiers.get(self.current_state, 1.0)
        access_modifier = 1.0 / (1.0 + 0.1 * node_data['access_count'])
        decay_rate = base_decay_rate * decay_modifier * access_modifier
        
        # Calculate decay over time
        projection_seconds = projection_time.total_seconds()
        projected_decay = node_data['decay_level'] + (decay_rate * projection_seconds)
        
        # Calculate time until full decay
        time_to_decay = (1.0 - node_data['decay_level']) / decay_rate if decay_rate > 0 else float('inf')
        
        return {
            'node_id': node_id,
            'current_decay': node_data['decay_level'],
            'projected_decay': min(1.0, projected_decay),
            'decay_rate': decay_rate,
            'time_to_decay': time_to_decay,
            'will_decay': projected_decay >= 1.0
        }
    
    def bulk_update_node_positions(self, position_updates):
        """
        Update positions for multiple nodes at once.
        
        Parameters:
            position_updates (dict): {node_id: new_position} mapping
            
        Returns:
            int: Number of nodes successfully updated
        """
        successful_updates = 0
        
        for node_id, new_position in position_updates.items():
            if self.update_node_position(node_id, new_position):
                successful_updates += 1
        
        logger.info(f"Bulk updated positions for {successful_updates}/{len(position_updates)} nodes")
        return successful_updates
    
    def get_decayed_nodes_by_type(self, memory_type=None):
        """
        Get decayed nodes filtered by memory type.
        
        Parameters:
            memory_type (str, optional): Memory type to filter by
            
        Returns:
            dict: Decayed nodes with their last known positions
        """
        if memory_type:
            return {
                node_id: data for node_id, data in self.last_known_positions.items()
                if data.get('memory_type') == memory_type
            }
        else:
            return self.last_known_positions
    
    def get_active_nodes_by_decay_level(self, min_decay=0.0, max_decay=1.0):
        """
        Get active nodes filtered by their current decay level.
        
        Parameters:
            min_decay (float): Minimum decay level
            max_decay (float): Maximum decay level
            
        Returns:
            dict: Nodes within the specified decay range
        """
        return {
            node_id: data for node_id, data in self.node_activity.items()
            if min_decay <= data['decay_level'] <= max_decay
        }
    
    def simulate_decay(self, time_period, time_step=timedelta(days=1)):
        """
        Simulate decay over a period of time without actually modifying the system.
        
        Parameters:
            time_period (timedelta): Total time period to simulate
            time_step (timedelta): Size of each simulation step
            
        Returns:
            dict: Simulation results
        """
        # Make copies of current data to avoid modifying the actual system
        simulated_nodes = {k: v.copy() for k, v in self.node_activity.items()}
        simulated_connections = {k: v.copy() for k, v in self.connection_strengths.items()}
        
        # Prepare results
        results = {
            'initial_node_count': len(simulated_nodes),
            'initial_connection_count': len(simulated_connections),
            'timeline': [],
            'final_node_count': 0,
            'final_connection_count': 0
        }
        
        # Get decay modifier for current state
        decay_modifier = self.state_decay_modifiers.get(self.current_state, 1.0)
        
        # Current simulation time
        current_time = datetime.now()
        end_time = current_time + time_period
        
        # Simulation loop
        while current_time < end_time:
            # Advance time
            current_time += time_step
            elapsed_seconds = time_step.total_seconds()
            
            # Simulate node decay
            for node_id, node_data in list(simulated_nodes.items()):
                # Get memory type and base decay rate
                memory_type = node_data.get('memory_type', 'semantic')
                base_decay_rate = self.memory_type_decay_rates.get(memory_type, 0.005)
                
                # Apply modifiers
                access_modifier = 1.0 / (1.0 + 0.1 * node_data['access_count'])
                decay_rate = base_decay_rate * decay_modifier * access_modifier
                
                # Calculate decay for this step
                decay_amount = decay_rate * elapsed_seconds
                node_data['decay_level'] += decay_amount
                
                # Check if node has decayed beyond threshold
                if node_data['decay_level'] >= 1.0:
                    del simulated_nodes[node_id]
            
            # Simulate connection decay
            base_connection_decay = 0.003
            for conn_key, conn_data in list(simulated_connections.items()):
                # Calculate decay rate
                access_modifier = 1.0 / (1.0 + 0.05 * conn_data['access_count'])
                decay_rate = base_connection_decay * decay_modifier * access_modifier
                
                # Calculate decay for this step
                decay_amount = decay_rate * elapsed_seconds
                conn_data['strength'] -= decay_amount
                
                # Check if connection has dissolved
                if conn_data['strength'] < self.min_connection_strength:
                    del simulated_connections[conn_key]
            
            # Record state at this point
            results['timeline'].append({
                'time': current_time,
                'nodes_remaining': len(simulated_nodes),
                'connections_remaining': len(simulated_connections)
            })
        
        # Record final state
        results['final_node_count'] = len(simulated_nodes)
        results['final_connection_count'] = len(simulated_connections)
        
        return results
    
    def recommend_memory_consolidation(self, decay_threshold=0.5, min_access_count=2):
        """
        Recommend nodes for memory consolidation based on decay level and access patterns.
        
        Parameters:
            decay_threshold (float): Decay threshold for consideration
            min_access_count (int): Minimum access count for consideration
            
        Returns:
            list: Nodes recommended for consolidation
        """
        recommendations = []
        
        for node_id, node_data in self.node_activity.items():
            # Check if node meets criteria
            if (node_data['decay_level'] > decay_threshold and 
                node_data['access_count'] >= min_access_count):
                
                # Calculate importance score
                # Nodes with high access count and moderate decay are prioritized
                importance = (node_data['access_count'] / 10) * (1.0 - node_data['decay_level'])
                
                recommendations.append({
                    'node_id': node_id,
                    'decay_level': node_data['decay_level'],
                    'access_count': node_data['access_count'],
                    'memory_type': node_data.get('memory_type', 'semantic'),
                    'importance': importance
                })
        
        # Sort by importance
        recommendations.sort(key=lambda x: x['importance'], reverse=True)
        
        return recommendations
    
    def get_related_nodes(self, node_id, max_distance=1):
        """
        Get nodes related to the specified node through connections.
        
        Parameters:
            node_id (str): The center node ID
            max_distance (int): Maximum connection distance to include
            
        Returns:
            dict: Related nodes by distance
        """
        if node_id not in self.node_activity:
            logger.warning(f"Cannot find related nodes for {node_id}: not found")
            return {}
        
        # Initialize results
        related_nodes = {
            0: [node_id],  # Distance 0 is the node itself
            1: []  # Distance 1 is directly connected nodes
        }
        
        # Find directly connected nodes (distance 1)
        for conn_key in self.connection_strengths:
            if node_id in conn_key:
                # Get the other node in the connection
                other_node = conn_key[0] if conn_key[1] == node_id else conn_key[1]
                
                if other_node in self.node_activity:
                    related_nodes[1].append(other_node)
        
        # If max_distance > 1, find nodes at greater distances
        if max_distance > 1:
            visited = set([node_id] + related_nodes[1])
            
            for distance in range(2, max_distance + 1):
                related_nodes[distance] = []
                
                # For each node at the previous distance
                for prev_node in related_nodes[distance - 1]:
                    # Find its connections
                    for conn_key in self.connection_strengths:
                        if prev_node in conn_key:
                            # Get the other node in the connection
                            other_node = conn_key[0] if conn_key[1] == prev_node else conn_key[1]
                            
                            # Add if not already visited
                            if other_node not in visited and other_node in self.node_activity:
                                related_nodes[distance].append(other_node)
                                visited.add(other_node)
        
        return related_nodes
    
    def get_metrics(self):
        """
        Get comprehensive metrics about the time decay system.
        
        Returns:
            dict: System metrics
        """
        metrics = {
            'active_nodes': {
                'total': len(self.node_activity),
                'by_memory_type': {}
            },
            'decayed_nodes': {
                'total': len(self.last_known_positions),
                'by_memory_type': {}
            },
            'connections': {
                'total': len(self.connection_strengths),
                'avg_strength': 0.0
            },
            'current_state': self.current_state,
            'decay_modifier': self.state_decay_modifiers.get(self.current_state, 1.0)
        }
        
        # Count nodes by memory type
        for node_data in self.node_activity.values():
            memory_type = node_data.get('memory_type', 'semantic')
            if memory_type not in metrics['active_nodes']['by_memory_type']:
                metrics['active_nodes']['by_memory_type'][memory_type] = 0
            metrics['active_nodes']['by_memory_type'][memory_type] += 1
        
        # Count decayed nodes by memory type
        for position_data in self.last_known_positions.values():
            memory_type = position_data.get('memory_type', 'semantic')
            if memory_type not in metrics['decayed_nodes']['by_memory_type']:
                metrics['decayed_nodes']['by_memory_type'][memory_type] = 0
            metrics['decayed_nodes']['by_memory_type'][memory_type] += 1
        
        # Calculate average connection strength
        if self.connection_strengths:
            avg_strength = sum(conn['strength'] for conn in self.connection_strengths.values()) / len(self.connection_strengths)
            metrics['connections']['avg_strength'] = avg_strength
        
        # Calculate average decay level of active nodes
        if self.node_activity:
            avg_decay = sum(node['decay_level'] for node in self.node_activity.values()) / len(self.node_activity)
            metrics['active_nodes']['avg_decay_level'] = avg_decay
        
        return metrics


# Factory function to create TimeDecay
def create_time_decay(brain_structure):
    """
    Create a new time decay mechanism.
    
    Parameters:
        brain_structure: The brain structure to manage decay for
        
    Returns:
        TimeDecay: A new time decay instance
    """
    logger.info("Creating new time decay mechanism")
    return TimeDecay(brain_structure)