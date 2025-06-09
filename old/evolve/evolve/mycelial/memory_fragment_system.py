# --- START OF FILE stage_1/evolve/memory_fragment_system.py ---

"""
Memory Fragment System (V4.5.0 - Stage 1 Implementation)

Handles the storage and processing of memory fragments in the brain.
Manages transfer of soul aspects to memory fragments and fragment interactions.
Follows proper physics principles with hard validation and no fallbacks.
"""

import numpy as np
import logging
import os
import sys
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import math
import random
from constants.constants import *
# --- Logging Setup ---
logger = logging.getLogger("MemoryFragmentSystem")
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

class MemoryFragmentSystem:
    """
    System for managing memory fragments in the brain.
    
    Distinguishes between fragments, active nodes, and inactive nodes.
    Handles storage, retrieval, and association of memory fragments.
    """
    
    def __init__(self, brain_structure=None):
        """
        Initialize the memory fragment system.
        
        Args:
            brain_structure: Optional brain structure object
        """
        self.system_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.last_updated = self.creation_time
        
        # Fragment storage
        self.fragments = {}  # Maps fragment_id to fragment data
        self.fragment_positions = {}  # Maps fragment_id to position in brain
        
        # Node storage (fragments with sufficient complexity)
        self.active_nodes = {}  # Maps node_id to active node data
        self.inactive_nodes = {}  # Maps node_id to inactive node data
        
        # Association tracking
        self.associations = {}  # Maps (fragment_id1, fragment_id2) to association data
        
        # Brain structure reference
        self.brain_structure = brain_structure
        
        # Metrics
        self.metrics = {
            'fragments_created': 0,
            'fragments_retrieved': 0,
            'associations_created': 0,
            'nodes_activated': 0,
            'nodes_deactivated': 0
        }
        
        logger.info(f"Memory Fragment System initialized with ID {self.system_id}")
    
    def add_fragment(self, content: Any, region: str, position: Optional[Tuple[int, int, int]] = None, 
                   frequency: Optional[float] = None, meta_tags: Optional[Dict[str, Any]] = None,
                   origin: str = "perception") -> str:
        """
        Add a memory fragment to the brain.
        
        Args:
            content: The content of the memory fragment
            region: Brain region to store the fragment
            position: Optional specific position in the region
            frequency: Optional frequency for the fragment
            meta_tags: Optional metadata tags for classification
            origin: Source of the fragment (perception, soul, etc.)
            
        Returns:
            Fragment ID
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if content is None:
            raise ValueError("Fragment content cannot be None")
        
        if not region:
            raise ValueError("Region must be provided")
        
        # Generate fragment ID
        fragment_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        
        # Set default frequency if not provided
        if frequency is None:
            if self.brain_structure is not None:
                # Use default frequency for region
                frequency = REGION_DEFAULT_FREQUENCIES.get(
                    region, SCHUMANN_FREQUENCY)
            else:
                frequency = SCHUMANN_FREQUENCY
        
        # Create meta tags if not provided
        if meta_tags is None:
            meta_tags = {}
        
        # Find position if not provided
        if position is None:
            if self.brain_structure is not None:
                position = self._find_optimal_position(region, frequency)
            else:
                # Default position (will be updated when brain structure is available)
                position = (0, 0, 0)
        
        # Create the fragment
        fragment = {
            "fragment_id": fragment_id,
            "content": content,
            "region": region,
            "position": position,
            "frequency_hz": frequency,
            "meta_tags": meta_tags,
            "origin": origin,
            "creation_time": creation_time,
            "last_accessed": creation_time,
            "access_count": 0,
            "energy": 0.1,  # Initial energy level
            "resonance": 0.5,  # Initial resonance
            "coherence": 0.5,  # Initial coherence
            "classification": None,  # Not classified yet
            "complexity": self._calculate_complexity(content),
            "associations": []  # List of associated fragment IDs
        }
        
        # Store the fragment
        self.fragments[fragment_id] = fragment
        self.fragment_positions[fragment_id] = position
        
        # Apply fragment to brain structure if available
        if self.brain_structure is not None:
            self._apply_fragment_to_brain(fragment_id, fragment)
        
        # Update metrics
        self.metrics['fragments_created'] += 1
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Added memory fragment {fragment_id} to {region} region at {position}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'fragment_id': fragment_id,
                    'region': region,
                    'position': position,
                    'frequency': frequency,
                    'origin': origin,
                    'complexity': fragment['complexity'],
                    'timestamp': creation_time
                }
                metrics.record_metrics("memory_fragment_creation", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment creation metrics: {e}")
        
        return fragment_id
    
    def add_stimulus_to_fragment(self, fragment_id: str, stimulus: Any) -> Dict[str, Any]:
        """
        Add stimulus data to an existing fragment.
        
        Args:
            fragment_id: ID of the fragment to update
            stimulus: Stimulus data to add
            
        Returns:
            Dict with update status
            
        Raises:
            ValueError: If fragment_id is invalid
        """
        # Validate fragment_id
        if fragment_id not in self.fragments:
            raise ValueError(f"Invalid fragment ID: {fragment_id}")
        
        # Get the fragment
        fragment = self.fragments[fragment_id]
        
        # Update access timestamp and count
        fragment["last_accessed"] = datetime.now().isoformat()
        fragment["access_count"] += 1
        
        # Add stimulus to content (depends on content type)
        if isinstance(fragment["content"], list):
            # Content is a list, append stimulus
            fragment["content"].append(stimulus)
        elif isinstance(fragment["content"], dict):
            # Content is a dict, update with stimulus
            if isinstance(stimulus, dict):
                fragment["content"].update(stimulus)
            else:
                # Add as a 'stimulus' field
                fragment["content"]["stimulus"] = stimulus
        else:
            # For other content types, convert to list
            fragment["content"] = [fragment["content"], stimulus]
        
        # Recalculate complexity
        fragment["complexity"] = self._calculate_complexity(fragment["content"])
        
        # Increase energy slightly
        fragment["energy"] = min(1.0, fragment["energy"] + 0.1)
        
        # Apply updated fragment to brain structure if available
        if self.brain_structure is not None:
            self._apply_fragment_to_brain(fragment_id, fragment)
        
        # Update the fragment in storage
        self.fragments[fragment_id] = fragment
        
        # Update timestamp
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Added stimulus to fragment {fragment_id}, new complexity: {fragment['complexity']}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'fragment_id': fragment_id,
                    'new_complexity': fragment['complexity'],
                    'timestamp': fragment["last_accessed"]
                }
                metrics.record_metrics("memory_fragment_stimulus", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment stimulus metrics: {e}")
        
        return {
            'success': True,
            'fragment_id': fragment_id,
            'complexity': fragment['complexity'],
            'energy': fragment['energy']
        }
    
    def associate_fragments(self, fragment_id1: str, fragment_id2: str, 
                          strength: float = 0.5, bidirectional: bool = True) -> Dict[str, Any]:
        """
        Create an association between two fragments.
        
        Args:
            fragment_id1: ID of the first fragment
            fragment_id2: ID of the second fragment
            strength: Association strength (0-1)
            bidirectional: Whether the association is bidirectional
            
        Returns:
            Dict with association data
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate fragment IDs
        if fragment_id1 not in self.fragments:
            raise ValueError(f"Invalid fragment ID: {fragment_id1}")
        
        if fragment_id2 not in self.fragments:
            raise ValueError(f"Invalid fragment ID: {fragment_id2}")
        
        if fragment_id1 == fragment_id2:
            raise ValueError("Cannot associate a fragment with itself")
        
        # Validate strength
        if not 0 <= strength <= 1:
            raise ValueError(f"Association strength must be between 0 and 1, got {strength}")
        
        # Create association ID
        association_id = f"{fragment_id1}_{fragment_id2}"
        if bidirectional:
            # Use consistent ID for bidirectional associations
            if fragment_id1 > fragment_id2:
                association_id = f"{fragment_id2}_{fragment_id1}"
        
        # Get the fragments
        fragment1 = self.fragments[fragment_id1]
        fragment2 = self.fragments[fragment_id2]
        
        # Update access timestamps
        fragment1["last_accessed"] = datetime.now().isoformat()
        fragment2["last_accessed"] = datetime.now().isoformat()
        
        # Create association data
        association_time = datetime.now().isoformat()
        
        association_data = {
            'association_id': association_id,
            'fragment1': fragment_id1,
            'fragment2': fragment_id2,
            'strength': strength,
            'bidirectional': bidirectional,
            'creation_time': association_time,
            'last_activation': association_time,
            'activation_count': 0
        }
        
        # Store the association
        self.associations[association_id] = association_data
        
        # Add association to fragment1
        fragment1['associations'].append({
            'fragment_id': fragment_id2,
            'association_id': association_id,
            'strength': strength,
            'creation_time': association_time
        })
        
        # Add association to fragment2 if bidirectional
        if bidirectional:
            fragment2['associations'].append({
                'fragment_id': fragment_id1,
                'association_id': association_id,
                'strength': strength,
                'creation_time': association_time
            })
        
        # Update fragments in storage
        self.fragments[fragment_id1] = fragment1
        self.fragments[fragment_id2] = fragment2
        
        # If brain structure available, create physical connection
        if self.brain_structure is not None:
            self._create_physical_association(fragment_id1, fragment_id2, strength)
        
        # Update metrics
        self.metrics['associations_created'] += 1
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Created association {association_id} between fragments {fragment_id1} and {fragment_id2} "
                   f"with strength {strength}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'association_id': association_id,
                    'fragment1': fragment_id1,
                    'fragment2': fragment_id2,
                    'strength': strength,
                    'bidirectional': bidirectional,
                    'timestamp': association_time
                }
                metrics.record_metrics("memory_fragment_association", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment association metrics: {e}")
        
        return association_data
    
    def get_fragment(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory fragment by ID.
        
        Args:
            fragment_id: ID of the fragment to retrieve
            
        Returns:
            Fragment data or None if not found
        """
        if fragment_id not in self.fragments:
            logger.warning(f"Fragment {fragment_id} not found")
            return None
        
        # Get the fragment
        fragment = self.fragments[fragment_id].copy()
        
        # Update access timestamp and count
        self.fragments[fragment_id]["last_accessed"] = datetime.now().isoformat()
        self.fragments[fragment_id]["access_count"] += 1
        
        # Update metrics
        self.metrics['fragments_retrieved'] += 1
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'fragment_id': fragment_id,
                    'access_count': self.fragments[fragment_id]["access_count"],
                    'timestamp': self.fragments[fragment_id]["last_accessed"]
                }
                metrics.record_metrics("memory_fragment_retrieval", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment retrieval metrics: {e}")
        
        return fragment
    
    def search_fragments_by_frequency(self, target_frequency: float, 
                                    tolerance: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for fragments with frequencies close to the target.
        
        Args:
            target_frequency: Target frequency to search for
            tolerance: Frequency tolerance range
            
        Returns:
            List of matching fragments
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(target_frequency, (int, float)) or target_frequency <= 0:
            raise ValueError(f"Target frequency must be positive, got {target_frequency}")
        
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError(f"Tolerance must be positive, got {tolerance}")
        
        # Search for fragments with frequency in range
        matching_fragments = []
        
        for fragment_id, fragment in self.fragments.items():
            fragment_freq = fragment.get('frequency_hz', 0)
            
            # Check if frequency is within tolerance
            if abs(fragment_freq - target_frequency) <= tolerance:
                # Add a copy of the fragment to results
                matching_fragments.append(fragment.copy())
        
        # Sort by frequency proximity (closest first)
        matching_fragments.sort(key=lambda f: abs(f.get('frequency_hz', 0) - target_frequency))
        
        logger.debug(f"Found {len(matching_fragments)} fragments with frequency {target_frequency} ± {tolerance} Hz")
        
        return matching_fragments
    
    def search_fragments_by_meta(self, meta_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for fragments matching metadata criteria.
        
        Args:
            meta_criteria: Dictionary of metadata criteria to match
            
        Returns:
            List of matching fragments
        """
        if not isinstance(meta_criteria, dict):
            raise ValueError("Meta criteria must be a dictionary")
        
        # Search for fragments matching criteria
        matching_fragments = []
        
        for fragment_id, fragment in self.fragments.items():
            meta_tags = fragment.get('meta_tags', {})
            
            # Check if all criteria match
            criteria_match = True
            for key, value in meta_criteria.items():
                if key not in meta_tags or meta_tags[key] != value:
                    criteria_match = False
                    break
            
            if criteria_match:
                # Add a copy of the fragment to results
                matching_fragments.append(fragment.copy())
        
        logger.debug(f"Found {len(matching_fragments)} fragments matching meta criteria {meta_criteria}")
        
        return matching_fragments
    
    def search_fragments_by_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Search for fragments in a specific brain region.
        
        Args:
            region: Brain region to search
            
        Returns:
            List of fragments in the region
        """
        if not region:
            raise ValueError("Region must be provided")
        
        # Search for fragments in region
        matching_fragments = []
        
        for fragment_id, fragment in self.fragments.items():
            if fragment.get('region') == region:
                # Add a copy of the fragment to results
                matching_fragments.append(fragment.copy())
        
        logger.debug(f"Found {len(matching_fragments)} fragments in region {region}")
        
        return matching_fragments
    
    def convert_fragment_to_node(self, fragment_id: str, 
                               classification: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert a fragment to a node when it has sufficient complexity.
        
        Args:
            fragment_id: ID of the fragment to convert
            classification: Optional classification for the node
            
        Returns:
            Dict with conversion status
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate fragment_id
        if fragment_id not in self.fragments:
            raise ValueError(f"Invalid fragment ID: {fragment_id}")
        
        # Get the fragment
        fragment = self.fragments[fragment_id]
        
        # Check if fragment has sufficient complexity
        complexity = fragment.get('complexity', 0)
        
        if complexity < MEMORY_FRAGMENT_ENERGY_THRESHOLD and classification is None:
            logger.warning(f"Fragment {fragment_id} complexity {complexity} is below threshold "
                         f"{MEMORY_FRAGMENT_ENERGY_THRESHOLD} and no classification provided")
            
            return {
                'success': False,
                'fragment_id': fragment_id,
                'message': f"Insufficient complexity ({complexity}) and no classification provided"
            }
        
        # Generate node ID
        node_id = str(uuid.uuid4())
        creation_time = datetime.now().isoformat()
        
        # Set classification if not provided
        if classification is None:
            classification = self._derive_classification(fragment)
        
        # Create the node
        node = {
            "node_id": node_id,
            "fragment_id": fragment_id,
            "content": fragment["content"],
            "region": fragment["region"],
            "position": fragment["position"],
            "frequency_hz": fragment["frequency_hz"],
            "meta_tags": fragment["meta_tags"].copy(),
            "origin": fragment["origin"],
            "creation_time": creation_time,
            "fragment_creation_time": fragment["creation_time"],
            "last_accessed": creation_time,
            "access_count": 0,
            "energy": fragment["energy"],
            "resonance": fragment["resonance"],
            "coherence": fragment["coherence"],
            "classification": classification,
            "complexity": complexity,
            "associations": [],  # Will be populated from fragment
            "is_active": True
        }
        
        # Copy associations from fragment
        for assoc in fragment['associations']:
            node['associations'].append(assoc.copy())
        
        # Store the node
        self.active_nodes[node_id] = node
        
        # Mark fragment as converted
        self.fragments[fragment_id]['converted_to_node'] = node_id
        
        # Apply node to brain structure if available
        if self.brain_structure is not None:
            self._apply_node_to_brain(node_id, node)
        
        # Update metrics
        self.metrics['nodes_activated'] += 1
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Converted fragment {fragment_id} to node {node_id} with classification '{classification}'")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'fragment_id': fragment_id,
                    'node_id': node_id,
                    'classification': classification,
                    'complexity': complexity,
                    'timestamp': creation_time
                }
                metrics.record_metrics("memory_fragment_to_node", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record fragment conversion metrics: {e}")
        
        return {
            'success': True,
            'fragment_id': fragment_id,
            'node_id': node_id,
            'classification': classification,
            'complexity': complexity
        }
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node data or None if not found
        """
        # Check active nodes first
        if node_id in self.active_nodes:
            # Get the node
            node = self.active_nodes[node_id].copy()
            
            # Update access timestamp and count
            self.active_nodes[node_id]["last_accessed"] = datetime.now().isoformat()
            self.active_nodes[node_id]["access_count"] += 1
            
            return node
        
        # Check inactive nodes
        if node_id in self.inactive_nodes:
            # Get the node
            node = self.inactive_nodes[node_id].copy()
            
            # Update access timestamp and count
            self.inactive_nodes[node_id]["last_accessed"] = datetime.now().isoformat()
            self.inactive_nodes[node_id]["access_count"] += 1
            
            return node
        
        logger.warning(f"Node {node_id} not found")
        return None
    
    def deactivate_node(self, node_id: str) -> Dict[str, Any]:
        """
        Deactivate a node.
        
        Args:
            node_id: ID of the node to deactivate
            
        Returns:
            Dict with deactivation status
            
        Raises:
            ValueError: If node_id is invalid
        """
        if node_id not in self.active_nodes:
            raise ValueError(f"Node {node_id} not found in active nodes")
        
        # Get the node
        node = self.active_nodes[node_id].copy()
        
        # Update status
        node["is_active"] = False
        node["last_accessed"] = datetime.now().isoformat()
        
        # Move to inactive nodes
        self.inactive_nodes[node_id] = node
        
        # Remove from active nodes
        del self.active_nodes[node_id]
        
        # Apply deactivation to brain structure if available
        if self.brain_structure is not None:
            self._apply_node_deactivation(node_id, node)
        
        # Update metrics
        self.metrics['nodes_deactivated'] += 1
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Deactivated node {node_id}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'node_id': node_id,
                    'timestamp': node["last_accessed"]
                }
                metrics.record_metrics("memory_node_deactivation", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record node deactivation metrics: {e}")
        
        return {
            'success': True,
            'node_id': node_id
        }
    
    def activate_node(self, node_id: str) -> Dict[str, Any]:
        """
        Activate an inactive node.
        
        Args:
            node_id: ID of the node to activate
            
        Returns:
            Dict with activation status
            
        Raises:
            ValueError: If node_id is invalid
        """
        if node_id not in self.inactive_nodes:
            raise ValueError(f"Node {node_id} not found in inactive nodes")
        
        # Get the node
        node = self.inactive_nodes[node_id].copy()
        
        # Update status
        node["is_active"] = True
        node["last_accessed"] = datetime.now().isoformat()
        
        # Move to active nodes
        self.active_nodes[node_id] = node
        
        # Remove from inactive nodes
        del self.inactive_nodes[node_id]
        
        # Apply activation to brain structure if available
        if self.brain_structure is not None:
            self._apply_node_activation(node_id, node)
        
        # Update metrics
        self.metrics['nodes_activated'] += 1
        self.last_updated = datetime.now().isoformat()
        
        logger.info(f"Activated node {node_id}")
        
        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                metrics_data = {
                    'system_id': self.system_id,
                    'node_id': node_id,
                    'timestamp': node["last_accessed"]
                }
                metrics.record_metrics("memory_node_activation", metrics_data)
            except Exception as e:
                logger.warning(f"Failed to record node activation metrics: {e}")
        
        return {
            'success': True,
            'node_id': node_id
        }
    
    def find_related_fragments(self, fragment_id: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        Find fragments related to the given fragment through associations.
        
        Args:
            fragment_id: ID of the fragment to find relations for
            max_distance: Maximum association distance to traverse
            
        Returns:
            List of related fragments with distance information
            
        Raises:
            ValueError: If fragment_id is invalid
        """
        if fragment_id not in self.fragments:
            raise ValueError(f"Invalid fragment ID: {fragment_id}")
        
        # Get the fragment
        fragment = self.fragments[fragment_id]
        
        # Initialize with the starting fragment
        related = {
            fragment_id: {
                'distance': 0,
                'path': [fragment_id],
                'data': fragment
            }
        }
        
        # Use breadth-first search to find related fragments
# Use breadth-first search to find related fragments
        queue = [(fragment_id, 0, [fragment_id])]  # (fragment_id, distance, path)
        visited = {fragment_id}
        
        while queue:
            current_id, distance, path = queue.pop(0)
            
            # Stop if we've reached max distance
            if distance >= max_distance:
                continue
            
            # Get current fragment's associations
            current = self.fragments[current_id]
            
            for association in current['associations']:
                neighbor_id = association['fragment_id']
                
                # Skip if already visited
                if neighbor_id in visited:
                    continue
                
                # Mark as visited
                visited.add(neighbor_id)
                
                # Get neighbor fragment
                if neighbor_id in self.fragments:
                    neighbor = self.fragments[neighbor_id]
                    
                    # Add to related fragments
                    related[neighbor_id] = {
                        'distance': distance + 1,
                        'path': path + [neighbor_id],
                        'data': neighbor,
                        'association_strength': association['strength']
                    }
                    
                    # Add to queue for further exploration
                    queue.append((neighbor_id, distance + 1, path + [neighbor_id]))
        
        # Convert to list and sort by distance
        result = list(related.values())
        result.sort(key=lambda x: (x['distance'], -x.get('association_strength', 0)))
        
        logger.debug(f"Found {len(result)} fragments related to {fragment_id} within distance {max_distance}")
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Returns:
            Dict with system metrics
        """
        # Calculate additional metrics
        active_node_count = len(self.active_nodes)
        inactive_node_count = len(self.inactive_nodes)
        fragment_count = len(self.fragments)
        association_count = len(self.associations)
        
        # Combine with tracked metrics
        metrics_data = {
            'system_id': self.system_id,
            'creation_time': self.creation_time,
            'last_updated': self.last_updated,
            'fragment_count': fragment_count,
            'active_node_count': active_node_count,
            'inactive_node_count': inactive_node_count,
            'association_count': association_count,
            'fragments_created': self.metrics['fragments_created'],
            'fragments_retrieved': self.metrics['fragments_retrieved'],
            'associations_created': self.metrics['associations_created'],
            'nodes_activated': self.metrics['nodes_activated'],
            'nodes_deactivated': self.metrics['nodes_deactivated']
        }
        
        return metrics_data
    
    def calculate_memory_coverage(self) -> Dict[str, Any]:
        """
        Calculate memory coverage metrics across brain regions.
        
        Returns:
            Dict with coverage metrics
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return {
                'coverage_available': False,
                'message': "Brain structure not available"
            }
        
        # Calculate fragment distribution by region
        region_fragments = {}
        region_nodes = {}
        
        # Count fragments by region
        for fragment_id, fragment in self.fragments.items():
            region = fragment.get('region')
            if region:
                if region not in region_fragments:
                    region_fragments[region] = 0
                region_fragments[region] += 1
        
        # Count nodes by region
        for node_id, node in self.active_nodes.items():
            region = node.get('region')
            if region:
                if region not in region_nodes:
                    region_nodes[region] = 0
                region_nodes[region] += 1
        
        # Calculate total region cells
        region_cells = {}
        for region_name in self.brain_structure.regions:
            region_indices = np.where(self.brain_structure.region_grid == region_name)
            region_cells[region_name] = len(region_indices[0])
        
        # Calculate coverage
        coverage = {}
        for region_name, cell_count in region_cells.items():
            fragment_count = region_fragments.get(region_name, 0)
            node_count = region_nodes.get(region_name, 0)
            
            coverage[region_name] = {
                'cell_count': cell_count,
                'fragment_count': fragment_count,
                'node_count': node_count,
                'memory_density': (fragment_count + node_count) / cell_count if cell_count > 0 else 0
            }
        
        return {
            'coverage_available': True,
            'coverage': coverage,
            'total_fragments': len(self.fragments),
            'total_nodes': len(self.active_nodes) + len(self.inactive_nodes),
            'active_nodes': len(self.active_nodes),
            'inactive_nodes': len(self.inactive_nodes)
        }
    
    # --- Internal Helper Methods ---
    
    def _find_optimal_position(self, region: str, frequency: float) -> Tuple[int, int, int]:
        """
        Find optimal position for a fragment in the given region.
        
        Args:
            region: Brain region to find position in
            frequency: Frequency of the fragment
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # Validate brain structure
        if self.brain_structure is None:
            logger.warning("Brain structure not available. Using default position (0, 0, 0).")
            return (0, 0, 0)
        
        # Find cells in the region
        region_indices = np.where(self.brain_structure.region_grid == region)
        
        if len(region_indices[0]) == 0:
            logger.warning(f"Region {region} not found in brain structure. Using default position (0, 0, 0).")
            return (0, 0, 0)
        
        # Find cells with matching frequency (or close to it)
        freq_tolerance = 1.0  # Tolerance range in Hz
        
        # Calculate frequency difference for all cells in the region
        freq_diffs = np.zeros_like(self.brain_structure.frequency_grid)
        
        for i in range(len(region_indices[0])):
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            cell_freq = self.brain_structure.frequency_grid[x, y, z]
            
            if cell_freq > 0:
                freq_diffs[x, y, z] = abs(cell_freq - frequency)
            else:
                freq_diffs[x, y, z] = float('inf')  # Infinite difference for zero frequency
        
        # Find cells with frequency difference within tolerance
        close_freq_indices = []
        
        for i in range(len(region_indices[0])):
            x, y, z = region_indices[0][i], region_indices[1][i], region_indices[2][i]
            
            if freq_diffs[x, y, z] <= freq_tolerance:
                # Consider resonance and energy as well
                resonance = self.brain_structure.resonance_grid[x, y, z]
                energy = self.brain_structure.energy_grid[x, y, z]
                
                score = (1.0 - freq_diffs[x, y, z] / freq_tolerance) * 0.6 + resonance * 0.3 + energy * 0.1
                
                close_freq_indices.append((x, y, z, score))
        
        # If no cells with close frequency, use any cell in the region
        if not close_freq_indices:
            # Sample a random cell from the region
            sample_index = np.random.randint(len(region_indices[0]))
            return (
                int(region_indices[0][sample_index]),
                int(region_indices[1][sample_index]),
                int(region_indices[2][sample_index])
            )
        
        # Sort by score (highest first)
        close_freq_indices.sort(key=lambda x: x[3], reverse=True)
        
        # Return the best position
        return (
            int(close_freq_indices[0][0]),
            int(close_freq_indices[0][1]),
            int(close_freq_indices[0][2])
        )
    
    def _calculate_complexity(self, content: Any) -> float:
        """
        Calculate the complexity of a memory fragment.
        
        Args:
            content: Fragment content
            
        Returns:
            Complexity score (0-1)
        """
        # Initialize with base complexity
        complexity = 0.1
        
        # Different calculation based on content type
        if isinstance(content, dict):
            # Count fields and nested fields
            complexity += min(0.5, len(content) * 0.05)
            
            # Add complexity from nested structures
            for value in content.values():
                if isinstance(value, (dict, list)):
                    complexity += self._calculate_complexity(value) * 0.1
        
        elif isinstance(content, list):
            # Count items
            complexity += min(0.5, len(content) * 0.05)
            
            # Add complexity from nested structures
            for item in content:
                if isinstance(item, (dict, list)):
                    complexity += self._calculate_complexity(item) * 0.1
        
        elif isinstance(content, str):
            # Length-based complexity for strings
            complexity += min(0.3, len(content) / 1000)
        
        # Ensure complexity is in [0, 1] range
        return min(1.0, complexity)
    
    def _derive_classification(self, fragment: Dict[str, Any]) -> str:
        """
        Derive a classification for a fragment based on its properties.
        
        Args:
            fragment: Fragment data
            
        Returns:
            Classification string
        """
        # Start with default classification
        classification = "memory"
        
        # Check meta tags for classification hints
        meta_tags = fragment.get('meta_tags', {})
        if 'classification' in meta_tags:
            return meta_tags['classification']
        
        # Check origin for classification hints
        origin = fragment.get('origin', '')
        if origin == 'perception':
            classification = "perception"
        elif origin == 'soul':
            classification = "soul_aspect"
        
        # Check region for classification hints
        region = fragment.get('region', '')
        if region == REGION_TEMPORAL:
            if classification == "memory":
                classification = "temporal_memory"
        elif region == REGION_LIMBIC:
            classification = "emotional_memory"
        elif region == REGION_FRONTAL:
            classification = "cognitive_memory"
        elif region == REGION_OCCIPITAL:
            classification = "visual_memory"
        
        # Check frequency for classification hints
        frequency = fragment.get('frequency_hz', 0)
        if frequency > 0:
            if frequency < 4.0:  # Delta wave
                classification += "_deep"
            elif frequency < 8.0:  # Theta wave
                classification += "_dream"
            elif frequency < 13.0:  # Alpha wave
                classification += "_resting"
            elif frequency < 30.0:  # Beta wave
                classification += "_active"
            else:  # Gamma wave
                classification += "_heightened"
        
        return classification
    
    def _apply_fragment_to_brain(self, fragment_id: str, fragment: Dict[str, Any]) -> None:
        """
        Apply a fragment to the brain structure.
        
        Args:
            fragment_id: ID of the fragment
            fragment: Fragment data
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return
        
        # Get fragment properties
        position = fragment.get('position')
        if position is None:
            return
        
        x, y, z = position
        
        # Validate position is within brain bounds
        if not (0 <= x < self.brain_structure.dimensions[0] and 
               0 <= y < self.brain_structure.dimensions[1] and 
               0 <= z < self.brain_structure.dimensions[2]):
            logger.warning(f"Fragment position {position} is outside brain bounds. Skipping application.")
            return
        
        # Get fragment properties
        frequency = fragment.get('frequency_hz', SCHUMANN_FREQUENCY)
        energy = fragment.get('energy', 0.1)
        resonance = fragment.get('resonance', 0.5)
        coherence = fragment.get('coherence', 0.5)
        
        # Create memory field around fragment position
        radius = 2  # Small radius for fragments
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < self.brain_structure.dimensions[0] and 
                           0 <= ny < self.brain_structure.dimensions[1] and 
                           0 <= nz < self.brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from center
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= radius:
                        # Calculate falloff
                        falloff = 1.0 - (dist / radius)
                        
                        # Apply fragment properties to brain
                        # Set frequency (blend with existing)
                        current_freq = self.brain_structure.frequency_grid[nx, ny, nz]
                        self.brain_structure.frequency_grid[nx, ny, nz] = (
                            current_freq * 0.7 + frequency * falloff * 0.3
                        )
                        
                        # Set mycelial density
                        self.brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                            self.brain_structure.mycelial_density_grid[nx, ny, nz],
                            0.3 * falloff  # Low density for fragments
                        )
                        
                        # Set resonance
                        self.brain_structure.resonance_grid[nx, ny, nz] = max(
                            self.brain_structure.resonance_grid[nx, ny, nz],
                            resonance * falloff
                        )
                        
                        # Set coherence
                        self.brain_structure.coherence_grid[nx, ny, nz] = max(
                            self.brain_structure.coherence_grid[nx, ny, nz],
                            coherence * falloff
                        )
                        
                        # Set energy
                        self.brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                            self.brain_structure.mycelial_energy_grid[nx, ny, nz],
                            energy * falloff
                        )
    
    def _apply_node_to_brain(self, node_id: str, node: Dict[str, Any]) -> None:
        """
        Apply a node to the brain structure.
        
        Args:
            node_id: ID of the node
            node: Node data
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return
        
        # Get node properties
        position = node.get('position')
        if position is None:
            return
        
        x, y, z = position
        
        # Validate position is within brain bounds
        if not (0 <= x < self.brain_structure.dimensions[0] and 
               0 <= y < self.brain_structure.dimensions[1] and 
               0 <= z < self.brain_structure.dimensions[2]):
            logger.warning(f"Node position {position} is outside brain bounds. Skipping application.")
            return
        
        # Get node properties
        frequency = node.get('frequency_hz', SCHUMANN_FREQUENCY)
        energy = node.get('energy', 0.1)
        resonance = node.get('resonance', 0.5)
        coherence = node.get('coherence', 0.5)
        
        # Create memory field around node position
        # Nodes have larger radius and stronger effect than fragments
        radius = 3
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < self.brain_structure.dimensions[0] and 
                           0 <= ny < self.brain_structure.dimensions[1] and 
                           0 <= nz < self.brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from center
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= radius:
                        # Calculate falloff
                        falloff = 1.0 - (dist / radius)
                        
                        # Apply node properties to brain
                        # Set frequency (stronger influence than fragments)
                        current_freq = self.brain_structure.frequency_grid[nx, ny, nz]
                        self.brain_structure.frequency_grid[nx, ny, nz] = (
                            current_freq * 0.5 + frequency * falloff * 0.5
                        )
                        
                        # Set mycelial density
                        self.brain_structure.mycelial_density_grid[nx, ny, nz] = max(
                            self.brain_structure.mycelial_density_grid[nx, ny, nz],
                            0.6 * falloff  # Higher density for nodes
                        )
                        
                        # Set resonance
                        self.brain_structure.resonance_grid[nx, ny, nz] = max(
                            self.brain_structure.resonance_grid[nx, ny, nz],
                            resonance * falloff
                        )
                        
                        # Set coherence
                        self.brain_structure.coherence_grid[nx, ny, nz] = max(
                            self.brain_structure.coherence_grid[nx, ny, nz],
                            coherence * falloff
                        )
                        
                        # Set energy
                        self.brain_structure.mycelial_energy_grid[nx, ny, nz] = max(
                            self.brain_structure.mycelial_energy_grid[nx, ny, nz],
                            energy * falloff
                        )
    
    def _apply_node_deactivation(self, node_id: str, node: Dict[str, Any]) -> None:
        """
        Apply node deactivation to the brain structure.
        
        Args:
            node_id: ID of the node
            node: Node data
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return
        
        # Get node position
        position = node.get('position')
        if position is None:
            return
        
        x, y, z = position
        
        # Validate position is within brain bounds
        if not (0 <= x < self.brain_structure.dimensions[0] and 
               0 <= y < self.brain_structure.dimensions[1] and 
               0 <= z < self.brain_structure.dimensions[2]):
            return
        
        # Reduce energy around node position
        radius = 3
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Skip if outside brain bounds
                    if not (0 <= nx < self.brain_structure.dimensions[0] and 
                           0 <= ny < self.brain_structure.dimensions[1] and 
                           0 <= nz < self.brain_structure.dimensions[2]):
                        continue
                    
                    # Calculate distance from center
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if dist <= radius:
                        # Calculate falloff
                        falloff = 1.0 - (dist / radius)
                        
                        # Reduce energy (deactivation)
                        self.brain_structure.mycelial_energy_grid[nx, ny, nz] *= (1.0 - 0.5 * falloff)
    
    def _apply_node_activation(self, node_id: str, node: Dict[str, Any]) -> None:
        """
        Apply node activation to the brain structure.
        
        Args:
            node_id: ID of the node
            node: Node data
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return
        
        # Get node position
        position = node.get('position')
        if position is None:
            return
        
        # Simply apply the node to brain again
        self._apply_node_to_brain(node_id, node)
    
    def _create_physical_association(self, fragment_id1: str, fragment_id2: str, strength: float) -> None:
        """
        Create a physical association path between two fragments in the brain.
        
        Args:
            fragment_id1: ID of the first fragment
            fragment_id2: ID of the second fragment
            strength: Association strength
        """
        # Skip if brain structure is not available
        if self.brain_structure is None:
            return
        
        # Get fragment positions
        pos1 = self.fragments[fragment_id1].get('position')
        pos2 = self.fragments[fragment_id2].get('position')
        
        if pos1 is None or pos2 is None:
            return
        
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        
        # Calculate direct distance
        direct_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # Skip if too far
        if direct_dist > 50:  # Maximum association distance
            logger.warning(f"Fragment positions too far apart ({direct_dist:.2f} units). "
                         f"Skipping physical association.")
            return
        
        # Create straight-line pathway
        path_points = []
        
        # Number of steps based on distance
        steps = int(direct_dist * 1.5)
        steps = max(5, min(30, steps))  # Between 5 and 30 steps
        
        for i in range(steps + 1):
            t = i / steps
            
            # Linear interpolation
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            z = int(z1 + t * (z2 - z1))
            
            # Constrain to brain bounds
            x = max(0, min(self.brain_structure.dimensions[0] - 1, x))
            y = max(0, min(self.brain_structure.dimensions[1] - 1, y))
            z = max(0, min(self.brain_structure.dimensions[2] - 1, z))
            
            path_points.append((x, y, z))
        
        # Apply association properties along pathway
        for x, y, z in path_points:
            # Enhance mycelial density along pathway
            self.brain_structure.mycelial_density_grid[x, y, z] = max(
                self.brain_structure.mycelial_density_grid[x, y, z],
                0.3 * strength  # Density proportional to association strength
            )
            
            # Add slight energy
            self.brain_structure.mycelial_energy_grid[x, y, z] = max(
                self.brain_structure.mycelial_energy_grid[x, y, z],
                0.05 * strength  # Small energy for association
            )


# --- Helper Functions ---

def add_soul_aspect_as_fragment(memory_system: MemoryFragmentSystem, 
                              brain_structure, 
                              aspect_data: Dict[str, Any],
                              temporal_region_position: Optional[Tuple[int, int, int]] = None) -> str:
    """
    Add a soul aspect as a memory fragment.
    
    Args:
        memory_system: Memory fragment system
        brain_structure: Brain structure
        aspect_data: Soul aspect data
        temporal_region_position: Optional specific position in temporal region
        
    Returns:
        Fragment ID
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Adding soul aspect as memory fragment")
    
    # Validate inputs
    if not isinstance(aspect_data, dict):
        raise ValueError("Invalid aspect_data. Must be a dictionary.")
    
    # Extract aspect properties
    aspect_id = aspect_data.get('id', str(uuid.uuid4()))
    aspect_content = aspect_data.get('content', aspect_data)
    aspect_frequency = aspect_data.get('frequency', SCHUMANN_FREQUENCY)
    
    # Create meta tags from aspect data
    meta_tags = {
        'aspect_id': aspect_id,
        'aspect_type': aspect_data.get('type', 'soul_aspect'),
        'origin': 'soul',
        'classification': 'soul_aspect'
    }
    
    # Find temporal region position if not provided
    if temporal_region_position is None and brain_structure is not None:
        temporal_indices = np.where(brain_structure.region_grid == REGION_TEMPORAL)
        
        if len(temporal_indices[0]) > 0:
            # Find position with good resonance
            best_resonance = -1
            best_position = None
            
            for i in range(min(100, len(temporal_indices[0]))):
                x, y, z = temporal_indices[0][i], temporal_indices[1][i], temporal_indices[2][i]
                resonance = brain_structure.resonance_grid[x, y, z]
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_position = (x, y, z)
            
            if best_position is not None:
                temporal_region_position = best_position
        else:
            logger.warning("Temporal region not found in brain structure.")
    
    # Add fragment to memory system
    fragment_id = memory_system.add_fragment(
        content=aspect_content,
        region=REGION_TEMPORAL,
        position=temporal_region_position,
        frequency=aspect_frequency,
        meta_tags=meta_tags,
        origin='soul'
    )
    
    logger.info(f"Added soul aspect as fragment {fragment_id}")
    
    return fragment_id


def distribute_soul_aspects(memory_system: MemoryFragmentSystem, 
                          brain_structure, 
                          aspects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Distribute soul aspects to memory fragments.
    
    Args:
        memory_system: Memory fragment system
        brain_structure: Brain structure
        aspects: Dictionary of soul aspects
        
    Returns:
        Dict with distribution metrics
    """
    logger.info(f"Distributing {len(aspects)} soul aspects to memory fragments")
    
    # Track distribution
    aspect_fragments = {}
    distribution_count = 0
    
    # Distribute each aspect
    for aspect_id, aspect_data in aspects.items():
        try:
            # Add as fragment
            fragment_id = add_soul_aspect_as_fragment(
                memory_system, brain_structure, aspect_data)
            
            # Store mapping
            aspect_fragments[aspect_id] = fragment_id
            distribution_count += 1
            
        except Exception as e:
            logger.error(f"Error distributing aspect {aspect_id}: {e}", exc_info=True)
    
    # Create associations between related aspects
    association_count = 0
    
    # Use aspect relations if available
    for aspect_id, aspect_data in aspects.items():
        if aspect_id not in aspect_fragments:
            continue
            
        fragment_id = aspect_fragments[aspect_id]
        
        # Check for relations
        relations = aspect_data.get('relations', [])
        
        for relation in relations:
            related_id = relation.get('related_aspect_id')
            
            if related_id in aspect_fragments:
                related_fragment_id = aspect_fragments[related_id]
                
                # Create association
                try:
                    memory_system.associate_fragments(
                        fragment_id, related_fragment_id,
                        strength=relation.get('strength', 0.5)
                    )
                    
                    association_count += 1
                    
                except Exception as e:
                    logger.error(f"Error creating association between {fragment_id} and "
                               f"{related_fragment_id}: {e}", exc_info=True)
    
    logger.info(f"Distributed {distribution_count} soul aspects with {association_count} associations")
    
    # Return metrics
    return {
        'aspects_distributed': distribution_count,
        'total_aspects': len(aspects),
        'associations_created': association_count,
        'aspect_mapping': aspect_fragments
    }


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("MemoryFragmentSystem module test execution")
    
    try:
        # Import necessary modules
        from stage_1.evolve.brain_structure import BrainGrid, create_brain_structure
        
        # Create test brain structure
        brain_structure = create_brain_structure(
            dimensions=(64, 64, 64),  # Smaller for testing
            initialize_regions=True,
            initialize_sound=True
        )
        
        # Create memory fragment system
        memory_system = MemoryFragmentSystem(brain_structure)
        
        # Test functions
        logger.info("Testing MemoryFragmentSystem module functions:")
        
        # 1. Add fragments
        logger.info("1. Testing add_fragment()...")
        fragment_ids = []
        for i in range(5):
            fragment_id = memory_system.add_fragment(
                content=f"Test fragment {i}",
                region=REGION_TEMPORAL,
                meta_tags={'test_index': i}
            )
            fragment_ids.append(fragment_id)
            logger.info(f"   Added fragment {fragment_id}")
        
        # 2. Add stimulus to fragment
        logger.info("2. Testing add_stimulus_to_fragment()...")
        stimulus_result = memory_system.add_stimulus_to_fragment(
            fragment_ids[0], "Additional stimulus data")
        logger.info(f"   Result: complexity={stimulus_result['complexity']}")
        
        # 3. Associate fragments
        logger.info("3. Testing associate_fragments()...")
        association = memory_system.associate_fragments(
            fragment_ids[0], fragment_ids[1], strength=0.8)
        logger.info(f"   Created association {association['association_id']}")
        
        # 4. Convert fragment to node
        logger.info("4. Testing convert_fragment_to_node()...")
        conversion = memory_system.convert_fragment_to_node(
            fragment_ids[0], classification="test_node")
        logger.info(f"   Converted to node {conversion['node_id']}")
        
        # 5. Find related fragments
        logger.info("5. Testing find_related_fragments()...")
        related = memory_system.find_related_fragments(fragment_ids[1])
        logger.info(f"   Found {len(related)} related fragments")
        
        # 6. Get metrics
        logger.info("6. Testing get_metrics()...")
        metrics_data = memory_system.get_metrics()
        logger.info(f"   System metrics: {metrics_data}")
        
    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)