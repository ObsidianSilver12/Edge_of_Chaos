"""
attentional_gating.py - Module for managing attentional gating mechanism.

This module implements the quantum threshold-based attentional gating mechanism
that determines when memory seeds activate based on specific resonance conditions.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AttentionalGating')

class AttentionalGating:
    """
    Manages the attentional gating mechanism that activates seeds 
    when they meet quantum thresholds.
    
    The gating mechanism keeps seeds dormant until specific resonance
    conditions are met, such as activation of a learning pathway or
    reaching a pattern recognition threshold.
    """
    
    def __init__(self):
        """Initialize the attentional gating mechanism."""
        # Dictionary to track active learning pathways
        self.active_pathways = {}
        
        # Dictionary to store dormant seeds by category
        self.dormant_seeds = {}
        
        # Dictionary to store activated seeds
        self.activated_seeds = {}
        
        # Dictionary to store resonance thresholds for different seed types
        self.resonance_thresholds = {
            'memory': 0.65,
            'pattern': 0.70,
            'concept': 0.75,
            'insight': 0.85,
            'creative': 0.60,
            'skill': 0.80,
            'knowledge': 0.72,
            'linguistic': 0.68,
            'spatial': 0.70,
            'emotional': 0.55
        }
        
        # Track activation history
        self.activation_history = []
        
        # Track current system state
        self.system_state = {
            'attention_focus': None,
            'cognitive_load': 0.0,
            'learning_mode': 'passive',
            'activation_threshold_modifier': 0.0
        }
        
        logger.info("Attentional gating mechanism initialized")
    
    def register_seed(self, seed_id, seed_type, metadata=None):
        """
        Register a new seed with the gating mechanism.
        
        Parameters:
            seed_id (str): Unique identifier for the seed
            seed_type (str): Type of seed (memory, pattern, concept, etc.)
            metadata (dict, optional): Additional seed metadata
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if not metadata:
            metadata = {}
        
        # Create basic seed structure
        seed = {
            'id': seed_id,
            'type': seed_type,
            'metadata': metadata,
            'creation_time': datetime.now(),
            'resonance_level': 0.0,
            'activation_threshold': self.resonance_thresholds.get(seed_type, 0.7),
            'activation_attempts': 0,
            'related_pathways': [],
            'quantum_state': 'potential'  # potential, activating, active
        }
        
        # Add type-specific attributes
        if seed_type == 'memory':
            seed['decay_rate'] = metadata.get('decay_rate', 0.01)
            seed['memory_type'] = metadata.get('memory_type', 'semantic')
            seed['emotional_valence'] = metadata.get('emotional_valence', 0.0)
        
        elif seed_type == 'pattern':
            seed['pattern_complexity'] = metadata.get('complexity', 0.5)
            seed['recognition_accuracy'] = metadata.get('accuracy', 0.0)
            seed['similarity_threshold'] = metadata.get('similarity_threshold', 0.7)
        
        elif seed_type == 'concept':
            seed['abstraction_level'] = metadata.get('abstraction_level', 0.5)
            seed['connections'] = metadata.get('connections', [])
            seed['formation_completeness'] = metadata.get('completeness', 0.0)
        
        # Store the seed in dormant seeds by type
        if seed_type not in self.dormant_seeds:
            self.dormant_seeds[seed_type] = {}
        
        self.dormant_seeds[seed_type][seed_id] = seed
        
        logger.info(f"Registered new {seed_type} seed with ID: {seed_id}")
        return True
    
    def register_learning_pathway(self, pathway_id, pathway_type, nodes=None, metadata=None):
        """
        Register a learning pathway for potential seed activation.
        
        Parameters:
            pathway_id (str): Unique identifier for the pathway
            pathway_type (str): Type of learning pathway
            nodes (list, optional): Nodes in the pathway
            metadata (dict, optional): Additional pathway metadata
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if not nodes:
            nodes = []
        
        if not metadata:
            metadata = {}
        
        # Create pathway structure
        pathway = {
            'id': pathway_id,
            'type': pathway_type,
            'nodes': nodes,
            'metadata': metadata,
            'creation_time': datetime.now(),
            'activation_level': 0.0,
            'stability': metadata.get('stability', 0.5),
            'related_seeds': []
        }
        
        # Store pathway
        self.active_pathways[pathway_id] = pathway
        
        logger.info(f"Registered new {pathway_type} learning pathway with ID: {pathway_id}")
        return True
    
    def evaluate_seed_activation(self, seed_id, seed_type, current_conditions):
        """
        Evaluate if a seed meets the quantum threshold for activation.
        
        Parameters:
            seed_id (str): Unique identifier for the seed
            seed_type (str): Type of seed
            current_conditions (dict): Current brain/soul state conditions
            
        Returns:
            tuple: (meets_threshold, resonance_level, factors)
        """
        # Check if seed exists
        if seed_type not in self.dormant_seeds or seed_id not in self.dormant_seeds[seed_type]:
            logger.warning(f"Seed {seed_id} of type {seed_type} not found")
            return False, 0.0, {}
        
        # Get seed
        seed = self.dormant_seeds[seed_type][seed_id]
        
        # Increment activation attempts
        seed['activation_attempts'] += 1
        
        # Calculate base resonance from current conditions
        resonance_factors = self._calculate_resonance_factors(seed, current_conditions)
        
        # Calculate total resonance
        base_resonance = sum(factor['contribution'] for factor in resonance_factors.values())
        
        # Apply system modifiers
        if self.system_state['attention_focus'] == seed_type:
            base_resonance *= 1.2  # 20% boost when attention is focused on this type
        
        # Apply threshold modifier
        modified_threshold = seed['activation_threshold'] + self.system_state['activation_threshold_modifier']
        
        # Determine if threshold is met
        meets_threshold = base_resonance >= modified_threshold
        
        # Store resonance level
        seed['resonance_level'] = base_resonance
        
        # Update quantum state if threshold is met
        if meets_threshold:
            seed['quantum_state'] = 'activating'
        
        logger.info(f"Seed {seed_id} resonance: {base_resonance:.2f}, threshold: {modified_threshold:.2f}, "
                   f"meets threshold: {meets_threshold}")
        
        return meets_threshold, base_resonance, resonance_factors
    
    def _calculate_resonance_factors(self, seed, conditions):
        """
        Calculate resonance factors between a seed and current conditions.
        
        Parameters:
            seed (dict): The seed to evaluate
            conditions (dict): Current brain/soul state conditions
            
        Returns:
            dict: Resonance factors with their contributions
        """
        factors = {}
        
        # Type-specific resonance calculations
        if seed['type'] == 'memory':
            # Frequency resonance
            if 'frequency' in conditions and 'frequency' in seed['metadata']:
                seed_freq = seed['metadata']['frequency']
                condition_freq = conditions['frequency']
                
                # Calculate frequency resonance (simplified)
                freq_ratio = max(seed_freq, condition_freq) / min(seed_freq, condition_freq)
                freq_resonance = 1.0 / (1.0 + abs(freq_ratio - 1.0))
                
                factors['frequency'] = {
                    'description': 'Frequency resonance',
                    'contribution': freq_resonance * 0.3  # 30% weight
                }
            
            # Memory context match
            if 'context' in conditions and 'context' in seed['metadata']:
                context_match = self._calculate_context_similarity(
                    seed['metadata']['context'], 
                    conditions['context']
                )
                
                factors['context'] = {
                    'description': 'Context similarity',
                    'contribution': context_match * 0.4  # 40% weight
                }
            
            # Emotional state match
            if 'emotional_state' in conditions and 'emotional_valence' in seed:
                emotional_match = 1.0 - min(1.0, abs(
                    conditions['emotional_state'] - seed['emotional_valence']
                ))
                
                factors['emotional'] = {
                    'description': 'Emotional state match',
                    'contribution': emotional_match * 0.3  # 30% weight
                }
        
        elif seed['type'] == 'pattern':
            # Pattern recognition match
            if 'pattern_input' in conditions and 'pattern_template' in seed['metadata']:
                pattern_match = self._calculate_pattern_similarity(
                    seed['metadata']['pattern_template'],
                    conditions['pattern_input']
                )
                
                factors['pattern_match'] = {
                    'description': 'Pattern similarity',
                    'contribution': pattern_match * 0.6  # 60% weight
                }
            
            # Processing mode match
            if 'processing_mode' in conditions and 'optimal_processing' in seed['metadata']:
                mode_match = 1.0 if conditions['processing_mode'] == seed['metadata']['optimal_processing'] else 0.3
                
                factors['processing_mode'] = {
                    'description': 'Processing mode compatibility',
                    'contribution': mode_match * 0.2  # 20% weight
                }
            
            # Complexity match
            if 'complexity_tolerance' in conditions and 'pattern_complexity' in seed:
                complexity_match = 1.0 - min(1.0, max(0.0, 
                    seed['pattern_complexity'] - conditions['complexity_tolerance']
                ))
                
                factors['complexity'] = {
                    'description': 'Complexity compatibility',
                    'contribution': complexity_match * 0.2  # 20% weight
                }
        
        elif seed['type'] == 'concept':
            # Concept relevance to current thought
            if 'current_concepts' in conditions and 'related_concepts' in seed['metadata']:
                concept_overlap = self._calculate_concept_overlap(
                    seed['metadata']['related_concepts'],
                    conditions['current_concepts']
                )
                
                factors['concept_relevance'] = {
                    'description': 'Concept relevance',
                    'contribution': concept_overlap * 0.5  # 50% weight
                }
            
            # Abstraction level match
            if 'abstraction_level' in conditions and 'abstraction_level' in seed:
                abstraction_match = 1.0 - min(1.0, abs(
                    conditions['abstraction_level'] - seed['abstraction_level']
                ))
                
                factors['abstraction'] = {
                    'description': 'Abstraction level match',
                    'contribution': abstraction_match * 0.3  # 30% weight
                }
            
            # Formation completeness (concepts need to be sufficiently formed)
            if 'formation_completeness' in seed:
                factors['formation'] = {
                    'description': 'Concept formation completeness',
                    'contribution': seed['formation_completeness'] * 0.2  # 20% weight
                }
        
        # Default factors if none calculated (fallback)
        if not factors:
            factors['default'] = {
                'description': 'Default resonance',
                'contribution': 0.4  # Default resonance
            }
        
        return factors
    
    def _calculate_context_similarity(self, seed_context, current_context):
        """Calculate similarity between memory contexts."""
        # Simplified implementation - would be more sophisticated in full model
        if isinstance(seed_context, dict) and isinstance(current_context, dict):
            # Check location match
            location_match = 1.0 if seed_context.get('location') == current_context.get('location') else 0.0
            
            # Check time period
            time_match = 0.0
            if 'time' in seed_context and 'time' in current_context:
                time_diff = abs(seed_context['time'] - current_context['time'])
                time_match = max(0.0, 1.0 - (time_diff / 86400))  # Within one day = full match
            
            # Check activity match
            activity_match = 1.0 if seed_context.get('activity') == current_context.get('activity') else 0.0
            
            # Weighted average
            return 0.4 * location_match + 0.3 * time_match + 0.3 * activity_match
        
        return 0.2  # Default minimal match
    
    def _calculate_pattern_similarity(self, template, input_pattern):
        """Calculate similarity between pattern template and input."""
        # Simplified implementation
        if isinstance(template, list) and isinstance(input_pattern, list):
            # Direct element comparison
            if len(template) == len(input_pattern):
                matches = sum(1 for a, b in zip(template, input_pattern) if a == b)
                return matches / len(template)
        
        return 0.1  # Default minimal match
    
    def _calculate_concept_overlap(self, seed_concepts, current_concepts):
        """Calculate overlap between concept sets."""
        # Convert to sets if they're lists
        if isinstance(seed_concepts, list) and isinstance(current_concepts, list):
            seed_set = set(seed_concepts)
            current_set = set(current_concepts)
            
            if not seed_set or not current_set:
                return 0.0
            
# Calculate Jaccard similarity
            intersection = len(seed_set.intersection(current_set))
            union = len(seed_set.union(current_set))
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0  # Default no match
    
    def activate_resonant_pathway(self, seed_id, seed_type, pathway_id):
        """
        Activate a resonant learning pathway when conditions are right.
        
        Parameters:
            seed_id (str): Seed to activate
            seed_type (str): Type of seed
            pathway_id (str): Related pathway ID
            
        Returns:
            bool: True if activation successful, False otherwise
        """
        # Check if seed exists
        if seed_type not in self.dormant_seeds or seed_id not in self.dormant_seeds[seed_type]:
            logger.warning(f"Cannot activate seed {seed_id}: not found")
            return False
        
        # Check if pathway exists
        if pathway_id not in self.active_pathways:
            logger.warning(f"Cannot activate pathway {pathway_id}: not found")
            return False
        
        # Get seed and pathway
        seed = self.dormant_seeds[seed_type][seed_id]
        pathway = self.active_pathways[pathway_id]
        
        # Check if seed is in activating state
        if seed['quantum_state'] != 'activating':
            logger.warning(f"Seed {seed_id} not in activating state. Current state: {seed['quantum_state']}")
            return False
        
        # Activate the seed
        seed['quantum_state'] = 'active'
        seed['activation_time'] = datetime.now()
        
        # Add pathway to seed's related pathways
        if pathway_id not in seed['related_pathways']:
            seed['related_pathways'].append(pathway_id)
        
        # Add seed to pathway's related seeds
        if seed_id not in pathway['related_seeds']:
            pathway['related_seeds'].append(seed_id)
        
        # Remove from dormant seeds and add to activated seeds
        self.dormant_seeds[seed_type].pop(seed_id)
        
        if seed_type not in self.activated_seeds:
            self.activated_seeds[seed_type] = {}
        
        self.activated_seeds[seed_type][seed_id] = seed
        
        # Record activation in history
        activation_record = {
            'seed_id': seed_id,
            'seed_type': seed_type,
            'pathway_id': pathway_id,
            'time': datetime.now(),
            'resonance_level': seed['resonance_level'],
            'activation_attempts': seed['activation_attempts']
        }
        
        self.activation_history.append(activation_record)
        
        logger.info(f"Activated {seed_type} seed {seed_id} via pathway {pathway_id}")
        return True
    
    def get_dormant_seeds_by_type(self, seed_type=None):
        """
        Get dormant seeds by type.
        
        Parameters:
            seed_type (str, optional): Type of seeds to retrieve
            
        Returns:
            dict: Dormant seeds, filtered by type if specified
        """
        if seed_type:
            return self.dormant_seeds.get(seed_type, {})
        return self.dormant_seeds
    
    def get_activated_seeds_by_type(self, seed_type=None):
        """
        Get activated seeds by type.
        
        Parameters:
            seed_type (str, optional): Type of seeds to retrieve
            
        Returns:
            dict: Activated seeds, filtered by type if specified
        """
        if seed_type:
            return self.activated_seeds.get(seed_type, {})
        return self.activated_seeds
    
    def set_system_state(self, state_updates):
        """
        Update system state parameters.
        
        Parameters:
            state_updates (dict): State parameters to update
            
        Returns:
            dict: Updated system state
        """
        for key, value in state_updates.items():
            if key in self.system_state:
                self.system_state[key] = value
        
        logger.info(f"Updated system state: {state_updates}")
        return self.system_state
    
    def process_learning_event(self, event):
        """
        Process a learning event that might trigger seed activation.
        
        Parameters:
            event (dict): Learning event data
            
        Returns:
            list: Seeds activated by this event
        """
        activated_seeds = []
        
        # Extract event conditions
        conditions = event.get('conditions', {})
        
        # Set attention focus if specified
        if 'focus' in event:
            self.system_state['attention_focus'] = event['focus']
        
        # Process event for each relevant seed type
        seed_types = event.get('relevant_types', list(self.dormant_seeds.keys()))
        
        for seed_type in seed_types:
            if seed_type not in self.dormant_seeds:
                continue
                
            # Check each dormant seed of this type
            for seed_id, seed in list(self.dormant_seeds[seed_type].items()):
                # Evaluate if seed meets activation threshold
                meets_threshold, resonance, _ = self.evaluate_seed_activation(
                    seed_id, seed_type, conditions
                )
                
                # If threshold met, activate seed with related pathway
                if meets_threshold and event.get('pathway_id'):
                    activation_success = self.activate_resonant_pathway(
                        seed_id, seed_type, event['pathway_id']
                    )
                    
                    if activation_success:
                        activated_seeds.append({
                            'seed_id': seed_id,
                            'seed_type': seed_type,
                            'resonance': resonance
                        })
        
        logger.info(f"Learning event processed, activated {len(activated_seeds)} seeds")
        return activated_seeds
    
    def check_pathways_for_patterns(self, pattern_threshold=0.75):
        """
        Check active learning pathways for emergent patterns.
        
        Parameters:
            pattern_threshold (float): Threshold for pattern detection
            
        Returns:
            list: Detected patterns
        """
        detected_patterns = []
        
        # Skip if no pathways or activated seeds
        if not self.active_pathways or not self.activated_seeds:
            return detected_patterns
        
        # Check for seed clusters in each pathway
        for pathway_id, pathway in self.active_pathways.items():
            if not pathway['related_seeds']:
                continue
                
            # Get all activated seeds related to this pathway
            related_seeds = []
            for seed_type in self.activated_seeds:
                for seed_id, seed in self.activated_seeds[seed_type].items():
                    if pathway_id in seed.get('related_pathways', []):
                        related_seeds.append(seed)
            
            # Skip if not enough seeds
            if len(related_seeds) < 3:
                continue
                
            # Look for patterns in seed activations
            # This is a simplified algorithm - a full implementation would be more sophisticated
            
            # Group seeds by type
            type_groups = {}
            for seed in related_seeds:
                seed_type = seed['type']
                if seed_type not in type_groups:
                    type_groups[seed_type] = []
                type_groups[seed_type].append(seed)
            
            # Check for patterns in each type group
            for seed_type, seeds in type_groups.items():
                if len(seeds) < 3:
                    continue
                
                # Check for temporal sequence pattern (activation timing)
                if seed_type in ['memory', 'concept']:
                    # Sort by activation time
                    sorted_seeds = sorted(seeds, key=lambda s: s.get('activation_time', datetime.min))
                    
                    # Check if activations follow a regular temporal pattern
                    activation_intervals = []
                    for i in range(1, len(sorted_seeds)):
                        if 'activation_time' in sorted_seeds[i] and 'activation_time' in sorted_seeds[i-1]:
                            interval = (sorted_seeds[i]['activation_time'] - 
                                       sorted_seeds[i-1]['activation_time']).total_seconds()
                            activation_intervals.append(interval)
                    
                    if activation_intervals:
                        # Calculate consistency of intervals
                        mean_interval = sum(activation_intervals) / len(activation_intervals)
                        interval_consistency = sum(1.0 - min(1.0, abs(interval - mean_interval) / mean_interval) 
                                               for interval in activation_intervals) / len(activation_intervals)
                        
                        if interval_consistency > pattern_threshold:
                            pattern = {
                                'type': 'temporal_sequence',
                                'seed_type': seed_type,
                                'pathway_id': pathway_id,
                                'consistency': interval_consistency,
                                'seeds': [s['id'] for s in sorted_seeds],
                                'mean_interval': mean_interval
                            }
                            detected_patterns.append(pattern)
                
                # Check for resonance frequency pattern
                if all('resonance_level' in seed for seed in seeds):
                    resonance_values = [seed['resonance_level'] for seed in seeds]
                    mean_resonance = sum(resonance_values) / len(resonance_values)
                    
                    # Calculate if resonance values follow a harmonic pattern
                    resonance_ratios = []
                    for i in range(1, len(resonance_values)):
                        if resonance_values[i-1] > 0:
                            ratio = resonance_values[i] / resonance_values[i-1]
                            resonance_ratios.append(ratio)
                    
                    if resonance_ratios:
                        # Check if ratios cluster around harmonics (1, 2, 3/2, etc.)
                        harmonic_targets = [1.0, 2.0, 1.5, 3.0/2.0, 4.0/3.0, 5.0/3.0]
                        harmonic_matches = 0
                        
                        for ratio in resonance_ratios:
                            for target in harmonic_targets:
                                if abs(ratio - target) < 0.1:
                                    harmonic_matches += 1
                                    break
                        
                        harmonic_consistency = harmonic_matches / len(resonance_ratios)
                        
                        if harmonic_consistency > pattern_threshold:
                            pattern = {
                                'type': 'harmonic_resonance',
                                'seed_type': seed_type,
                                'pathway_id': pathway_id,
                                'consistency': harmonic_consistency,
                                'seeds': [s['id'] for s in seeds],
                                'resonance_values': resonance_values
                            }
                            detected_patterns.append(pattern)
        
        return detected_patterns
    
    def deactivate_seed(self, seed_id, seed_type, reason='decay'):
        """
        Deactivate a previously activated seed.
        
        Parameters:
            seed_id (str): ID of the seed to deactivate
            seed_type (str): Type of the seed
            reason (str): Reason for deactivation
            
        Returns:
            bool: True if deactivation successful, False otherwise
        """
        # Check if seed exists in activated seeds
        if seed_type not in self.activated_seeds or seed_id not in self.activated_seeds[seed_type]:
            logger.warning(f"Cannot deactivate seed {seed_id}: not found in activated seeds")
            return False
        
        # Get seed
        seed = self.activated_seeds[seed_type][seed_id]
        
        # Prepare deactivation record
        deactivation_record = {
            'seed_id': seed_id,
            'seed_type': seed_type,
            'activation_time': seed.get('activation_time', datetime.min),
            'deactivation_time': datetime.now(),
            'reason': reason
        }
        
        # Check if any learning pathways should be updated
        for pathway_id in seed.get('related_pathways', []):
            if pathway_id in self.active_pathways:
                if seed_id in self.active_pathways[pathway_id]['related_seeds']:
                    self.active_pathways[pathway_id]['related_seeds'].remove(seed_id)
        
        # Remove seed from activated seeds
        self.activated_seeds[seed_type].pop(seed_id)
        
        # If seed should return to dormant state (e.g., for reactivation later)
        if reason in ['temporary', 'reset', 'reprocess']:
            seed['quantum_state'] = 'potential'
            seed['resonance_level'] = 0.0
            
            if seed_type not in self.dormant_seeds:
                self.dormant_seeds[seed_type] = {}
            
            self.dormant_seeds[seed_type][seed_id] = seed
        
        logger.info(f"Deactivated {seed_type} seed {seed_id} for reason: {reason}")
        return True
    
    def get_system_metrics(self):
        """
        Get metrics about the attentional gating system.
        
        Returns:
            dict: System metrics
        """
        metrics = {
            'dormant_seeds': {
                'total': sum(len(seeds) for seeds in self.dormant_seeds.values()),
                'by_type': {stype: len(seeds) for stype, seeds in self.dormant_seeds.items()}
            },
            'activated_seeds': {
                'total': sum(len(seeds) for seeds in self.activated_seeds.values()),
                'by_type': {stype: len(seeds) for stype, seeds in self.activated_seeds.items()}
            },
            'pathways': {
                'total': len(self.active_pathways),
                'connection_density': self._calculate_connection_density()
            },
            'activation_history': {
                'total_activations': len(self.activation_history),
                'recent_activations': sum(1 for act in self.activation_history 
                                        if (datetime.now() - act['time']).total_seconds() < 3600)  # Last hour
            },
            'system_state': self.system_state.copy()
        }
        
        # Calculate average resonance levels for activated seeds
        if any(self.activated_seeds.values()):
            resonance_values = []
            for seed_type in self.activated_seeds:
                for seed in self.activated_seeds[seed_type].values():
                    if 'resonance_level' in seed:
                        resonance_values.append(seed['resonance_level'])
            
            if resonance_values:
                metrics['average_resonance'] = sum(resonance_values) / len(resonance_values)
        
        return metrics
    
    def _calculate_connection_density(self):
        """Calculate connection density between pathways and seeds."""
        if not self.active_pathways:
            return 0.0
        
        total_seeds = sum(len(seeds) for seeds in self.activated_seeds.values())
        if total_seeds == 0:
            return 0.0
        
        # Count all pathway-seed connections
        connection_count = 0
        for pathway in self.active_pathways.values():
            connection_count += len(pathway.get('related_seeds', []))
        
        # Maximum possible connections = pathways * seeds
        max_connections = len(self.active_pathways) * total_seeds
        
        return connection_count / max_connections if max_connections > 0 else 0.0


# Factory function to create AttentionalGating
def create_attentional_gating():
    """
    Create a new attentional gating mechanism.
    
    Returns:
        AttentionalGating: A new attentional gating instance
    """
    logger.info("Creating new attentional gating mechanism")
    return AttentionalGating()