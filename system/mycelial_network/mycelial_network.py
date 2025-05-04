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
            union = len