# --- processing.py (V6.0.0 - Integrated Brain Processing) ---

"""
Brain Processing - Integrated with developed brain structure.

Simple and flexible processing for baby brain neuroplastic system.
Monitors synaptic connections, thought patterns, learning, and memory processing.
Works with the developed neural and mycelial networks.
"""

import logging
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import constants
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("BrainProcessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SynapticProcessing:
    """
    Process synaptic connections made by the neural network.
    Monitor patterns, health, energy consumption, and decay.
    """
    
    def __init__(self, brain_structure=None):
        """Initialize synaptic processing with brain structure reference."""
        self.processing_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        
        # Processing state
        self.monitoring_active = False
        self.processing_cycles = 0
        
        # Synaptic health tracking
        self.synaptic_health = {}
        self.connection_strengths = {}
        self.energy_consumption = {}
        
        logger.info(f"Synaptic processing initialized: {self.processing_id[:8]}")
    
    def start_monitoring(self):
        """Start synaptic monitoring."""
        self.monitoring_active = True
        logger.info("Synaptic monitoring started")
    
    def stop_monitoring(self):
        """Stop synaptic monitoring."""
        self.monitoring_active = False
        logger.info("Synaptic monitoring stopped")
    
    def process_synaptic_activity(self) -> Dict[str, Any]:
        """
        Process current synaptic activity patterns.
        Monitor connection health and energy consumption.
        """
        if not self.brain_structure or not self.monitoring_active:
            return {'success': False, 'reason': 'not_ready'}
        
        try:
            self.processing_cycles += 1
            
            # Find all neural nodes with synapses
            neural_nodes = []
            total_synapses = 0
            active_synapses = 0
            total_energy_used = 0.0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural' and 'synapses' in cell_data:
                    neural_nodes.append(position)
                    
                    synapses = cell_data['synapses']
                    total_synapses += len(synapses)
                    
                    # Process each synapse
                    for synapse in synapses:
                        synapse_id = synapse['id']
                        strength = synapse['strength']
                        target_pos = synapse['target']
                        
                        # Check if synapse is active (strength above threshold)
                        if strength > 0.3:
                            active_synapses += 1
                            
                            # Calculate energy consumption
                            energy_used = strength * 0.1  # Energy proportional to strength
                            total_energy_used += energy_used
                            
                            # Update energy consumption tracking
                            self.energy_consumption[synapse_id] = energy_used
                            
                            # Update connection strength tracking
                            self.connection_strengths[synapse_id] = strength
                            
                            # Basic health assessment
                            if strength > 0.7:
                                health = 'strong'
                            elif strength > 0.5:
                                health = 'moderate'
                            else:
                                health = 'weak'
                            
                            self.synaptic_health[synapse_id] = health
            
            # Calculate network statistics
            firing_rate = active_synapses / total_synapses if total_synapses > 0 else 0
            avg_energy_per_synapse = total_energy_used / active_synapses if active_synapses > 0 else 0
            
            # Assess overall network health
            strong_synapses = sum(1 for health in self.synaptic_health.values() if health == 'strong')
            network_health = strong_synapses / len(self.synaptic_health) if self.synaptic_health else 0
            
            processing_result = {
                'success': True,
                'processing_cycle': self.processing_cycles,
                'neural_nodes_count': len(neural_nodes),
                'total_synapses': total_synapses,
                'active_synapses': active_synapses,
                'firing_rate': firing_rate,
                'total_energy_used': total_energy_used,
                'avg_energy_per_synapse': avg_energy_per_synapse,
                'network_health_score': network_health,
                'strong_synapses': strong_synapses,
                'processing_time': datetime.now().isoformat()
            }
            
            logger.debug(f"Synaptic processing cycle {self.processing_cycles}: "
                        f"{active_synapses}/{total_synapses} active synapses")
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Synaptic processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def detect_decay_candidates(self) -> List[str]:
        """
        Detect synapses that are candidates for decay.
        Returns list of synapse IDs that should be weakened.
        """
        decay_candidates = []
        
        for synapse_id, strength in self.connection_strengths.items():
            energy_usage = self.energy_consumption.get(synapse_id, 0)
            
            # Decay if weak and low energy usage
            if strength < 0.4 and energy_usage < 0.05:
                decay_candidates.append(synapse_id)
        
        logger.debug(f"Found {len(decay_candidates)} synapses for decay")
        return decay_candidates
    
    def apply_synaptic_decay(self, decay_candidates: List[str]) -> Dict[str, Any]:
        """
        Apply decay to weak synapses and return energy to mycelial store.
        """
        if not self.brain_structure:
            return {'success': False, 'reason': 'no_brain_structure'}
        
        synapses_decayed = 0
        energy_recovered = 0.0
        
        try:
            # Find and decay synapses
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural' and 'synapses' in cell_data:
                    synapses = cell_data['synapses']
                    
                    for synapse in synapses[:]:  # Copy list to modify during iteration
                        if synapse['id'] in decay_candidates:
                            # Recover energy
                            energy_recovered += synapse['strength'] * 0.05
                            
                            # Remove synapse
                            synapses.remove(synapse)
                            synapses_decayed += 1
                            
                            # Clean up tracking
                            self.connection_strengths.pop(synapse['id'], None)
                            self.energy_consumption.pop(synapse['id'], None)
                            self.synaptic_health.pop(synapse['id'], None)
            
            # Return energy to mycelial storage
            if energy_recovered > 0:
                self._return_energy_to_mycelial_storage(energy_recovered)
            
            decay_result = {
                'success': True,
                'synapses_decayed': synapses_decayed,
                'energy_recovered': energy_recovered,
                'decay_time': datetime.now().isoformat()
            }
            
            logger.info(f"Synaptic decay applied: {synapses_decayed} synapses, "
                       f"{energy_recovered:.3f} energy recovered")
            
            return decay_result
            
        except Exception as e:
            logger.error(f"Synaptic decay failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _return_energy_to_mycelial_storage(self, energy_amount: float):
        """Return recovered energy to mycelial storage."""
        if not self.brain_structure:
            return
        
        # Find brain stem storage cells
        for position, cell_data in self.brain_structure.active_cells.items():
            if cell_data.get('storage_type') == 'mycelial_energy':
                current_energy = cell_data.get('storage_energy', 0.0)
                new_energy = current_energy + (energy_amount * 0.1)  # Distribute energy
                self.brain_structure.set_field_value(position, 'storage_energy', new_energy)


class ThoughtProcessing:
    """
    Process thought formation and patterns in neural network activity.
    Basic pattern recognition for baby brain.
    """
    
    def __init__(self, brain_structure=None):
        """Initialize thought processing."""
        self.processing_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        
        # Thought tracking
        self.active_thoughts = {}
        self.thought_patterns = {}
        self.thought_frequency_map = {}
        
        logger.info(f"Thought processing initialized: {self.processing_id[:8]}")
    
    def detect_thought_formation(self) -> Dict[str, Any]:
        """
        Detect when neural activity forms coherent thought patterns.
        """
        if not self.brain_structure:
            return {'success': False, 'reason': 'no_brain_structure'}
        
        try:
            thoughts_detected = []
            
            # Look for synchronized neural activity
            active_regions = {}
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural' and cell_data.get('active'):
                    region = self.brain_structure.get_region_at_position(position)
                    if region:
                        if region not in active_regions:
                            active_regions[region] = []
                        active_regions[region].append({
                            'position': position,
                            'frequency': cell_data.get('frequency', 0),
                            'energy': cell_data.get('energy', 0)
                        })
            
            # Detect thought patterns
            for region, nodes in active_regions.items():
                if len(nodes) >= 3:  # Need at least 3 nodes for pattern
                    # Check for frequency synchronization
                    frequencies = [node['frequency'] for node in nodes]
                    avg_frequency = sum(frequencies) / len(frequencies)
                    
                    # Check if frequencies are synchronized (within 5Hz)
                    synchronized = all(abs(freq - avg_frequency) < 5.0 for freq in frequencies)
                    
                    if synchronized:
                        thought_id = str(uuid.uuid4())
                        thought = {
                            'id': thought_id,
                            'region': region,
                            'node_count': len(nodes),
                            'avg_frequency': avg_frequency,
                            'total_energy': sum(node['energy'] for node in nodes),
                            'synchronization': True,
                            'formation_time': datetime.now().isoformat()
                        }
                        
                        thoughts_detected.append(thought)
                        self.active_thoughts[thought_id] = thought
                        
                        # Track frequency patterns
                        freq_key = f"{region}_{int(avg_frequency)}"
                        self.thought_frequency_map[freq_key] = self.thought_frequency_map.get(freq_key, 0) + 1
            
            detection_result = {
                'success': True,
                'thoughts_detected': len(thoughts_detected),
                'thoughts': thoughts_detected,
                'active_regions': list(active_regions.keys()),
                'detection_time': datetime.now().isoformat()
            }
            
            if thoughts_detected:
                logger.info(f"Thought formation detected: {len(thoughts_detected)} thoughts")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Thought detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_thought_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in thought formation for learning insights.
        """
        pattern_analysis = {
            'total_thoughts': len(self.active_thoughts),
            'frequency_patterns': self.thought_frequency_map.copy(),
            'regional_activity': {},
            'dominant_frequencies': []
        }
        
        # Analyze regional activity
        for thought in self.active_thoughts.values():
            region = thought['region']
            pattern_analysis['regional_activity'][region] = pattern_analysis['regional_activity'].get(region, 0) + 1
        
        # Find dominant frequencies
        if self.thought_frequency_map:
            sorted_frequencies = sorted(self.thought_frequency_map.items(), key=lambda x: x[1], reverse=True)
            pattern_analysis['dominant_frequencies'] = sorted_frequencies[:5]  # Top 5
        
        logger.debug(f"Thought pattern analysis: {pattern_analysis['total_thoughts']} thoughts analyzed")
        
        return pattern_analysis


class LearningProcessing:
    """
    Process learning events and connection strengthening.
    Monitor how neural connections adapt and strengthen.
    """
    
    def __init__(self, brain_structure=None, memory_system=None):
        """Initialize learning processing."""
        self.processing_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        self.memory_system = memory_system
        
        # Learning tracking
        self.learning_events = {}
        self.connection_changes = {}
        self.strengthened_pathways = set()
        
        logger.info(f"Learning processing initialized: {self.processing_id[:8]}")
    
    def process_learning_event(self, stimulus_type: str, content: str, 
                              involved_regions: List[str] = None) -> Dict[str, Any]:
        """
        Process a learning event and strengthen relevant connections.
        """
        if not self.brain_structure:
            return {'success': False, 'reason': 'no_brain_structure'}
        
        try:
            learning_id = str(uuid.uuid4())
            
            # Create memory fragment for this learning
            if self.memory_system:
                memory_fragment = self.memory_system.add_memory_fragment(content)
                fragment_id = memory_fragment['id']
            else:
                fragment_id = None
            
            # Find neural nodes involved in processing this stimulus
            involved_nodes = []
            if involved_regions:
                for position, cell_data in self.brain_structure.active_cells.items():
                    if cell_data.get('node_type') == 'neural':
                        region = self.brain_structure.get_region_at_position(position)
                        if region in involved_regions:
                            involved_nodes.append(position)
            
            # Strengthen connections between involved nodes
            connections_strengthened = 0
            for node_pos in involved_nodes:
                cell_data = self.brain_structure.active_cells.get(node_pos, {})
                synapses = cell_data.get('synapses', [])
                
                for synapse in synapses:
                    target_pos = synapse['target']
                    if target_pos in involved_nodes:
                        # Strengthen this connection
                        old_strength = synapse['strength']
                        new_strength = min(1.0, old_strength * 1.1)  # 10% increase, max 1.0
                        synapse['strength'] = new_strength
                        
                        connections_strengthened += 1
                        self.connection_changes[synapse['id']] = {
                            'old_strength': old_strength,
                            'new_strength': new_strength,
                            'change': new_strength - old_strength
                        }
                        
                        self.strengthened_pathways.add((node_pos, target_pos))
            
            # Create learning event record
            learning_event = {
                'id': learning_id,
                'stimulus_type': stimulus_type,
                'content': content,
                'involved_regions': involved_regions or [],
                'involved_nodes': len(involved_nodes),
                'connections_strengthened': connections_strengthened,
                'memory_fragment_id': fragment_id,
                'learning_time': datetime.now().isoformat()
            }
            
            self.learning_events[learning_id] = learning_event
            
            learning_result = {
                'success': True,
                'learning_id': learning_id,
                'stimulus_type': stimulus_type,
                'nodes_involved': len(involved_nodes),
                'connections_strengthened': connections_strengthened,
                'memory_created': fragment_id is not None,
                'strengthening_factor': 1.1
            }
            
            logger.info(f"Learning event processed: {stimulus_type}, "
                       f"{connections_strengthened} connections strengthened")
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Learning processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning processing statistics."""
        return {
            'total_learning_events': len(self.learning_events),
            'total_connection_changes': len(self.connection_changes),
            'strengthened_pathways': len(self.strengthened_pathways),
            'avg_connections_per_event': (
                sum(event['connections_strengthened'] for event in self.learning_events.values()) /
                len(self.learning_events) if self.learning_events else 0
            )
        }


class Memory3DProcessing:
    """
    Process memory storage and retrieval in 3D coordinate system.
    Handle memory placement, associations, and pattern recognition.
    """
    
    def __init__(self, brain_structure=None, memory_system=None):
        """Initialize memory processing."""
        self.processing_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        self.memory_system = memory_system
        
        # Memory processing state
        self.processing_active = False
        
        logger.info(f"Memory3D processing initialized: {self.processing_id[:8]}")
    
    def start_processing(self):
        """Start memory processing."""
        self.processing_active = True
        logger.info("Memory3D processing started")
    
    def stop_processing(self):
        """Stop memory processing."""
        self.processing_active = False
        logger.info("Memory3D processing stopped")
    
    def process_incoming_stimulus(self, content: str, context: str = None) -> Dict[str, Any]:
        """
        Process incoming stimulus and store as memory fragment.
        Simple processing for baby brain.
        """
        if not self.memory_system or not self.processing_active:
            return {'success': False, 'reason': 'not_ready'}
        
        try:
            # Classify memory type
            memory_type = self.memory_system.classify_memory_type(content, context)
            
            # Create memory fragment
            fragment = self.memory_system.add_memory_fragment(content, memory_type)
            
            # Check if any existing memories are related
            related_memories = self.memory_system.retrieve_memory(content_search=content[:10])
            
            processing_result = {
                'success': True,
                'fragment_id': fragment['id'],
                'memory_type': memory_type,
                'coordinates': fragment['coordinates'],
                'related_memories': len(related_memories) - 1,  # Exclude the just-created one
                'processing_time': datetime.now().isoformat()
            }
            
            logger.debug(f"Stimulus processed: {memory_type} memory at {fragment['coordinates']}")
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_memory_associations(self) -> Dict[str, Any]:
        """
        Check for memory associations and potential node conversions.
        """
        if not self.memory_system:
            return {'success': False, 'reason': 'no_memory_system'}
        
        try:
            # Check for fragments ready for node conversion
            conversions = self.memory_system.memory_to_node_trigger()
            
            # Run pattern recognition
            patterns = self.memory_system.memory_pattern_recognition()
            
            # Check for memory overflow in regions
            stats = self.memory_system.get_memory_statistics()
            
            association_result = {
                'success': True,
                'conversions_made': len(conversions),
                'patterns_found': patterns['patterns_found'],
                'total_active_memories': stats['total_active'],
                'fragment_to_node_ratio': stats['total_nodes'] / (stats['total_fragments'] + 1),
                'check_time': datetime.now().isoformat()
            }
            
            if conversions:
                logger.info(f"Memory associations processed: {len(conversions)} conversions")
            
            return association_result
            
        except Exception as e:
            logger.error(f"Memory association check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_memory_processing_status(self) -> Dict[str, Any]:
        """Get memory processing status."""
        if not self.memory_system:
            return {'processing_active': False, 'memory_system_connected': False}
        
        stats = self.memory_system.get_memory_statistics()
        
        return {
            'processing_id': self.processing_id,
            'processing_active': self.processing_active,
            'memory_system_connected': True,
            'total_memories': stats['total_active'],
            'memory_fragments': stats['total_fragments'],
            'memory_nodes': stats['total_nodes'],
            'brain_structure_connected': self.brain_structure is not None
        }


# === MAIN EXECUTION ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demonstrate processing systems
    print("=== Brain Processing Systems Demonstration ===")
    
    try:
        print("1. Creating processing systems...")
        
        # Create mock brain structure
        class MockBrainStructure:
            def __init__(self):
                self.active_cells = {
                    (10, 10, 10): {
                        'node_type': 'neural',
                        'frequency': 144.0,
                        'energy': 1.5,
                        'active': True,
                        'synapses': [
                            {'id': 'syn1', 'target': (11, 10, 10), 'strength': 0.8},
                            {'id': 'syn2', 'target': (10, 11, 10), 'strength': 0.6}
                        ]
                    },
                    (11, 10, 10): {
                        'node_type': 'neural',
                        'frequency': 146.0,
                        'energy': 1.3,
                        'active': True,
                        'synapses': []
                    },
                    (50, 50, 25): {
                        'storage_type': 'mycelial_energy',
                        'storage_energy': 100.0
                    }
                }
            
            def get_region_at_position(self, pos):
                return 'frontal'
            
            def set_field_value(self, pos, field, value):
                if pos in self.active_cells:
                    self.active_cells[pos][field] = value
        
        mock_brain = MockBrainStructure()
        
        # Create processing systems
        synaptic_proc = SynapticProcessing(mock_brain)
        thought_proc = ThoughtProcessing(mock_brain)
        learning_proc = LearningProcessing(mock_brain)
        memory_proc = Memory3DProcessing(mock_brain)
        
        print("2. Testing synaptic processing...")
        synaptic_proc.start_monitoring()
        syn_result = synaptic_proc.process_synaptic_activity()
        print(f"   Synaptic result: {syn_result['success']}, "
              f"{syn_result.get('active_synapses', 0)} active synapses")
        
        print("3. Testing thought processing...")
        thought_result = thought_proc.detect_thought_formation()
        print(f"   Thought result: {thought_result['success']}, "
              f"{thought_result.get('thoughts_detected', 0)} thoughts")
        
        print("4. Testing learning processing...")
        learn_result = learning_proc.process_learning_event(
            'visual', 'I see a red circle', ['frontal']
        )
        print(f"   Learning result: {learn_result['success']}, "
              f"{learn_result.get('connections_strengthened', 0)} connections strengthened")
        
        print("5. Testing memory processing...")
        memory_proc.start_processing()
        mem_result = memory_proc.process_incoming_stimulus('I remember seeing colors')
        print(f"   Memory result: {mem_result.get('success', False)}")
        
        print("\nProcessing systems demonstration completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Processing demonstration failed: {e}")

# --- End of processing.py ---





# # --- processing.py ---

# # Imports
# import logging

# # Import constants
# from constants.constants import *

# # --- Logging Setup ---
# logger = logging.getLogger("BrainSeed")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


# class SynapticProcessing:
#     """
#     Process synaptic connections made by the neural network to discover patterns in the neural network
#     activity. Use this to determine decay of synaptic connections and to ensure that the neural network
#     is healthy and not consuming too much energy. log this information and metrics and do calculations
#     to determine health and need for neural network decay of synapsis. if decay modulation is needed,
#     then modulate the decay of synapsis and update the values.return the energy of that connection 
#     to mycelial store
#     """
#     def __init__(self):

# class ThoughtProcessing:
#     """
#     process connections made in various ways to understand how thoughts are formed and processed. at
#     this stage may be very basic check activated nodes for each thought, for frequency matches between 
#     activated nodes, for patterns in the classifications or words that are stored in the node and check if
#     any glyphs associated and what exif or steganography is present or measure the prediction accuracy of 
#     the neural network. how thought and learning differ: thought is a triggered process often the result 
#     of learning or simulation training or from responce to input. Different to the process in memory
#     where we look more at the subconscious memories and classifications
#     """
#     def __init__(self):

# class LearningProcessing:
#     """
#     process connections made in various ways to understand how learning occurs. at
#     this stage may be very basic check activated nodes for each thought, for frequency matches between 
#     activated nodes, for patterns in the classifications or words that are stored in the node and check if
#     any glyphs associated and what exif or steganography is present. how thought and learning differ: learning
#     is a process of creating new connections between nodes and strengthening existing connections.while thought
#     is more about what patterns are activated in the neural network and whether they are divergent or not.
#     """
#     def __init__(self):

# class Memory3DProcessing:
#     """
#     process incoming stimuli or data and store it in the 3D memory coordinate system. unassociated memory is
#     stored in a temporary sub region if the sub region gets full it can overflow to surrounding sub regions.
#     if the memory is associated with a thought or learning then it is stored in the appropriate sub region as per
#     the coordinate classification. it is important to note that the way memory is classified has no correlation
#     to sub region storage. memories get stored in the coordinate that makes sense based on the classification.
#     3d memory classification is defined under system/mycelial_network/memory_3d/memory_structure.however note 
#     this is not the fixed/100% accurate classification and will be refined over time as the brain develops.For now
#     we keep the structure but define very simple dictionaries to map the classification to the sub region.
#     """
#     def __init__(self):

