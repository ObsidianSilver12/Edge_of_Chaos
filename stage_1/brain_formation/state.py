# --- state.py (V6.0.0 - Integrated Brain State Management) ---

"""
Brain State Management - Integrated with developed brain structure.

Simple trigger-based state management for baby brain neuroplastic system.
Monitors brain health, triggers interventions, manages consciousness states.
Works with the developed neural and mycelial networks.
"""

import logging
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime
from enum import Enum

# Import constants
from constants.constants import *

# --- Logging Setup ---
logger = logging.getLogger("BrainState")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ConsciousnessState(Enum):
    """Brain consciousness states."""
    DORMANT = "dormant"
    FORMATION = "formation"
    AWARE_RESTING = "aware_resting"
    AWARE_PROCESSING = "aware_processing"
    ACTIVE_THOUGHT = "active_thought"
    DREAMING = "dreaming"
    LIMINAL = "liminal"
    MEDITATION = "meditation"


class BrainHealthLevel(Enum):
    """Brain health assessment levels."""
    CRITICAL = "critical"
    POOR = "poor"
    MODERATE = "moderate"
    GOOD = "good"
    EXCELLENT = "excellent"


class BrainState:
    """
    Brain State Management - monitor and manage brain state and health.
    """
    
    def __init__(self, brain_structure=None, memory_system=None, processing_systems=None):
        """Initialize brain state management."""
        self.state_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        
        # System references
        self.brain_structure = brain_structure
        self.memory_system = memory_system
        self.processing_systems = processing_systems or {}
        
        # Current state
        self.current_consciousness_state = ConsciousnessState.DORMANT
        self.current_health_level = BrainHealthLevel.MODERATE
        self.last_state_change = datetime.now().isoformat()
        
        # Monitoring flags
        self.monitoring_active = False
        self.auto_triggers_enabled = True
        
        # Health metrics
        self.stress_level = 0.0
        self.energy_level = 1.0
        self.neural_activity = 0.0
        self.mycelial_activity = 0.0
        
        # Trigger thresholds
        self.trigger_thresholds = {
            'stress_critical': 0.8,
            'stress_high': 0.6,
            'energy_low': 0.3,
            'energy_critical': 0.1,
            'neural_overload': 0.9,
            'mycelial_congestion': 0.8
        }
        
        # Trigger history
        self.trigger_history = []
        
        # Intervention functions
        self.intervention_functions = {}
        self._setup_default_interventions()
        
        logger.info(f"Brain state management initialized: {self.state_id[:8]}")
    
    def start_monitoring(self):
        """Start brain state monitoring."""
        self.monitoring_active = True
        logger.info("Brain state monitoring started")
    
    def stop_monitoring(self):
        """Stop brain state monitoring."""
        self.monitoring_active = False
        logger.info("Brain state monitoring stopped")
    
    def _setup_default_interventions(self):
        """Setup default intervention functions."""
        self.intervention_functions = {
            'healing': self.mycelial_network_healing_trigger,
            'mother_influence': self.mycelial_network_mothers_influence_trigger,
            'meditation': self.mycelial_network_meditation_trigger,
            'learning': self.mycelial_network_learning_trigger,
            'energy': self.mycelial_network_energy_trigger,
            'mental_health': self.mycelial_network_mental_health_trigger,
            'dream': self.mycelial_network_dream_trigger,
            'liminal': self.mycelial_network_liminal_trigger,
            'awareness': self.mycelial_network_awareness_trigger,
            'pruning': self.mycelial_network_pruning_trigger
        }
    
    # === MONITORING METHODS ===
    
    def mycelial_neural_network_health_state_monitoring(self) -> Dict[str, Any]:
        """
        Monitor neural network health through mycelial observation.
        Basic monitoring to ensure synapses fire and nodes activate/deactivate properly.
        """
        if not self.brain_structure or not self.monitoring_active:
            return {'success': False, 'reason': 'not_ready'}
        
        try:
            monitoring_events = []
            
            # Monitor neural nodes
            neural_nodes_active = 0
            neural_nodes_total = 0
            synaptic_fires = 0
            synaptic_failures = 0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    neural_nodes_total += 1
                    
                    # Check if node is active
                    if cell_data.get('active', False):
                        neural_nodes_active += 1
                        
                        # Monitor synaptic activity
                        synapses = cell_data.get('synapses', [])
                        for synapse in synapses:
                            if synapse.get('strength', 0) > 0.3:
                                synaptic_fires += 1
                                
                                # Create monitoring event
                                event = {
                                    'event_id': str(uuid.uuid4()),
                                    'coordinates': position,
                                    'event_type': 'synaptic_fire',
                                    'event_time': datetime.now().isoformat(),
                                    'event_source': 'neural_monitoring',
                                    'strength': synapse['strength'],
                                    'target': synapse.get('target')
                                }
                                monitoring_events.append(event)
                            else:
                                synaptic_failures += 1
            
            # Calculate health metrics
            neural_activity_rate = neural_nodes_active / neural_nodes_total if neural_nodes_total > 0 else 0
            synaptic_success_rate = synaptic_fires / (synaptic_fires + synaptic_failures) if (synaptic_fires + synaptic_failures) > 0 else 0
            
            self.neural_activity = neural_activity_rate
            
            monitoring_result = {
                'success': True,
                'neural_nodes_active': neural_nodes_active,
                'neural_nodes_total': neural_nodes_total,
                'neural_activity_rate': neural_activity_rate,
                'synaptic_fires': synaptic_fires,
                'synaptic_failures': synaptic_failures,
                'synaptic_success_rate': synaptic_success_rate,
                'monitoring_events': monitoring_events,
                'monitoring_time': datetime.now().isoformat()
            }
            
            logger.debug(f"Neural network monitoring: {neural_nodes_active}/{neural_nodes_total} nodes active, "
                        f"{synaptic_success_rate:.2f} synaptic success rate")
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Neural network monitoring failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_health_state_monitoring(self) -> Dict[str, Any]:
        """
        Monitor mycelial network self-state including quantum entanglement and energy transfer.
        """
        if not self.brain_structure or not self.monitoring_active:
            return {'success': False, 'reason': 'not_ready'}
        
        try:
            monitoring_events = []
            
            # Monitor mycelial seeds
            mycelial_seeds_active = 0
            mycelial_seeds_total = 0
            quantum_entanglements = 0
            energy_transfers = 0
            field_effects = 0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('seed_type') == 'mycelial':
                    mycelial_seeds_total += 1
                    
                    # Check seed activity
                    seed_energy = cell_data.get('energy', 0)
                    if seed_energy > 0.5:
                        mycelial_seeds_active += 1
                        
                        # Monitor quantum entanglement
                        entangled_with = cell_data.get('entangled_with', [])
                        quantum_entanglements += len(entangled_with)
                        
                        # Test quantum communication
                        for entangled_pos in entangled_with:
                            if entangled_pos in self.brain_structure.active_cells:
                                # Create quantum communication event
                                event = {
                                    'event_id': str(uuid.uuid4()),
                                    'coordinates': position,
                                    'event_type': 'quantum_communication',
                                    'event_time': datetime.now().isoformat(),
                                    'event_source': 'mycelial_monitoring',
                                    'entangled_with': entangled_pos,
                                    'communication_strength': seed_energy
                                }
                                monitoring_events.append(event)
                        
                        # Monitor energy transfer capability
                        connections = cell_data.get('connections', [])
                        for connection in connections:
                            if connection.get('distance', 100) <= 25:  # Close enough for energy transfer
                                energy_transfers += 1
                        
                        # Check field effect
                        if seed_energy > 1.0:  # High energy creates field effect
                            field_effects += 1
                            
                            event = {
                                'event_id': str(uuid.uuid4()),
                                'coordinates': position,
                                'event_type': 'field_effect',
                                'event_time': datetime.now().isoformat(),
                                'event_source': 'mycelial_monitoring',
                                'field_strength': seed_energy
                            }
                            monitoring_events.append(event)
            
            # Calculate mycelial health metrics
            mycelial_activity_rate = mycelial_seeds_active / mycelial_seeds_total if mycelial_seeds_total > 0 else 0
            quantum_stability = quantum_entanglements / mycelial_seeds_active if mycelial_seeds_active > 0 else 0
            
            self.mycelial_activity = mycelial_activity_rate
            
            monitoring_result = {
                'success': True,
                'mycelial_seeds_active': mycelial_seeds_active,
                'mycelial_seeds_total': mycelial_seeds_total,
                'mycelial_activity_rate': mycelial_activity_rate,
                'quantum_entanglements': quantum_entanglements,
                'quantum_stability': quantum_stability,
                'energy_transfers': energy_transfers,
                'field_effects': field_effects,
                'monitoring_events': monitoring_events,
                'monitoring_time': datetime.now().isoformat()
            }
            
            logger.debug(f"Mycelial network monitoring: {mycelial_seeds_active}/{mycelial_seeds_total} seeds active, "
                        f"{quantum_entanglements} entanglements, {field_effects} field effects")
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Mycelial network monitoring failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def neural_network_state_monitoring(self) -> Dict[str, Any]:
        """
        Monitor the consciousness level of neural network (dreaming, awareness, liminal).
        Determined by correlating field changes in sub-regions.
        """
        if not self.brain_structure or not self.monitoring_active:
            return {'success': False, 'reason': 'not_ready'}
        
        try:
            # Monitor field changes across regions
            region_activities = {}
            total_field_strength = 0.0
            
            for region_name, region in self.brain_structure.regions.items():
                region_activity = 0.0
                region_cells = 0
                
                # Check cells in this region
                bounds = region['bounds']
                for x in range(bounds[0], bounds[1]):
                    for y in range(bounds[2], bounds[3]):
                        for z in range(bounds[4], bounds[5]):
                            position = (x, y, z)
                            if position in self.brain_structure.active_cells:
                                cell_data = self.brain_structure.active_cells[position]
                                energy = cell_data.get('energy', 0)
                                frequency = cell_data.get('frequency', 0)
                                
                                # Calculate field activity
                                field_activity = energy * frequency / 1000.0  # Normalize
                                region_activity += field_activity
                                region_cells += 1
                
                if region_cells > 0:
                    region_activities[region_name] = region_activity / region_cells
                    total_field_strength += region_activities[region_name]
                else:
                    region_activities[region_name] = 0.0
            
            # Determine consciousness state based on field patterns
            avg_field_strength = total_field_strength / len(region_activities) if region_activities else 0.0
            
            # Simple consciousness state detection
            if avg_field_strength < 0.1:
                detected_state = ConsciousnessState.DORMANT
            elif avg_field_strength < 0.3:
                detected_state = ConsciousnessState.AWARE_RESTING
            elif avg_field_strength < 0.5:
                detected_state = ConsciousnessState.AWARE_PROCESSING
            elif avg_field_strength < 0.7:
                detected_state = ConsciousnessState.ACTIVE_THOUGHT
            elif avg_field_strength < 0.9:
                detected_state = ConsciousnessState.DREAMING
            else:
                detected_state = ConsciousnessState.LIMINAL
            
            # Update current state if different
            if detected_state != self.current_consciousness_state:
                old_state = self.current_consciousness_state
                self.current_consciousness_state = detected_state
                self.last_state_change = datetime.now().isoformat()
                
                logger.info(f"Consciousness state changed: {old_state.value} -> {detected_state.value}")
            
            monitoring_result = {
                'success': True,
                'current_state': detected_state.value,
                'avg_field_strength': avg_field_strength,
                'region_activities': region_activities,
                'state_change_time': self.last_state_change,
                'monitoring_time': datetime.now().isoformat()
            }
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Neural state monitoring failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # === TRIGGER SYSTEM ===
    
    def mycelial_network_trigger_switch_case(self, trigger_type: str, **kwargs) -> Dict[str, Any]:
        """
        Switch-case trigger system for mycelial network responses.
        """
        if not self.auto_triggers_enabled:
            return {'success': False, 'reason': 'triggers_disabled'}
        
        try:
            # Get appropriate intervention function
            intervention_func = self.intervention_functions.get(trigger_type)
            
            if not intervention_func:
                available_triggers = list(self.intervention_functions.keys())
                logger.warning(f"Unknown trigger type: {trigger_type}. Available: {available_triggers}")
                return {'success': False, 'error': f'unknown_trigger_type: {trigger_type}'}
            
            # Execute intervention
            intervention_result = intervention_func(**kwargs)
            
            # Record trigger event
            trigger_event = {
                'trigger_id': str(uuid.uuid4()),
                'trigger_type': trigger_type,
                'trigger_time': datetime.now().isoformat(),
                'intervention_result': intervention_result,
                'kwargs': kwargs
            }
            
            self.trigger_history.append(trigger_event)
            
            logger.info(f"Trigger executed: {trigger_type}, success: {intervention_result.get('success', False)}")
            
            return {
                'success': True,
                'trigger_type': trigger_type,
                'intervention_result': intervention_result,
                'trigger_id': trigger_event['trigger_id']
            }
            
        except Exception as e:
            logger.error(f"Trigger switch-case failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def neural_network_state_trigger(self) -> Dict[str, Any]:
        """
        Trigger state changes based on neural network conditions.
        Monitors stress, learning needs, energy states, and triggers appropriate responses.
        """
        if not self.monitoring_active:
            return {'success': False, 'reason': 'monitoring_not_active'}
        
        try:
            # Monitor current neural state
            neural_monitoring = self.mycelial_neural_network_health_state_monitoring()
            mycelial_monitoring = self.mycelial_network_health_state_monitoring()
            
            triggers_fired = []
            
            # Check stress levels
            synaptic_success_rate = neural_monitoring.get('synaptic_success_rate', 1.0)
            if synaptic_success_rate < 0.5:
                self.stress_level = 1.0 - synaptic_success_rate
                
                if self.stress_level >= self.trigger_thresholds['stress_critical']:
                    # Critical stress - trigger healing
                    result = self.mycelial_network_trigger_switch_case('healing', stress_level=self.stress_level)
                    triggers_fired.append(('healing', result))
                    
                elif self.stress_level >= self.trigger_thresholds['stress_high']:
                    # High stress - trigger mother influence
                    result = self.mycelial_network_trigger_switch_case('mother_influence', stress_level=self.stress_level)
                    triggers_fired.append(('mother_influence', result))
            
            # Check energy levels
            mycelial_activity = mycelial_monitoring.get('mycelial_activity_rate', 1.0)
            if mycelial_activity < 0.5:
                self.energy_level = mycelial_activity
                
                if self.energy_level <= self.trigger_thresholds['energy_critical']:
                    # Critical energy - trigger energy restoration
                    result = self.mycelial_network_trigger_switch_case('energy', energy_level=self.energy_level)
                    triggers_fired.append(('energy', result))
            
            # Check for neural overload
            neural_activity = neural_monitoring.get('neural_activity_rate', 0.0)
            if neural_activity >= self.trigger_thresholds['neural_overload']:
                # Neural overload - trigger meditation/rest
                result = self.mycelial_network_trigger_switch_case('meditation', neural_activity=neural_activity)
                triggers_fired.append(('meditation', result))
            
            # Check for learning opportunities
            if (self.current_consciousness_state == ConsciousnessState.AWARE_PROCESSING and
                neural_activity > 0.6 and synaptic_success_rate > 0.8):
                # Good conditions for learning
                result = self.mycelial_network_trigger_switch_case('learning', readiness=True)
                triggers_fired.append(('learning', result))
            
            state_trigger_result = {
                'success': True,
                'triggers_fired': len(triggers_fired),
                'trigger_details': triggers_fired,
                'current_stress_level': self.stress_level,
                'current_energy_level': self.energy_level,
                'neural_activity': neural_activity,
                'trigger_time': datetime.now().isoformat()
            }
            
            if triggers_fired:
                logger.info(f"State triggers fired: {[t[0] for t in triggers_fired]}")
            
            return state_trigger_result
            
        except Exception as e:
            logger.error(f"Neural state trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # === INTERVENTION METHODS ===
    
    def mycelial_network_healing_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger healing of neural network through field recalibration and energy replenishment.
        """
        try:
            healing_applied = 0
            energy_restored = 0.0
            
            # Apply healing to stressed neural nodes
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    # Check if node needs healing (low energy or failed synapses)
                    current_energy = cell_data.get('energy', 0)
                    if current_energy < 0.5:
                        # Apply healing energy
                        healing_energy = 0.3
                        new_energy = min(2.0, current_energy + healing_energy)
                        self.brain_structure.set_field_value(position, 'energy', new_energy)
                        
                        energy_restored += healing_energy
                        healing_applied += 1
                        
                        # Recalibrate frequency if needed
                        current_freq = cell_data.get('frequency', 144.0)
                        stable_freq = 144.0  # Standard neural frequency
                        if abs(current_freq - stable_freq) > 5.0:
                            self.brain_structure.set_field_value(position, 'frequency', stable_freq)
            
            # Reduce stress level
            self.stress_level = max(0.0, self.stress_level - 0.3)
            
            healing_result = {
                'success': True,
                'healing_applied': healing_applied,
                'energy_restored': energy_restored,
                'stress_reduction': 0.3,
                'new_stress_level': self.stress_level,
                'healing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Healing applied to {healing_applied} nodes, restored {energy_restored:.2f} energy")
            
            return healing_result
            
        except Exception as e:
            logger.error(f"Healing trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_mothers_influence_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Apply mother's resonance to neural network with recursive frequency adjustments.
        """
        try:
            mothers_love_freq = 528.0  # Hz - love frequency
            mothers_voice_freq = 432.0  # Hz - comfort frequency
            
            stress_level = kwargs.get('stress_level', self.stress_level)
            
            # Determine number of cycles based on stress level
            if stress_level >= 0.8:
                cycles = 12  # Maximum cycles for high stress
            elif stress_level >= 0.6:
                cycles = 8
            elif stress_level >= 0.4:
                cycles = 5
            else:
                cycles = 3
            
            cells_influenced = 0
            total_resonance_applied = 0.0
            
            for cycle in range(cycles):
                cycle_resonance = 0.0
                
                # Apply mother's influence in order: love -> voice -> frequency
                for position, cell_data in self.brain_structure.active_cells.items():
                    if cell_data.get('node_type') == 'neural':
                        current_freq = cell_data.get('frequency', 144.0)
                        
                        # Apply love frequency harmonization
                        love_adjustment = (mothers_love_freq - current_freq) * 0.05  # Gentle adjustment
                        adjusted_freq = current_freq + love_adjustment
                        
                        # Apply voice frequency comfort
                        voice_adjustment = (mothers_voice_freq - adjusted_freq) * 0.03
                        final_freq = adjusted_freq + voice_adjustment
                        
                        # Update frequency
                        self.brain_structure.set_field_value(position, 'frequency', final_freq)
                        self.brain_structure.set_field_value(position, 'mother_influence', True)
                        
                        cells_influenced += 1
                        cycle_resonance += abs(love_adjustment) + abs(voice_adjustment)
                
                total_resonance_applied += cycle_resonance
                
                # Stop if good resonance level achieved
                if cycle_resonance < 50.0 and cycle >= 3:  # Minimum 3 cycles
                    break
            
            # Reduce stress significantly
            stress_reduction = min(0.5, total_resonance_applied / 1000.0)
            self.stress_level = max(0.0, self.stress_level - stress_reduction)
            
            mothers_influence_result = {
                'success': True,
                'cycles_applied': cycle + 1,
                'cells_influenced': cells_influenced,
                'total_resonance_applied': total_resonance_applied,
                'stress_reduction': stress_reduction,
                'new_stress_level': self.stress_level,
                'mothers_love_freq': mothers_love_freq,
                'mothers_voice_freq': mothers_voice_freq,
                'influence_time': datetime.now().isoformat()
            }
            
            logger.info(f"Mother's influence applied over {cycle + 1} cycles to {cells_influenced} cells")
            
            return mothers_influence_result
            
        except Exception as e:
            logger.error(f"Mother's influence trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_meditation_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger meditation state to calm overactive neural networks.
        """
        try:
            # Shift to meditation state
            old_state = self.current_consciousness_state
            self.current_consciousness_state = ConsciousnessState.MEDITATION
            self.last_state_change = datetime.now().isoformat()
            
            # Reduce neural activity
            nodes_calmed = 0
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural' and cell_data.get('active', False):
                    # Reduce frequency for calming
                    current_freq = cell_data.get('frequency', 144.0)
                    meditation_freq = current_freq * 0.7  # Reduce by 30%
                    self.brain_structure.set_field_value(position, 'frequency', meditation_freq)
                    
                    # Reduce energy consumption
                    current_energy = cell_data.get('energy', 1.0)
                    calmed_energy = current_energy * 0.8  # Reduce by 20%
                    self.brain_structure.set_field_value(position, 'energy', calmed_energy)
                    
                    nodes_calmed += 1
            
            meditation_result = {
                'success': True,
                'old_state': old_state.value,
                'new_state': self.current_consciousness_state.value,
                'nodes_calmed': nodes_calmed,
                'meditation_time': datetime.now().isoformat()
            }
            
            logger.info(f"Meditation state triggered: {old_state.value} -> {self.current_consciousness_state.value}")
            
            return meditation_result
            
        except Exception as e:
            logger.error(f"Meditation trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_learning_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger learning state to enhance neural plasticity.
        """
        try:
            readiness = kwargs.get('readiness', False)
            
            if not readiness:
                return {'success': False, 'reason': 'not_ready_for_learning'}
            
            # Enhance neural plasticity
            nodes_enhanced = 0
            synapses_strengthened = 0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    # Increase energy for learning
                    current_energy = cell_data.get('energy', 1.0)
                    learning_energy = min(2.0, current_energy * 1.2)  # 20% boost, max 2.0
                    self.brain_structure.set_field_value(position, 'energy', learning_energy)
                    
                    # Strengthen synapses
                    synapses = cell_data.get('synapses', [])
                    for synapse in synapses:
                        if synapse.get('strength', 0) > 0.5:  # Only strengthen good synapses
                            old_strength = synapse['strength']
                            new_strength = min(1.0, old_strength * 1.1)  # 10% boost, max 1.0
                            synapse['strength'] = new_strength
                            synapses_strengthened += 1
                    
                    nodes_enhanced += 1
            
            learning_result = {
                'success': True,
                'nodes_enhanced': nodes_enhanced,
                'synapses_strengthened': synapses_strengthened,
                'enhancement_factor': 1.2,
                'learning_time': datetime.now().isoformat()
            }
            
            logger.info(f"Learning triggered: {nodes_enhanced} nodes enhanced, {synapses_strengthened} synapses strengthened")
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Learning trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_energy_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger energy restoration from mycelial storage.
        """
        try:
            energy_level = kwargs.get('energy_level', self.energy_level)
            
            # Find mycelial energy storage
            storage_cells = []
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('storage_type') == 'mycelial_energy':
                    storage_cells.append((position, cell_data))
            
            if not storage_cells:
                return {'success': False, 'reason': 'no_energy_storage_found'}
            
            # Calculate energy needed
            energy_deficit = 1.0 - energy_level
            total_energy_available = sum(cell[1].get('storage_energy', 0) for cell in storage_cells)
            
            if total_energy_available < energy_deficit:
                return {'success': False, 'reason': 'insufficient_stored_energy'}
            
            # Distribute energy to neural nodes
            energy_distributed = 0.0
            nodes_energized = 0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural' and energy_distributed < energy_deficit:
                    current_energy = cell_data.get('energy', 0)
                    if current_energy < 1.0:
                        energy_boost = min(0.2, energy_deficit - energy_distributed)
                        new_energy = min(2.0, current_energy + energy_boost)
                        self.brain_structure.set_field_value(position, 'energy', new_energy)
                        
                        energy_distributed += energy_boost
                        nodes_energized += 1
            
            # Deduct energy from storage
            energy_per_storage = energy_distributed / len(storage_cells)
            for position, cell_data in storage_cells:
                current_storage = cell_data.get('storage_energy', 0)
                new_storage = max(0.0, current_storage - energy_per_storage)
                self.brain_structure.set_field_value(position, 'storage_energy', new_storage)
            
            # Update energy level
            self.energy_level = min(1.0, self.energy_level + energy_distributed)
            
            energy_result = {
                'success': True,
                'energy_distributed': energy_distributed,
                'nodes_energized': nodes_energized,
                'storage_cells_used': len(storage_cells),
                'new_energy_level': self.energy_level,
                'energy_time': datetime.now().isoformat()
            }
            
            logger.info(f"Energy restored: {energy_distributed:.3f} to {nodes_energized} nodes")
            
            return energy_result
            
        except Exception as e:
            logger.error(f"Energy trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_mental_health_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger mental health maintenance and balance.
        """
        try:
            # Balance neural frequencies across regions
            region_frequencies = {}
            nodes_balanced = 0
            
            # Calculate average frequency per region
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    region = self.brain_structure.get_region_at_position(position)
                    if region:
                        if region not in region_frequencies:
                            region_frequencies[region] = []
                        region_frequencies[region].append(cell_data.get('frequency', 144.0))
            
            # Calculate balanced frequencies
            for region, frequencies in region_frequencies.items():
                if frequencies:
                    avg_freq = sum(frequencies) / len(frequencies)
                    
                    # Apply gentle balancing to outliers
                    for position, cell_data in self.brain_structure.active_cells.items():
                        if (cell_data.get('node_type') == 'neural' and 
                            self.brain_structure.get_region_at_position(position) == region):
                            
                            current_freq = cell_data.get('frequency', 144.0)
                            if abs(current_freq - avg_freq) > 10.0:  # Outlier
                                balanced_freq = current_freq + (avg_freq - current_freq) * 0.1  # Gentle adjustment
                                self.brain_structure.set_field_value(position, 'frequency', balanced_freq)
                                nodes_balanced += 1
            
            mental_health_result = {
                'success': True,
                'regions_balanced': len(region_frequencies),
                'nodes_balanced': nodes_balanced,
                'region_frequencies': {k: sum(v)/len(v) for k, v in region_frequencies.items()},
                'mental_health_time': datetime.now().isoformat()
            }
            
            logger.info(f"Mental health trigger: {nodes_balanced} nodes balanced across {len(region_frequencies)} regions")
            
            return mental_health_result
            
        except Exception as e:
            logger.error(f"Mental health trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_dream_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger dream state for memory processing and pattern recognition.
        """
        try:
            # Shift to dream state
            old_state = self.current_consciousness_state
            self.current_consciousness_state = ConsciousnessState.DREAMING
            self.last_state_change = datetime.now().isoformat()
            
            # Run memory pattern recognition if available
            patterns_found = 0
            if self.memory_system:
                try:
                    pattern_result = self.memory_system.memory_pattern_recognition()
                    patterns_found = pattern_result.get('patterns_found', 0)
                except Exception as e:
                    logger.warning(f"Dream pattern recognition failed: {e}")
            
            # Modify neural activity for dream processing
            dream_nodes = 0
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    # Random frequency variations for dream-like activity
                    base_freq = cell_data.get('frequency', 144.0)
                    dream_freq = base_freq * (0.5 + np.random.random() * 0.5)  # 50-100% of base
                    self.brain_structure.set_field_value(position, 'frequency', dream_freq)
                    self.brain_structure.set_field_value(position, 'dream_state', True)
                    dream_nodes += 1
            
            dream_result = {
                'success': True,
                'old_state': old_state.value,
                'new_state': self.current_consciousness_state.value,
                'dream_nodes': dream_nodes,
                'patterns_found': patterns_found,
                'dream_time': datetime.now().isoformat()
            }
            
            logger.info(f"Dream state triggered: {patterns_found} patterns found")
            
            return dream_result
            
        except Exception as e:
            logger.error(f"Dream trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_liminal_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger liminal state for creative and transitional processing.
        """
        try:
            # Shift to liminal state
            old_state = self.current_consciousness_state
            self.current_consciousness_state = ConsciousnessState.LIMINAL
            self.last_state_change = datetime.now().isoformat()
            
            # Enhance cross-regional connections in liminal state
            cross_connections = 0
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    # Increase energy for cross-regional communication
                    current_energy = cell_data.get('energy', 1.0)
                    liminal_energy = min(2.0, current_energy * 1.3)  # 30% boost
                    self.brain_structure.set_field_value(position, 'energy', liminal_energy)
                    
                    # Mark as liminal processing
                    self.brain_structure.set_field_value(position, 'liminal_state', True)
                    cross_connections += 1
            
            liminal_result = {
                'success': True,
                'old_state': old_state.value,
                'new_state': self.current_consciousness_state.value,
                'cross_connections_enhanced': cross_connections,
                'energy_boost_factor': 1.3,
                'liminal_time': datetime.now().isoformat()
            }
            
            logger.info(f"Liminal state triggered: {cross_connections} nodes enhanced for cross-regional processing")
            
            return liminal_result
            
        except Exception as e:
            logger.error(f"Liminal trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_awareness_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger awareness state for conscious processing.
        """
        try:
            # Shift to aware processing state
            old_state = self.current_consciousness_state
            self.current_consciousness_state = ConsciousnessState.AWARE_PROCESSING
            self.last_state_change = datetime.now().isoformat()
            
            # Optimize neural networks for awareness
            aware_nodes = 0
            synapses_optimized = 0
            
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    # Set optimal frequency for awareness
                    awareness_freq = 144.0  # Standard awareness frequency
                    self.brain_structure.set_field_value(position, 'frequency', awareness_freq)
                    
                    # Optimize synaptic strengths
                    synapses = cell_data.get('synapses', [])
                    for synapse in synapses:
                        current_strength = synapse.get('strength', 0.5)
                        if 0.3 < current_strength < 0.9:  # Optimize mid-range synapses
                            optimized_strength = 0.7  # Optimal awareness strength
                            synapse['strength'] = optimized_strength
                            synapses_optimized += 1
                    
                    # Clear special state flags
                    self.brain_structure.set_field_value(position, 'dream_state', False)
                    self.brain_structure.set_field_value(position, 'liminal_state', False)
                    aware_nodes += 1
            
            awareness_result = {
                'success': True,
                'old_state': old_state.value,
                'new_state': self.current_consciousness_state.value,
                'aware_nodes': aware_nodes,
                'synapses_optimized': synapses_optimized,
                'awareness_frequency': 144.0,
                'awareness_time': datetime.now().isoformat()
            }
            
            logger.info(f"Awareness state triggered: {aware_nodes} nodes optimized for conscious processing")
            
            return awareness_result
            
        except Exception as e:
            logger.error(f"Awareness trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def mycelial_network_pruning_trigger(self, **kwargs) -> Dict[str, Any]:
        """
        Trigger pruning of weak connections and inactive memories.
        """
        try:
            # Prune weak synapses
            synapses_pruned = 0
            for position, cell_data in self.brain_structure.active_cells.items():
                if cell_data.get('node_type') == 'neural':
                    synapses = cell_data.get('synapses', [])
                    pruned_synapses = []
                    
                    for synapse in synapses[:]:  # Copy to modify during iteration
                        if synapse.get('strength', 0) < 0.2:  # Weak synapse
                            synapses.remove(synapse)
                            synapses_pruned += 1
                        else:
                            pruned_synapses.append(synapse)
                    
                    # Update synapses list
                    cell_data['synapses'] = pruned_synapses
            
            # Prune memories if memory system available
            memories_pruned = 0
            if self.memory_system:
                try:
                    pruning_result = self.memory_system.memory_pruning()
                    memories_pruned = pruning_result.get('total_pruned', 0)
                except Exception as e:
                    logger.warning(f"Memory pruning failed: {e}")
            
            pruning_result = {
                'success': True,
                'synapses_pruned': synapses_pruned,
                'memories_pruned': memories_pruned,
                'pruning_time': datetime.now().isoformat()
            }
            
            logger.info(f"Pruning completed: {synapses_pruned} synapses, {memories_pruned} memories pruned")
            
            return pruning_result
            
        except Exception as e:
            logger.error(f"Pruning trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def memory_to_node_trigger(self) -> Dict[str, Any]:
        """
        Trigger event for memory fragment to node conversion.
        """
        if not self.memory_system:
            return {'success': False, 'reason': 'no_memory_system'}
        
        try:
            # Check for memory fragments ready for node conversion
            conversions = self.memory_system.memory_to_node_trigger()
            
            conversion_result = {
                'success': True,
                'conversions_made': len(conversions),
                'conversion_details': conversions,
                'conversion_time': datetime.now().isoformat()
            }
            
            if conversions:
                logger.info(f"Memory to node conversions: {len(conversions)} fragments converted")
            
            return conversion_result
            
        except Exception as e:
            logger.error(f"Memory to node trigger failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # === STATUS AND CONTROL METHODS ===
    
    def update_health_level(self):
        """Update overall brain health level based on current metrics."""
        try:
            # Calculate health score
            health_factors = [
                (1.0 - self.stress_level) * 0.3,  # Low stress is good
                self.energy_level * 0.3,          # High energy is good
                self.neural_activity * 0.2,       # Moderate neural activity is good
                self.mycelial_activity * 0.2      # High mycelial activity is good
            ]
            
            health_score = sum(health_factors)
            
            # Determine health level
            if health_score >= 0.9:
                self.current_health_level = BrainHealthLevel.EXCELLENT
            elif health_score >= 0.7:
                self.current_health_level = BrainHealthLevel.GOOD
            elif health_score >= 0.5:
                self.current_health_level = BrainHealthLevel.MODERATE
            elif health_score >= 0.3:
                self.current_health_level = BrainHealthLevel.POOR
            else:
                self.current_health_level = BrainHealthLevel.CRITICAL
            
            return {
                'health_level': self.current_health_level.value,
                'health_score': health_score,
                'health_factors': {
                    'stress_factor': (1.0 - self.stress_level) * 0.3,
                    'energy_factor': self.energy_level * 0.3,
                    'neural_factor': self.neural_activity * 0.2,
                    'mycelial_factor': self.mycelial_activity * 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"Health level update failed: {e}")
            return {'error': str(e)}
    
    def get_brain_state_status(self) -> Dict[str, Any]:
        """Get complete brain state status."""
        # Update health level
        health_info = self.update_health_level()
        
        return {
            'state_id': self.state_id,
            'creation_time': self.creation_time,
            'monitoring_active': self.monitoring_active,
            'auto_triggers_enabled': self.auto_triggers_enabled,
            'current_consciousness_state': self.current_consciousness_state.value,
            'current_health_level': self.current_health_level.value,
            'last_state_change': self.last_state_change,
            'metrics': {
                'stress_level': self.stress_level,
                'energy_level': self.energy_level,
                'neural_activity': self.neural_activity,
                'mycelial_activity': self.mycelial_activity
            },
            'health_info': health_info,
            'trigger_history_count': len(self.trigger_history),
            'available_interventions': list(self.intervention_functions.keys()),
            'brain_structure_connected': self.brain_structure is not None,
            'memory_system_connected': self.memory_system is not None,
            'processing_systems_connected': len(self.processing_systems)
        }


# === MAIN EXECUTION ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demonstrate brain state management
    print("=== Brain State Management Demonstration ===")
    
    try:
        # Create mock brain structure
        class MockBrainStructure:
            def __init__(self):
                self.active_cells = {
                    (10, 10, 10): {
                        'node_type': 'neural',
                        'frequency': 144.0,
                        'energy': 0.8,
                        'active': True,
                        'synapses': [
                            {'id': 'syn1', 'target': (11, 10, 10), 'strength': 0.6},
                            {'id': 'syn2', 'target': (10, 11, 10), 'strength': 0.1}  # Weak synapse
                        ]
                    },
                    (20, 20, 20): {
                        'seed_type': 'mycelial',
                        'frequency': 987.0,
                        'energy': 1.5,
                        'entangled_with': [(25, 25, 25)],
                        'connections': [{'target': (25, 25, 25), 'distance': 15}]
                    },
                    (50, 50, 25): {
                        'storage_type': 'mycelial_energy',
                        'storage_energy': 100.0
                    }
                }
                self.regions = {
                    'frontal': {'bounds': (0, 50, 0, 50, 0, 50)}
                }
            
            def get_region_at_position(self, pos):
                return 'frontal'
            
            def set_field_value(self, pos, field, value):
                if pos in self.active_cells:
                    self.active_cells[pos][field] = value
        
        mock_brain = MockBrainStructure()
        
        print("1. Creating brain state management system...")
        brain_state = BrainState(brain_structure=mock_brain)
        brain_state.start_monitoring()
        
        print("2. Testing neural network monitoring...")
        neural_result = brain_state.mycelial_neural_network_health_state_monitoring()
        print(f"   Neural monitoring: {neural_result['success']}, "
              f"{neural_result.get('neural_nodes_active', 0)} active nodes")
        
        print("3. Testing mycelial network monitoring...")
        mycelial_result = brain_state.mycelial_network_health_state_monitoring()
        print(f"   Mycelial monitoring: {mycelial_result['success']}, "
              f"{mycelial_result.get('mycelial_seeds_active', 0)} active seeds")
        
        print("4. Testing state triggers...")
        trigger_result = brain_state.neural_network_state_trigger()
        print(f"   State triggers: {trigger_result['success']}, "
              f"{trigger_result.get('triggers_fired', 0)} triggers fired")
        
        print("5. Testing individual interventions...")
        
        # Test healing trigger
        healing_result = brain_state.mycelial_network_healing_trigger()
        print(f"   Healing: {healing_result.get('success', False)}")
        
        # Test mother's influence trigger
        influence_result = brain_state.mycelial_network_mothers_influence_trigger(stress_level=0.7)
        print(f"   Mother's influence: {influence_result.get('success', False)}")
        
        # Test pruning trigger
        pruning_result = brain_state.mycelial_network_pruning_trigger()
        print(f"   Pruning: {pruning_result.get('success', False)}")
        
        print("6. Checking final brain state...")
        status = brain_state.get_brain_state_status()
        print(f"   State: {status['current_consciousness_state']}")
        print(f"   Health: {status['current_health_level']}")
        print(f"   Stress: {status['metrics']['stress_level']:.3f}")
        print(f"   Energy: {status['metrics']['energy_level']:.3f}")
        
        print("\nBrain state management demonstration completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Brain state demonstration failed: {e}")

# --- End of state.py ---






























# # --- state.py ---

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


# class BrainState:
#     """
#     Brain State - process of brain development after the initial seed is planted.
#     """
#     def __init__(self):
#         """
#         Initialize the BrainState class.
#         """

#     # Develop Mycelial Network State Monitoring
#     def mycelial_neural_network_health_state_monitoring(self):
#         """
#         develop the mycelial network state monitoring that will monitor the neural network. This will be 
#         basic monitoring at this stage to ensure that synapses fire, that nodes can be activated and deactivated
#         and that the mycelial network is able to control this activation and deactivation.
#         Ensure the monitoring is stored as a list of events with their x y z coordinates, event type,
#         event time, event source and a unique id.
#         """
#     def mycelial_network_health_state_monitoring(self):
#         """
#         develop the mycelial network self state monitoring that will monitor the mycelial network. This will be 
#         basic monitoring at this stage to ensure that quantum entanglement is active and stable, that the energy
#         can be properly transferred from mycelial storage to seed and that there is a field effect within that
#         sub region due to increased energy within the region. Ensure these values are logged and reported in
#         console to ensure that appropriate adjustments can be made. Add a test to see if quantum communication occurs
#         between seeds and that the field effect is stable. Future system state checks will monitor mycelial network
#         activity within regions to determine decay of network /seed. Like nodes become inactive do we do the same for the
#         seeds and mycelial paths and then recalculate the energy being used by the network
#         """
#     def mycelial_network_trigger_switch_case(self):
#         """
#         develop the mycelial triggers for the defined triggers (below). triggers will possibly work on a switch-case 
#         basis to ensure that the correct case is applied depending on the feedback from the neural network or 
#         mycelial self reporting. Ensure testing, logging and error handling is in place for the triggers.
#         This will ensure that the mycelial network can respond to the neural network state and that the neural network
#         can respond to the mycelial network state. This will be basic at this stage to ensure that the mycelial network
#         can respond to the neural network state and that the neural network can respond to the mycelial network state.
#         """
#     def mycelial_network_health_state_triggering_monitoring(self):
#         """
#         develop the mycelial triggers and monitoring and error logging for the defined triggers (below). triggers
#         must be tested. triggers will possibly work on a switch-case basis to ensure that the correct case is applied
#         depending on the feedback from the neural network or mycelial self reporting
#         """
#     def neural_network_state_monitoring():
#         """
#         This function will monitor the level of awareness of the neural network i.e. dreaming, awareness or liminal.
#         It will determine this by monitoring correlating field changes in sub regions. 
#         """

#     def neural_network_state_trigger():
#         """
#         This function will trigger the level of awareness of the neural network i.e. dreaming, awareness or liminal.
#         It will determine if there are issues within the neural network fields or energy states or stress from
#         excessive learning or compute or emotional stress, learning needs, physical stress or spiritual needs
#         and may trigger a state change to replenish energy or to ensure that the neural network does not stagnate when needs
#         are not being met. This means that we must have some basic parameters or rules to follow to ensure a base spiritual, learning,
#         physical, emotional and energy state is maintained. If these are not met then the neural network will trigger a state change.
#         This will be a basic state change to ensure that the neural network can continue to develop and grow.
#         """
    
#     def mycelial_network_healing_trigger():
#         """
#         This function will trigger the healing of the neural network.Healing means that fields may be recalibrated or that
#         energy stores may need to be replenished through standing waves/phi or sacred geometry to ensure a field / sub region
#         is in a healthy state.
#         """
#     def mycelial_network_mothers_influence_trigger():
#         """
#         This function will trigger the mothers resonance on the neural network. This means that the mothers 
#         resonance frequency, voice and love will be applied to the neural network temporarily adjusting vibrations
#         to a higher or lower state in a recursive way. If the mothers influence does not have a positive effect
#         then the healing trigger must be applied.
#         """

#     def mycelial_network_meditation_trigger():
#         """
        
#         """
    
#     def mycelial_network_learning_trigger():
#         """
        
#         """

#     def mycelial_network_energy_trigger():
#         """
        
#         """

#     def mycelial_network_mental_health_trigger():
#         """
        
#         """
    
#     def mycelial_network_spiritual_health_trigger():
#         """
        
#         """

#     def mycelial_network_identity_crisis_trigger():
#         """
        
#         """
    
#     def mycelial_network_morality_trigger():
#         """
        
#         """

#     def mycelial_network_psychic_trigger():
#         """
        
#         """

#     def mycelial_network_psychic_trigger():
#         """
        
#         """

#     def mycelial_network_glyph_trigger():
#         """
        
#         """
#     def mycelial_network_knowledge_trigger():
#         """
        
#         """
#     def mycelial_network_gateway_trigger():
#         """
        
#         """
#     def mycelial_network_memory_trigger():
#         """
        
#         """

#     def mycelial_network_dream_trigger():
#         """
        
#         """

#     def mycelial_network_liminal_trigger():
#         """
        
#         """

#     def mycelial_network_awareness_trigger():
#         """
        
#         """

#     def mycelial_network_pruning_trigger():
#         """
        
#         """

#     def memory_to_node_trigger():
#         """
#         the trigger event for monitoring when a memory is classified as a node because it meets this criteria: it has a level 1,2 
#         and 3 classification and is not a node already. it is then assigned a node coordinate and added to the brain grid. the 
#         assignment function is in memory def memory_to_node():
#         """