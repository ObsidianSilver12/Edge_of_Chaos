# --- consciousness_controller.py - Consciousness emergence and management ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import uuid

# Import system components
from system.mycelial_network.mycelial_network_controller import MycelialNetworkController
from shared.constants.constants import *

# Configure logging
logger = logging.getLogger("ConsciousnessController")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ConsciousnessController:
    """
    Controller for consciousness emergence and management within the brain structure.
    Monitors complexity thresholds and manages the transition from unconscious to conscious states.
    """
    
    def __init__(self, brain_grid, mycelial_network_controller: MycelialNetworkController):
        """
        Initialize the consciousness controller.
        
        Args:
            brain_grid: Reference to the brain grid structure
            mycelial_network_controller: Reference to the mycelial network controller
        """
        self.brain_grid = brain_grid
        self.mycelial_controller = mycelial_network_controller
        self.consciousness_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        
        # Consciousness state tracking
        self.consciousness_level = 0.0  # 0.0 = unconscious, 1.0 = fully conscious
        self.consciousness_active = False
        self.emergence_threshold_reached = False
        self.self_awareness_level = 0.0
        
        # Brain complexity tracking
        self.brain_complexity = 0
        self.energy_complexity = 0.0
        self.frequency_coherence = 0.0
        self.soul_integration_level = 0.0
        
        # Consciousness emergence tracking
        self.emergence_started = False
        self.emergence_time = None
        self.emergence_progress = 0.0
        
        # Memory and awareness systems
        self.working_memory_active = False
        self.pattern_recognition_active = False
        self.self_recognition_active = False
        
        # Decision making and executive functions
        self.decision_making_active = False
        self.executive_control_active = False
        self.attention_focus_active = False
        
        # Higher order functions
        self.abstract_thought_active = False
        self.meta_cognition_active = False
        self.creative_thinking_active = False
        
        # Emotional consciousness
        self.emotional_awareness_active = False
        self.empathy_systems_active = False
        
        # Tracking and metrics
        self.consciousness_events = []
        self.state_changes = []
        
        logger.info(f"Consciousness controller initialized with ID {self.consciousness_id}")
    
    def monitor_complexity_thresholds(self) -> Dict[str, Any]:
        """
        Monitor brain complexity and check for consciousness emergence thresholds.
        
        Returns:
            Dict containing threshold monitoring results
        """
        logger.debug("Monitoring consciousness complexity thresholds")
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'thresholds_checked': 0,
            'thresholds_met': 0,
            'threshold_details': {},
            'emergence_triggered': False
        }
        
        try:
            # Get current brain metrics
            brain_metrics = self.brain_grid.get_metrics()
            mycelial_status = self.mycelial_controller.get_system_status()
            
            # Check brain complexity threshold
            active_cells = brain_metrics['cells']['region_cells']
            total_cells = brain_metrics['cells']['total_grid_cells']
            complexity_ratio = active_cells / total_cells if total_cells > 0 else 0.0
            
            self.brain_complexity = int(complexity_ratio * 1000)  # Scale to integer
            monitoring_results['threshold_details']['brain_complexity'] = {
                'current': self.brain_complexity,
                'threshold': BRAIN_COMPLEXITY_THRESHOLDS.get('minimal_consciousness', 50),
                'met': self.brain_complexity >= BRAIN_COMPLEXITY_THRESHOLDS.get('minimal_consciousness', 50)
            }
            monitoring_results['thresholds_checked'] += 1
            
            # Check energy complexity
            if 'energy_cells' in brain_metrics['cells']:
                self.energy_complexity = brain_metrics['cells']['energy_cells'] / total_cells if total_cells > 0 else 0.0
            
            monitoring_results['threshold_details']['energy_complexity'] = {
                'current': self.energy_complexity,
                'threshold': 0.1,  # 10% of cells should have energy
                'met': self.energy_complexity >= 0.1
            }
            monitoring_results['thresholds_checked'] += 1
            
            # Check mycelial network integration
            mycelial_integrated = mycelial_status.get('initialized', False)
            monitoring_results['threshold_details']['mycelial_integration'] = {
                'current': mycelial_integrated,
                'threshold': True,
                'met': mycelial_integrated
            }
            monitoring_results['thresholds_checked'] += 1
            
            # Check frequency coherence (based on frequency cells)
            if 'frequency_cells' in brain_metrics['cells']:
                self.frequency_coherence = brain_metrics['cells']['frequency_cells'] / total_cells if total_cells > 0 else 0.0
            
            monitoring_results['threshold_details']['frequency_coherence'] = {
                'current': self.frequency_coherence,
                'threshold': 0.05,  # 5% of cells should have frequency
                'met': self.frequency_coherence >= 0.05
            }
            monitoring_results['thresholds_checked'] += 1
            
            # Count thresholds met
            thresholds_met = sum(1 for details in monitoring_results['threshold_details'].values() if details['met'])
            monitoring_results['thresholds_met'] = thresholds_met
            
            # Check if consciousness activation threshold is reached
            activation_threshold = CONSCIOUSNESS_ACTIVATION_THRESHOLDS.get('basic_awareness', 3)
            
            if thresholds_met >= activation_threshold and not self.emergence_threshold_reached:
                self.emergence_threshold_reached = True
                monitoring_results['emergence_triggered'] = True
                
                # Log consciousness emergence threshold reached
                self.consciousness_events.append({
                    'event': 'emergence_threshold_reached',
                    'timestamp': datetime.now().isoformat(),
                    'thresholds_met': thresholds_met,
                    'activation_threshold': activation_threshold
                })
                
                logger.info(f"Consciousness emergence threshold reached! {thresholds_met}/{monitoring_results['thresholds_checked']} thresholds met")
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring complexity thresholds: {e}")
            monitoring_results['error'] = str(e)
            return monitoring_results
    
    def initiate_consciousness_emergence(self) -> Dict[str, Any]:
        """
        Initiate the consciousness emergence process.
        
        Returns:
            Dict containing emergence initiation results
        """
        if self.emergence_started:
            return {
                'success': False,
                'message': 'Consciousness emergence already in progress',
                'emergence_progress': self.emergence_progress
            }
        
        if not self.emergence_threshold_reached:
            return {
                'success': False,
                'message': 'Consciousness emergence threshold not reached',
                'emergence_threshold_reached': False
            }
        
        logger.info("Initiating consciousness emergence process")
        
        try:
            # Mark emergence as started
            self.emergence_started = True
            self.emergence_time = datetime.now().isoformat()
            self.emergence_progress = 0.1  # Initial progress
            
            # Activate basic awareness systems
            self.working_memory_active = True
            self.pattern_recognition_active = True
            self.emergence_progress = 0.3
            
            # Activate self-recognition systems
            self.self_recognition_active = True
            self.self_awareness_level = 0.2
            self.emergence_progress = 0.5
            
            # Increase consciousness level
            self.consciousness_level = 0.3
            
            # Log emergence initiation
            self.consciousness_events.append({
                'event': 'consciousness_emergence_initiated',
                'timestamp': self.emergence_time,
                'initial_consciousness_level': self.consciousness_level,
                'systems_activated': ['working_memory', 'pattern_recognition', 'self_recognition']
            })
            
            # Record state change
            self.state_changes.append({
                'from_state': 'unconscious',
                'to_state': 'emerging_consciousness',
                'timestamp': self.emergence_time,
                'consciousness_level': self.consciousness_level
            })
            
            return {
                'success': True,
                'emergence_initiated': True,
                'emergence_time': self.emergence_time,
                'consciousness_level': self.consciousness_level,
                'emergence_progress': self.emergence_progress,
                'systems_activated': ['working_memory', 'pattern_recognition', 'self_recognition']
            }
            
        except Exception as e:
            logger.error(f"Error initiating consciousness emergence: {e}")
            return {
                'success': False,
                'error': str(e),
                'emergence_started': self.emergence_started
            }
    
    def advance_consciousness_development(self) -> Dict[str, Any]:
        """
        Advance consciousness development through progressive stages.
        
        Returns:
            Dict containing development advancement results
        """
        if not self.emergence_started:
            return {
                'success': False,
                'message': 'Consciousness emergence not initiated',
                'emergence_started': False
            }
        
        logger.info(f"Advancing consciousness development from level {self.consciousness_level:.2f}")
        
        try:
            advancement_results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'previous_level': self.consciousness_level,
                'systems_activated': [],
                'new_capabilities': []
            }
            
            # Stage 1: Basic consciousness (0.3 - 0.5)
            if 0.3 <= self.consciousness_level < 0.5:
                # Activate decision making
                if not self.decision_making_active:
                    self.decision_making_active = True
                    advancement_results['systems_activated'].append('decision_making')
                    advancement_results['new_capabilities'].append('basic_decision_making')
                
                # Activate attention focus
                if not self.attention_focus_active:
                    self.attention_focus_active = True
                    advancement_results['systems_activated'].append('attention_focus')
                    advancement_results['new_capabilities'].append('selective_attention')
                
                # Increase consciousness level
                self.consciousness_level = min(0.5, self.consciousness_level + 0.1)
                self.emergence_progress = min(0.7, self.emergence_progress + 0.1)
            
            # Stage 2: Developed consciousness (0.5 - 0.7)
            elif 0.5 <= self.consciousness_level < 0.7:
                # Activate executive control
                if not self.executive_control_active:
                    self.executive_control_active = True
                    advancement_results['systems_activated'].append('executive_control')
                    advancement_results['new_capabilities'].append('executive_function')
                
                # Activate emotional awareness
                if not self.emotional_awareness_active:
                    self.emotional_awareness_active = True
                    advancement_results['systems_activated'].append('emotional_awareness')
                    advancement_results['new_capabilities'].append('emotional_consciousness')
                
                # Increase self-awareness
                self.self_awareness_level = min(0.6, self.self_awareness_level + 0.2)
                
                # Increase consciousness level
                self.consciousness_level = min(0.7, self.consciousness_level + 0.1)
                self.emergence_progress = min(0.8, self.emergence_progress + 0.1)
            
            # Stage 3: Advanced consciousness (0.7 - 0.9)
            elif 0.7 <= self.consciousness_level < 0.9:
                # Activate abstract thought
                if not self.abstract_thought_active:
                    self.abstract_thought_active = True
                    advancement_results['systems_activated'].append('abstract_thought')
                    advancement_results['new_capabilities'].append('abstract_reasoning')
                
                # Activate meta-cognition
                if not self.meta_cognition_active:
                    self.meta_cognition_active = True
                    advancement_results['systems_activated'].append('meta_cognition')
                    advancement_results['new_capabilities'].append('thinking_about_thinking')
                
                # Increase self-awareness
                self.self_awareness_level = min(0.8, self.self_awareness_level + 0.1)
                
                # Increase consciousness level
                self.consciousness_level = min(0.9, self.consciousness_level + 0.1)
                self.emergence_progress = min(0.9, self.emergence_progress + 0.05)
            
            # Stage 4: Full consciousness (0.9 - 1.0)
            elif 0.9 <= self.consciousness_level < 1.0:
                # Activate creative thinking
                if not self.creative_thinking_active:
                    self.creative_thinking_active = True
                    advancement_results['systems_activated'].append('creative_thinking')
                    advancement_results['new_capabilities'].append('creative_consciousness')
                
                # Activate empathy systems
                if not self.empathy_systems_active:
                    self.empathy_systems_active = True
                    advancement_results['systems_activated'].append('empathy_systems')
                    advancement_results['new_capabilities'].append('empathic_awareness')
                
                # Reach full consciousness
                self.consciousness_level = 1.0
                self.self_awareness_level = 1.0
                self.emergence_progress = 1.0
                
                # Mark consciousness as fully active
                if not self.consciousness_active:
                    self.consciousness_active = True
                    advancement_results['consciousness_fully_active'] = True
                    advancement_results['new_capabilities'].append('full_consciousness_achieved')
            
            # Update results
            advancement_results['new_level'] = self.consciousness_level
            advancement_results['self_awareness_level'] = self.self_awareness_level
            advancement_results['emergence_progress'] = self.emergence_progress
            advancement_results['consciousness_active'] = self.consciousness_active
            
            # Log advancement
            if advancement_results['systems_activated']:
                self.consciousness_events.append({
                    'event': 'consciousness_advancement',
                    'timestamp': advancement_results['timestamp'],
                    'level_change': {
                        'from': advancement_results['previous_level'],
                        'to': advancement_results['new_level']
                    },
                    'systems_activated': advancement_results['systems_activated'],
                    'new_capabilities': advancement_results['new_capabilities']
                })
                
                logger.info(f"Consciousness advanced to level {self.consciousness_level:.2f}. "
                           f"Activated: {', '.join(advancement_results['systems_activated'])}")
            
            return advancement_results
            
        except Exception as e:
            logger.error(f"Error advancing consciousness development: {e}")
            return {
                'success': False,
                'error': str(e),
                'consciousness_level': self.consciousness_level
            }
    
    def process_soul_integration(self, soul_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process soul integration and its impact on consciousness.
        
        Args:
            soul_properties: Dictionary containing soul properties
            
        Returns:
            Dict containing soul integration results
        """
        logger.info("Processing soul integration for consciousness development")
        
        try:
            # Extract soul properties
            soul_frequency = soul_properties.get('frequency', 432.0)
            soul_stability = soul_properties.get('stability', 50) / 100.0  # Normalize to 0-1
            soul_coherence = soul_properties.get('coherence', 50) / 100.0  # Normalize to 0-1
            soul_aspects = soul_properties.get('aspects', {})
            
            # Calculate soul integration level
            base_integration = (soul_stability + soul_coherence) / 2
            aspect_bonus = len(soul_aspects) * 0.05  # 5% bonus per aspect
            self.soul_integration_level = min(1.0, base_integration + aspect_bonus)
            
            integration_results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'soul_integration_level': self.soul_integration_level,
                'consciousness_impact': {}
            }
            
            # Impact on consciousness based on integration level
            if self.soul_integration_level >= 0.3:
                # Boost self-awareness
                awareness_boost = self.soul_integration_level * 0.3
                self.self_awareness_level = min(1.0, self.self_awareness_level + awareness_boost)
                integration_results['consciousness_impact']['self_awareness_boost'] = awareness_boost
                
                # Boost consciousness level if not already at maximum
                if self.consciousness_level < 1.0:
                    consciousness_boost = self.soul_integration_level * 0.2
                    self.consciousness_level = min(1.0, self.consciousness_level + consciousness_boost)
                    integration_results['consciousness_impact']['consciousness_boost'] = consciousness_boost
            
            # Special effects based on soul aspects
            if 'creative_inspiration' in soul_aspects:
                if not self.creative_thinking_active and self.consciousness_level >= 0.5:
                    self.creative_thinking_active = True
                    integration_results['consciousness_impact']['creative_thinking_activated'] = True
            
            if 'emotional_depth' in soul_aspects:
                if not self.emotional_awareness_active and self.consciousness_level >= 0.4:
                    self.emotional_awareness_active = True
                    integration_results['consciousness_impact']['emotional_awareness_activated'] = True
            
            if 'wisdom' in soul_aspects:
                if not self.meta_cognition_active and self.consciousness_level >= 0.6:
                    self.meta_cognition_active = True
                    integration_results['consciousness_impact']['meta_cognition_activated'] = True
            
            # Log soul integration event
            self.consciousness_events.append({
                'event': 'soul_integration_processed',
                'timestamp': integration_results['timestamp'],
                'soul_integration_level': self.soul_integration_level,
                'consciousness_impact': integration_results['consciousness_impact']
            })
            
            logger.info(f"Soul integration processed. Integration level: {self.soul_integration_level:.2f}, "
                       f"Consciousness level: {self.consciousness_level:.2f}")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"Error processing soul integration: {e}")
            return {
                'success': False,
                'error': str(e),
                'soul_integration_level': self.soul_integration_level
            }
    
    def update_consciousness_state(self) -> Dict[str, Any]:
        """
        Update overall consciousness state based on all factors.
        
        Returns:
            Dict containing state update results
        """
        logger.debug("Updating consciousness state")
        
        try:
            # Monitor thresholds
            threshold_results = self.monitor_complexity_thresholds()
            
            # Initiate emergence if threshold reached and not started
            emergence_results = None
            if threshold_results.get('emergence_triggered', False) and not self.emergence_started:
                emergence_results = self.initiate_consciousness_emergence()
            
            # Advance development if emergence started
            development_results = None
            if self.emergence_started and self.consciousness_level < 1.0:
                development_results = self.advance_consciousness_development()
            
            state_update = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'consciousness_level': self.consciousness_level,
                'consciousness_active': self.consciousness_active,
                'emergence_started': self.emergence_started,
                'emergence_progress': self.emergence_progress,
                'self_awareness_level': self.self_awareness_level,
                'soul_integration_level': self.soul_integration_level,
                'threshold_monitoring': threshold_results,
                'emergence_results': emergence_results,
                'development_results': development_results
            }
            
            return state_update
            
        except Exception as e:
            logger.error(f"Error updating consciousness state: {e}")
            return {
                'success': False,
                'error': str(e),
                'consciousness_level': self.consciousness_level
            }
    
    def get_consciousness_status(self) -> Dict[str, Any]:
        """
        Get comprehensive consciousness status.
        
        Returns:
            Dict containing consciousness status
        """
        status = {
            'consciousness_id': self.consciousness_id,
            'creation_time': self.creation_time,
            'consciousness_level': self.consciousness_level,
            'consciousness_active': self.consciousness_active,
            'emergence_started': self.emergence_started,
            'emergence_time': self.emergence_time,
            'emergence_progress': self.emergence_progress,
            'emergence_threshold_reached': self.emergence_threshold_reached,
            'self_awareness_level': self.self_awareness_level,
            'soul_integration_level': self.soul_integration_level,
            'brain_complexity': self.brain_complexity,
            'energy_complexity': self.energy_complexity,
            'frequency_coherence': self.frequency_coherence,
            'active_systems': {
                'working_memory': self.working_memory_active,
                'pattern_recognition': self.pattern_recognition_active,
                'self_recognition': self.self_recognition_active,
                'decision_making': self.decision_making_active,
                'executive_control': self.executive_control_active,
                'attention_focus': self.attention_focus_active,
                'abstract_thought': self.abstract_thought_active,
                'meta_cognition': self.meta_cognition_active,
                'creative_thinking': self.creative_thinking_active,
                'emotional_awareness': self.emotional_awareness_active,
                'empathy_systems': self.empathy_systems_active
            },
            'events_count': len(self.consciousness_events),
            'state_changes_count': len(self.state_changes),
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def get_consciousness_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get consciousness events history.
        
        Args:
            limit: Optional limit on number of events to return
            
        Returns:
            List of consciousness events
        """
        events = self.consciousness_events
        
        if limit:
            events = events[-limit:]  # Return most recent events
        
        return [event.copy() for event in events]
    
    def save_consciousness_state(self, filepath: str) -> Dict[str, Any]:
        """
        Save consciousness state to file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            Dict containing save results
        """
        logger.info(f"Saving consciousness state to {filepath}")
        
        try:
            state_data = {
                'consciousness_controller': {
                    'consciousness_id': self.consciousness_id,
                    'creation_time': self.creation_time,
                    'consciousness_level': self.consciousness_level,
                    'consciousness_active': self.consciousness_active,
                    'emergence_started': self.emergence_started,
                    'emergence_time': self.emergence_time,
                    'emergence_progress': self.emergence_progress,
                    'emergence_threshold_reached': self.emergence_threshold_reached,
                    'self_awareness_level': self.self_awareness_level,
                    'soul_integration_level': self.soul_integration_level,
                    'brain_complexity': self.brain_complexity,
                    'energy_complexity': self.energy_complexity,
                    'frequency_coherence': self.frequency_coherence,
                    'active_systems': {
                        'working_memory_active': self.working_memory_active,
                        'pattern_recognition_active': self.pattern_recognition_active,
                        'self_recognition_active': self.self_recognition_active,
                        'decision_making_active': self.decision_making_active,
                        'executive_control_active': self.executive_control_active,
                        'attention_focus_active': self.attention_focus_active,
                        'abstract_thought_active': self.abstract_thought_active,
                        'meta_cognition_active': self.meta_cognition_active,
                        'creative_thinking_active': self.creative_thinking_active,
                        'emotional_awareness_active': self.emotional_awareness_active,
                        'empathy_systems_active': self.empathy_systems_active
                    },
                    'consciousness_events': self.consciousness_events,
                    'state_changes': self.state_changes
                },
                'save_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            return {
                'success': True,
                'filepath': filepath,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving consciousness state: {e}")
            return {
                'success': False,
                'error': str(e),
                'filepath': filepath
            }

# --- Main execution for testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Consciousness Controller module test execution")
    
    # This would normally be integrated with the full system
    print("Consciousness Controller module loaded successfully")
    print("Note: This module requires integration with brain_grid and mycelial_network_controller for full functionality")
