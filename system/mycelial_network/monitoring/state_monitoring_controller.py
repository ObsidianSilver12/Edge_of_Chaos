# --- state_monitoring_controller.py - Basic state monitoring system ---

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger("StateMonitoring")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class StateMonitoringController:
    """
    Controller for monitoring various brain states,
    including awareness, emotions, dreaming, and other operational states.
    """
    
    def __init__(self, brain_grid=None, mycelial_network=None):
        """Initialize the state monitoring controller"""
        self.brain_grid = brain_grid
        self.mycelial_network = mycelial_network
        self.initialized = False
        self.creation_time = datetime.now().isoformat()
        
        # Current states (0-1 scale)
        self.states = {
            "awareness": 0.8,  # Default to good awareness
            "dreaming": 0.0,   # Not dreaming
            "liminal": 0.0,    # Not in liminal state
            "meditation": 0.0, # Not meditating
            "healing": 0.0,    # Default healing processes
            "survival": 0.1,   # Low survival state
            
            # Emotional states
            "emotions": {
                "joy": 0.5,
                "sadness": 0.0,
                "fear": 0.0,
                "anger": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
                "trust": 0.7,
                "anticipation": 0.3
            },
            
            # Physical state
            "physical": {
                "energy": 0.7,
                "health": 0.8,
                "fatigue": 0.1
            },
            
            # Personality aspects (example)
            "personality": {
                "openness": 0.7,
                "conscientiousness": 0.6,
                "extraversion": 0.5,
                "agreeableness": 0.7,
                "neuroticism": 0.3
            }
        }
        
        # Monitoring settings
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
        self.alerts_enabled = True
        self.alert_thresholds = {
            "awareness": 0.3,  # Alert if awareness drops below this
            "survival": 0.8,   # Alert if survival need rises above this
            "emotions.fear": 0.7,  # Alert if fear rises above this
            "emotions.anger": 0.7,  # Alert if anger rises above this
            "physical.energy": 0.2  # Alert if energy drops below this
        }
        
        # State history for tracking changes
        self.state_history = []
        self.alerts_history = []
        
        logger.info("State monitoring controller initialized")
    
    def initialize_monitoring(self):
        """Start active state monitoring"""
        if self.initialized:
            return {"success": False, "error": "Already initialized"}
        
        # Bootstrap initial state
        self._update_awareness_state()
        self._update_emotional_state()
        self._update_physical_state()
        
        # Start monitoring
        self.monitoring_active = True
        self.initialized = True
        
        logger.info("State monitoring started")
        return {
            "success": True,
            "monitoring_active": self.monitoring_active,
            "monitoring_interval": self.monitoring_interval,
            "alerts_enabled": self.alerts_enabled,
            "initial_states": self.states
        }
    
    def _update_awareness_state(self):
        """Update awareness state based on brain activity"""
        # In a full implementation, this would analyze various brain activity patterns
        # For now, use a simpler approach
        
        # If brain grid is available, check energy levels in prefrontal and limbic regions
        if self.brain_grid is not None:
            # Find prefrontal and limbic regions
            pf_indices = np.where(self.brain_grid.sub_region_grid == "prefrontal")
            limbic_indices = np.where(self.brain_grid.region_grid == "limbic")
            
            if len(pf_indices[0]) > 0 and len(limbic_indices[0]) > 0:
                # Check energy levels
                pf_energy = np.mean(self.brain_grid.energy_grid[pf_indices])
                limbic_energy = np.mean(self.brain_grid.energy_grid[limbic_indices])
                
                # Calculate awareness based on energy levels
                pf_factor = min(1.0, pf_energy * 5.0)  # Scale factor
                limbic_factor = min(1.0, limbic_energy * 5.0)  # Scale factor
                
                # Weighted average for awareness
                new_awareness = 0.7 * pf_factor + 0.3 * limbic_factor
                
                # Update state
                self.states["awareness"] = new_awareness
        
        # Add randomness to simulate fluctuations
        fluctuation = np.random.normal(0, 0.05)
        self.states["awareness"] = max(0.0, min(1.0, self.states["awareness"] + fluctuation))
    
    def _update_emotional_state(self):
        """Update emotional state based on limbic activity"""
        # In a full implementation, this would analyze limbic system activity
        # For now, use a simpler approach with some random fluctuations
        
        # Small random changes to emotional states
        for emotion in self.states["emotions"]:
            fluctuation = np.random.normal(0, 0.03)
            current = self.states["emotions"][emotion]
            self.states["emotions"][emotion] = max(0.0, min(1.0, current + fluctuation))
        
        # If brain grid is available, check amygdala for fear/anger
        if self.brain_grid is not None:
            amygdala_indices = np.where(self.brain_grid.sub_region_grid == "amygdala")
            if len(amygdala_indices[0]) > 0:
                amygdala_energy = np.mean(self.brain_grid.energy_grid[amygdala_indices])
                
                # Higher amygdala energy correlates with fear/anger
                if amygdala_energy > 0.5:
                    self.states["emotions"]["fear"] += 0.1
                    self.states["emotions"]["anger"] += 0.1
    
    def _update_physical_state(self):
        """Update physical state based on cerebellum and brain stem"""
        # In a full implementation, this would analyze cerebellum and brain stem activity
        # For now, use a simpler approach with some consistency in fluctuations
        
        # Energy decreases slowly over time (simulating regular energy use)
        self.states["physical"]["energy"] -= 0.01
        self.states["physical"]["fatigue"] += 0.005
        
        # Keep within bounds
        self.states["physical"]["energy"] = max(0.1, min(1.0, self.states["physical"]["energy"]))
        self.states["physical"]["fatigue"] = max(0.0, min(1.0, self.states["physical"]["fatigue"]))
        
        # If brain grid is available, check cerebellum and brain stem
        if self.brain_grid is not None:
            cereb_indices = np.where(self.brain_grid.region_grid == "cerebellum")
            stem_indices = np.where(self.brain_grid.region_grid == "brain_stem")
            
            if len(cereb_indices[0]) > 0 and len(stem_indices[0]) > 0:
                # Get energy levels
                cereb_energy = np.mean(self.brain_grid.energy_grid[cereb_indices])
                stem_energy = np.mean(self.brain_grid.energy_grid[stem_indices])
                
                # Higher brain stem energy improves physical states
                if stem_energy > 0.5:
                    self.states["physical"]["energy"] += 0.02
                    self.states["physical"]["health"] += 0.01
                    self.states["physical"]["fatigue"] -= 0.02

    def update_states(self):
        """Update all states based on current conditions"""
        if not self.initialized or not self.monitoring_active:
            return {"success": False, "error": "Monitoring not active"}
        
        # Store previous state for change detection
        previous_states = self._copy_states()
        
        # Update each state category
        self._update_awareness_state()
        self._update_emotional_state()
        self._update_physical_state()
        
        # Special states updated based on awareness and emotions
        # Dreaming: high when awareness is low
        self.states["dreaming"] = max(0.0, 0.8 - self.states["awareness"])
        
        # Liminal: between awareness and dreaming
        awareness = self.states["awareness"]
        dreaming = self.states["dreaming"]
        if 0.3 < awareness < 0.6 and dreaming > 0.2:
            self.states["liminal"] = 0.7
        else:
            self.states["liminal"] = 0.0
        
        # Meditation: high awareness, low emotional activation
        emotional_activation = sum(self.states["emotions"].values()) / len(self.states["emotions"])
        if awareness > 0.7 and emotional_activation < 0.3:
            self.states["meditation"] = 0.8
        else:
            self.states["meditation"] = 0.0
        
        # Survival: based on fear and physical energy
        fear = self.states["emotions"]["fear"]
        energy = self.states["physical"]["energy"]
        self.states["survival"] = (fear * 0.7) + ((1.0 - energy) * 0.3)
        
        # Healing: increases when awareness is medium and survival is low
        if 0.4 < awareness < 0.7 and self.states["survival"] < 0.2:
            self.states["healing"] += 0.05
        else:
            self.states["healing"] -= 0.01
        
        self.states["healing"] = max(0.0, min(1.0, self.states["healing"]))
        
        # Record state history
        state_record = {
            "timestamp": datetime.now().isoformat(),
            "states": self._copy_states(),
            "changes": self._calculate_changes(previous_states)
        }
        self.state_history.append(state_record)
        
        # Check for alerts
        if self.alerts_enabled:
            self._check_alerts()
        
        return {
            "success": True,
            "current_states": self.states,
            "changes": state_record["changes"]
        }

    def _copy_states(self):
        """Create a deep copy of the current states"""
        states_copy = {}
        for key, value in self.states.items():
            if isinstance(value, dict):
                states_copy[key] = value.copy()
            else:
                states_copy[key] = value
        return states_copy

    def _calculate_changes(self, previous_states):
        """Calculate changes between previous and current states"""
        changes = {}
        
        # Check top-level states
        for key, value in self.states.items():
            if key in previous_states:
                if isinstance(value, dict):
                    # Handle nested dictionaries (emotions, physical, personality)
                    sub_changes = {}
                    for sub_key, sub_value in value.items():
                        if sub_key in previous_states[key]:
                            prev_val = previous_states[key][sub_key]
                            change = sub_value - prev_val
                            if abs(change) > 0.05:  # Only report significant changes
                                sub_changes[sub_key] = change
                    
                    if sub_changes:
                        changes[key] = sub_changes
                else:
                    # Handle simple values
                    prev_val = previous_states[key]
                    change = value - prev_val
                    if abs(change) > 0.05:  # Only report significant changes
                        changes[key] = change
        
        return changes

    def _check_alerts(self):
        """Check for state values crossing alert thresholds"""
        alerts = []
        
        # Check each alert threshold
        for path, threshold in self.alert_thresholds.items():
            # Parse the path (e.g., "emotions.fear" -> self.states["emotions"]["fear"])
            parts = path.split('.')
            if len(parts) == 1:
                # Top-level state
                value = self.states.get(parts[0], 0.0)
            elif len(parts) == 2:
                # Nested state
                if parts[0] in self.states and isinstance(self.states[parts[0]], dict):
                    value = self.states[parts[0]].get(parts[1], 0.0)
                else:
                    continue
            else:
                continue
            
            # Check if threshold crossed (greater than or less than depending on configuration)
            threshold_crossed = False
            if parts[0] in ["awareness", "physical.energy"]:
                # These should alert when dropping below threshold
                threshold_crossed = value < threshold
            else:
                # Others alert when rising above threshold
                threshold_crossed = value > threshold
            
            if threshold_crossed:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "path": path,
                    "value": value,
                    "threshold": threshold,
                    "message": f"State {path} {'below' if parts[0] in ['awareness', 'physical.energy'] else 'above'} threshold ({value:.2f} vs {threshold:.2f})"
                }
                alerts.append(alert)
                self.alerts_history.append(alert)
                logger.warning(f"ALERT: {alert['message']}")
        
        return alerts

    def get_dominant_state(self):
        """Get the currently dominant state"""
        # Simple states
        simple_states = {
            "awareness": self.states["awareness"],
            "dreaming": self.states["dreaming"],
            "liminal": self.states["liminal"],
            "meditation": self.states["meditation"],
            "healing": self.states["healing"],
            "survival": self.states["survival"]
        }
        
        # Find the highest value state
        dominant_state = max(simple_states.items(), key=lambda x: x[1])
        
        # Only consider it dominant if it's above a threshold
        if dominant_state[1] > 0.5:
            # Also include dominant emotion if it's significant
            dominant_emotion = max(self.states["emotions"].items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0.5:
                return {
                    "primary_state": dominant_state[0],
                    "primary_state_value": dominant_state[1],
                    "dominant_emotion": dominant_emotion[0],
                    "dominant_emotion_value": dominant_emotion[1]
                }
            else:
                return {
                    "primary_state": dominant_state[0],
                    "primary_state_value": dominant_state[1]
                }
        else:
            # No clearly dominant state
            return {
                "primary_state": "balanced",
                "state_values": simple_states
            }

    def get_state_for_frequency(self):
        """Get the appropriate frequency based on current dominant state"""
        dominant = self.get_dominant_state()
        primary_state = dominant.get("primary_state", "balanced")
        
        # Map states to frequencies
        frequency_map = {
            "awareness": 14.0,  # Beta
            "dreaming": 5.0,     # Theta
            "liminal": 7.5,      # Alpha/Theta border
            "meditation": 9.0,   # Alpha
            "healing": 4.0,      # Theta/Delta border
            "survival": 18.0,    # Beta/Gamma border
            "balanced": 10.0     # Alpha
        }
        
        # Get base frequency for state
        base_frequency = frequency_map.get(primary_state, 10.0)
        
        # Modify based on emotion if present
        if "dominant_emotion" in dominant:
            emotion = dominant["dominant_emotion"]
            emotion_value = dominant["dominant_emotion_value"]
            
            # Emotional frequency modifiers
            emotion_mods = {
                "joy": 2.0,
                "sadness": -2.0,
                "fear": 3.0,
                "anger": 4.0,
                "surprise": 1.5,
                "disgust": -1.0,
                "trust": 0.0,
                "anticipation": 1.0
            }
            
            # Apply emotion modifier scaled by its intensity
            modifier = emotion_mods.get(emotion, 0.0) * emotion_value
            modified_frequency = base_frequency + modifier
        else:
            modified_frequency = base_frequency
        
        return {
            "base_frequency": base_frequency,
            "modified_frequency": modified_frequency,
            "dominant_state": primary_state
        }

    def get_monitoring_status(self):
        """Get current monitoring status and statistics"""
        return {
            "initialized": self.initialized,
            "monitoring_active": self.monitoring_active,
            "monitoring_interval": self.monitoring_interval,
            "alerts_enabled": self.alerts_enabled,
            "alerts_count": len(self.alerts_history),
            "recent_alerts": self.alerts_history[-5:] if self.alerts_history else [],
            "history_entries": len(self.state_history),
            "creation_time": self.creation_time,
            "current_dominant_state": self.get_dominant_state(),
            "current_frequency": self.get_state_for_frequency()["modified_frequency"]
        }

    def get_homeostatic_feedback(self):
        """
        Generate homeostatic feedback for energy and mycelial systems.
        This provides guidance for energy redistribution based on current states.
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "stress_level": 0.0,
            "energy_balance": 0.0,
            "regional_imbalances": {},
            "suggested_actions": []
        }
        
        # Calculate stress level based on negative states
        stress_contributors = [
            self.states["survival"],
            self.states["emotions"]["fear"],
            self.states["emotions"]["anger"],
            self.states["physical"]["fatigue"]
        ]
        feedback["stress_level"] = sum(stress_contributors) / len(stress_contributors)
        
        # Calculate energy balance
        energy_balance = self.states["physical"]["energy"] - self.states["physical"]["fatigue"]
        feedback["energy_balance"] = energy_balance
        
        # Determine suggestions based on states
        if feedback["stress_level"] > 0.6:
            feedback["suggested_actions"].append("redistribute_energy")
            feedback["suggested_actions"].append("reinforce_pathways")
        
        if energy_balance < 0.3:
            feedback["suggested_actions"].append("conserve_energy")
        
        # Check for regional imbalances based on dominant states
        if self.states["awareness"] < 0.4:
            feedback["regional_imbalances"]["prefrontal"] = -0.3  # Need more energy here
        
        if self.states["emotions"]["fear"] > 0.6:
            feedback["regional_imbalances"]["amygdala"] = 0.3  # Need to reduce energy here
        
        if self.states["survival"] > 0.7:
            feedback["regional_imbalances"]["brain_stem"] = 0.2  # Need to reduce energy here
            feedback["regional_imbalances"]["limbic"] = -0.2  # Need more energy here
        
        return feedback
