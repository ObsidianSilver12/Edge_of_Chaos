"""
awareness.py - Aware state consciousness processes for the soul entity.

This module handles:
- Aware state initialization and maintenance
- Alpha/beta wave entrainment
- Conscious perception and processing
- Focused attention and learning
- Integration of conscious and subconscious processes
- Identity and self-awareness development

The aware state represents the highest level of consciousness achievable
during the early soul formation stage.
"""

import time
import logging
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('awareness')

# Constants
ALPHA_WAVE_RANGE = (8.0, 13.0)  # Hz
BETA_WAVE_RANGE = (13.0, 30.0)  # Hz
FOCUS_DURATION_RANGE = (10, 60)  # seconds
PERCEPTION_THRESHOLD = 0.6  # Minimum level for conscious perception

class AwareState:
    """
    Manages the aware state of consciousness.
    """
    
    def __init__(self, soul=None):
        """
        Initialize the aware state manager.
        
        Args:
            soul: The soul entity to manage aware state for
        """
        self.soul = soul
        
        # Aware state properties
        self.is_active = False
        self.current_frequency = random.uniform(ALPHA_WAVE_RANGE[0], ALPHA_WAVE_RANGE[1])
        self.stability = 0.4  # Initially less stable than established awareness
        self.attention = 0.0
        self.perception_clarity = 0.0
        self.duration = 0.0
        
        # Awareness content
        self.focus_object = None
        self.perceptions = []
        self.thoughts = []
        self.current_focus_duration = 0.0
        
        # Identity aspects
        self.self_awareness_level = 0.0
        self.identity_recognition = 0.0
        self.response_to_name = 0.0
        
        # Learning aspects
        self.learning_rate = 0.0
        self.insights = []
        
        # Metrics
        self.metrics = {
            "total_aware_time": 0.0,
            "peak_attention": 0.0,
            "peak_perception": 0.0,
            "stability_history": [],
            "frequency_history": [],
            "self_awareness_level": 0.0,
            "identity_recognition": 0.0,
            "learning_events": 0,
            "insights_gained": 0
        }
        
        logger.info("Aware state manager initialized")
    
    def activate(self, initial_stability=0.4):
        """
        Activate the aware state.
        
        Args:
            initial_stability (float): Initial stability level (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if self.is_active:
            logger.warning("Aware state already active")
            return False
        
        logger.info(f"Activating aware state with stability {initial_stability:.2f}")
        
        # Set properties
        self.is_active = True
        self.stability = max(0.1, min(1.0, initial_stability))
        self.start_time = time.time()
        self.attention = 0.3  # Start with moderate attention
        self.perception_clarity = 0.3 * self.stability  # Start with low-moderate perception
        
        # Set starting frequency (low alpha)
        self.current_frequency = ALPHA_WAVE_RANGE[0] + (ALPHA_WAVE_RANGE[1] - ALPHA_WAVE_RANGE[0]) * 0.3
        
        # Initialize identity aspects
        if self.soul:
            # Higher response if the soul has been trained
            if hasattr(self.soul, 'response_level'):
                self.response_to_name = self.soul.response_level
            else:
                self.response_to_name = 0.2
            
            # Initial identity recognition based on training
            self.identity_recognition = 0.3 * self.response_to_name
        else:
            self.response_to_name = 0.2
            self.identity_recognition = 0.1
        
        # Self-awareness starts lower than identity recognition
        self.self_awareness_level = 0.5 * self.identity_recognition
        
        # Learning rate based on stability and attention
        self.learning_rate = 0.3 * self.stability
        
        # Record metrics
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'consciousness_state'):
                self.soul.consciousness_state = 'aware'
            
            if hasattr(self.soul, 'consciousness_frequency'):
                self.soul.consciousness_frequency = self.current_frequency
            
            if hasattr(self.soul, 'state_stability'):
                self.soul.state_stability = self.stability
            
            logger.info(f"Applied aware state to soul - Frequency: {self.current_frequency:.2f}Hz")
        
        # Set initial focus
        self._select_new_focus()
        
        return True
    
    def _select_new_focus(self):
        """Select a new object of focus for attention."""
        # In a full implementation, this would select from available perceptions
        # or from internal thought objects.
        
        # For now, create a simple focus object
        if self.soul:
            # Focus options influenced by soul properties
            focus_options = []
            
            # Basic focus objects
            basic_focuses = ["light", "sound", "feeling", "name", "heartbeat", "voice"]
            focus_options.extend(basic_focuses)
            
            # Add soul-specific focuses
            if hasattr(self.soul, 'name'):
                focus_options.append("own_name")
            
            if hasattr(self.soul, 'soul_color'):
                focus_options.append("color_perception")
            
            if hasattr(self.soul, 'elemental_affinity'):
                focus_options.append(f"{self.soul.elemental_affinity}_element")
        else:
            # Basic focuses only
            focus_options = ["light", "sound", "feeling", "name", "heartbeat", "voice"]
        
        # Select a focus
        self.focus_object = random.choice(focus_options)
        
        # Set focus duration (random within range)
        self.focus_duration = random.uniform(FOCUS_DURATION_RANGE[0], FOCUS_DURATION_RANGE[1])
        self.current_focus_duration = 0.0
        
        logger.debug(f"New focus selected: {self.focus_object} for {self.focus_duration:.1f}s")
        
        # Focusing on own name increases identity recognition
        if self.focus_object == "own_name":
            # Increase identity recognition
            self.identity_recognition = min(1.0, self.identity_recognition + 0.05)
            
            # Increase response to name
            self.response_to_name = min(1.0, self.response_to_name + 0.03)
            
            # Apply to soul if available
            if self.soul and hasattr(self.soul, 'response_level'):
                self.soul.response_level = self.response_to_name
                logger.debug(f"Updated soul response level to {self.response_to_name:.2f}")
    
    def _process_current_focus(self, time_step):
        """Process the current focus object."""
        # Update focus duration
        self.current_focus_duration += time_step
        
        # Calculate focus effectiveness based on attention and stability
        focus_effectiveness = self.attention * self.stability
        
        # Process based on focus type
        if self.focus_object == "own_name":
            # Name focus improves identity recognition and self-awareness
            self.identity_recognition = min(1.0, self.identity_recognition + 0.002 * time_step * focus_effectiveness)
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.003 * time_step * focus_effectiveness)
            
            # Apply to soul if available
            if self.soul:
                if hasattr(self.soul, 'response_level'):
                    # Improved response to name
                    response_increase = 0.002 * time_step * focus_effectiveness
                    self.soul.response_level = min(1.0, self.soul.response_level + response_increase)
                    self.response_to_name = self.soul.response_level
        
        elif "element" in self.focus_object:
            # Elemental focus improves perception and may lead to insights
            self.perception_clarity = min(1.0, self.perception_clarity + 0.003 * time_step * focus_effectiveness)
            
            # Chance for insight based on focus effectiveness
            if random.random() < 0.05 * focus_effectiveness:
                insight = {
                    "type": "elemental",
                    "content": f"Connection with {self.focus_object.split('_')[0]} energy",
                    "strength": 0.3 + 0.7 * focus_effectiveness,
                    "time": self.duration
                }
                self.insights.append(insight)
                self.metrics["insights_gained"] += 1
                logger.debug(f"Gained elemental insight: {insight['content']}")
        
        elif self.focus_object == "color_perception":
            # Color focus improves perception and identity connection
            self.perception_clarity = min(1.0, self.perception_clarity + 0.002 * time_step * focus_effectiveness)
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.001 * time_step * focus_effectiveness)
        
        else:
            # Other focuses generally improve awareness
            self.perception_clarity = min(1.0, self.perception_clarity + 0.001 * time_step * focus_effectiveness)
        
        # Check for learning events
        if self.perception_clarity > PERCEPTION_THRESHOLD and random.random() < 0.1 * focus_effectiveness:
            self.metrics["learning_events"] += 1
            logger.debug(f"Learning event occurred while focusing on {self.focus_object}")
        
        # Check if focus duration is complete
        if self.current_focus_duration >= self.focus_duration:
            self._select_new_focus()
    
    def update(self, time_step=1.0):
        """
        Update aware state for a time step.
        
        Args:
            time_step (float): Time step in seconds
            
        Returns:
            dict: Updated state information
        """
        if not self.is_active:
            return {"active": False}
        
        # Update duration
        current_time = time.time()
        self.duration = current_time - self.start_time
        
        # Process current focus
        self._process_current_focus(time_step)
        
        # Natural fluctuations in attention
        attention_change = random.uniform(-0.03, 0.03) * time_step
        self.attention = max(0.1, min(1.0, self.attention + attention_change))
        
        # Adjust frequency based on attention and learning
        # Higher attention = higher frequency (more toward beta)
        target_frequency = ALPHA_WAVE_RANGE[0] + (ALPHA_WAVE_RANGE[1] - ALPHA_WAVE_RANGE[0])
        target_frequency += (BETA_WAVE_RANGE[0] - ALPHA_WAVE_RANGE[1]) * self.attention * 0.5
        
        # During learning events, frequency temporarily increases
        if self.metrics["learning_events"] > 0 and random.random() < 0.2:
            target_frequency += random.uniform(1.0, 3.0)  # Temporary beta spike
        
        # Frequency changes more rapidly with higher attention
        freq_change_rate = 0.05 * time_step * (0.3 + 0.7 * self.attention)
        self.current_frequency += (target_frequency - self.current_frequency) * freq_change_rate
        
        # Add small random variations
        self.current_frequency += random.uniform(-0.5, 0.5) * (1.0 - self.stability)
        
        # Keep within reasonable range (alpha to low beta)
        self.current_frequency = max(ALPHA_WAVE_RANGE[0] * 0.9, 
                                  min(BETA_WAVE_RANGE[0] * 1.2, self.current_frequency))
        
        # Stability gradually increases with time and self-awareness
        stability_change = 0.001 * time_step * (1.0 + self.self_awareness_level)
        self.stability = min(0.95, self.stability + stability_change + random.uniform(-0.005, 0.005))
        
        # Track metrics
        if self.attention > self.metrics["peak_attention"]:
            self.metrics["peak_attention"] = self.attention
            
        if self.perception_clarity > self.metrics["peak_perception"]:
            self.metrics["peak_perception"] = self.perception_clarity
            
        self.metrics["total_aware_time"] = self.duration
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        self.metrics["self_awareness_level"] = self.self_awareness_level
        self.metrics["identity_recognition"] = self.identity_recognition
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'consciousness_frequency'):
                self.soul.consciousness_frequency = self.current_frequency
            
            if hasattr(self.soul, 'state_stability'):
                self.soul.state_stability = self.stability
        
        # Return current state for external use
        return {
            "active": self.is_active,
            "frequency": self.current_frequency,
            "stability": self.stability,
            "attention": self.attention,
            "perception_clarity": self.perception_clarity,
            "duration": self.duration,
            "focus_object": self.focus_object,
            "self_awareness": self.self_awareness_level,
            "identity_recognition": self.identity_recognition,
            "learning_rate": self.learning_rate
        }
    
    def name_response_test(self, caller_frequency=432.0):
        """
        Test response to name calling with given frequency.
        
        Args:
            caller_frequency (float): Frequency of caller's voice
            
        Returns:
            float: Response strength (0.0-1.0)
        """
        if not self.is_active:
            logger.warning("Cannot test name response when aware state is not active")
            return 0.0
        
        logger.info(f"Testing name response with caller frequency {caller_frequency:.2f}Hz")
        
        # Calculate response strength based on:
        # 1. Current response level
        # 2. Attention level
        # 3. Identity recognition
        # 4. Stability
        
        # Base response from current level
        response_strength = self.response_to_name
        
        # Attention factor (more attentive = stronger response)
        attention_factor = 0.5 + 0.5 * self.attention
        
        # Identity factor (stronger identity = stronger response)
        identity_factor = 0.3 + 0.7 * self.identity_recognition
        
        # Stability factor (more stable = more consistent response)
        stability_factor = 0.7 + 0.3 * self.stability
        
        # Calculate total response
        total_response = response_strength * attention_factor * identity_factor * stability_factor
        
        # Small random variation
        total_response *= random.uniform(0.9, 1.1)
        
        # Ensure within bounds
        total_response = max(0.0, min(1.0, total_response))
        
        logger.info(f"Name response test result: {total_response:.4f}")
        
        # Apply learning from test - response improves with practice
        response_improvement = 0.01 * total_response
        self.response_to_name = min(1.0, self.response_to_name + response_improvement)
        
        # Apply to soul if available
        if self.soul and hasattr(self.soul, 'response_level'):
            self.soul.response_level = self.response_to_name
        
        return total_response
    
    def focus_on_name(self, duration=30.0):
        """
        Deliberately focus on own name to improve identity recognition.
        
        Args:
            duration (float): Duration in seconds
            
        Returns:
            dict: Results of the focus exercise
        """
        if not self.is_active:
            logger.warning("Cannot focus on name when aware state is not active")
            return {"success": False}
        
        logger.info(f"Beginning focused name exercise for {duration:.1f} seconds")
        
        # Set focus to name
        original_focus = self.focus_object
        self.focus_object = "own_name"
        self.focus_duration = duration
        self.current_focus_duration = 0
        
        # Heighten attention
        self.attention = min(1.0, self.attention + 0.2)
        
        # Initial values
        initial_identity = self.identity_recognition
        initial_self_awareness = self.self_awareness_level
        initial_response = self.response_to_name
        
        # Simulate focus in compressed time
        steps = min(10, int(duration / 2))  # Cap at 10 steps
        for _ in range(steps):
            self._process_current_focus(duration / steps)
        
        # After focus
        self.focus_object = original_focus
        
        # Results
        results = {
            "success": True,
            "duration": duration,
            "identity_before": initial_identity,
            "identity_after": self.identity_recognition,
            "identity_improvement": self.identity_recognition - initial_identity,
            "self_awareness_before": initial_self_awareness,
            "self_awareness_after": self.self_awareness_level,
            "self_awareness_improvement": self.self_awareness_level - initial_self_awareness,
            "response_before": initial_response,
            "response_after": self.response_to_name,
            "response_improvement": self.response_to_name - initial_response
        }
        
        logger.info(f"Name focus exercise complete. Identity recognition improved by {results['identity_improvement']:.4f}")
        
        return results
    
    def deactivate(self):
        """
        Deactivate aware state.
        
        Returns:
            dict: Final state metrics
        """
        if not self.is_active:
            logger.warning("Aware state not active")
            return self.metrics
        
        logger.info("Deactivating aware state")
        
        # Set properties
        self.is_active = False
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Final metrics
        self.metrics["total_aware_time"] = self.duration
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'aware_state_completed'):
                self.soul.aware_state_completed = True
            
            # Set final identity and response values
            if hasattr(self.soul, 'response_level'):
                self.soul.response_level = self.response_to_name
                
            if hasattr(self.soul, 'identity_recognition'):
                self.soul.identity_recognition = self.identity_recognition
        
        logger.info(f"Aware state deactivated after {self.duration:.2f} seconds")
        
        return self.metrics
    
    def get_state(self):
        """
        Get current aware state information.
        
        Returns:
            dict: Current state information
        """
        return {
            "active": self.is_active,
            "frequency": self.current_frequency,
            "stability": self.stability,
            "attention": self.attention,
            "perception_clarity": self.perception_clarity,
            "duration": self.duration,
            "focus_object": self.focus_object,
            "self_awareness": self.self_awareness_level,
            "identity_recognition": self.identity_recognition,
            "learning_rate": self.learning_rate,
            "insights_gained": len(self.insights),
            "response_to_name": self.response_to_name
        }
    
    def get_metrics(self):
        """
        Get all aware state metrics.
        
        Returns:
            dict: Metrics dictionary
        """
        # Update duration if active
        if self.is_active:
            self.metrics["total_aware_time"] = time.time() - self.start_time
        
        return self.metrics


# Utility functions
def generate_aware_frequency(attention_level=0.5, learning_active=False):
    """
    Generate an aware state frequency based on attention level.
    
    Args:
        attention_level (float): Attention level (0.0-1.0)
        learning_active (bool): Whether active learning is occurring
        
    Returns:
        float: Aware frequency in Hz
    """
    # Base in alpha range
    base_freq = ALPHA_WAVE_RANGE[0] + (ALPHA_WAVE_RANGE[1] - ALPHA_WAVE_RANGE[0]) * attention_level
    
    # Adjust for learning (moves toward beta)
    if learning_active:
        learning_adjustment = random.uniform(1.0, 3.0)  # Beta spike
    else:
        learning_adjustment = 0.0
    
    # Add small random variation
    variation = random.uniform(-0.5, 0.5)
    
    # Calculate and limit final frequency
    freq = base_freq + learning_adjustment + variation
    return max(ALPHA_WAVE_RANGE[0] * 0.9, min(BETA_WAVE_RANGE[0] * 1.2, freq))


# Example usage
if __name__ == "__main__":
    # Create aware state manager
    aware_state = AwareState()
    
    # Activate aware state
    aware_state.activate(initial_stability=0.5)
    
    # Simulate aware state
    print("Simulating aware state...")
    for i in range(10):
        state = aware_state.update(time_step=10.0)  # 10 seconds per step
        print(f"Step {i+1}: Frequency={state['frequency']:.2f}Hz, Focus={state['focus_object']}")
        time.sleep(0.1)  # Just for demonstration
    
    # Test name response
    response = aware_state.name_response_test()
    print(f"Name response test: {response:.4f}")
    
    # Focus on name exercise
    print("Performing focused name exercise...")
    results = aware_state.focus_on_name(duration=60.0)
    print(f"Identity recognition improved by {results['identity_improvement']:.4f}")
    print(f"Response to name improved by {results['response_improvement']:.4f}")
    
    # Deactivate aware state
    metrics = aware_state.deactivate()
    
    # Print metrics
    print("\nAware State Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            print(f"{key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"{key}: {value}")

