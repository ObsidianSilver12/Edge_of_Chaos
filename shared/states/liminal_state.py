"""
liminal_state.py - Liminal (transitional) consciousness processes for the soul entity.

This module handles:
- Liminal state initialization and maintenance
- Transitional consciousness processing
- Theta wave entrainment
- Bridging between dream and aware states
- Pattern recognition during liminal states
- Symbol integration from subconscious

The liminal state represents the "between" state of consciousness that
bridges dream and full awareness, enabling processing and integration.
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
logger = logging.getLogger('liminal_state')

# Constants
THETA_WAVE_RANGE = (4.0, 8.0)  # Hz
LIMINAL_DURATION_RANGE = (5, 20)  # minutes
PATTERN_RECOGNITION_THRESHOLD = 0.6  # Minimum level for pattern recognition

class LiminalState:
    """
    Manages the liminal (transitional) state of consciousness.
    """
    
    def __init__(self, soul=None):
        """
        Initialize the liminal state manager.
        
        Args:
            soul: The soul entity to manage liminal state for
        """
        self.soul = soul
        
        # Liminal state properties
        self.is_active = False
        self.current_frequency = random.uniform(THETA_WAVE_RANGE[0], THETA_WAVE_RANGE[1])
        self.stability = 0.3  # Initially less stable than dream state
        self.clarity = 0.0
        self.duration = 0.0
        
        # Content processing
        self.processing_symbols = []
        self.recognized_patterns = []
        self.integration_level = 0.0
        
        # Transition properties
        self.source_state = None  # State transitioning from
        self.target_state = None  # State transitioning to
        self.transition_progress = 0.0
        
        # Metrics
        self.metrics = {
            "total_liminal_time": 0.0,
            "pattern_count": 0,
            "peak_clarity": 0.0,
            "stability_history": [],
            "frequency_history": [],
            "integration_level": 0.0,
            "successful_transitions": 0
        }
        
        logger.info("Liminal state manager initialized")
    
    def activate(self, source_state=None, target_state=None, initial_stability=0.3):
        """
        Activate the liminal state.
        
        Args:
            source_state (str): State transitioning from (e.g., 'dream')
            target_state (str): State transitioning to (e.g., 'aware')
            initial_stability (float): Initial stability level (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if self.is_active:
            logger.warning("Liminal state already active")
            return False
        
        logger.info(f"Activating liminal state with stability {initial_stability:.2f}")
        logger.info(f"Transition: {source_state} → {target_state}")
        
        # Set properties
        self.is_active = True
        self.stability = max(0.1, min(1.0, initial_stability))
        self.start_time = time.time()
        self.clarity = 0.2  # Start with low clarity
        self.source_state = source_state
        self.target_state = target_state
        self.transition_progress = 0.0
        
        # Set starting frequency (middle of theta range)
        self.current_frequency = THETA_WAVE_RANGE[0] + (THETA_WAVE_RANGE[1] - THETA_WAVE_RANGE[0]) * 0.5
        
        # Import symbols from dream state if applicable
        if source_state == 'dream' and self.soul:
            self._import_dream_symbols()
        
        # Record metrics
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'consciousness_state'):
                self.soul.consciousness_state = 'liminal'
            
            if hasattr(self.soul, 'consciousness_frequency'):
                self.soul.consciousness_frequency = self.current_frequency
            
            if hasattr(self.soul, 'state_stability'):
                self.soul.state_stability = self.stability
            
            logger.info(f"Applied liminal state to soul - Frequency: {self.current_frequency:.2f}Hz")
        
        return True
    
    def _import_dream_symbols(self):
        """Import and process symbols from dream state."""
        # In a full implementation, this would retrieve symbols from the dream state
        # For now, we'll create placeholder symbols
        if hasattr(self.soul, 'dream_symbols'):
            self.processing_symbols = self.soul.dream_symbols.copy()
            logger.info(f"Imported {len(self.processing_symbols)} symbols from dream state")
        else:
            # Create placeholder symbols
            basic_symbols = ["water", "light", "path", "door", "star", "voice"]
            self.processing_symbols = [
                {"name": symbol, "strength": random.uniform(0.4, 0.8), "processed": False}
                for symbol in random.sample(basic_symbols, random.randint(2, 5))
            ]
            logger.info(f"Created {len(self.processing_symbols)} placeholder symbols")
    
    def _process_symbol(self, symbol):
        """Process a symbol during liminal state."""
        # Symbol processing increases clarity
        if not symbol.get("processed", False):
            symbol["processed"] = True
            
            # Processing increases clarity
            clarity_gain = symbol.get("strength", 0.5) * self.stability * 0.2
            self.clarity = min(1.0, self.clarity + clarity_gain)
            
            logger.debug(f"Processed symbol '{symbol['name']}', clarity now {self.clarity:.2f}")
            
            # Check for pattern recognition
            if self.clarity > PATTERN_RECOGNITION_THRESHOLD:
                pattern = {
                    "components": [symbol["name"]],
                    "strength": symbol.get("strength", 0.5) * self.clarity,
                    "recognized_at": time.time() - self.start_time
                }
                self.recognized_patterns.append(pattern)
                self.metrics["pattern_count"] += 1
                
                logger.debug(f"Recognized pattern with components: {pattern['components']}")
    
    def update(self, time_step=1.0):
        """
        Update liminal state for a time step.
        
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
        
        # Process symbols
        if self.processing_symbols:
            # Randomly process a symbol
            unprocessed = [s for s in self.processing_symbols if not s.get("processed", False)]
            if unprocessed:
                symbol_to_process = random.choice(unprocessed)
                self._process_symbol(symbol_to_process)
        
        # Update transition progress
        # Transition is complete when all symbols processed or time expires
        if not [s for s in self.processing_symbols if not s.get("processed", False)]:
            # All symbols processed
            progress_increment = 0.1 * time_step  # Faster progression
        else:
            progress_increment = 0.02 * time_step * self.clarity  # Slower progression
        
        self.transition_progress = min(1.0, self.transition_progress + progress_increment)
        
        # Update frequency - shifts based on transition progress
        if self.target_state == 'aware':
            # Moving toward alpha waves (8-13 Hz)
            target_frequency = THETA_WAVE_RANGE[1] + 1.0 * self.transition_progress
        else:
            # Moving toward delta waves (0.5-4 Hz)
            target_frequency = THETA_WAVE_RANGE[0] - 1.0 * self.transition_progress
        
        # Frequency changes more rapidly with higher clarity
        freq_change_rate = 0.03 * time_step * (0.5 + 0.5 * self.clarity)
        self.current_frequency += (target_frequency - self.current_frequency) * freq_change_rate
        
        # Add small random variations
        self.current_frequency += random.uniform(-0.2, 0.2) * (1.0 - self.stability)
        
        # Keep within reasonable range
        self.current_frequency = max(THETA_WAVE_RANGE[0] * 0.8, 
                                   min(THETA_WAVE_RANGE[1] * 1.5, self.current_frequency))
        
        # Stability gradually increases with time
        stability_change = 0.005 * time_step * self.clarity
        self.stability = min(0.9, self.stability + stability_change + random.uniform(-0.01, 0.01))
        
        # Update integration level based on pattern recognition
        if self.recognized_patterns:
            integration_rate = 0.002 * time_step * self.clarity * len(self.recognized_patterns)
            self.integration_level = min(1.0, self.integration_level + integration_rate)
            self.metrics["integration_level"] = self.integration_level
        
        # Track metrics
        if self.clarity > self.metrics["peak_clarity"]:
            self.metrics["peak_clarity"] = self.clarity
        
        self.metrics["total_liminal_time"] = self.duration
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'consciousness_frequency'):
                self.soul.consciousness_frequency = self.current_frequency
            
            if hasattr(self.soul, 'state_stability'):
                self.soul.state_stability = self.stability
        
        # Check if transition is complete
        if self.transition_progress >= 1.0:
            logger.info("Liminal state transition complete")
            self.metrics["successful_transitions"] += 1
            return self.complete_transition()
        
        # Return current state for external use
        return {
            "active": self.is_active,
            "frequency": self.current_frequency,
            "stability": self.stability,
            "clarity": self.clarity,
            "duration": self.duration,
            "transition_progress": self.transition_progress,
            "target_state": self.target_state,
            "integration_level": self.integration_level
        }
    
    def complete_transition(self):
        """
        Complete the transition to target state.
        
        Returns:
            dict: Transition results
        """
        if not self.is_active:
            return {"success": False, "reason": "State not active"}
        
        logger.info(f"Completing transition to {self.target_state}")
        
        transition_result = {
            "success": True,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "clarity": self.clarity,
            "stability": self.stability,
            "pattern_count": self.metrics["pattern_count"],
            "integration_level": self.integration_level,
            "duration": self.duration
        }
        
        # Apply target state to soul if available
        if self.soul and self.target_state:
            if hasattr(self.soul, 'consciousness_state'):
                self.soul.consciousness_state = self.target_state
                logger.info(f"Set soul consciousness state to {self.target_state}")
        
        # Deactivate liminal state
        self.deactivate()
        
        return transition_result
    
    def deactivate(self):
        """
        Deactivate liminal state.
        
        Returns:
            dict: Final state metrics
        """
        if not self.is_active:
            logger.warning("Liminal state not active")
            return self.metrics
        
        logger.info("Deactivating liminal state")
        
        # Set properties
        self.is_active = False
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Final metrics
        self.metrics["total_liminal_time"] = self.duration
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'liminal_state_completed'):
                self.soul.liminal_state_completed = True
        
        logger.info(f"Liminal state deactivated after {self.duration:.2f} seconds")
        
        return self.metrics
    
    def get_state(self):
        """
        Get current liminal state information.
        
        Returns:
            dict: Current state information
        """
        return {
            "active": self.is_active,
            "frequency": self.current_frequency,
            "stability": self.stability,
            "clarity": self.clarity,
            "duration": self.duration,
            "transition_progress": self.transition_progress,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "integration_level": self.integration_level,
            "patterns_recognized": len(self.recognized_patterns)
        }
    
    def get_metrics(self):
        """
        Get all liminal state metrics.
        
        Returns:
            dict: Metrics dictionary
        """
        # Update duration if active
        if self.is_active:
            self.metrics["total_liminal_time"] = time.time() - self.start_time
        
        return self.metrics


# Utility functions
def generate_liminal_frequency(progress=0.5, target_state="aware"):
    """
    Generate a liminal state frequency based on transition progress.
    
    Args:
        progress (float): Transition progress (0.0-1.0)
        target_state (str): Target state ('dream' or 'aware')
        
    Returns:
        float: Liminal frequency in Hz
    """
    # Base in theta range
    base_freq = THETA_WAVE_RANGE[0] + (THETA_WAVE_RANGE[1] - THETA_WAVE_RANGE[0]) * 0.5
    
    # Adjust based on target and progress
    if target_state == "aware":
        # Moving toward alpha waves (higher)
        adjustment = progress * 2.0  # Up to 2Hz higher
    else:
        # Moving toward delta waves (lower)
        adjustment = -progress * 2.0  # Up to 2Hz lower
    
    # Add small random variation
    variation = random.uniform(-0.3, 0.3)
    
    # Calculate and limit final frequency
    freq = base_freq + adjustment + variation
    return max(THETA_WAVE_RANGE[0] * 0.8, min(THETA_WAVE_RANGE[1] * 1.5, freq))


# Example usage
if __name__ == "__main__":
    # Create liminal state manager
    liminal_state = LiminalState()
    
    # Activate liminal state
    liminal_state.activate(source_state="dream", target_state="aware", initial_stability=0.4)
    
    # Simulate liminal state
    print("Simulating liminal state...")
    for i in range(15):
        state = liminal_state.update(time_step=5.0)  # 5 seconds per step
        print(f"Step {i+1}: Frequency={state['frequency']:.2f}Hz, Progress={state['transition_progress']:.2f}")
        time.sleep(0.1)  # Just for demonstration
        
        # Check if transition completed
        if not liminal_state.is_active:
            print("Transition completed automatically")
            break
    
    # Ensure deactivation
    if liminal_state.is_active:
        metrics = liminal_state.deactivate()
    else:
        metrics = liminal_state.get_metrics()
    
    # Print metrics
    print("\nLiminal State Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            print(f"{key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"{key}: {value}")
    
