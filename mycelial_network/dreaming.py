"""
dreaming.py - Dream state consciousness processes for the soul entity.

This module handles:
- Dream state initialization and maintenance
- Dream content generation and processing
- Subconscious symbol processing
- Memory integration during dream state
- Delta wave entrainment
- Regenerative processes

The dream state is the soul's first state of consciousness and provides 
foundation for further development.
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
logger = logging.getLogger('dreaming')

# Constants
DELTA_WAVE_RANGE = (0.5, 4.0)  # Hz
REM_CYCLE_MINUTES = 90         # Average REM cycle in minutes
DREAM_SYMBOL_STRENGTH = 0.7    # Base strength of dream symbols

class DreamState:
    """
    Manages the dream state consciousness of the soul.
    """
    
    def __init__(self, soul=None):
        """
        Initialize the dream state manager.
        
        Args:
            soul: The soul entity to manage dream state for
        """
        self.soul = soul
        
        # Dream state properties
        self.is_active = False
        self.current_frequency = random.uniform(DELTA_WAVE_RANGE[0], DELTA_WAVE_RANGE[1])
        self.stability = 0.5
        self.intensity = 0.0
        self.duration = 0.0
        
        # Dream content
        self.dream_symbols = []
        self.dream_narrative = None
        self.recurring_patterns = {}
        
        # Dream cycles
        self.cycle_count = 0
        self.rem_phase = False
        
        # Metrics
        self.metrics = {
            "total_dream_time": 0.0,
            "rem_cycles": 0,
            "peak_intensity": 0.0,
            "symbol_count": 0,
            "stability_history": [],
            "frequency_history": [],
            "rejuvenation_level": 0.0
        }
        
        logger.info("Dream state manager initialized")
    
    def activate(self, initial_stability=0.5):
        """
        Activate the dream state.
        
        Args:
            initial_stability (float): Initial stability level (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if self.is_active:
            logger.warning("Dream state already active")
            return False
        
        logger.info(f"Activating dream state with stability {initial_stability:.2f}")
        
        # Set properties
        self.is_active = True
        self.stability = max(0.1, min(1.0, initial_stability))
        self.start_time = time.time()
        
        # Initialize with low intensity
        self.intensity = 0.2 * self.stability
        
        # Set starting frequency (deeper delta for more stable start)
        self.current_frequency = DELTA_WAVE_RANGE[0] + (DELTA_WAVE_RANGE[1] - DELTA_WAVE_RANGE[0]) * (1.0 - self.stability*0.7)
        
        # Record metrics
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'consciousness_state'):
                self.soul.consciousness_state = 'dream'
            
            if hasattr(self.soul, 'consciousness_frequency'):
                self.soul.consciousness_frequency = self.current_frequency
            
            if hasattr(self.soul, 'state_stability'):
                self.soul.state_stability = self.stability
            
            logger.info(f"Applied dream state to soul - Frequency: {self.current_frequency:.2f}Hz")
        
        # Initialize first dream cycle
        self._begin_new_cycle()
        
        return True
    
    def _begin_new_cycle(self):
        """Begin a new dream cycle."""
        self.cycle_count += 1
        self.rem_phase = False
        self.cycle_start_time = time.time()
        
        logger.info(f"Beginning dream cycle {self.cycle_count}")
        
        # Generate new dream content based on soul properties
        self._generate_dream_symbols()
        self._generate_dream_narrative()
        
        # Record metrics
        self.metrics["rem_cycles"] = self.cycle_count
    
    def _generate_dream_symbols(self):
        """Generate dream symbols based on soul properties and previous experiences."""
        self.dream_symbols = []
        
        # Number of symbols based on stability and cycle count
        num_symbols = 3 + int(self.stability * 5) + min(self.cycle_count, 4)
        
        # Simple dream symbol generation - would be expanded in real implementation
        basic_symbols = ["water", "light", "darkness", "path", "door", "sky", 
                         "tree", "mountain", "star", "animal", "voice", "color"]
        
        # Get soul properties if available to influence symbols
        if self.soul:
            if hasattr(self.soul, 'elemental_affinity') and self.soul.elemental_affinity:
                # Add element-related symbols
                element_symbols = {
                    "fire": ["flame", "sun", "heat", "transformation"],
                    "water": ["ocean", "river", "flow", "reflection"],
                    "air": ["wind", "breath", "clouds", "flight"],
                    "earth": ["mountain", "crystal", "tree", "ground"],
                    "aether": ["stars", "cosmos", "void", "ethereal"]
                }
                if self.soul.elemental_affinity in element_symbols:
                    basic_symbols.extend(element_symbols[self.soul.elemental_affinity])
        
        # Select symbols
        selected_symbols = random.sample(basic_symbols, min(num_symbols, len(basic_symbols)))
        
        # Add strength and meaning to each symbol
        for symbol in selected_symbols:
            symbol_data = {
                "name": symbol,
                "strength": random.uniform(0.4, DREAM_SYMBOL_STRENGTH),
                "recurrence": random.randint(1, 3)
            }
            self.dream_symbols.append(symbol_data)
        
        self.metrics["symbol_count"] = len(self.dream_symbols)
        
        logger.debug(f"Generated {len(self.dream_symbols)} dream symbols")
    
    def _generate_dream_narrative(self):
        """Generate a simple dream narrative from the symbols."""
        if not self.dream_symbols:
            self.dream_narrative = None
            return
        
        # This is a placeholder for actual dream narrative generation
        # In a full implementation, this would create more complex narratives
        
        # Select a subset of symbols for this narrative
        active_symbols = random.sample(
            self.dream_symbols, 
            min(len(self.dream_symbols), random.randint(2, 4))
        )
        
        # Create simple narrative structure (very basic)
        scene_count = random.randint(1, 3)
        narrative = {
            "scenes": [],
            "symbols_used": [s["name"] for s in active_symbols],
            "emotional_tone": random.choice(["neutral", "positive", "mysterious", "challenging"])
        }
        
        # Create scenes
        for i in range(scene_count):
            scene = {
                "setting": random.choice(["void", "light", "earth", "cosmos", "abstract"]),
                "symbols": random.sample([s["name"] for s in active_symbols], 
                                       min(len(active_symbols), random.randint(1, 3))),
                "intensity": random.uniform(0.3, 0.8)
            }
            narrative["scenes"].append(scene)
        
        self.dream_narrative = narrative
        
        logger.debug(f"Generated dream narrative with {scene_count} scenes")
    
    def update(self, time_step=1.0):
        """
        Update dream state for a time step.
        
        Args:
            time_step (float): Time step in seconds
            
        Returns:
            dict: Updated state information
        """
        if not self.is_active:
            return {"active": False}
        
        # Update dream duration
        current_time = time.time()
        self.duration = current_time - self.start_time
        
        # Calculate time in current cycle
        cycle_time = current_time - self.cycle_start_time
        cycle_duration_minutes = cycle_time / 60.0
        
        # Check if we should transition to REM phase
        if not self.rem_phase and cycle_duration_minutes >= REM_CYCLE_MINUTES * 0.8:
            self.rem_phase = True
            self._enter_rem_phase()
        
        # Check if we should start a new cycle
        if cycle_duration_minutes >= REM_CYCLE_MINUTES:
            self._begin_new_cycle()
        
        # Update metrics
        self.metrics["total_dream_time"] = self.duration
        
        # Small random variations in frequency to simulate natural fluctuations
        frequency_variation = random.uniform(-0.2, 0.2)
        
        if self.rem_phase:
            # Move toward theta waves in REM
            target_frequency = DELTA_WAVE_RANGE[1] * 1.1  # Just above delta into theta
            # More intensity during REM
            self.intensity = min(1.0, self.intensity + 0.01 * time_step)
        else:
            # Stay in delta range during non-REM
            target_frequency = DELTA_WAVE_RANGE[0] + 0.3 * (DELTA_WAVE_RANGE[1] - DELTA_WAVE_RANGE[0])
            # Less intensity during non-REM
            self.intensity = max(0.2, self.intensity - 0.005 * time_step)
        
        # Move current frequency toward target
        freq_change_rate = 0.05 * time_step
        self.current_frequency += (target_frequency - self.current_frequency) * freq_change_rate
        self.current_frequency += frequency_variation * self.stability
        
        # Keep within delta range
        self.current_frequency = max(DELTA_WAVE_RANGE[0], 
                                   min(DELTA_WAVE_RANGE[1] * 1.2, self.current_frequency))
        
        # Stability gradually increases as cycles progress (learning to dream)
        # But with small fluctuations
        stability_change = 0.002 * time_step * (1.0 + 0.1 * self.cycle_count)
        self.stability = min(0.95, self.stability + stability_change + random.uniform(-0.01, 0.01))
        
        # Track peak intensity
        if self.intensity > self.metrics["peak_intensity"]:
            self.metrics["peak_intensity"] = self.intensity
        
        # Record history
        self.metrics["stability_history"].append(self.stability)
        self.metrics["frequency_history"].append(self.current_frequency)
        
        # Calculate rejuvenation effects
        rejuvenation_rate = 0.001 * time_step * self.stability * (1.0 + 0.5 * self.cycle_count)
        self.metrics["rejuvenation_level"] = min(1.0, self.metrics["rejuvenation_level"] + rejuvenation_rate)
        
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
            "intensity": self.intensity,
            "duration": self.duration,
            "cycle": self.cycle_count,
            "rem_phase": self.rem_phase,
            "rejuvenation": self.metrics["rejuvenation_level"]
        }
    
    def _enter_rem_phase(self):
        """Handle transition to REM phase."""
        logger.info(f"Entering REM phase in cycle {self.cycle_count}")
        
        # In REM, dream narrative becomes more vivid
        if self.dream_narrative:
            # Enhance existing narrative
            for scene in self.dream_narrative["scenes"]:
                scene["intensity"] = min(1.0, scene["intensity"] * 1.3)
            
            # Add an additional scene in REM
            if self.dream_symbols:
                new_scene = {
                    "setting": random.choice(["void", "light", "earth", "cosmos", "abstract"]),
                    "symbols": random.sample([s["name"] for s in self.dream_symbols], 
                                           min(len(self.dream_symbols), random.randint(1, 3))),
                    "intensity": random.uniform(0.6, 0.9)
                }
                self.dream_narrative["scenes"].append(new_scene)
    
    def deactivate(self):
        """
        Deactivate dream state.
        
        Returns:
            dict: Final state metrics
        """
        if not self.is_active:
            logger.warning("Dream state not active")
            return self.metrics
        
        logger.info("Deactivating dream state")
        
        # Set properties
        self.is_active = False
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Final metrics
        self.metrics["total_dream_time"] = self.duration
        
        # Apply to soul if available
        if self.soul:
            if hasattr(self.soul, 'dream_state_completed'):
                self.soul.dream_state_completed = True
            
            # Note: we don't reset consciousness_state here as that should
            # be managed by the state transition controller
        
        logger.info(f"Dream state deactivated after {self.duration:.2f} seconds, {self.cycle_count} cycles")
        
        return self.metrics
    
    def get_state(self):
        """
        Get current dream state information.
        
        Returns:
            dict: Current state information
        """
        return {
            "active": self.is_active,
            "frequency": self.current_frequency,
            "stability": self.stability,
            "intensity": self.intensity,
            "duration": self.duration,
            "cycle": self.cycle_count,
            "rem_phase": self.rem_phase,
            "symbols": [s["name"] for s in self.dream_symbols],
            "rejuvenation": self.metrics["rejuvenation_level"]
        }
    
    def get_metrics(self):
        """
        Get all dream state metrics.
        
        Returns:
            dict: Metrics dictionary
        """
        # Update duration if active
        if self.is_active:
            self.metrics["total_dream_time"] = time.time() - self.start_time
        
        return self.metrics


# Utility functions
def generate_dream_frequency(stability=0.5):
    """
    Generate a dream state frequency based on stability.
    
    Args:
        stability (float): Stability level (0.0-1.0)
        
    Returns:
        float: Dream frequency in Hz
    """
    # More stable dreams have deeper delta waves
    base = DELTA_WAVE_RANGE[0]
    range_width = DELTA_WAVE_RANGE[1] - DELTA_WAVE_RANGE[0]
    
    # Higher stability = deeper delta (lower frequency)
    freq = base + range_width * (1.0 - stability*0.7)
    
    # Add small random variation
    freq += random.uniform(-0.3, 0.3)
    
    # Keep within range
    return max(DELTA_WAVE_RANGE[0], min(DELTA_WAVE_RANGE[1], freq))


# Example usage
if __name__ == "__main__":
    # Create dream state manager
    dream_state = DreamState()
    
    # Activate dream state
    dream_state.activate(initial_stability=0.6)
    
    # Simulate dream cycles
    print("Simulating dream state...")
    for i in range(20):
        state = dream_state.update(time_step=30.0)  # 30 seconds per step
        print(f"Step {i+1}: Frequency={state['frequency']:.2f}Hz, Stability={state['stability']:.2f}")
        time.sleep(0.1)  # Just for demonstration
    
    # Deactivate dream state
    metrics = dream_state.deactivate()
    
    # Print metrics
    print("\nDream State Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            print(f"{key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"{key}: {value}")

