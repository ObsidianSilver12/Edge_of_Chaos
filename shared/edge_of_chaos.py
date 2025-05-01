"""
Edge of Chaos

This module implements the edge of chaos mechanics, which are critical for the
emergence of complex patterns and soul development. It manages the delicate balance
between order and chaos needed for optimal soul emergence and evolution.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='edge_of_chaos.log'
)
logger = logging.getLogger('edge_of_chaos')

# Add parent directory to path to import from parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
try:
    from stage_1.soul_spark.soul_spark import SoulSpark
    import metrics_tracking as metrics
    from constants.constants import *
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # Define fallback constants in case imports fail
    GOLDEN_RATIO = 1.618033988749895
    EDGE_OF_CHAOS_RATIO = 0.618  # Critical value for emergence (1/φ)

# Constants
MIN_COMPLEXITY = 0.2
MAX_COMPLEXITY = 0.9


class EdgeOfChaos:
    """
    Edge of Chaos Manager
    
    This class implements the edge of chaos mechanics, managing the balance between
    order and chaos needed for optimal soul emergence and evolution.
    """
    
    def __init__(self, soul_spark=None, field=None):
        """
        Initialize the edge of chaos manager.
        
        Args:
            soul_spark: Optional soul spark to manage
            field: Optional field environment to use
        """
        self.soul_spark = soul_spark
        self.field = field
        
        # System state tracking
        self.chaos_level = 0.5  # Start at middle point
        self.order_level = 0.5  # Start at middle point
        self.complexity = 0.0   # Emergent complexity
        
        # Process tracking
        self.last_adjustment_time = 0.0
        self.adjustment_history = []
        
        # Metrics
        self.metrics = {
            "initial_chaos": self.chaos_level,
            "initial_order": self.order_level,
            "edge_proximity": 0.0,
            "adjustments": []
        }
        
        logger.info("Edge of Chaos manager initialized")
    
    def calculate_current_state(self):
        """
        Calculate the current state of chaos, order, and complexity.
        
        Returns:
            tuple: (chaos_level, order_level, complexity)
        """
        # If we have a soul spark, use its properties
        if self.soul_spark:
            # Calculate chaos from soul properties
            coherence = getattr(self.soul_spark, "coherence", 0.5)
            stability = getattr(self.soul_spark, "stability", 0.5)
            
            # Chaos is inverse of stability and coherence
            self.chaos_level = 1.0 - (stability * 0.6 + coherence * 0.4)
            
            # Order is directly related to stability and coherence
            self.order_level = stability * 0.6 + coherence * 0.4
            
        # If we have a field, incorporate field properties
        elif self.field:
            # Calculate from field properties
            field_entropy = getattr(self.field, "entropy", 0.5)
            field_coherence = getattr(self.field, "coherence", 0.5)
            field_stability = getattr(self.field, "stability", 0.5)
            
            # Calculate chaos and order levels
            self.chaos_level = field_entropy
            self.order_level = (field_coherence + field_stability) / 2
        
        # Calculate complexity as a function of chaos and order
        # Complexity is highest at the edge of chaos
        # Uses a modified logistic function centered at the edge of chaos
        chaos_order_balance = 1.0 - abs(self.chaos_level - (1.0 - self.order_level))
        edge_proximity = 1.0 - abs(self.chaos_level - EDGE_OF_CHAOS_RATIO)
        
        # Complexity is highest when chaos is near target and when chaos/order are balanced
        self.complexity = chaos_order_balance * 0.4 + edge_proximity * 0.6
        self.complexity = max(MIN_COMPLEXITY, min(MAX_COMPLEXITY, self.complexity))
        
        # Store edge proximity
        self.metrics["edge_proximity"] = edge_proximity
        
        logger.info(f"Current state - Chaos: {self.chaos_level:.2f}, Order: {self.order_level:.2f}, Complexity: {self.complexity:.2f}")
        
        return (self.chaos_level, self.order_level, self.complexity)
    
    def adjust_toward_edge(self, intensity=0.5):
        """
        Adjust the system toward the edge of chaos.
        
        Args:
            intensity (float): Intensity of adjustment (0.1-1.0)
            
        Returns:
            tuple: (new_chaos_level, new_order_level, new_complexity)
        """
        logger.info(f"Adjusting toward edge with intensity {intensity}")
        self.last_adjustment_time = time.time()
        
        # Calculate current state
        self.calculate_current_state()
        
        # Calculate how far we are from the edge
        distance_from_edge = abs(self.chaos_level - EDGE_OF_CHAOS_RATIO)
        
        # Determine direction of adjustment
        if self.chaos_level > EDGE_OF_CHAOS_RATIO:
            # Too chaotic, need more order
            chaos_adjustment = -distance_from_edge * intensity
            order_adjustment = distance_from_edge * intensity * 0.8  # Slightly less to avoid oscillation
        else:
            # Too ordered, need more chaos
            chaos_adjustment = distance_from_edge * intensity
            order_adjustment = -distance_from_edge * intensity * 0.8  # Slightly less to avoid oscillation
        
        # Apply adjustments
        new_chaos = max(0.1, min(0.9, self.chaos_level + chaos_adjustment))
        new_order = max(0.1, min(0.9, self.order_level + order_adjustment))
        
        # Record adjustment
        adjustment = {
            "time": self.last_adjustment_time,
            "old_chaos": self.chaos_level,
            "old_order": self.order_level,
            "new_chaos": new_chaos,
            "new_order": new_order,
            "intensity": intensity
        }
        
        self.adjustment_history.append(adjustment)
        self.metrics["adjustments"].append(adjustment)
        
        # Update state
        self.chaos_level = new_chaos
        self.order_level = new_order
        
        # Recalculate complexity
        self.calculate_current_state()
        
        # Apply changes to soul or field if available
        self._apply_changes()
        
        logger.info(f"Adjusted to - Chaos: {self.chaos_level:.2f}, Order: {self.order_level:.2f}, Complexity: {self.complexity:.2f}")
        
        return (self.chaos_level, self.order_level, self.complexity)
    
    def maintain_edge_state(self, duration=10, intensity=0.3, interval=1.0):
        """
        Maintain the system at the edge of chaos for a specified duration.
        
        Args:
            duration (float): Duration to maintain edge state (seconds)
            intensity (float): Intensity of adjustments (0.1-1.0)
            interval (float): Time between adjustments (seconds)
            
        Returns:
            float: Average complexity during maintenance
        """
        logger.info(f"Maintaining edge state for {duration} seconds")
        start_time = time.time()
        end_time = start_time + duration
        
        complexity_samples = []
        
        while time.time() < end_time:
            # Calculate current state
            self.calculate_current_state()
            complexity_samples.append(self.complexity)
            
            # Adjust toward edge if needed
            distance_from_edge = abs(self.chaos_level - EDGE_OF_CHAOS_RATIO)
            if distance_from_edge > 0.05:  # Only adjust if we're far from edge
                self.adjust_toward_edge(intensity)
            
            # Wait for interval
            time.sleep(interval)
        
        # Calculate average complexity
        avg_complexity = sum(complexity_samples) / len(complexity_samples) if complexity_samples else 0
        
        logger.info(f"Maintained edge state with average complexity: {avg_complexity:.2f}")
        
        # Record metrics
        self.metrics["maintenance_duration"] = duration
        self.metrics["average_complexity"] = avg_complexity
        
        # Record to central metrics
        try:
            metrics.record_metric(
                "edge_of_chaos",
                "maintenance_duration",
                duration
            )
            
            metrics.record_metric(
                "edge_of_chaos",
                "average_complexity",
                avg_complexity
            )
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
        
        return avg_complexity
    
    def induce_phase_transition(self, target_complexity=0.8, max_attempts=5):
        """
        Induce a phase transition by pushing the system to high complexity.
        
        Args:
            target_complexity (float): Target complexity level (0.1-1.0)
            max_attempts (int): Maximum number of adjustment attempts
            
        Returns:
            bool: Success status
        """
        logger.info("Attempting to induce phase transition to complexity %s", target_complexity)
        
        # Calculate current state
        self.calculate_current_state()
        
        # If already at target, just return success
        if self.complexity >= target_complexity:
            logger.info("Already at target complexity")
            return True
        
        # Try to reach target complexity
        for attempt in range(max_attempts):
            # Calculate required chaos level for target complexity
            # This is a simplified model - in reality the relationship is more complex
            required_chaos = EDGE_OF_CHAOS_RATIO
            
            # Calculate required order level
            required_order = 1.0 - required_chaos
            
            # Calculate adjustment amounts
            chaos_adjustment = required_chaos - self.chaos_level
            order_adjustment = required_order - self.order_level
            
            # Apply adjustments with increasing intensity
            intensity = 0.5 + ((attempt + 1) / max_attempts) * 0.5
            
            self.chaos_level = max(0.1, min(0.9, self.chaos_level + chaos_adjustment * intensity))
            self.order_level = max(0.1, min(0.9, self.order_level + order_adjustment * intensity))
            
            # Recalculate complexity
            self.calculate_current_state()
            
            # Record attempt
            self.metrics["phase_transition_attempts"] = attempt + 1
            
            # Check if we've reached target
            if self.complexity >= target_complexity:
                logger.info("Phase transition successful after %s attempts", attempt + 1)
                
                # Apply changes to soul or field
                self._apply_changes()
                
                # Record success metrics
                self.metrics["phase_transition_success"] = True
                self.metrics["final_complexity"] = self.complexity
                
                try:
                    metrics.record_metric(
                        "edge_of_chaos",
                        "phase_transition_success",
                        True
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning("Failed to record metrics: %s", e)
                
                return True
        
        logger.warning("Failed to induce phase transition after %s attempts", max_attempts)
        
        # Record failure metrics
        self.metrics["phase_transition_success"] = False
        self.metrics["final_complexity"] = self.complexity
        
        try:
            metrics.record_metric(
                "edge_of_chaos",
                "phase_transition_success",
                False
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("Failed to record metrics: %s", e)
        
        return False    
    def _apply_changes(self):
        """Apply the current chaos/order state to the soul or field."""
        # Apply to soul if available
        if self.soul_spark:
            # Calculate new soul stability and coherence
            new_stability = self.order_level * 0.8 + 0.2
            new_coherence = self.order_level * 0.7 + 0.3
            
            # Apply changes
            setattr(self.soul_spark, "stability", new_stability)
            setattr(self.soul_spark, "coherence", new_coherence)
            setattr(self.soul_spark, "complexity", self.complexity)
            
            logger.info(f"Applied changes to soul - Stability: {new_stability:.2f}, Coherence: {new_coherence:.2f}")
        
        # Apply to field if available
        elif self.field:
            # Calculate new field properties
            new_entropy = self.chaos_level
            new_coherence = self.order_level * 0.9 + 0.1
            new_stability = self.order_level * 0.8 + 0.2
            
            # Apply changes
            setattr(self.field, "entropy", new_entropy)
            setattr(self.field, "coherence", new_coherence)
            setattr(self.field, "stability", new_stability)
            setattr(self.field, "complexity", self.complexity)
            
            logger.info(f"Applied changes to field - Entropy: {new_entropy:.2f}, Coherence: {new_coherence:.2f}")
    
    def get_complexity_metrics(self):
        """
        Get detailed complexity metrics.
        
        Returns:
            dict: Complexity metrics
        """
        # Calculate current state
        self.calculate_current_state()
        
        # Calculate additional metrics
        edge_distance = abs(self.chaos_level - EDGE_OF_CHAOS_RATIO)
        order_chaos_ratio = self.order_level / self.chaos_level if self.chaos_level > 0 else float('inf')
        phi_alignment = 1.0 - abs(order_chaos_ratio - GOLDEN_RATIO) / GOLDEN_RATIO
        
        complexity_metrics = {
            "chaos_level": self.chaos_level,
            "order_level": self.order_level,
            "complexity": self.complexity,
            "edge_distance": edge_distance,
            "edge_proximity": 1.0 - edge_distance,
            "order_chaos_ratio": order_chaos_ratio,
            "phi_alignment": phi_alignment
        }
        
        return complexity_metrics


# Example usage
if __name__ == "__main__":
    # Create a soul spark
    try:
        from stage_1.soul_spark.soul_spark import SoulSpark
        soul = SoulSpark()
    except ImportError:
        # Create a simple test object if SoulSpark is not available
        class TestSoul:
            def __init__(self):
                self.stability = 0.7
                self.coherence = 0.6
        soul = TestSoul()
    
    # Set initial properties
    soul.stability = 0.7
    soul.coherence = 0.6
    
    # Initialize edge of chaos manager
    eoc = EdgeOfChaos(soul_spark=soul)
    
    # Calculate initial state
    chaos, order, complexity = eoc.calculate_current_state()
    print(f"Initial state - Chaos: {chaos:.2f}, Order: {order:.2f}, Complexity: {complexity:.2f}")
    
    # Adjust toward edge
    chaos, order, complexity = eoc.adjust_toward_edge(intensity=0.6)
    print(f"After adjustment - Chaos: {chaos:.2f}, Order: {order:.2f}, Complexity: {complexity:.2f}")
    
    # Get detailed metrics
    metrics = eoc.get_complexity_metrics()
    print("\nComplexity Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

